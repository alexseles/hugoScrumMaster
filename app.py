import json
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains import create_stuff_documents_chain
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da aplicação Flask
app = Flask(__name__)
UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
VECTORSTORE_PATH = Path("vectorstore/faiss_index.pkl")
KNOWLEDGE_BASE_DIR = Path("data")
PROMPT_FILE_PATH = Path("prompts/scrum_master_prompt.json")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.75))

# Função para carregar o LLM
def load_llm() -> Optional[ChatGoogleGenerativeAI]:
    """Carrega e configura o modelo de linguagem do Google Gemini."""
    google_api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-latest")

    if not google_api_key:
        print("ERRO: A variável de ambiente GEMINI_API_KEY não está definida.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=TEMPERATURE,
            google_api_key=google_api_key,
        )
        test_response = llm.invoke("Teste de conexão - responda 'OK'")
        if "OK" in test_response.content:
            print("Conexão com Gemini API: OK")
            return llm
        print("ERRO: Teste de conexão com Gemini falhou")
        return None
    except Exception as e:
        print(f"ERRO: Falha ao conectar com Gemini API: {e}")
        return None

# Função para carregar o prompt
def load_prompt_from_json(file_path: Path) -> tuple[str, str]:
    """Carrega o template do prompt e as diretrizes de um arquivo JSON."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_config = json.load(f)

        guidelines_str = "\n".join([f"- {g}" for g in prompt_config["guidelines"]])
        system_prompt_rag = (
            f"{prompt_config['system_prompt']}\n\n"
            f"**Diretrizes:**\n{guidelines_str}\n\n"
            f"**Contexto Fornecido:**\n{{context}}"
        )
        system_prompt_no_rag = (
            f"{prompt_config['system_prompt']}\n\n"
            f"**Diretrizes:**\n{guidelines_str}"
        )
        return system_prompt_rag, system_prompt_no_rag
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERRO: Não foi possível carregar o arquivo de prompt: {e}")
        return (
            "Você é um assistente prestativo. Use o contexto fornecido para responder. Contexto: {context}",
            "Você é um assistente prestativo.",
        )

# Função para carregar documentos
def load_documents_from_folder(folder_path: Path) -> List[Document]:
    """Extrai o conteúdo de arquivos PDF e Markdown em uma pasta."""
    if not folder_path.is_dir():
        print(f"AVISO: O diretório '{folder_path}' não foi encontrado.")
        return []

    all_files = list(folder_path.glob("*.pdf")) + list(folder_path.glob("*.md"))
    if not all_files:
        print(f"AVISO: Nenhum arquivo PDF ou Markdown encontrado em '{folder_path}'.")
        return []

    all_docs = []
    for file_path in all_files:
        try:
            loader = PyMuPDFLoader(str(file_path)) if file_path.suffix.lower() == ".pdf" else TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
            all_docs.extend(docs)
        except Exception as e:
            print(f"ERRO: Falha ao carregar o arquivo {file_path.name}: {e}")
    print(f"Carregados {len(all_docs)} documentos de {len(all_files)} arquivos.")
    return all_docs

# Função para processar arquivo carregado
def process_uploaded_file(file_path: Path) -> List[Document]:
    """Carrega e extrai texto de um arquivo PDF, TXT ou MD."""
    try:
        if file_path.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(str(file_path))
        elif file_path.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            print(f"AVISO: Tipo de arquivo não suportado: {file_path.suffix}")
            return []
        return loader.load()
    except Exception as e:
        print(f"ERRO: Falha ao processar o arquivo {file_path.name}: {e}")
        return []

# Função para criar o retriever
def create_retriever(folder_path: Path):
    """Cria ou carrega um retriever a partir dos documentos."""
    if VECTORSTORE_PATH.exists():
        try:
            with open(VECTORSTORE_PATH, "rb") as f:
                vectorstore = pickle.load(f)
            print(f"Vectorstore carregado de {VECTORSTORE_PATH}")
            return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 7})
        except Exception as e:
            print(f"ERRO: Falha ao carregar o vectorstore: {e}")

    loaded_documents = load_documents_from_folder(folder_path)
    if not loaded_documents:
        print("AVISO: Nenhum documento carregado para criar o retriever.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_documents(loaded_documents)
    print(f"Gerados {len(chunks)} chunks para embedding.")

    try:
        embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        VECTORSTORE_PATH.parent.mkdir(exist_ok=True)
        with open(VECTORSTORE_PATH, "wb") as f:
            pickle.dump(vectorstore, f)
        print(f"Vectorstore salvo em {VECTORSTORE_PATH}")
        return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 7})
    except Exception as e:
        print(f"ERRO: Falha ao criar o VectorStore ou Retriever: {e}")
        return None

# Função para criar a cadeia RAG
def create_rag_chain(llm, retriever, system_prompt_template_rag):
    """Cria a cadeia de RAG para respostas baseadas em documentos."""
    context_q_system_prompt = (
        "Dada uma conversa e uma nova pergunta, reformule a pergunta para que seja independente "
        "e possa ser compreendida sem o histórico do chat. NÃO responda à pergunta, apenas a reformule "
        "se necessário; caso contrário, devolva-a como está."
    )
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template_rag),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Carregamento global
print("Iniciando o servidor e configurando o agente de IA...")
try:
    LLM = load_llm()
    SYSTEM_PROMPT_RAG, SYSTEM_PROMPT_NO_RAG = load_prompt_from_json(PROMPT_FILE_PATH)
    RETRIEVER = create_retriever(KNOWLEDGE_BASE_DIR)

    if RETRIEVER is None:
        print("AVISO: O Retriever não foi criado. O chat funcionará sem acesso a documentos.")
        RAG_CHAIN = None
    else:
        RAG_CHAIN = create_rag_chain(LLM, RETRIEVER, SYSTEM_PROMPT_RAG)
        print("Agente de IA com RAG está pronto.")

except Exception as e:
    print(f"ERRO CRÍTICO na inicialização do agente de IA: {e}")
    LLM = None
    RAG_CHAIN = None
    SYSTEM_PROMPT_RAG, SYSTEM_PROMPT_NO_RAG = (
        "Você é um assistente.",
        "Você é um assistente."
    )

# Rotas da aplicação
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("message", "").strip()
    chat_history_json_str = request.form.get("history", "[]")
    uploaded_file = request.files.get("file")

    print(f"DEBUG: Recebida mensagem: '{user_input}'")
    print(f"DEBUG: Arquivo recebido: {uploaded_file.filename if uploaded_file else 'Nenhum'}")

    if not user_input and (not uploaded_file or uploaded_file.filename == ""):
        return jsonify({"error": "A mensagem e o arquivo estão vazios"}), 400

    try:
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in json.loads(chat_history_json_str)
        ]
        print(f"DEBUG: Histórico carregado: {len(chat_history)} mensagens")
    except json.JSONDecodeError as e:
        print(f"ERRO: JSON decode error: {e}")
        return jsonify({"error": "Formato de histórico inválido"}), 400

    try:
        ai_response = "Desculpe, não consigo processar o seu pedido neste momento."

        if LLM is None:
            print("ERRO: LLM não está disponível")
            ai_response = "O serviço de IA não está disponível no momento."
            return jsonify({"response": ai_response})

        if uploaded_file and uploaded_file.filename != "":
            filename = secure_filename(uploaded_file.filename)
            file_path = UPLOAD_FOLDER / filename
            uploaded_file.save(file_path)

            print(f"Arquivo '{filename}' recebido. A processar...")
            file_docs = process_uploaded_file(file_path)

            if not file_docs:
                ai_response = f"Não consegui extrair conteúdo do arquivo '{filename}'."
            else:
                file_content_for_prompt = "\n\n---\n\n".join([doc.page_content for doc in file_docs])
                max_file_context_chars = 15000
                if len(file_content_for_prompt) > max_file_context_chars:
                    file_content_for_prompt = file_content_for_prompt[:max_file_context_chars] + "\n\n[...conteúdo truncado...]"

                prompt_with_file_context = ChatPromptTemplate.from_messages([
                    SystemMessage(content=f"{SYSTEM_PROMPT_NO_RAG}\n\n**Analisa o seguinte contexto do arquivo anexo ({filename}):**\n{file_content_for_prompt}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessage(content="{input}")
                ])

                chain = prompt_with_file_context | LLM
                response = chain.invoke({
                    "chat_history": chat_history,
                    "input": user_input if user_input else f"Faz um resumo do conteúdo deste arquivo: {filename}."
                })
                ai_response = response.content

            os.remove(file_path)

        elif RAG_CHAIN:
            print("DEBUG: Usando RAG_CHAIN para resposta")
            response = RAG_CHAIN.invoke({"input": user_input, "chat_history": chat_history})
            ai_response = response.get("answer", "Erro ao processar com a base de conhecimento.")
            print(f"DEBUG: Resposta RAG: {ai_response[:100]}...")
        elif LLM:
            print("DEBUG: Usando LLM direto para resposta")
            messages = [SystemMessage(content=SYSTEM_PROMPT_NO_RAG)] + chat_history + [HumanMessage(content=user_input)]
            response = LLM.invoke(messages)
            ai_response = response.content
            print(f"DEBUG: Resposta LLM: {ai_response[:100]}...")
        else:
            ai_response = "O modelo de IA não está disponível."

    except Exception as e:
        print(f"ERRO CRÍTICO ao invocar a cadeia de IA: {str(e)}")
        ai_response = "Ocorreu um erro interno. Tente novamente mais tarde."

    return jsonify({"response": ai_response})

@app.route("/reload_knowledge", methods=["POST"])
def reload_knowledge():
    """Recarrega a base de conhecimento."""
    global RETRIEVER, RAG_CHAIN

    try:
        if VECTORSTORE_PATH.exists():
            VECTORSTORE_PATH.unlink()
            print("Vectorstore antigo removido.")

        RETRIEVER = create_retriever(KNOWLEDGE_BASE_DIR)
        if RETRIEVER is None:
            return jsonify({"status": "error", "message": "Falha ao recriar o retriever"}), 500

        RAG_CHAIN = create_rag_chain(LLM, RETRIEVER, SYSTEM_PROMPT_RAG)
        return jsonify({"status": "success", "message": "Base de conhecimento recarregada com sucesso"})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Erro ao recarregar: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Verifica a saúde do sistema."""
    return jsonify({
        "llm_available": LLM is not None,
        "rag_available": RAG_CHAIN is not None,
        "retriever_available": RETRIEVER is not None,
        "knowledge_base_exists": KNOWLEDGE_BASE_DIR.exists() and any(KNOWLEDGE_BASE_DIR.iterdir())
    })

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(debug=debug_mode, port=port)