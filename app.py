import json
import os
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "uploads"))
    VECTORSTORE_PATH = Path(os.getenv("VECTORSTORE_PATH", "vectorstore/faiss_index"))
    KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "data"))
    PROMPT_FILE_PATH = Path(os.getenv("PROMPT_FILE_PATH", "prompts/scrum_master_prompt.json"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.75))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10 MB
    MAX_FILE_CONTEXT_CHARS = int(os.getenv("MAX_FILE_CONTEXT_CHARS", 15000))
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Initialize Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_FILE_SIZE
Config.UPLOAD_FOLDER.mkdir(exist_ok=True)

# Global state
class AppState:
    def __init__(self):
        self.llm = None
        self.retriever = None
        self.rag_chain = None
        self.system_prompt_rag = None
        self.system_prompt_no_rag = None

    def initialize(self):
        """Initialize the LLM, retriever, and RAG chain."""
        self.llm = load_llm()
        self.system_prompt_rag, self.system_prompt_no_rag = load_prompt_from_json(Config.PROMPT_FILE_PATH)
        self.retriever = create_retriever(Config.KNOWLEDGE_BASE_DIR)
        if self.retriever and self.llm:
            self.rag_chain = create_rag_chain(self.llm, self.retriever, self.system_prompt_rag)
            logger.info("Agente de IA com RAG está pronto.")
        else:
            logger.warning("Retriever ou LLM não disponível. O chat funcionará sem acesso a documentos.")

state = AppState()

def load_llm() -> Optional[ChatGoogleGenerativeAI]:
    """Load and configure the Google Gemini language model."""
    google_api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "gemini-1.5-flash-latest")

    if not google_api_key:
        logger.error("GEMINI_API_KEY não está definida.")
        return None

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=Config.LLM_TEMPERATURE,
            google_api_key=google_api_key,
        )
        test_response = llm.invoke("Teste de conexão - responda 'OK'")
        if "OK" in test_response.content:
            logger.info("Conexão com Gemini API: OK")
            return llm
        logger.error("Teste de conexão com Gemini falhou")
        return None
    except Exception as e:
        logger.error(f"Falha ao conectar com Gemini API: {e}")
        return None

def load_prompt_from_json(file_path: Path) -> tuple[str, str]:
    """Load the prompt template and guidelines from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_config = json.load(f)
        guidelines_str = "\n".join([f"- {g}" for g in prompt_config.get("guidelines", [])])
        system_prompt_rag = (
            f"{prompt_config.get('system_prompt', 'Você é um assistente prestativo.')}\n\n"
            f"**Diretrizes:**\n{guidelines_str}\n\n"
            f"**Contexto Fornecido:**\n{{context}}"
        )
        system_prompt_no_rag = (
            f"{prompt_config.get('system_prompt', 'Você é um assistente prestativo.')}\n\n"
            f"**Diretrizes:**\n{guidelines_str}"
        )
        return system_prompt_rag, system_prompt_no_rag
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Não foi possível carregar o arquivo de prompt: {e}")
        return (
            "Você é um assistente prestativo. Use o contexto fornecido para responder. Contexto: {context}",
            "Você é um assistente prestativo.",
        )

def load_documents_from_folder(folder_path: Path) -> List[Document]:
    """Extract content from PDF and Markdown files in a folder."""
    if not folder_path.is_dir():
        logger.warning(f"Diretório '{folder_path}' não foi encontrado.")
        return []

    all_files = list(folder_path.glob("*.pdf")) + list(folder_path.glob("*.md"))
    if not all_files:
        logger.warning(f"Nenhum arquivo PDF ou Markdown encontrado em '{folder_path}'.")
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
            logger.error(f"Falha ao carregar o arquivo {file_path.name}: {e}")
    logger.info(f"Carregados {len(all_docs)} documentos de {len(all_files)} arquivos.")
    return all_docs

def process_uploaded_file(file_path: Path) -> List[Document]:
    """Load and extract text from a PDF, TXT, or MD file."""
    if file_path.suffix.lower() not in Config.ALLOWED_EXTENSIONS:
        logger.warning(f"Tipo de arquivo não suportado: {file_path.suffix}")
        return []
    try:
        if file_path.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()
    except Exception as e:
        logger.error(f"Falha ao processar o arquivo {file_path.name}: {e}")
        return []

def create_retriever(folder_path: Path):
    """Create or load a retriever from documents."""
    if Config.VECTORSTORE_PATH.exists():
        try:
            embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            vectorstore = FAISS.load_local(Config.VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Vectorstore carregado de {Config.VECTORSTORE_PATH}")
            return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 7})
        except Exception as e:
            logger.error(f"Falha ao carregar o vectorstore: {e}")

    loaded_documents = load_documents_from_folder(folder_path)
    if not loaded_documents:
        logger.warning("Nenhum documento carregado para criar o retriever.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_documents(loaded_documents)
    logger.info(f"Gerados {len(chunks)} chunks para embedding.")

    try:
        embeddings = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        Config.VECTORSTORE_PATH.parent.mkdir(exist_ok=True)
        vectorstore.save_local(Config.VECTORSTORE_PATH)
        logger.info(f"Vectorstore salvo em {Config.VECTORSTORE_PATH}")
        return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 7})
    except Exception as e:
        logger.error(f"Falha ao criar o VectorStore ou Retriever: {e}")
        return None

def create_rag_chain(llm, retriever, system_prompt_template_rag):
    """Create the RAG chain for document-based responses."""
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

# Initialize application state
state.initialize()

# Flask routes
@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests with optional file uploads."""
    user_input = request.form.get("message", "").strip()
    chat_history_json_str = request.form.get("history", "[]")
    uploaded_file = request.files.get("file")

    logger.debug(f"Recebida mensagem: '{user_input}'")
    logger.debug(f"Arquivo recebido: {uploaded_file.filename if uploaded_file else 'Nenhum'}")

    if not user_input and (not uploaded_file or uploaded_file.filename == ""):
        return jsonify({"error": "A mensagem e o arquivo estão vazios"}), 400

    try:
        chat_history = [
            HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            for msg in json.loads(chat_history_json_str)
        ]
        logger.debug(f"Histórico carregado: {len(chat_history)} mensagens")
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return jsonify({"error": "Formato de histórico inválido"}), 400

    try:
        ai_response = "Desculpe, não consigo processar o seu pedido neste momento."

        if state.llm is None:
            logger.error("LLM não está disponível")
            ai_response = "O serviço de IA não está disponível no momento."
            return jsonify({"response": ai_response})

        if uploaded_file and uploaded_file.filename != "":
            if uploaded_file.filename.rsplit(".", 1)[-1].lower() not in [ext.lstrip(".") for ext in Config.ALLOWED_EXTENSIONS]:
                logger.warning(f"Tipo de arquivo não permitido: {uploaded_file.filename}")
                return jsonify({"error": f"Tipo de arquivo não suportado. Use: {', '.join(Config.ALLOWED_EXTENSIONS)}"}), 400

            filename = secure_filename(uploaded_file.filename)
            file_path = Config.UPLOAD_FOLDER / filename
            uploaded_file.save(file_path)

            logger.info(f"Arquivo '{filename}' recebido. A processar...")
            file_docs = process_uploaded_file(file_path)

            try:
                if not file_docs:
                    ai_response = f"Não consegui extrair conteúdo do arquivo '{filename}'."
                else:
                    file_content_for_prompt = "\n\n---\n\n".join([doc.page_content for doc in file_docs])
                    if len(file_content_for_prompt) > Config.MAX_FILE_CONTEXT_CHARS:
                        file_content_for_prompt = file_content_for_prompt[:Config.MAX_FILE_CONTEXT_CHARS] + "\n\n[...conteúdo truncado...]"

                    prompt_with_file_context = ChatPromptTemplate.from_messages([
                        SystemMessage(content=f"{state.system_prompt_no_rag}\n\n**Analisa o seguinte contexto do arquivo anexo ({filename}):**\n{file_content_for_prompt}"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessage(content="{input}")
                    ])

                    chain = prompt_with_file_context | state.llm
                    response = chain.invoke({
                        "chat_history": chat_history,
                        "input": user_input if user_input else f"Faz um resumo do conteúdo deste arquivo: {filename}."
                    })
                    ai_response = response.content
            finally:
                try:
                    os.remove(file_path)
                    logger.info(f"Arquivo '{file_path}' removido.")
                except Exception as e:
                    logger.error(f"Falha ao remover o arquivo {file_path}: {e}")

        elif state.rag_chain:
            logger.debug("Usando RAG_CHAIN para resposta")
            response = state.rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            ai_response = response.get("answer", "Erro ao processar com a base de conhecimento.")
            logger.debug(f"Resposta RAG: {ai_response[:100]}...")
        elif state.llm:
            logger.debug("Usando LLM direto para resposta")
            messages = [SystemMessage(content=state.system_prompt_no_rag)] + chat_history + [HumanMessage(content=user_input)]
            response = state.llm.invoke(messages)
            ai_response = response.content
            logger.debug(f"Resposta LLM: {ai_response[:100]}...")
        else:
            ai_response = "O modelo de IA não está disponível."

    except Exception as e:
        logger.error(f"Erro ao invocar a cadeia de IA: {str(e)}")
        ai_response = "Ocorreu um erro interno. Tente novamente mais tarde."

    return jsonify({"response": ai_response})

@app.route("/reload_knowledge", methods=["POST"])
def reload_knowledge():
    """Reload the knowledge base."""
    try:
        if Config.VECTORSTORE_PATH.exists():
            Config.VECTORSTORE_PATH.unlink()
            logger.info("Vectorstore antigo removido.")

        state.retriever = create_retriever(Config.KNOWLEDGE_BASE_DIR)
        if state.retriever is None:
            return jsonify({"status": "error", "message": "Falha ao recriar o retriever"}), 500

        state.rag_chain = create_rag_chain(state.llm, state.retriever, state.system_prompt_rag)
        return jsonify({"status": "success", "message": "Base de conhecimento recarregada com sucesso"})

    except Exception as e:
        logger.error(f"Erro ao recarregar: {str(e)}")
        return jsonify({"status": "error", "message": f"Erro ao recarregar: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Check the system health."""
    return jsonify({
        "llm_available": state.llm is not None,
        "rag_available": state.rag_chain is not None,
        "retriever_available": state.retriever is not None,
        "knowledge_base_exists": Config.KNOWLEDGE_BASE_DIR.exists() and any(Config.KNOWLEDGE_BASE_DIR.iterdir())
    })

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    port = int(os.getenv("FLASK_PORT", 5000))
    app.run(debug=debug_mode, port=port, host="0.0.0.0")