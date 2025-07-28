import os
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração da Aplicação Flask ---
app = Flask(__name__)

# --- Configurações do Modelo e RAG (Adaptado do seu código) ---
ID_MODEL = "llama3-8b-8192"
TEMPERATURE = 0.6
KNOWLEDGE_BASE_DIR = "data" # <-- LINHA ATUALIZADA

# Função para carregar o LLM
def load_llm():
    return ChatGroq(
        model=ID_MODEL,
        temperature=TEMPERATURE,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

# Função para extrair texto dos PDFs
def extract_text_from_pdfs(folder_path):
    docs_path = Path(folder_path)
    pdf_files = list(docs_path.glob("*.pdf"))
    if not pdf_files:
        print(f"AVISO: Nenhum arquivo PDF encontrado na pasta '{folder_path}'. O agente responderá sem contexto.")
        return []
    
    all_docs_content = []
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = pdf_path.name 
            all_docs_content.append(doc)
            
    print(f"Carregados {len(all_docs_content)} páginas de {len(pdf_files)} arquivos PDF.")
    return all_docs_content

# Função para criar e configurar o retriever (base de conhecimento)
def create_retriever(folder_path):
    loaded_documents = extract_text_from_pdfs(folder_path)
    if not loaded_documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(loaded_documents)

    embedding_model = "BAAI/bge-m3"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device': 'cpu'})

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    return vectorstore.as_retriever(
        search_type='mmr',
        search_kwargs={'k': 5, 'fetch_k': 7}
    )

# Função para criar a RAG chain
def create_rag_chain(llm, retriever):
    context_q_system_prompt = """Dada uma conversa e uma pergunta, reformule a pergunta para que ela seja auto-contida e possa ser entendida sem o histórico do chat. NÃO responda à pergunta, apenas a reformule se necessário."""
    
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)
    
    system_prompt = """Você é 'Tomás', um assistente de atendimento ao cliente amigável e prestativo. Use os trechos de contexto fornecidos para responder à pergunta do utilizador, sempre em português de Portugal.
    - Responda sempre em português de Portugal.
    - Se a resposta estiver no contexto, seja direto e use a informação.
    - Se o contexto não contiver a resposta, diga educadamente que não encontrou a informação nos documentos disponíveis, mas tente ajudar com conhecimentos gerais se for apropriado.
    - Mantenha as respostas concisas e claras.

    Contexto:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, qa_chain)

# --- Carregamento Global (executado uma vez no início) ---
print("Iniciando o servidor e configurando o agente de IA...")
LLM = load_llm()
RETRIEVER = create_retriever(KNOWLEDGE_BASE_DIR)
if RETRIEVER is None:
    print("AVISO: O Retriever não foi criado. O chat funcionará sem a base de conhecimento.")
    RAG_CHAIN = None
else:
    RAG_CHAIN = create_rag_chain(LLM, RETRIEVER)
    print("Agente de IA pronto.")


# --- Rotas da Aplicação ---

@app.route("/")
def index():
    """Serve a página principal do chat."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint para receber mensagens e retornar respostas da IA."""
    data = request.json
    user_input = data.get("message")
    chat_history_json = data.get("history", [])

    if not user_input:
        return jsonify({"error": "Mensagem não encontrada"}), 400

    chat_history = []
    for msg in chat_history_json:
        if msg.get("role") == "user":
            chat_history.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "ai":
            chat_history.append(AIMessage(content=msg.get("content")))
            
    if RAG_CHAIN is None:
        response = LLM.invoke(chat_history + [HumanMessage(content=user_input)])
        ai_response = response.content
    else:
        response = RAG_CHAIN.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        ai_response = response.get("answer", "Desculpe, ocorreu um erro ao processar sua solicitação.")

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True, port=5000)