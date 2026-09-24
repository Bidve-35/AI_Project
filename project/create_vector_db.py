from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


PDF_PATH = "D:\Think_quotient\Gen-AI\project\document\company_policy.pdf"
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

loader =PyPDFLoader(PDF_PATH)
documents =loader.load()

print("Page loaded :" , len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=50
)

chunks =splitter.split_documents(documents)

print(" Chunks created :",len(chunks))

embeddings = OllamaEmbeddings(
    model= EMBEDDING_MODEL
)

vector_db = Chroma.from_documents(
    documents = chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

print("Vector database created successfully .")
print("Database location : ",DB_PATH)