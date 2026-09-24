from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DB_PATH ="croma_db"
EMBEDDING_MODEL = "nomic-embed-text"

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

vector_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

question = "How many casual leaves are available?"

results = vector_db.similarity_search(
    question,
    k=3
)

print("QUESTION:")
print(question)

for i, doc in enumerate(results):
    print(f"\n--- RESULT {i + 1} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)
