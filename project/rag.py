from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

DB_PATH = "D:\Think_quotient\Gen-AI\chroma_db"
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

# 1. Embedding model
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# 2. Load existing ChromaDB
vector_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

# 3. Create retriever
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# 4. Local LLM
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0
)

# 5. Prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a company policy assistant.

Answer the question using ONLY the context provided below.

If the answer cannot be found in the context, say:

"I could not find this information in the provided documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)

# 6. Ask a question
question = "How many casual leaves are available?"

# 7. Retrieve relevant chunks
documents = retriever.invoke(question)

# 8. Combine retrieved chunks
context = "\n\n".join(
    doc.page_content
    for doc in documents
)

# 9. Build prompt
messages = prompt.invoke({
    "context": context,
    "question": question
})

# 10. Generate answer
response = llm.invoke(messages)

print("\nANSWER:")
print(response.content)

print("\nSOURCES:")
for doc in documents:
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "Unknown")
    print(f"- {source}, page {page}")
