import streamlit as st

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

DB_PATH = "D:\Think_quotient\Gen-AI\chroma_db"
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

st.set_page_config(
    page_title="Company Policy Chatbot",
    page_icon="R"
)

st.title("Company Policy Chatbot")
st.write(
    "Ask questions about the company policy document. "
    "Answers are generated from the indexed document using a local LLM."
)

@st.cache_resource
def load_rag_components():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0
    )

    return retriever, llm

try:
    retriever, llm = load_rag_components()
except Exception as e:
    st.error("Could not load the RAG components.")
    st.exception(e)
    st.stop()

prompt = ChatPromptTemplate.from_template(
    """
You are a company policy assistant.

Use ONLY the information provided in the context.

If the answer is not available in the context, say:

"I could not find this information in the provided documents."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)

question = st.text_input(
    "Enter your question:",
    placeholder="Example: How many casual leaves are available?"
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            try:
                documents = retriever.invoke(question)

                context = "\n\n".join(
                    doc.page_content
                    for doc in documents
                )

                messages = prompt.invoke({
                    "context": context,
                    "question": question
                })

                response = llm.invoke(messages)

                st.subheader("Answer")
                st.write(response.content)

                st.subheader("Retrieved Sources")

                for i, doc in enumerate(documents):
                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                    page = doc.metadata.get(
                        "page",
                        "Unknown"
                    )

                    with st.expander(
                        f"Source {i + 1}: {source} | Page {page}"
                    ):
                        st.write(doc.page_content)

            except Exception as e:
                st.error("An error occurred while processing the question.")
                st.exception(e)
