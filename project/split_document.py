from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = r"D:\Think_quotient\Gen-AI\project\document\company_policy.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Original pages:", len(documents))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:10]):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk.page_content)
    print("Metadata:", chunk.metadata)
