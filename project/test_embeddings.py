import ollama

MODEL = "nomic-embed-text"

text = "Employees are entitled to 12 casual leaves per year."

response = ollama.embeddings(
    model=MODEL,
    prompt=text
)

embedding = response["embedding"]

print("MODEL:", MODEL)
print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])
