import ollama

MODEL = "llama3.2"

response = ollama.chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in three simple sentences."
        }
    ]
)

print("MODEL:", MODEL)
print("\nRESPONSE:")
print(response["message"]["content"])
