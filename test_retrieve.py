from src.retrieve import search_qdrant

query = "What is transformer architecture?"
results = search_qdrant(query)

for i, hit in enumerate(results):
    print(f"\nResult {i+1}:")
    print(hit.payload["text"])
