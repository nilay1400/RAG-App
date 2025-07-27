from qdrant_client import QdrantClient

qdrant = QdrantClient(host="localhost", port=6333)
collection_names = [c.name for c in qdrant.get_collections().collections]

print(collection_names)
