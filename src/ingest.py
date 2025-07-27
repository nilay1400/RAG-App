import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import tiktoken
import uuid
from utils.cleaning import extract_text_from_pdf

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)
qdrant = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "papers"

def create_collection():
    if COLLECTION_NAME not in qdrant.get_collections().collections:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

def get_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-3-small")
    return response.data[0].embedding

def tokenize(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    return enc.encode(text)

def chunk_text(text, chunk_size=1000, overlap=200):

    enc = tiktoken.encoding_for_model("gpt-4")

    tokens = enc.encode(text)

    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):

        chunk = tokens[i:i + chunk_size]

        decoded = enc.decode(chunk) 

        chunks.append(decoded)

    return chunks



def ingest_pdfs():
    create_collection()
    for filename in os.listdir("pdfs"):
        if filename.endswith(".pdf"):
            path = os.path.join("pdfs", filename)
            raw_text = extract_text_from_pdf(path)
            chunks = chunk_text(raw_text)
            points = []

            for i, chunk in enumerate(chunks):
                vector = get_embedding(chunk)
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "filename": filename,
                        "chunk_index": i,
                        "text": chunk,
                        "source": filename  
                    },
                ))

            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"✔ Ingested {len(points)} chunks from {filename}")
            visualize_tsne(points)



from sklearn.manifold import TSNE

import numpy as np

import matplotlib.pyplot as plt



def visualize_tsne(points):

    vectors = [point.vector for point in points]

    if len(vectors) < 3:

        print("Not enough points for t-SNE visualization.")

        return



    tsne = TSNE(n_components=2, perplexity=min(5, len(vectors)-1), random_state=42)

    reduced = tsne.fit_transform(np.array(vectors))



    plt.figure(figsize=(10, 8))

    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)

    plt.title("t-SNE of Embedded Chunks")

    plt.savefig("tsne_visualization.png")

    print("t-SNE visualization saved as tsne_visualization.png")




if __name__ == "__main__":
    ingest_pdfs()
