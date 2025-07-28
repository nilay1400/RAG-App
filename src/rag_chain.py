from qdrant_client import QdrantClient
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
#qdrant = QdrantClient(host="localhost", port=6333)
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

bibtex_lookup = {
    "TDMR_24___AdAM.pdf": "Taheri, M., Cherezova, N., Nazari, S., Azarpeyvand, A., Ghasempouri, T., Daneshtalab, M. and Raik, J., 2024. Adam: Adaptive approximate multiplier for fault tolerance in dnn accelerators. IEEE Transactions on Device and Materials Reliability.",
    "ANPiler.pdf": "Nazari, S., Azarpeyvand, A. and Ghasempouri, T., 2025, February. A Double Approximate Neural Hardware Accelerator. In 2025 29th International Computer Conference, Computer Society of Iran (CSICC) (pp. 1-5). IEEE.",
    "ATS_25___FORTUNE.pdf":"Nazari, S., Taheri, M., Azarpeyvand, A., Afsharchi, M., Ghasempouri, T., Herglotz, C., Daneshtalab, M. and Jenihhin, M., 2024, December. Fortune: A negative memory overhead hardware-agnostic fault tolerance technique in dnns. In 2024 IEEE 33rd Asian Test Symposium (ATS) (pp. 1-6). IEEE.",
    "Genetic_Edited.pdf":", Nazari, S., Taheri, M., Azarpeyvand, A., Afsharchi, M., Herglotz, C. and Jenihhin, M.. Genie. In 2025 ICCAI. IEEE.",
    "LATS_2025__Genquant.pdf":", Nazari, S., Taheri, M., Azarpeyvand, A., Afsharchi, M., Herglotz, C. and Jenihhin, M., 2025, March. Reliability-aware performance optimization of DNN HW accelerators through heterogeneous quantization. In 2025 IEEE 26th Latin American Test Symposium (LATS) (pp. 1-6). IEEE.",
    "Taltech___Zanjan_DFT24__.pdf":"Parchekani, B., Nazari, S., Ahmadilivani, M.H., Azarpeyvand, A., Raik, J., Ghasempouri, T. and Daneshtalab, M., 2024, October. Zero-Memory-Overhead Clipping-Based Fault Tolerance for LSTM Deep Neural Networks. In 2024 IEEE International Symposium on Defect and Fault Tolerance in VLSI and Nanotechnology Systems (DFT) (pp. 1-4). IEEE.",
}



def retrieve_chunks(query, collection_name="papers", k=5):
    embedding = client.embeddings.create(model="text-embedding-3-small", input=[query])
    query_vector = embedding.data[0].embedding
    results = qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=k
    )
    return results


def build_prompt(question, top_chunks):
    # Extract unique sources and map to BibTeX
    seen_sources = {}
    citations_list = []
    for hit in top_chunks:
        raw_source = hit.payload.get("source", "")
        bibtex_key = bibtex_lookup.get(raw_source, raw_source)
        if bibtex_key not in seen_sources:
            seen_sources[bibtex_key] = len(seen_sources) + 1
            citations_list.append(f"[{seen_sources[bibtex_key]}]: {bibtex_key}")

    # Replace citations in the source text with updated numbers
    sources = "\n".join(
        [f"[{seen_sources[bibtex_lookup.get(hit.payload.get('source', ''), hit.payload.get('source', 'chunk'))]}] {hit.payload.get('text', '')[:500]}..."
        for hit in top_chunks]
    )

    citations = "\n".join(citations_list)

    return f"""
You are a helpful academic assistant. Use the following sources to answer the question.

{sources}

Question: {question}

Answer with citations in [x] format, and list them at the end.

{citations}
""".strip()


def answer_question(question):
    top_chunks = retrieve_chunks(question)
    prompt = build_prompt(question, top_chunks)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Convert to list of plain dicts for the UI to consume safely
    #formatted_sources = [
     #   {
      #      "text": doc.payload.get("text", ""),
      #      "source": doc.payload.get("source", f"chunk {i+1}")
      #  }
      #  for i, doc in enumerate(top_chunks)
    #]

    formatted_sources = [
        {
            "text": doc.payload.get("text", ""),
            "source": bibtex_lookup.get(doc.payload.get("source", ""), doc.payload.get("source", "chunk"))
        }
        for doc in top_chunks
    ]


    return response.choices[0].message.content.strip(), formatted_sources
