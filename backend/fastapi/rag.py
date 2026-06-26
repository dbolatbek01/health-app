import config
import chromadb
import requests
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
collection = client.get_collection(config.COLLECTION)
embed_model = SentenceTransformer(config.EMBED_MODEL, device="cuda")

def embed_query(query):
    query_embedding = embed_model.encode(query).tolist()
    return query_embedding

def retrieve(query_embedding, top_k=config.TOP_K):
    query_result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return query_result
    
def build_prompt(query, query_result):
    context = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(query_result["documents"][0])])
    return f"""Du bist ein medizinischer Assistent für klinische SOPs am DHZC.
Beantworte die Frage ausschließlich auf Basis der folgenden Dokumente.
Wenn die Antwort nicht im Kontext enthalten ist, antworte: "Diese Information ist in den SOPs nicht vorhanden."
Antworte auf Deutsch.

Kontext:
{context}

Frage: {query}
Antwort:"""

def generate_answer(prompt, model=config.LLM_MODEL):
    response = requests.post(config.OLLAMA_URL + "/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 512}
    })
    data = response.json()
    if "error" in data:
        print(f"  FEHLER: {data['error']}")
        return "FEHLER: Modell nicht verfügbar"
    else:
        return data["response"]

def answer_query(query, model=config.LLM_MODEL):
    query_embedding = embed_query(query)
    query_result = retrieve(query_embedding)
    prompt = build_prompt(query, query_result)
    answer = generate_answer(prompt, model)

    sources = []

    for text, meta in zip(query_result["documents"][0], query_result["metadatas"][0]):
        sources.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"]
        })

    return answer, sources