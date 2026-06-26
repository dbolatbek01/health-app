from pathlib import Path
import chromadb
import requests
import json
import time
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(r"C:\Masterarbeit\App\chromadb")
EMBED_MODEL = "BAAI/bge-m3"
COLLECTION = "small_bge_m3"
OLLAMA_URL = "http://localhost:11434/api/generate"
QA = Path(r"C:\Masterarbeit\App\QA\test_set_final.json")
RESULT_DIR = Path(r"C:\Masterarbeit\App\generation\results")
TOP_K = 5

MODELS = [
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:7b",
    "meditron:7b",
    "hf.co/BioMistral/BioMistral-7B-GGUF:Q4_K_M",
]

def build_prompt(question, chunks):
    context = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(chunks)])
    return f"""Du bist ein medizinischer Assistent für klinische SOPs am DHZC.
Beantworte die Frage ausschließlich auf Basis der folgenden Dokumente.
Wenn die Antwort nicht im Kontext enthalten ist, antworte: "Diese Information ist in den SOPs nicht vorhanden."
Antworte auf Deutsch.

Kontext:
{context}

Frage: {question}
Antwort:"""

def run():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION)
    embed_model = SentenceTransformer(EMBED_MODEL, device="cuda")

    with open(QA, "r", encoding="utf-8") as file:
        questions = json.load(file)

    for model in MODELS:
        results = []
        print(f"{model}" + " - Starte Evaluation...")

        for item in questions["questions"]:
            question = item["question"]
            print(f"  [{item['id']}/50] {question[:50]}...")
            question_embedding = embed_model.encode(question).tolist()

            query_result = collection.query(
                query_embeddings=[question_embedding],
                n_results = TOP_K
            )
            chunks = query_result["documents"][0]

            prompt = build_prompt(question, chunks)

            response = requests.post(OLLAMA_URL, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 512}
            })
            data = response.json()
            if "error" in data:
                print(f"  FEHLER: {data['error']}")
                answer = "FEHLER: Modell nicht verfügbar"
            else:
                answer = data["response"]

            results.append({
                "id": item["id"],
                "question": question,
                "reference_answer": item["reference_answer"],
                "answer": answer,
                "sources": query_result["metadatas"][0],
                "chunks": query_result["documents"][0]
            })

        output_file = RESULT_DIR / f"{model.replace(':', '_').replace('/', '_')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Gespeichert: " + f"{output_file}")

if __name__ == "__main__":
    run() 