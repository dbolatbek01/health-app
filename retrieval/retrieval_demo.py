from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

INPUT_DIR = Path(r"C:\Masterarbeit\App\chromadb")

MODELS = {
    "1": ("sop_bge_m3", "BAAI/bge-m3", "BGE-M3 (multilingual, 1024d)"),
    "2": ("sop_e5_large", "intfloat/multilingual-e5-large-instruct", "E5-Large-Instruct (multilingual, 1024d)"),
    "3": ("sop_granite", "ibm-granite/granite-embedding-311m-multilingual-r2", "Granite-311M (multilingual, 768d)"),
}

def search(query, collection_name, model_name, top_k = 10):  
    client = chromadb.PersistentClient(path=str(INPUT_DIR))

    collection = client.get_collection(collection_name)

    model = SentenceTransformer(model_name, device="cuda")

    if "e5" in model_name.lower():
       query_text = f"query: {query}"
    elif "bge" in model_name.lower():
        query_text = f"Represent this sentence for searching relevant passages: {query}"
    else:
        query_text = query

    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return(results)


def run():
    query = input("Frage: ")

    print("\nVerfügbare Modelle:")
    print("  1. BGE-M3 (multilingual, 1024d)")
    print("  2. E5-Large-Instruct (multilingual, 1024d)")
    print("  3. Granite-311M (multilingual, 768d)")

    choice = input("\nModell (1/2/3): ")

    models = {
        "1": ("sop_bge_m3", "BAAI/bge-m3"),
        "2": ("sop_e5_large", "intfloat/multilingual-e5-large-instruct"),
        "3": ("sop_granite", "ibm-granite/granite-embedding-311m-multilingual-r2"),
    }

    collection_name, model_name = models[choice]
    results = search(query, collection_name, model_name)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n--- Ergebnis {i+1} ---")
        print(f"Quelle: {meta['source']}")
        print(f"Score: {round(1 - dist, 3)}")
        print(f"Text: {doc[:300]}")

if __name__ == "__main__":
    run()

