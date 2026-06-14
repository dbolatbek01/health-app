from pathlib import Path
import numpy
import chromadb
import json

JSON = Path(r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\parser\chunks.jsonl")
INPUT_DIR = Path(r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\embeddings\models") 
CHROMA_DIR = Path(r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\chromadb")

MODELS = [
    ("sop_bge_m3", INPUT_DIR / "embeddings_bge_m3.npz"),
    ("sop_e5_large", INPUT_DIR / "embeddings_e5_large.npz"),
    ("sop_granite", INPUT_DIR / "embeddings_granite.npz")
]

def get_texts():
    texts = []
    with open(JSON, "r", encoding="utf-8") as file:
        for line in file:
            texts.append(json.loads(line))
    return(texts)

def setup_db():
    chunks = get_texts()
    ids = [chunk["id"] for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    for collection_name, emb_file in MODELS:
        collection = client.get_or_create_collection(
            name = collection_name,
            metadata={"hnsw:space":"cosine"}
        )
        
        data = numpy.load(emb_file, allow_pickle=True)
        embeddings=data["embeddings"].tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents = texts,
            metadatas = metadatas
        )

def run():
    setup_db()

if __name__ == "__main__":
    run()