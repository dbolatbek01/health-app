from pathlib import Path
import json
import numpy
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path(r"C:\Masterarbeit\App\parser\chunks.jsonl") 
OUTPUT_DIR = Path(r"C:\Masterarbeit\App\embeddings\models")

MODELS = [
    ("BAAI/bge-m3", OUTPUT_DIR / "embeddings_bge_m3.npz"),
    ("intfloat/multilingual-e5-large-instruct", OUTPUT_DIR / "embeddings_e5_large.npz"),
    ("ibm-granite/granite-embedding-311m-multilingual-r2", OUTPUT_DIR / "embeddings_granite.npz"),
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def split_into_vectors():
    vectors_list = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        for line in file:
            vectors_list.append(json.loads(line))

    texts = []
    for chunk in vectors_list:
        texts.append(chunk["text"])

    for model_name, output_file in MODELS:
        if Path(output_file).exists():
            print(f"{output_file} already exists, skipping...")
            continue

        model = SentenceTransformer(model_name, device="cuda")

        embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        )

        ids = [chunk["id"] for chunk in vectors_list]

        numpy.savez(output_file, embeddings=embeddings, ids=ids)
        print("Saved")

        del model
   
def run():
    split_into_vectors()

if __name__ == "__main__":
    run()