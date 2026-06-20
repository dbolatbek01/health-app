"""
Creates 9 ChromaDB collections: 3 chunk configs x 3 embedding models.

Collection naming: {config}_{model}
  e.g. small_bge_m3, medium_e5_large, large_granite

Usage:
  python setup_chroma.py                  # all 9 collections
  python setup_chroma.py --config medium  # only medium (3 collections)
"""

from pathlib import Path
import numpy
import chromadb
import json
import argparse

CHUNKS_DIR   = Path(r"C:\Masterarbeit\App\parser")
EMBEDDINGS_DIR = Path(r"C:\Masterarbeit\App\embeddings\models")
CHROMA_DIR   = Path(r"C:\Masterarbeit\App\chromadb")

CHUNK_CONFIGS = ["small", "medium", "large"]

MODELS = [
    ("bge_m3",    "embeddings_bge_m3.npz"),
    ("e5_large",  "embeddings_e5_large.npz"),
    ("granite",   "embeddings_granite.npz"),
]


def load_chunks(jsonl_path: Path) -> tuple[list, list, list]:
    ids, texts, metadatas = [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d["id"])
            texts.append(d["text"])
            metadatas.append(d["metadata"])
    return ids, texts, metadatas


def run_config(config: str, client: chromadb.PersistentClient):
    chunks_path = CHUNKS_DIR / f"chunks_{config}.jsonl"
    if not chunks_path.exists():
        print(f"  WARNING: {chunks_path} not found, skipping")
        return

    print(f"\n{'='*60}")
    print(f"Config: {config.upper()}")
    print(f"{'='*60}")

    ids, texts, metadatas = load_chunks(chunks_path)
    print(f"Loaded {len(ids)} chunks")

    for model_key, emb_filename in MODELS:
        collection_name = f"{config}_{model_key}"
        emb_path = EMBEDDINGS_DIR / config / emb_filename

        if not emb_path.exists():
            print(f"  [{collection_name}] WARNING: {emb_path} not found, skipping")
            continue

        print(f"  [{collection_name}] loading embeddings...")

        data = numpy.load(emb_path, allow_pickle=True)
        embeddings = data["embeddings"].tolist()

        # Delete existing collection if it exists (fresh rebuild)
        try:
            client.delete_collection(collection_name)
            print(f"  [{collection_name}] deleted existing collection")
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Add in batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
            )

        print(f"  [{collection_name}] done — {len(ids)} vectors indexed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=CHUNK_CONFIGS,
        default=None,
        help="Run only one config (default: all three)"
    )
    args = parser.parse_args()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    configs = [args.config] if args.config else CHUNK_CONFIGS

    for config in configs:
        run_config(config, client)

    print(f"\nDone. Collections in DB:")
    for col in client.list_collections():
        print(f"  {col.name}")


if __name__ == "__main__":
    main()