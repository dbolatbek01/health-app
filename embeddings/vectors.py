"""
Generates embeddings for all three chunk size variants x three embedding models.

Output structure:
  C:\Masterarbeit\App\embeddings\
    small\
      embeddings_bge_m3.npz
      embeddings_e5_large.npz
      embeddings_granite.npz
    medium\
      ...
    large\
      ...

Usage:
  python vectors.py
  python vectors.py --config medium   # only one config
"""

from pathlib import Path
import json
import argparse
import numpy
from sentence_transformers import SentenceTransformer

CHUNKS_DIR  = Path(r"C:\Masterarbeit\App\parser")
EMBEDDINGS_DIR = Path(r"C:\Masterarbeit\App\embeddings")

CHUNK_CONFIGS = {
    "small":  CHUNKS_DIR / "chunks_small.jsonl",
    "medium": CHUNKS_DIR / "chunks_medium.jsonl",
    "large":  CHUNKS_DIR / "chunks_large.jsonl",
}

MODELS = [
    ("BAAI/bge-m3",                                     "embeddings_bge_m3.npz"),
    ("intfloat/multilingual-e5-large-instruct",         "embeddings_e5_large.npz"),
    ("ibm-granite/granite-embedding-311m-multilingual-r2", "embeddings_granite.npz"),
]


def load_chunks(jsonl_path: Path) -> tuple[list[str], list[str]]:
    texts, ids = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
            ids.append(d["id"])
    return texts, ids


def run_config(config_name: str, chunks_path: Path):
    output_dir = EMBEDDINGS_DIR / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Config: {config_name.upper()} | {chunks_path.name}")
    print(f"{'='*60}")

    texts, ids = load_chunks(chunks_path)
    print(f"Loaded {len(texts)} chunks")

    for model_name, output_filename in MODELS:
        output_file = output_dir / output_filename

        if output_file.exists():
            print(f"  [{model_name}] already exists, skipping...")
            continue

        print(f"  [{model_name}] encoding {len(texts)} chunks...")

        model = SentenceTransformer(model_name, device="cuda")

        embeddings = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        numpy.savez(output_file, embeddings=embeddings, ids=ids)
        print(f"  [{model_name}] saved -> {output_file.name}")

        del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["small", "medium", "large"],
        default=None,
        help="Run only one config (default: all three)"
    )
    args = parser.parse_args()

    configs = (
        {args.config: CHUNK_CONFIGS[args.config]}
        if args.config
        else CHUNK_CONFIGS
    )

    for name, path in configs.items():
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        run_config(name, path)

    print("\nDone.")


if __name__ == "__main__":
    main()