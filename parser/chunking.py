"""
Generates three chunk size variants for FF1 experiments.

Configs:
  small:  ~512 chars / ~170 tokens  | overlap ~100 chars  (cf. Arzideh et al. 2026: 450 chars)
  medium: ~1536 chars / ~512 tokens | overlap ~384 chars  (cf. Gao et al. 2024: 512 tokens)
  large:  ~3072 chars / ~1024 tokens| overlap ~768 chars  (cf. arXiv:2505.21700: 512-1024 tokens)

Usage:
  python chunking_variants.py
"""

from pathlib import Path
import json
import uuid

INPUT_DIR  = Path(r"C:\Masterarbeit\App\parser\parsed_clean")
OUTPUT_DIR = Path(r"C:\Masterarbeit\App\parser")

CONFIGS = [
    {
        "name":       "small",
        "chunk_size": 512,
        "overlap":    100,
        "min_chunk":  50,
        "output":     OUTPUT_DIR / "chunks_small.jsonl",
    },
    {
        "name":       "medium",
        "chunk_size": 1536,
        "overlap":    384,
        "min_chunk":  100,
        "output":     OUTPUT_DIR / "chunks_medium.jsonl",
    },
    {
        "name":       "large",
        "chunk_size": 3072,
        "overlap":    768,
        "min_chunk":  200,
        "output":     OUTPUT_DIR / "chunks_large.jsonl",
    },
]


def split_into_chunks(content: str, chunk_size: int, overlap: int, min_chunk: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        if len(chunk) >= min_chunk:
            chunks.append(chunk)
        start = start + chunk_size - overlap
    return chunks


def run_config(cfg: dict, md_files: list) -> int:
    total_chunks = 0
    with cfg["output"].open("w", encoding="utf-8") as out:
        for f in md_files:
            content = f.read_text(encoding="utf-8")
            chunks = split_into_chunks(
                content, cfg["chunk_size"], cfg["overlap"], cfg["min_chunk"]
            )
            for i, chunk in enumerate(chunks):
                record = {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        "source":      f.stem,
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "chunk_config": cfg["name"],
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
    return total_chunks


def main():
    md_files = list(INPUT_DIR.glob("*.md"))
    print(f"Found {len(md_files)} files in {INPUT_DIR}\n")

    for cfg in CONFIGS:
        print(f"[{cfg['name'].upper()}] chunk={cfg['chunk_size']} chars, overlap={cfg['overlap']} chars")
        total = run_config(cfg, md_files)
        print(f"  -> {total} chunks saved to {cfg['output'].name}\n")

    print("Done. Three chunk files ready for embeddings.")


if __name__ == "__main__":
    main()