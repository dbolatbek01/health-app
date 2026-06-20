"""
Recall@k and MRR evaluation across all 9 ChromaDB collections.

Usage:
  python retrieval_eval.py
  python retrieval_eval.py --k 1 3 5 10
  python retrieval_eval.py --collection medium_bge_m3  # single collection
"""

from pathlib import Path
import json
import torch
import argparse
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR   = Path(r"C:\Masterarbeit\App\chromadb")
TEST_SET     = Path(r"C:\Masterarbeit\QA\test_set_final.json")
RESULTS_DIR  = Path(r"C:\Masterarbeit\retrieval\retrieval_results")

COLLECTIONS = [
    ("small_bge_m3",    "BAAI/bge-m3"),
    ("small_e5_large",  "intfloat/multilingual-e5-large-instruct"),
    ("small_granite",   "ibm-granite/granite-embedding-311m-multilingual-r2"),
    ("medium_bge_m3",   "BAAI/bge-m3"),
    ("medium_e5_large", "intfloat/multilingual-e5-large-instruct"),
    ("medium_granite",  "ibm-granite/granite-embedding-311m-multilingual-r2"),
    ("large_bge_m3",    "BAAI/bge-m3"),
    ("large_e5_large",  "intfloat/multilingual-e5-large-instruct"),
    ("large_granite",   "ibm-granite/granite-embedding-311m-multilingual-r2"),
]

# Query prefixes required by some models
QUERY_PREFIXES = {
    "intfloat/multilingual-e5-large-instruct": "query: ",
    "BAAI/bge-m3": "",
    "ibm-granite/granite-embedding-311m-multilingual-r2": "",
}


def load_test_set(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def encode_query(model: SentenceTransformer, model_name: str, question: str):
    prefix = QUERY_PREFIXES.get(model_name, "")
    return model.encode(prefix + question, normalize_embeddings=True).tolist()


def evaluate_collection(
    collection_name: str,
    model_name: str,
    questions: list,
    k_values: list,
    client: chromadb.PersistentClient,
) -> dict:

    collection = client.get_collection(collection_name)
    model = SentenceTransformer(model_name, device="cuda")

    max_k = max(k_values)
    per_question = []

    for q in questions:
        query_emb = encode_query(model, model_name, q["question"])
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=max_k,
            include=["metadatas"],
        )
        retrieved_sources = [m["source"] for m in results["metadatas"][0]]
        ground_truth = q["source_doc"]

        # Find rank of first hit
        rank = None
        for i, src in enumerate(retrieved_sources, 1):
            if src == ground_truth:
                rank = i
                break

        per_question.append({
            "id": q["id"],
            "ground_truth": ground_truth,
            "rank": rank,
            "retrieved_sources": retrieved_sources,
        })

    # Compute metrics
    metrics = {"collection": collection_name}
    for k in k_values:
        hits = sum(1 for r in per_question if r["rank"] is not None and r["rank"] <= k)
        metrics[f"recall@{k}"] = round(hits / len(per_question), 4)

    mrr = sum(
        1 / r["rank"] for r in per_question if r["rank"] is not None
    ) / len(per_question)
    metrics["mrr"] = round(mrr, 4)

    del model
    torch.cuda.empty_cache()
    return metrics, per_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--collection", default=None, help="Evaluate only one collection")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_test_set(TEST_SET)
    print(f"Loaded {len(questions)} questions")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collections = (
        [(c, m) for c, m in COLLECTIONS if c == args.collection]
        if args.collection
        else COLLECTIONS
    )

    all_metrics = []
    for collection_name, model_name in collections:
        print(f"\n[{collection_name}] evaluating...")
        metrics, per_question = evaluate_collection(
            collection_name, model_name, questions, args.k, client
        )
        all_metrics.append(metrics)

        # Save per-question results
        out = RESULTS_DIR / f"{collection_name}_results.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"metrics": metrics, "per_question": per_question}, f, ensure_ascii=False, indent=2)

        k_str = "  ".join(f"R@{k}={metrics[f'recall@{k}']}" for k in args.k)
        print(f"  {k_str}  MRR={metrics['mrr']}")

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    # Print summary table
    print(f"\n{'Collection':<22} " + "  ".join(f"R@{k:<4}" for k in args.k) + "  MRR")
    print("-" * 70)
    for m in all_metrics:
        row = f"{m['collection']:<22} "
        row += "  ".join(f"{m[f'recall@{k}']:.3f}" for k in args.k)
        row += f"  {m['mrr']:.3f}"
        print(row)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()