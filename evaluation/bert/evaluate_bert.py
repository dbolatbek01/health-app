"""
Phase 2 Evaluation: BERTScore (primary) + BLEU + ROUGE (baseline)

Input:  JSON result files from generate.py  (list of dicts with keys:
        id, question, reference_answer, answer, sources)
Output: results/eval_bertscore.json  — per-question scores per model
        results/eval_bertscore_summary.json — mean scores per model (for thesis table)

Usage:
    python evaluate_bert.py
"""

import json
from pathlib import Path

import nltk
nltk.download("punkt_tab", quiet=True)

from bert_score import score as bert_score
from rouge_score import rouge_scorer as rouge_lib
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(r"C:\Masterarbeit\App\generation\results")
OUTPUT_DIR  = Path(r"C:\Masterarbeit\App\evaluation\bert\results")

# model_label -> filename
MODEL_FILES = {
    "llama3.1:8b":   "llama3.1_8b.json",
    "mistral:7b":    "mistral_7b.json",
    "qwen2.5:7b":    "qwen2.5_7b.json",
    "meditron:7b":   "meditron_7b.json",
    "BioMistral-7B": "hf.co_BioMistral_BioMistral-7B-GGUF_Q4_K_M.json",
}
 
# BERTScore model — mBERT, stable multilingual model, works well for German
# Used in Arzideh et al. (2026) for German clinical RAG evaluation
BERTSCORE_MODEL = "bert-base-multilingual-cased"
 
# ── helpers ────────────────────────────────────────────────────────────────────
def compute_rouge(hypothesis: str, reference: str) -> dict:
    """Returns ROUGE-1, ROUGE-2, ROUGE-L F1."""
    scorer = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }
 
 
def compute_bleu(hypothesis: str, reference: str) -> float:
    """Sentence-level BLEU-4 with smoothing (method1)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    return round(score, 4)
 
 
def evaluate_model(model_label: str, filepath: Path) -> list[dict]:
    """
    Compute per-question metrics for one model.
    BERTScore is batched (all 50 at once) for speed.
    BLEU + ROUGE are computed per question.
    """
    print(f"\n[{model_label}] loading {filepath.name}")
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
 
    hypotheses = [item["answer"] for item in data]
    references  = [item["reference_answer"] for item in data]
 
    # BERTScore — single batch call, returns tensors with F1 per sample
    print(f"[{model_label}] computing BERTScore ({BERTSCORE_MODEL}) ...")
    P, R, F1 = bert_score(
        hypotheses,
        references,
        model_type=BERTSCORE_MODEL,
        verbose=False,
        device="cuda",      # falls back to CPU automatically if no GPU
    )
 
    results = []
    for i, item in enumerate(data):
        rouge = compute_rouge(hypotheses[i], references[i])
        bleu  = compute_bleu(hypotheses[i], references[i])
 
        results.append({
            "id":               item["id"],
            "question":         item["question"],
            "reference_answer": item["reference_answer"],
            "generated_answer": hypotheses[i],
            "model":            model_label,
            "bertscore_P":      round(P[i].item(), 4),
            "bertscore_R":      round(R[i].item(), 4),
            "bertscore_F1":     round(F1[i].item(), 4),
            "bleu":             bleu,
            **rouge,
        })
 
    print(f"[{model_label}] done. mean BERTScore F1 = {F1.mean().item():.4f}")
    return results
 
 
def summarize(all_results: dict[str, list[dict]]) -> dict:
    """Compute mean of each metric per model."""
    summary = {}
    metrics = ["bertscore_P", "bertscore_R", "bertscore_F1", "bleu", "rouge1", "rouge2", "rougeL"]
    for model_label, rows in all_results.items():
        n = len(rows)
        summary[model_label] = {
            m: round(sum(r[m] for r in rows) / n, 4)
            for m in metrics
        }
        summary[model_label]["n"] = n
    return summary
 
 
# ── main ───────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
 
    for model_label, filename in MODEL_FILES.items():
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"WARNING: {filepath} not found — skipping {model_label}")
            continue
        all_results[model_label] = evaluate_model(model_label, filepath)
 
    # save per-question results (one entry per model×question)
    flat = [row for rows in all_results.values() for row in rows]
    out_detail = OUTPUT_DIR / "eval_bertscore.json"
    with open(out_detail, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)
    print(f"\nPer-question results saved -> {out_detail}")
 
    # save summary table
    summary = summarize(all_results)
    out_summary = OUTPUT_DIR / "eval_bertscore_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary saved -> {out_summary}")
 
    # print summary to console
    print("\n=== SUMMARY ===")
    header = f"{'Model':<20} {'BS-P':>6} {'BS-R':>6} {'BS-F1':>6} {'BLEU':>6} {'R-1':>6} {'R-2':>6} {'R-L':>6}"
    print(header)
    print("-" * len(header))
    for model, s in summary.items():
        print(
            f"{model:<20} {s['bertscore_P']:>6.4f} {s['bertscore_R']:>6.4f} "
            f"{s['bertscore_F1']:>6.4f} {s['bleu']:>6.4f} "
            f"{s['rouge1']:>6.4f} {s['rouge2']:>6.4f} {s['rougeL']:>6.4f}"
        )
 
 
if __name__ == "__main__":
    main()