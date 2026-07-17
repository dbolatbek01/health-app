"""
Bootstrap 95% confidence intervals fuer die Modell-Metriken.
Liest die bereits vorhandenen per-question Ergebnisdateien:
  - BERTScore: evaluation/bert/results/eval_bertscore.json
  - RAGAS:     evaluation/ragas/results/ragas_detail_{model}.csv

Output: confidence_intervals.json + Konsolen-Tabelle.

Usage:  python confidence_intervals.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths (an dein Projekt anpassen) ─────────────────────────────────────────
BERT_DETAIL = Path(r"C:\Masterarbeit\App\evaluation\bert\results\eval_bertscore.json")
RAGAS_DIR   = Path(r"C:\Masterarbeit\App\evaluation\ragas\results")
OUTPUT      = Path(r"C:\Masterarbeit\App\evaluation\confidence_intervals.json")

MODELS = ["mistral:7b", "qwen2.5:7b", "llama3.1:8b", "BioMistral-7B", "meditron:7b"]
N_BOOT = 10000
SEED   = 42


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    """95% Perzentil-Bootstrap-CI fuer den Mittelwert. NaNs werden entfernt."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(SEED)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (v.mean(), lo, hi, len(v))


# ── BERTScore F1 aus eval_bertscore.json ─────────────────────────────────────
def load_bertscore(metric="bertscore_F1"):
    data = json.loads(BERT_DETAIL.read_text(encoding="utf-8"))
    out = {}
    for m in MODELS:
        vals = [r[metric] for r in data if r["model"] == m]
        out[m] = vals
    return out


# ── RAGAS faithfulness aus den detail-CSVs ───────────────────────────────────
def load_ragas(metric="faithfulness"):
    out = {}
    for m in MODELS:
        safe = m.replace(":", "_").replace("/", "_")
        path = RAGAS_DIR / f"ragas_detail_{safe}.csv"
        if not path.exists():
            print(f"WARNUNG: {path.name} nicht gefunden -- {m} uebersprungen")
            continue
        df = pd.read_csv(path)
        out[m] = df[metric].tolist()
    return out


def report(title, per_model):
    print(f"\n=== {title} (95% Bootstrap-CI, {N_BOOT} resamples) ===")
    print(f"{'Model':<16} {'Mean':>7} {'CI low':>8} {'CI high':>8} {'+/-':>7} {'n':>4}")
    print("-" * 54)
    result = {}
    for m, vals in per_model.items():
        mean, lo, hi, n = bootstrap_ci(vals)
        half = (hi - lo) / 2
        print(f"{m:<16} {mean:>7.4f} {lo:>8.4f} {hi:>8.4f} {half:>7.4f} {n:>4}")
        result[m] = {"mean": round(mean, 4), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "n": n}
    return result


def main():
    full = {}
    full["bertscore_f1"] = report("BERTScore F1", load_bertscore("bertscore_F1"))
    full["bleu"] = report("BLEU", load_bertscore("bleu"))
    full["rouge_l"] = report("ROUGE-L", load_bertscore("rougeL"))
    full["ragas_faithfulness"] = report("RAGAS Faithfulness", load_ragas("faithfulness"))
    full["ragas_answer_relevancy"] = report("RAGAS Answer Relevancy", load_ragas("answer_relevancy"))
    full["ragas_context_recall"] = report("RAGAS Context Recall", load_ragas("context_recall"))

    OUTPUT.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGespeichert -> {OUTPUT}")


if __name__ == "__main__":
    main()
