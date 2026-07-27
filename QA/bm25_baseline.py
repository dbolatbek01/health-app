"""
BM25-Baseline zum Vergleich mit Dense Retrieval (Kapitel 5.1 / 6.1)
=====================================================================
Berechnet Recall@k (Success-Rate: mind. ein Treffer aus dem korrekten
Quelldokument in den Top-k) für BM25 auf demselben Testdatensatz, mit
dem auch die Dense-Retrieval-Ergebnisse aus Tabelle 9 erzeugt wurden.

Voraussetzungen:
  pip install rank_bm25 --break-system-packages

Erwartete Eingaben (Pfade unten anpassen):
  - chunks.jsonl   : dieselbe Chunk-Datei, die auch für ChromaDB verwendet
                     wurde (small-Konfiguration, 512 Zeichen).
  - test_set_final.json : Frage + zugehörige Quelldokument(e)
"""

import json
import re
from rank_bm25 import BM25Okapi

CHUNKS_PFAD = r"C:\Masterarbeit\App\parser\chunks_small.jsonl"
TESTFRAGEN_PFAD = r"C:\Masterarbeit\App\QA\test_set_final.json"
K_WERTE = [1, 3, 5, 10]


def tokenize(text: str):
    # einfache Tokenisierung, analog zur Vorverarbeitung im übrigen Projekt
    return re.findall(r"\w+", text.lower())


def main():
    # 1) Chunks laden
    chunks = []
    with open(CHUNKS_PFAD, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    corpus_tokens = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)

    # 2) Testfragen + Ground-Truth-Quelldokumente laden
    with open(TESTFRAGEN_PFAD, "r", encoding="utf-8") as f:
        data = json.load(f)
    testfragen = data["questions"]

    recall_hits = {k: 0 for k in K_WERTE}
    n = len(testfragen)

    for item in testfragen:
        frage = item["question"]
        # Mehrere Quelldokumente sind durch "; " getrennt (siehe Anhang A)
        relevante_quellen = {q.strip() for q in item["source_doc"].split(";")}

        scores = bm25.get_scores(tokenize(frage))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Возвращенный цикл for k in K_WERTE:
        for k in K_WERTE:
            top_k_sources = {chunks[i]["metadata"]["source"] for i in ranked_idx[:k]}
            if top_k_sources & relevante_quellen:
                recall_hits[k] += 1

    print("BM25-Baseline — Success-Rate@k (identische Definition wie Tabelle 9):")
    for k in K_WERTE:
        print(f"  R@{k}: {recall_hits[k] / n:.3f}  ({recall_hits[k]}/{n})")

if __name__ == "__main__":
    main()