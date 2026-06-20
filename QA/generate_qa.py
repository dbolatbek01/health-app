"""
Generates 25 QA pairs from the first half of sources in chunks.jsonl via local Ollama REST API.
Saves the second half of sources to manual_sources.json for manual question creation.

Usage:
  # All single chunks (default)
  python generate_qa.py --chunks chunks.jsonl --model qwen2.5:7b --output test_set_llm.json

  # Mix: 60% single + 40% boundary
  python generate_qa.py --chunks chunks.jsonl --model qwen2.5:7b --boundary-ratio 0.4 --output test_set_llm.json
"""

import json
import random
import argparse
import requests
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"
N_QUESTIONS = 25

PROMPT_TEMPLATE = """Du bist ein medizinischer Experte für klinische Standard Operating Procedures (SOPs).

Lies folgenden Textabschnitt aus einem klinischen SOP-Dokument:

---
{chunk_text}
---

Erstelle genau eine konkrete Frage zum klinischen Inhalt dieses Textes.
Dann beantworte die Frage ausschließlich mit Informationen aus dem gegebenen Text.
Füge KEINE Informationen hinzu, die nicht explizit im Text stehen.

WICHTIG: Wenn der Text keine klaren klinischen Informationen enthält, um eine sinnvolle Frage zu formulieren, MÜSSEN die Felder "question" und "answer" den Wert null haben! Erfinde keine Fakten.

Die Frage MUSS sich auf klinische Inhalte beziehen, z.B.:
- Medizinische Prozeduren oder Vorgehensweisen
- Dosierungen oder Medikamente
- Indikationen oder Kontraindikationen
- Zuständigkeiten von medizinischem Personal
- Zeitangaben für medizinische Maßnahmen
- Kriterien für klinische Entscheidungen

Die Frage darf sich NICHT beziehen auf:
- Seitenzahlen, Abschnittsnummern oder Dokumentstruktur
- Versionsdaten oder Publikationsdaten des Dokuments
- Inhaltsverzeichnisse oder Literaturlisten

Antworte NUR in folgendem JSON-Format (kein Text davor oder danach):
{{
  "question": "...",
  "answer": "..."
}}"""


def query_ollama(prompt: str, model: str, timeout: int = 120) -> str | None:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        print("  [timeout]")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [error: {e}]")
        return None


def parse_json_response(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return None


def truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for punct in (".", "!", "?"):
        pos = truncated.rfind(punct)
        if pos > max_chars * 0.5:
            return truncated[:pos + 1]
    pos = truncated.rfind(" ")
    return truncated[:pos] if pos > 0 else truncated


def load_sources(jsonl_path: str) -> dict:
    sources = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            src = d["metadata"]["source"]
            if src not in sources:
                sources[src] = []
            sources[src].append(d)
    for src in sources:
        sources[src].sort(key=lambda x: x["metadata"]["chunk_index"])
    return sources


def split_sources(sources: dict, seed: int) -> tuple:
    all_sources = sorted(sources.keys())
    rng = random.Random(seed)
    rng.shuffle(all_sources)
    mid = len(all_sources) // 2
    return all_sources[:mid], all_sources[mid:]


def sample_single(sources: dict, source_pool: list, n: int, min_len: int = 300) -> list:
    candidates = []
    for src in source_pool:
        chunks = sources[src]
        total = len(chunks)
        for c in chunks:
            if len(c["text"]) >= min_len:
                idx = c["metadata"]["chunk_index"]
                is_middle = 0 < idx < total - 1
                candidates.append((c, is_middle))

    candidates.sort(key=lambda x: not x[1])

    selected_sources = set()
    selected = []
    for chunk, _ in candidates:
        src = chunk["metadata"]["source"]
        if src not in selected_sources:
            selected_sources.add(src)
            selected.append(chunk)
        if len(selected) >= n:
            break

    random.shuffle(selected)
    return selected[:n]


def sample_boundary(sources: dict, source_pool: list, n: int, min_len: int = 200) -> list:
    pairs = []
    for src in source_pool:
        chunks = sources[src]
        for i in range(len(chunks) - 1):
            a, b = chunks[i], chunks[i + 1]
            combined = a["text"] + "\n\n" + b["text"]
            if len(combined) >= min_len:
                pairs.append((a, b, combined))
                break  # one pair per source found, move to next source

    if len(pairs) < n:
        print(f"  [warning] only {len(pairs)} boundary pairs available, using all")
        n = len(pairs)

    random.shuffle(pairs)

    selected_sources = set()
    selected = []

    for a, b, combined in pairs:
        src = a["metadata"]["source"]
        if src not in selected_sources:
            selected_sources.add(src)
            selected.append({
                "id": f"{a['id']}+{b['id']}",
                "text": truncate_at_sentence(combined, 3500),
                "metadata": {
                    "source": src,
                    "chunk_index": f"{a['metadata']['chunk_index']}+{b['metadata']['chunk_index']}",
                    "boundary": True
                }
            })
        if len(selected) >= n:
            break

    if len(selected) < n:
        used = {s["id"] for s in selected}
        for a, b, combined in pairs:
            sid = f"{a['id']}+{b['id']}"
            if sid not in used:
                selected.append({
                    "id": sid,
                    "text": truncate_at_sentence(combined, 3500),
                    "metadata": {
                        "source": a["metadata"]["source"],
                        "chunk_index": f"{a['metadata']['chunk_index']}+{b['metadata']['chunk_index']}",
                        "boundary": True
                    }
                })
                used.add(sid)
            if len(selected) >= n:
                break

    random.shuffle(selected)
    return selected[:n]


def generate_qa(chunks: list, model: str) -> list:
    results = []
    for i, chunk in enumerate(chunks, 1):
        src = chunk["metadata"]["source"]
        chunk_idx = chunk["metadata"]["chunk_index"]
        is_boundary = chunk["metadata"].get("boundary", False)
        mode_tag = "boundary" if is_boundary else "single"

        print(f"[{i}/{len(chunks)}] [{mode_tag}] {src[:45]} (chunk {chunk_idx}) ... ", end="", flush=True)

        prompt = PROMPT_TEMPLATE.format(chunk_text=chunk["text"])
        raw = query_ollama(prompt, model)

        entry = {
            "id": i,
            "source_doc": src,
            "chunk_id": chunk["id"],
            "chunk_index": str(chunk_idx),
            "boundary": is_boundary,
            "generated_by": "llm",
            "model": model,
            "timestamp": datetime.now().isoformat()
        }

        if raw:
            parsed = parse_json_response(raw)
            if parsed and "question" in parsed and "answer" in parsed:
                entry["question"] = parsed["question"]
                entry["reference_answer"] = parsed["answer"]
                print("✓")
            else:
                entry["question"] = "[PARSE ERROR — edit manually]"
                entry["reference_answer"] = raw[:500]
                print("✗ (bad JSON)")
        else:
            entry["question"] = "[NO RESPONSE]"
            entry["reference_answer"] = ""
            print("✗ (no response)")

        results.append(entry)

    return results


def save_llm(results: list, output_path: str, model: str, mode_desc: str):
    boundary_count = sum(1 for r in results if r.get("boundary"))
    data = {
        "metadata": {
            "total": len(results),
            "single_chunk": len(results) - boundary_count,
            "boundary_chunk": boundary_count,
            "generated_by": "llm",
            "model": model,
            "mode": mode_desc,
            "created": datetime.now().isoformat()
        },
        "questions": results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(results)} LLM QA pairs → {output_path}")


def save_manual_sources(manual_sources: list, output_path: str):
    data = {
        "metadata": {
            "total": len(manual_sources),
            "note": "These sources are reserved for manual question creation (25 needed)",
            "created": datetime.now().isoformat()
        },
        "sources": sorted(manual_sources)
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(manual_sources)} manual sources → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral", help="Ollama model name")
    parser.add_argument("--chunks", default=r"C:\Masterarbeit\App\parser\chunks_medium.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--output", default="test_set_llm.json", help="Output JSON for LLM questions")
    parser.add_argument("--manual-output", default="manual_sources.json", help="Output JSON for manual sources")
    parser.add_argument("--boundary-ratio", type=float, default=0.0,
                        help="Fraction of boundary questions (0.0-1.0). E.g. 0.4 = 10 boundary + 15 single")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if not 0.0 <= args.boundary_ratio <= 1.0:
        print("Error: --boundary-ratio must be between 0.0 and 1.0")
        return

    random.seed(args.seed)

    n_boundary = round(N_QUESTIONS * args.boundary_ratio)
    n_single = N_QUESTIONS - n_boundary
    mode_desc = f"single={n_single}, boundary={n_boundary}"

    print(f"Model: {args.model} | {mode_desc} | Seed: {args.seed}\n")

    sources = load_sources(args.chunks)
    llm_pool, manual_pool = split_sources(sources, args.seed)

    print(f"Total sources: {len(sources)}")
    print(f"  LLM pool:    {len(llm_pool)} sources (generating {N_QUESTIONS} questions)")
    print(f"  Manual pool: {len(manual_pool)} sources (for your {N_QUESTIONS} manual questions)\n")

    # Split llm_pool so single and boundary don't reuse same sources
    chunks = []
    if n_single > 0:
        chunks += sample_single(sources, llm_pool, n_single)
    if n_boundary > 0:
        chunks += sample_boundary(sources, llm_pool, n_boundary)

    random.shuffle(chunks)

    unique = len(set(c["metadata"]["source"] for c in chunks))
    print(f"Sampled {len(chunks)} chunks from {unique} unique sources\n")

    results = generate_qa(chunks, args.model)

    save_llm(results, args.output, args.model, mode_desc)
    save_manual_sources(manual_pool, args.manual_output)


if __name__ == "__main__":
    main()