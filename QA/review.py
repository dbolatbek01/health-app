"""
Opens test_set_llm.json alongside original chunk text for manual review.
For each question shows: question, current reference_answer, and original chunk text.
Allows you to accept, edit, or skip each entry.
Output: test_set_reviewed.json

Usage:
  python review_qa.py --qa test_set_llm.json --chunks chunks.jsonl --output test_set_reviewed.json
"""

import json
import argparse

def load_chunks_by_id(jsonl_path: str) -> dict:
    """Load chunks indexed by id."""
    chunks = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks[d["id"]] = d["text"]
    return chunks


def get_chunk_text(chunks: dict, chunk_id: str) -> str:
    """Handle both single and boundary chunk IDs."""
    if "+" in chunk_id:
        id_a, id_b = chunk_id.split("+", 1)
        text_a = chunks.get(id_a, "[chunk not found]")
        text_b = chunks.get(id_b, "[chunk not found]")
        return text_a + "\n\n--- [chunk boundary] ---\n\n" + text_b
    return chunks.get(chunk_id, "[chunk not found]")


def review(qa_path: str, chunks_path: str, output_path: str):
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = load_chunks_by_id(chunks_path)
    questions = data["questions"]
    reviewed = []

    print(f"\nReviewing {len(questions)} questions. Commands: [Enter]=accept, e=edit answer, s=skip, q=quit\n")

    for q in questions:
        print("=" * 80)
        print(f"[{q['id']}/{len(questions)}] {'[BOUNDARY]' if q.get('boundary') else '[SINGLE]'} {q['source_doc']}")
        print()
        print("ORIGINAL CHUNK TEXT:")
        print("-" * 40)
        chunk_text = get_chunk_text(chunks, q["chunk_id"])
        print(chunk_text[:3000])
        if len(chunk_text) > 3000:
            print("... [truncated]")
        print()
        print("QUESTION:")
        print(q.get("question", "[missing]"))
        print()
        print("CURRENT REFERENCE ANSWER:")
        print(q.get("reference_answer", "[missing]")[:500])
        print()

        cmd = input("Action [Enter=accept / e=edit / s=skip / q=quit]: ").strip().lower()

        if cmd == "q":
            print("Quitting — saving progress so far.")
            break
        elif cmd == "s":
            print("Skipped.\n")
            continue
        elif cmd == "e":
            print("Enter new reference answer (type END on a new line to finish):")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            q["reference_answer"] = "\n".join(lines)
            q["manually_reviewed"] = True
            print("Updated.\n")
        else:
            q["manually_reviewed"] = True

        reviewed.append(q)

    data["questions"] = reviewed
    data["metadata"]["total"] = len(reviewed)
    data["metadata"]["manually_reviewed"] = True

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(reviewed)} reviewed questions → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", default="test_set_llm.json", help="Input QA JSON file")
    parser.add_argument("--chunks", default=r"C:\Masterarbeit\App\parser\chunks_medium.jsonl", help="Path to chunks.jsonl")
    parser.add_argument("--output", default="test_set_reviewed.json", help="Output reviewed JSON file")
    args = parser.parse_args()

    review(args.qa, args.chunks, args.output)


if __name__ == "__main__":
    main()