import json
from pathlib import Path
from bert_score import score

RESULTS_DIR = Path(r"C:\Masterarbeit\App\generation\results")
MODEL_TYPE = "bert-base-multilingual-cased"  

MODEL_FILES = {
    "Qwen2.5-7b": "qwen2.5_7b.json",
    "Mistral-7b": "mistral_7b.json",
    "Llama3.1-8b": "llama3.1_8b.json",
    "BioMistral-7B": "hf.co_BioMistral_BioMistral-7B-GGUF_Q4_K_M.json",
    "Meditron-7b": "meditron_7b.json",
}

ORIGINAL_WERTE = {
    "Qwen2.5-7b": 0.7514,
    "Mistral-7b": 0.7484,
    "Llama3.1-8b": 0.7360,
    "BioMistral-7B": 0.6722,
    "Meditron-7b": 0.6156,
}


def load_model_data(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        refs = data.get("refs", [])
        cands = data.get("cands", [])
    elif isinstance(data, list):
        refs = [
            item.get("reference_answer")
            or item.get("reference")
            or item.get("ref")
            for item in data
        ]
        cands = [
            item.get("generated_answer")
            or item.get("response")
            or item.get("answer")
            or item.get("cand")
            for item in data
        ]
    else:
        raise ValueError(f"Unbekanntes Format in {filepath}")

    return refs, cands


def main():
    print(
        f"{'Modell':<15} {'Original (Tabelle 12)':<24} {'Ohne Rescale (neu)':<20} {'Mit Rescale':<20}"
    )

    for name, filename in MODEL_FILES.items():
        filepath = RESULTS_DIR / filename

        if not filepath.exists():
            print(f"{name:<15} [Fehler: Datei {filename} nicht gefunden]")
            continue

        refs, cands = load_model_data(filepath)

        _, _, f1_ohne = score(
            cands,
            refs,
            model_type=MODEL_TYPE,
            lang="de",
            rescale_with_baseline=False,
            verbose=False,
        )
        _, _, f1_mit = score(
            cands,
            refs,
            model_type=MODEL_TYPE,
            lang="de",
            rescale_with_baseline=True,
            verbose=False,
        )

        orig = ORIGINAL_WERTE.get(name, 0.0)
        print(
            f"{name:<15} {orig:<24.4f} {f1_ohne.mean().item():<20.4f} {f1_mit.mean().item():<20.4f}"
        )


if __name__ == "__main__":
    main()