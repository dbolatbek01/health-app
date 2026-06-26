from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
import json
from pathlib import Path

run_config = RunConfig(timeout=3600, max_workers=1, max_retries=3)
INPUT_DIR = Path(r"C:\Masterarbeit\App\generation\results")
RESULT_DIR = Path(r"C:\Masterarbeit\App\evaluation\ragas\results")
JUDGE_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"

MODEL_FILES = {
    "llama3.1:8b":   "llama3.1_8b.json",
    "mistral:7b":    "mistral_7b.json",
    "qwen2.5:7b":    "qwen2.5_7b.json",
    "meditron:7b":   "meditron_7b.json",
    "BioMistral-7B": "hf.co_BioMistral_BioMistral-7B-GGUF_Q4_K_M.json",
}

def run():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    llm = LangchainLLMWrapper(ChatOllama(
       model=JUDGE_MODEL,
       base_url=OLLAMA_URL,
       temperature=0,
       num_predict=2048
    ))


    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    ))

    metrics = [faithfulness, answer_relevancy, context_recall]
    for metric in metrics:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    summary = {}
    for model_name, filename in MODEL_FILES.items():
        filepath = INPUT_DIR / filename
        if not filepath.exists():
            print(f"WARNUNG: {filepath} nicht gefunden -- uebersprungen")
            continue

        data = json.load(open(filepath, encoding="utf-8"))
    
        dataset = Dataset.from_dict({
            "question": [item["question"] for item in data],
            "answer": [item["answer"] for item in data],
            "contexts": [item["chunks"] for item in data],
            "ground_truth": [item["reference_answer"] for item in data],
        })

        results = evaluate(dataset, metrics=metrics, run_config=run_config)

        safe_name = model_name.replace(":", "_").replace("/", "_")
        df = results.to_pandas()
        df.to_csv(RESULT_DIR / f"ragas_detail_{safe_name}.csv", index=False, encoding="utf-8")

        summary[model_name] = {
            "faithfulness": round(df["faithfulness"].mean(skipna=True), 4),
            "answer_relevancy": round(df["answer_relevancy"].mean(skipna=True), 4),
            "context_recall": round(df["context_recall"].mean(skipna=True), 4),
            "n_valid_faithfulness": int(df["faithfulness"].notna().sum()),
            "n_valid_relevancy": int(df["answer_relevancy"].notna().sum()),
            "n_valid_recall": int(df["context_recall"].notna().sum()),
        }

        print(f"{model_name}: {summary[model_name]}")

        with open(RESULT_DIR / "ragas_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nRAGAS summary saved -> {RESULT_DIR / 'ragas_summary.json'}")

if __name__ == "__main__":
    run()