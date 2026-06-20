from pathlib import Path
import chromadb
import requests
import json
import time
from sentence_transformers import SentenceTransformer

COLLECTION = "small_bge_m3"
OLLAMA_URL = "http://localhost:11434/api/generate"
QA = Path(r"C:\Masterarbeit\App\QA\test_set_final.json")
RESULT_DIR = Path(r"C:\Masterarbeit\App\backend\results")
TOP_K = 5

MODELS = [
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:7b",
    "meditron:7b",
    "biomistral:7b",
]

def run():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    run()