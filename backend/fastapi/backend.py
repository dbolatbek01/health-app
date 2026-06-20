import fastapi
from pathlib import Path
import chromadb
import torch
import requests
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(r"C:\Masterarbeit\App\chromadb")
OLLAMA_URL = "http://localhost:11434/api/generate"

