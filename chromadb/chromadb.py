from pathlib import Path
import chromadb

INPUT_DIR = Path(r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\embeddings\models") 
CHROMA_DIR = Path(r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\chromadb")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))