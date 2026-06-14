from pathlib import Path
import json
import uuid

INPUT_DIR = Path(r"C:\Masterarbeit\App\parser\parsed_clean")
OUTPUT_FILE = Path(r"C:\Masterarbeit\App\parser\chunks.jsonl")

CHUNK_SIZE =1536   #  ≈ 512 tokens
OVERLAP = 384       #  ≈ 128 tokens

MIN_CHUNK = 100

def splint_into_chunks(content, chunk_size, overlap):
        chunks = []
        start = 0

        while start < len(content):
                end = start + CHUNK_SIZE
                chunk = content[start:end]
                if len(chunk) >= MIN_CHUNK:
                    chunks.append(chunk)
                start = start + chunk_size - overlap
        return chunks
        
        
def run():
        with OUTPUT_FILE.open("w", encoding="utf-8") as output:
            for file in INPUT_DIR.glob("*.md"): 
                content = file.read_text(encoding="utf-8")
                chunks = splint_into_chunks(content, CHUNK_SIZE, OVERLAP)

                for i, chunk in enumerate(chunks):
                    record = {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                        "metadata": {
                            "source": file.stem,
                            "chunk_index": i,
                            "chunk_total": len(chunks)
                        }
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(file.name, "->", len(chunks), "chunks")
                      
if __name__ == "__main__":
    run()
        
        



    
        

    
    
             


