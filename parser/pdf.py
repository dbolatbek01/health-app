import fitz
from pathlib import Path

path = r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\testpdf\test.pdf"

doc = fitz.open(path)

with open("result.txt", "w", encoding="utf-8") as f:
    for page in doc:
        text = page.get_text()
        f.write(f"--- Page {page.number + 1} ---\n")
        f.write(text)
        f.write("\n\n")

        print(f" {page.number + 1} pages written")

doc.close()
