from pathlib import Path
import pymupdf4llm

PDF_DIR = Path(r"C:\Users\Данат\Documents\Svetofor API")
OUTPUT_DIR = Path(r"C:\Masterarbeit\App\parsed_ohne_ki")
OUTPUT_DIR.mkdir(exist_ok=True)

for pdf_path in PDF_DIR.glob("**/*.pdf"):  # ** — рекурсивно по подпапкам
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    out_path = OUTPUT_DIR / pdf_path.with_suffix(".md").name
    out_path.write_text(md_text, encoding="utf-8")
    print(f"✓ {pdf_path.name}")
