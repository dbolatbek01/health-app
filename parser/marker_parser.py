from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

def main():
    PDF_DIR = Path(r"C:\Users\Данат\Documents\Svetofor API")
    OUTPUT_DIR = Path(r"C:\Masterarbeit\App\parsed")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Загрузка моделей...")
    config = ConfigParser({"disable_ocr": True})
    models = create_model_dict()

    converter = PdfConverter(
        artifact_dict=models,
        config=config.generate_config_dict()
    )

    pdf_files = list(PDF_DIR.glob("**/*.pdf"))
    print(f"Найдено файлов: {len(pdf_files)}\n---")

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}...")
        try:
            rendered = converter(str(pdf_path))
            md_text, _, _ = text_from_rendered(rendered)
            out_path = OUTPUT_DIR / pdf_path.with_suffix(".md").name
            out_path.write_text(md_text, encoding="utf-8")
            print(f"  ✓ готово")
        except Exception as e:
            print(f"  ❌ {e}")

    print("\nГотово!")

if __name__ == "__main__":
    main()