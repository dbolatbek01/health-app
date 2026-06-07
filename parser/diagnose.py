import re
import os
from pathlib import Path


def evaluate_parser_quality(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return f"Папка не найдена: {folder_path}"

    md_files = list(folder.glob("*.md"))
    if not md_files:
        return f"MD файлов не найдено в: {folder_path}"

    content = ""
    for f in md_files:
        content += f.read_text(encoding="utf-8")

    h1_count = len(re.findall(r"^# ", content, re.MULTILINE))
    h2_count = len(re.findall(r"^## ", content, re.MULTILINE))
    h3_count = len(re.findall(r"^### ", content, re.MULTILINE))
    tables_count = len(re.findall(r"\|.*\|", content))
    marker_blind_images = len(re.findall(r"!\[\]\(.*?\)", content))
    pymupdf_text_soups = len(re.findall(r"----- Start of picture text -----", content))
    footer_junk = len(re.findall(r"Freigabe am|Nächste Überprüfung|Letzte Überprüfung", content))
    broken_tables = sum(
        1
        for m in re.finditer(r"(\|[^\n]+\|)\n\|[-\s|]+\|", content)
        if m.group(1).count("|") - 1 > 8
    )
    total_chars = len(content)

    return {
        "Файлов обработано": len(md_files),
        "Заголовки H1": h1_count,
        "Заголовки H2": h2_count,
        "Заголовки H3": h3_count,
        "Строки таблиц": tables_count,
        "Битые таблицы (>8 col)": broken_tables,
        "Footer-мусор": footer_junk,
        "Пустые картинки (Marker)": marker_blind_images,
        "Каша из схем (PyMuPDF)": pymupdf_text_soups,
        "Объем символов": total_chars,
    }


def print_report(marker_folder, pymupdf_folder):
    print("=" * 50)
    print("АНАЛИЗ КАЧЕСТВА ПАРСИНГА")
    print("=" * 50)

    metrics_marker = evaluate_parser_quality(marker_folder)
    metrics_pymupdf = evaluate_parser_quality(pymupdf_folder)

    if isinstance(metrics_marker, str) or isinstance(metrics_pymupdf, str):
        print(metrics_marker if isinstance(metrics_marker, str) else metrics_pymupdf)
        return

    print(f"{'Метрика':<30} | {'Marker':<15} | {'PyMuPDF4LLM':<15}")
    print("-" * 65)
    for key in metrics_marker.keys():
        print(f"{key:<30} | {metrics_marker[key]:<15} | {metrics_pymupdf[key]:<15}")
    print("=" * 50)


MARKER_FOLDER = r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\parsed"
PYMUPDF_FOLDER = r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\parsed_ohne_ki"

if __name__ == "__main__":
    print_report(MARKER_FOLDER, PYMUPDF_FOLDER)