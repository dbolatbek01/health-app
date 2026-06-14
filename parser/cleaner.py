import re
from pathlib import Path

INPUT_DIR = Path(r"C:\Masterarbeit\App\parser\parsed_ohne_ki")
OUTPUT_DIR = Path(r"C:\Masterarbeit\App\parser\parsed_clean")
OUTPUT_DIR.mkdir(exist_ok=True)

FULL_LINE_JUNK = [
    # Номера страниц
    r"Seite\s*(<br>)?\s*\d+\s+von\s+\d+",
    # Пустые картинки
    r"\*\*==>\s*picture\s*\[\d+\s*x\s*\d+\]\s*intentionally omitted\s*<==\*\*",
    # Маркеры picture text — только одиночные (без данных)
    r"\*\*-+\s*(Start|End) of picture text\s*-+\*\*(<br>)?",
    # br-склеенные footer-артефакты: Version ... <br> --- End of picture text ---
    r"(Änderungshistorie<br>)?(\w[\w ]+<br>)*\*\*-+\s*End of picture text\s*-+\*\*(<br>)?",
    # Идентификаторы документов DHZC/DHZB
    r"(DHZC|DHZB)[\w/]*\s*[I|/].*?Version\s+Nr\.\s*[\d\.]+.*",
    # Таблицы истории версий (все варианты заголовков)
    r"(\*\*)?Änderungshistorie(\*\*)?",
    r"(\*\*)?Version\s+(Freigabe am|Freigabedatum|Erstellt|gültig ab|Erstellung).*",
    r"## \*\*Version\s+Freigabe am\*\*",
    r"\*\*Änderungshistorie\s+Version\s+Freigabe am\*\*",
    # Многокомпонентные footer-строки:
    # "Version: N.N Letzte Überprüfung ...", "Version N.N Letzte Überprüfung SOP NNN ..."
    # "Version: SOP-Nr.: 1006 Letzte Überprüfung 2.0 PS-SOP-PI-01 Freigabe am ..."
    r"Version[\s:]+.*?(Letzte|Nächste)\s+Überprüfung.*",
    r"Version[\s:]+.*?SOP.Nr.*Freigabe am.*",
    r"SOP-Nr\.\s+\d+\s+Freigabe am",
    r"\*\*Version:\s+Freigabe am\*\*",
    # Одиночные footer-элементы (только если вся строка)
    r"Freigabe am",
    r"Nächste Überprüfung",
    r"Letzte Überprüfung",
]
FULL_LINE_RE = re.compile(r"^\s*(?:" + "|".join(FULL_LINE_JUNK) + r")\s*$")

# Footer внутри таблиц
TABLE_FOOTER_PATTERNS = [
    re.compile(r"Freigabe am"),
    re.compile(r"(Nächste|Letzte)\s+Überprüfung"),
    re.compile(r"Seite\s*(<br>)?\s*\d+\s+von\s+\d+"),
    re.compile(r"(DHZC|DHZB)[\w/]*\s*[I|/].*?Version"),
    re.compile(r"PS-SOP|PS-TP"),
]

CAMPUS_LINE_RE = re.compile(r"^\s*Campus:.*", re.IGNORECASE)
BOLD_FRAGMENT_RE = re.compile(r"\*\*(.+?)\*\*")


def is_table_footer(line):
    if not line.strip().startswith("|"):
        return False
    footer_hits = sum(1 for p in TABLE_FOOTER_PATTERNS if p.search(line))
    if footer_hits == 0:
        return False
    cells = [c.strip() for c in line.split("|") if c.strip()]
    real_cells = []
    for c in cells:
        is_junk = any(p.search(c) for p in TABLE_FOOTER_PATTERNS)
        is_version = bool(re.match(r"^[\d\.]+$", c))
        if not is_junk and not is_version and len(c) > 1:
            real_cells.append(c)
    return len(real_cells) == 0


def clean_campus_line(line):
    bolds = BOLD_FRAGMENT_RE.findall(line)
    if not bolds:
        return None
    title = " ".join(b.strip() for b in bolds if b.strip())
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) < 4:
        return None
    return f"## {title}"


def clean_text(text):
    lines = text.split("\n")
    out = []
    removed = 0
    rescued = 0

    for line in lines:
        stripped = line.strip()

        if not stripped:
            out.append(line)
            continue

        if FULL_LINE_RE.match(stripped):
            removed += 1
            continue

        if CAMPUS_LINE_RE.match(stripped):
            rescued_title = clean_campus_line(stripped)
            if rescued_title:
                out.append(rescued_title)
                rescued += 1
            else:
                removed += 1
            continue

        if is_table_footer(stripped):
            removed += 1
            continue

        out.append(line)

    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned)
    return cleaned, removed, rescued


def main():
    md_files = list(INPUT_DIR.glob("*.md"))
    print(f"Найдено файлов: {len(md_files)}\n")

    total_removed = 0
    total_rescued = 0

    for f in md_files:
        text = f.read_text(encoding="utf-8")
        cleaned, removed, rescued = clean_text(text)
        out_path = OUTPUT_DIR / f.name
        out_path.write_text(cleaned, encoding="utf-8")
        total_removed += removed
        total_rescued += rescued
        print(f"  {f.name}: удалено {removed}, спасено {rescued}")

    print(f"\nИТОГО: удалено {total_removed} строк, спасено {total_rescued} названий документов")
    print(f"Очищенные файлы в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()