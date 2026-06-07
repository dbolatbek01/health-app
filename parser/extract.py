import fitz
import pdfplumber

PDF_PATH = r"C:\Users\Данат\Documents\Wildau\Masterarbeit\App\testpdf\test.pdf"

HEADER_PATTERNS = [
    "Behandlungsstandards",
    "Notfallambulanz",
    "Version  2017",
    "Version 2017",
]


def is_header_line(text):
    t = text.strip()
    if t.startswith("Seite "):
        return True
    return t in HEADER_PATTERNS


def format_table(table):
    rows = [
        [cell.replace("\n", " ") if cell else "" for cell in row]
        for row in table
        if any(cell for cell in row)
    ]
    if not rows:
        return ""
    col_count = max(len(row) for row in rows)
    rows = [row + [""] * (col_count - len(row)) for row in rows]

    non_empty_cols = [i for i in range(col_count) if any(row[i] for row in rows)]
    rows = [[row[i] for i in non_empty_cols] for row in rows]
    col_count = len(non_empty_cols)

    if col_count == 0:
        return ""

    col_widths = [max(len(row[i]) for row in rows) for i in range(col_count)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [sep]
    for j, row in enumerate(rows):
        cells = [f" {row[i].ljust(col_widths[i])} " for i in range(col_count)]
        lines.append("|" + "|".join(cells) + "|")
        if j == 0:
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)


fitz_doc = fitz.open(PDF_PATH)

with pdfplumber.open(PDF_PATH) as plumber_doc, open(
    "result.txt", "w", encoding="utf-8"
) as out:
    for i, fitz_page in enumerate(fitz_doc):
        out.write(f"--- Page {i + 1} ---\n")

        plumber_page = plumber_doc.pages[i]
        found_tables = plumber_page.find_tables()

        if found_tables:
            elements = []

            for t in found_tables:
                elements.append((t.bbox[1], "table", format_table(t.extract())))

            def not_in_any_table(obj):
                for t in found_tables:
                    x0, top, x1, bottom = t.bbox
                    if (
                        obj.get("x0", 0) >= x0 - 1
                        and obj.get("top", 0) >= top - 1
                        and obj.get("x1", 0) <= x1 + 1
                        and obj.get("bottom", 0) <= bottom + 1
                    ):
                        return False
                return True

            words = plumber_page.filter(not_in_any_table).extract_words()

            lines: dict[int, list] = {}
            for word in words:
                key = next(
                    (k for k in lines if abs(k - word["top"]) < 3), round(word["top"])
                )
                lines.setdefault(key, []).append(word)

            for top_y, line_words in lines.items():
                line_text = " ".join(
                    w["text"] for w in sorted(line_words, key=lambda w: w["x0"])
                )
                if not is_header_line(line_text):
                    elements.append((top_y, "text", line_text))

            for _, elem_type, content in sorted(elements, key=lambda e: e[0]):
                out.write(content + "\n")
                if elem_type == "table":
                    out.write("\n")
        else:
            for line in fitz_page.get_text().splitlines():
                if not is_header_line(line):
                    out.write(line + "\n")

        out.write("\n")

fitz_doc.close()
print(f"Done: {i + 1} pages written to result.txt")
