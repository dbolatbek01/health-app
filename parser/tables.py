import pdfplumber

with pdfplumber.open("test.pdf") as pdf:
    page = pdf.pages[9]
    tables = page.extract_tables()
    for table in tables:
        for row in table:
            cleaned = [cell.replace("\n", " ") for cell in row if cell] 
            if cleaned:  
                print(" | ".join(cleaned))
        print(15*"-")
                
