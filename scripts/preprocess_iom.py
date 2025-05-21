from docx import Document
import re

doc = Document('data/raw/IOM.docx')

raw_text = '\n'.join(para.text for para in doc.paragraphs)

clean_text = re.sub(
    r'INVESTIGATIONS OPERATIONS MANUAL\s*2024',
    '',
    raw_text,
    flags=re.IGNORECASE
)
clean_text = re.sub(r'\n\d+\s*\n', '\n', clean_text)
clean_text = re.sub(r'\n{2,}', '\n', clean_text)

segments = []
for part in re.split(r'CHAPTER \d+', clean_text):
    part = part.strip()
    if not part:
        continue
    paragraphs = [p.strip() for p in part.split('\n') if p.strip()]
    segments.extend(paragraphs)

with open('data/processed/iom_segments.txt', 'w', encoding='utf-8') as out:
    for seg in segments:
        out.write(seg + "\n")
