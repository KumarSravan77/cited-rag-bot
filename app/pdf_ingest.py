import io
import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_page(text: str, size: int = 900, overlap: int = 120) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def extract_pdf(data: bytes, filename: str) -> List[Tuple[str, int, str]]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")
    rows = []
    safe_name = Path(filename).name
    for page_number, page in enumerate(reader.pages, start=1):
        for chunk in chunk_page(page.extract_text() or ""):
            rows.append((safe_name, page_number, chunk))
    if not rows:
        raise ValueError("The PDF contains no extractable text; OCR is required")
    return rows
