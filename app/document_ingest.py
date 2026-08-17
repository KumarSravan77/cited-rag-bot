import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"}


@dataclass(frozen=True)
class ExtractedChunk:
    document: str
    page: int
    page_end: int
    text: str
    headings: List[str] = field(default_factory=list)
    content_types: List[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _page_numbers(chunk) -> List[int]:
    pages = set()
    for item in getattr(chunk.meta, "doc_items", []) or []:
        for provenance in getattr(item, "prov", []) or []:
            page_number = getattr(provenance, "page_no", None)
            if isinstance(page_number, int) and page_number >= 1:
                pages.add(page_number)
    return sorted(pages) or [1]


def _headings(chunk) -> List[str]:
    return [
        clean_text(str(value))
        for value in (getattr(chunk.meta, "headings", []) or [])
        if clean_text(str(value))
    ]


def _content_types(chunk) -> List[str]:
    labels = {
        str(getattr(item, "label", "text")).split(".")[-1].lower()
        for item in (getattr(chunk.meta, "doc_items", []) or [])
    }
    return sorted(labels or {"text"})


def chunks_from_docling(document, filename: str, chunker=None) -> List[ExtractedChunk]:
    active_chunker = chunker or HybridChunker(merge_peers=True, repeat_table_header=True)
    rows = []
    for chunk in active_chunker.chunk(dl_doc=document):
        text = clean_text(active_chunker.contextualize(chunk=chunk))
        if not text:
            continue
        pages = _page_numbers(chunk)
        rows.append(
            ExtractedChunk(
                document=Path(filename).name,
                page=pages[0],
                page_end=pages[-1],
                text=text,
                headings=_headings(chunk),
                content_types=_content_types(chunk),
            )
        )
    return rows


def extract_document(
    data: bytes,
    filename: str,
    converter: Optional[DocumentConverter] = None,
    chunker=None,
) -> List[ExtractedChunk]:
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix or 'missing extension'}")
    if not data:
        raise ValueError("The uploaded document is empty")

    active_converter = converter or DocumentConverter()
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as source:
            source.write(data)
            source.flush()
            result = active_converter.convert(source.name)
            rows = chunks_from_docling(result.document, safe_name, chunker)
    except Exception as exc:
        raise ValueError(f"Document conversion failed: {type(exc).__name__}") from exc
    if not rows:
        raise ValueError("The document produced no indexable content")
    return rows
