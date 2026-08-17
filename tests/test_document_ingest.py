from types import SimpleNamespace

import pytest

from app.document_ingest import chunks_from_docling, extract_document


class FakeChunker:
    def __init__(self):
        item = SimpleNamespace(
            label="DocItemLabel.TABLE",
            prov=[SimpleNamespace(page_no=2), SimpleNamespace(page_no=3)],
        )
        self.value = SimpleNamespace(
            text="Quarter Revenue\nQ1 $4M",
            meta=SimpleNamespace(doc_items=[item], headings=["Financial results"]),
        )

    def chunk(self, dl_doc):
        return [self.value]

    def contextualize(self, chunk):
        return chunk.text


class FakeConverter:
    def convert(self, source):
        return SimpleNamespace(document=object())


def test_docling_chunks_preserve_page_range_heading_and_type():
    rows = chunks_from_docling(object(), "../../annual-report.pdf", FakeChunker())
    assert rows[0].document == "annual-report.pdf"
    assert (rows[0].page, rows[0].page_end) == (2, 3)
    assert rows[0].headings == ["Financial results"]
    assert rows[0].content_types == ["table"]


def test_extraction_uses_converter_and_sanitizes_filename():
    rows = extract_document(
        b"not-real-pdf", "../../annual-report.pdf", FakeConverter(), FakeChunker()
    )
    assert rows[0].document == "annual-report.pdf"


def test_unsupported_extension_is_rejected():
    with pytest.raises(ValueError, match="Unsupported document type"):
        extract_document(b"content", "payload.exe", FakeConverter(), FakeChunker())
