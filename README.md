# Cited Document RAG Bot

Layout-aware document question answering with Docling ingestion, page and section
provenance, hybrid retrieval, reranking, and citation validation.

## Pipeline

1. Parse PDF, DOCX, PPTX, XLSX, or HTML with Docling, including layout, OCR, and tables.
2. Use Docling's `HybridChunker` to preserve sections and repeat table headers.
3. Carry page ranges, headings, and content types into every indexed chunk.
4. Combine BM25 lexical scores with embedding similarity.
5. Rerank candidates using direct query-term coverage.
6. Generate from retrieved context only.
7. Reject citation identifiers that do not map to retrieved chunks.
8. Apply the shared production guardrail policy to questions and final JSON answers.
9. Remove retrieved chunks containing indirect injection, harmful instructions, or secrets.
10. Sanitize PII in retrieved evidence before prompt construction.

With `OPENAI_API_KEY`, the service uses `text-embedding-3-small` and the
Responses API. Without a key, it degrades to deterministic local embeddings and
extractive answers, which makes the complete pipeline locally testable.

In production, set `LLM_GATEWAY_URL` and `LLM_GATEWAY_API_KEY` so answer generation
runs through the portfolio's Production LLM Gateway. The guardrails package is pinned
to an immutable commit and is enforced independently in both services. This provides
defense in depth: the RAG service filters untrusted retrieved content, while the gateway
protects the shared model boundary.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export OPENAI_API_KEY="your-key"  # optional
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`, upload a raw document body through
`POST /v1/documents?filename=manual.pdf`,
then ask questions through `POST /v1/ask`.

Docling performs OCR for supported scanned documents. Uploaded documents and the
local SQLite index are excluded from Git. Production deployments should run conversion
in isolated workers with file scanning, bounded CPU/memory/time, durable object storage,
and an external vector database rather than processing untrusted documents in the API process.
