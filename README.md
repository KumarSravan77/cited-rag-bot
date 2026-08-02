# Cited PDF RAG Bot

PDF question answering with page-level provenance, hybrid retrieval, reranking,
and citation validation.

## Pipeline

1. Extract text without losing PDF page boundaries.
2. Split each page independently so chunks never claim the wrong page.
3. Combine BM25 lexical scores with embedding similarity.
4. Rerank candidates using direct query-term coverage.
5. Generate from retrieved context only.
6. Reject citation identifiers that do not map to retrieved chunks.

With `OPENAI_API_KEY`, the service uses `text-embedding-3-small` and the
Responses API. Without a key, it degrades to deterministic local embeddings and
extractive answers, which makes the complete pipeline locally testable.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
export OPENAI_API_KEY="your-key"  # optional
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`, upload a raw `application/pdf` body through
`POST /v1/documents?filename=manual.pdf`,
then ask questions through `POST /v1/ask`.

Scanned PDFs require OCR before ingestion. Uploaded documents and the local
SQLite index are excluded from Git.
