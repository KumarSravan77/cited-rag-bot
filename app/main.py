import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.models import AskRequest, AskResponse, Chunk, Citation
from app.pdf_ingest import extract_pdf
from app.providers import build_embedder, synthesize
from app.retrieval import hybrid_search
from app.store import ChunkStore

store = ChunkStore(Path(os.getenv("RAG_DATABASE_PATH", "data/rag.db")))
app = FastAPI(title="Cited PDF RAG Bot", version="0.1.0")


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/v1/documents")
async def ingest(request: Request, filename: str):
    if request.headers.get("content-type", "").split(";")[0] != "application/pdf":
        raise HTTPException(415, "Only PDF files are supported")
    data = await request.body()
    if len(data) > int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024:
        raise HTTPException(413, "PDF exceeds the configured upload limit")
    try:
        rows = extract_pdf(data, filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    embedder = build_embedder()
    vectors = embedder.embed([text for _, _, text in rows])
    chunks = [
        Chunk(document=document, page=page, text=text, embedding=vector)
        for (document, page, text), vector in zip(rows, vectors)
    ]
    return {"document": Path(filename).name, "pages": len(set(c.page for c in chunks)),
            "chunks": store.replace_document(Path(filename).name, chunks)}


@app.post("/v1/ask", response_model=AskResponse)
def ask(request: AskRequest):
    hits = hybrid_search(request.question, store.all(), build_embedder(), request.top_k)
    answer = synthesize(request.question, hits)
    referenced = sorted(set(int(value) for value in re.findall(r"\[(\d+)\]", answer)))
    valid_ids = [value for value in referenced if 1 <= value <= len(hits)]
    citations = [
        Citation(id=value, document=hits[value - 1].document, page=hits[value - 1].page,
                 excerpt=hits[value - 1].text[:280])
        for value in valid_ids
    ]
    grounded = bool(citations) and referenced == valid_ids
    if referenced != valid_ids:
        answer = "The generated answer contained an unsupported citation and was rejected."
        citations, grounded = [], False
    return AskResponse(answer=answer, citations=citations, grounded=grounded)
