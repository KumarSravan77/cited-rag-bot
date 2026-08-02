from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main
from app.models import Chunk
from app.retrieval import HashEmbedder
from app.store import ChunkStore


def test_answer_returns_page_citation(tmp_path: Path, monkeypatch):
    main.store = ChunkStore(tmp_path / "rag.db")
    embedder = HashEmbedder()
    text = "Backups are retained for ninety days."
    main.store.replace_document("operations.pdf", [
        Chunk(document="operations.pdf", page=7, text=text, embedding=embedder.embed([text])[0])
    ])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = TestClient(main.app).post("/v1/ask", json={"question": "How long are backups retained?"})
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"][0]["page"] == 7
    assert body["citations"][0]["document"] == "operations.pdf"


def test_empty_index_is_not_grounded(tmp_path: Path):
    main.store = ChunkStore(tmp_path / "empty.db")
    body = TestClient(main.app).post("/v1/ask", json={"question": "What is the policy?"}).json()
    assert body["grounded"] is False
    assert body["citations"] == []
