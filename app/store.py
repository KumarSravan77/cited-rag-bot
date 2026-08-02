import json
import sqlite3
from pathlib import Path
from typing import List

from app.models import Chunk


class ChunkStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, document TEXT NOT NULL, page INTEGER NOT NULL,
                text TEXT NOT NULL, embedding TEXT NOT NULL
            )""")

    def connect(self):
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def replace_document(self, document: str, chunks: List[Chunk]) -> int:
        with self.connect() as db:
            db.execute("DELETE FROM chunks WHERE document=?", (document,))
            db.executemany(
                "INSERT INTO chunks(document,page,text,embedding) VALUES(?,?,?,?)",
                [(c.document, c.page, c.text, json.dumps(c.embedding)) for c in chunks],
            )
        return len(chunks)

    def all(self) -> List[Chunk]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        chunks = []
        for row in rows:
            values = dict(row)
            values["embedding"] = json.loads(values["embedding"])
            chunks.append(Chunk(**values))
        return chunks
