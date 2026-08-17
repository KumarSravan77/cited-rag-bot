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
                page_end INTEGER NOT NULL DEFAULT 1, headings TEXT NOT NULL DEFAULT '[]',
                content_types TEXT NOT NULL DEFAULT '[]', text TEXT NOT NULL, embedding TEXT NOT NULL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(chunks)")}
            migrations = {
                "page_end": "ALTER TABLE chunks ADD COLUMN page_end INTEGER NOT NULL DEFAULT 1",
                "headings": "ALTER TABLE chunks ADD COLUMN headings TEXT NOT NULL DEFAULT '[]'",
                "content_types": "ALTER TABLE chunks ADD COLUMN content_types TEXT NOT NULL DEFAULT '[]'",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    db.execute(statement)
            db.execute("UPDATE chunks SET page_end=page WHERE page_end < page")

    def connect(self):
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def replace_document(self, document: str, chunks: List[Chunk]) -> int:
        with self.connect() as db:
            db.execute("DELETE FROM chunks WHERE document=?", (document,))
            db.executemany(
                """INSERT INTO chunks(document,page,page_end,headings,content_types,text,embedding)
                   VALUES(?,?,?,?,?,?,?)""",
                [(c.document, c.page, c.page_end, json.dumps(c.headings),
                  json.dumps(c.content_types), c.text, json.dumps(c.embedding)) for c in chunks],
            )
        return len(chunks)

    def all(self) -> List[Chunk]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM chunks ORDER BY id").fetchall()
        chunks = []
        for row in rows:
            values = dict(row)
            values["embedding"] = json.loads(values["embedding"])
            values["headings"] = json.loads(values["headings"])
            values["content_types"] = json.loads(values["content_types"])
            chunks.append(Chunk(**values))
        return chunks
