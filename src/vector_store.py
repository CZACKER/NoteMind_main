from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .types import Chunk


class VectorStore:
    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = config.COLLECTION_NAME,
    ) -> None:
        if persist_dir is None:
            self.db_path = str(config.VECTOR_DB_PATH)
        else:
            self.db_path = str(Path(persist_dir))
        self.collection_name = collection_name
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        rows = []
        for chunk, vector in zip(chunks, vectors, strict=False):
            rows.append(
                (
                    chunk.chunk_id,
                    chunk.text,
                    json.dumps(self._metadata_from_chunk(chunk), ensure_ascii=False),
                    json.dumps(vector),
                )
            )
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT INTO vectors (chunk_id, text, metadata_json, vector_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
              text=excluded.text,
              metadata_json=excluded.metadata_json,
              vector_json=excluded.vector_json
            """,
            rows,
        )
        self._conn.commit()

    def search(self, query_vector: list[float], top_k: int = config.TOP_K) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT chunk_id, text, metadata_json, vector_json FROM vectors")
        rows = cur.fetchall()
        hits: list[dict[str, Any]] = []
        if not rows:
            return hits

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return hits

        for chunk_id, text, metadata_json, vector_json in rows:
            v = np.array(json.loads(vector_json), dtype=np.float32)
            v_norm = np.linalg.norm(v)
            similarity = 0.0 if v_norm == 0 else float(np.dot(q, v) / (q_norm * v_norm))
            distance = float(1.0 - similarity)
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": json.loads(metadata_json) if metadata_json else {},
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        hits.sort(key=lambda x: x["similarity"], reverse=True)
        return hits[:top_k]

    def count(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM vectors")
        return int(cur.fetchone()[0])

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                vector_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _metadata_from_chunk(chunk: Chunk) -> dict[str, Any]:
        data = {
            "doc_id": chunk.doc_id,
            "filename": chunk.filename,
            "page_num": chunk.page_num,
            "token_count": chunk.token_count,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }
        data.update(chunk.metadata)
        return data
