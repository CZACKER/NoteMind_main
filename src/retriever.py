from __future__ import annotations

from . import config
from .embeddings import EmbeddingEncoder
from .types import Chunk
from .vector_store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, encoder: EmbeddingEncoder) -> None:
        self.store = store
        self.encoder = encoder

    def search(self, query: str, top_k: int = config.TOP_K) -> tuple[list[Chunk], list[float]]:
        query_vector = self.encoder.encode_query(query)
        hits = self.store.search(query_vector=query_vector, top_k=top_k)
        chunks = [self._hit_to_chunk(hit) for hit in hits]
        scores = [float(hit["similarity"]) for hit in hits]
        return chunks, scores

    @staticmethod
    def _hit_to_chunk(hit: dict) -> Chunk:
        meta = hit.get("metadata", {})
        return Chunk(
            chunk_id=str(hit.get("chunk_id", "")),
            doc_id=str(meta.get("doc_id", "")),
            filename=str(meta.get("filename", "")),
            page_num=int(meta.get("page_num", 0)),
            text=str(hit.get("text", "")),
            token_count=int(meta.get("token_count", 0)),
            char_start=int(meta.get("char_start", 0)),
            char_end=int(meta.get("char_end", 0)),
            metadata={k: v for k, v in meta.items() if k not in {"doc_id", "filename", "page_num", "token_count", "char_start", "char_end"}},
        )

