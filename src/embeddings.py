from __future__ import annotations

import hashlib
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config
from .types import Chunk


class EmbeddingEncoder:
    def __init__(self, model_name: str = config.EMBED_MODEL_NAME) -> None:
        self.model_name = model_name
        self._backend = "sentence_transformers"
        self._dim = 384
        self._model = None
        try:
            self._model = SentenceTransformer(model_name)
        except Exception:
            # Offline-safe fallback: deterministic local hashing embeddings.
            self._backend = "hash_fallback"

    def encode_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        texts = [chunk.text for chunk in chunks]
        if not texts:
            return []
        return self._encode_texts(texts)

    def encode_query(self, query: str) -> list[float]:
        return self._encode_texts([query])[0]

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        if self._backend == "sentence_transformers" and self._model is not None:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return vectors.tolist()
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> list[float]:
        vector = np.zeros(self._dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        if not tokens:
            return vector.tolist()

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            weight = 1.0 + (digest[5] % 3) * 0.2
            vector[idx] += sign * weight

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()
