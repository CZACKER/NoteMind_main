from __future__ import annotations

import hashlib

from . import config
from .types import Chunk, Document


def make_chunks(
    document: Document,
    chunk_size_words: int = config.CHUNK_SIZE_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in document.pages:
        page_chunks = _chunk_page_text(
            doc_id=document.doc_id,
            filename=document.filename,
            page_num=page.page_num,
            text=page.text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
        chunks.extend(page_chunks)
    return chunks


def _chunk_page_text(
    doc_id: str,
    filename: str,
    page_num: int,
    text: str,
    chunk_size_words: int,
    overlap_words: int,
) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    cursor = 0

    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if not chunk_text:
            break

        char_start = text.find(chunk_words[0], cursor)
        if char_start < 0:
            char_start = cursor
        char_end = char_start + len(chunk_text)
        cursor = char_end

        chunk_id = _make_chunk_id(doc_id, page_num, chunk_text, len(chunks))
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            filename=filename,
            page_num=page_num,
            text=chunk_text,
            token_count=len(chunk_words),
            char_start=char_start,
            char_end=char_end,
            metadata={},
        )
        chunks.append(chunk)

        if end == len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def _make_chunk_id(doc_id: str, page_num: int, text: str, offset: int) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}_p{page_num}_c{offset}_{digest}"

