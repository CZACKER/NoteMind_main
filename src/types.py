from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageText:
    page_num: int
    text: str


@dataclass
class Document:
    doc_id: str
    filename: str
    pages: list[PageText] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_num: int
    text: str
    token_count: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerResult:
    answer: str
    model: str
    raw_response: str


@dataclass
class FinalResponse:
    answer: str
    citations: list[dict[str, Any]]
    confidence: str
    rejected: bool
    reason: str

