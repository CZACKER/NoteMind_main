from __future__ import annotations

import re
from collections import Counter

from . import config
from .types import Chunk, FinalResponse


NO_INFO_TEXT = "I don't have enough information in the provided notes."


def validate(answer: str, contexts: list[Chunk], scores: list[float]) -> FinalResponse:
    if not contexts or not scores:
        return _reject("No retrieved context.")

    top_score = max(scores)
    if top_score < config.MIN_RETRIEVAL_SCORE:
        return _reject(f"Low retrieval confidence ({top_score:.2f}).")

    if answer.strip() == NO_INFO_TEXT:
        return FinalResponse(
            answer=NO_INFO_TEXT,
            citations=[],
            confidence=_confidence_label(scores, evidence_ok=False),
            rejected=False,
            reason="Model reported insufficient information.",
        )

    evidence_ok = _is_answer_supported(answer, contexts)
    if not evidence_ok:
        return _reject("Answer is not sufficiently grounded in retrieved evidence.", scores=scores)

    citations = [
        {
            "filename": chunk.filename,
            "page_num": chunk.page_num,
            "chunk_id": chunk.chunk_id,
            "preview": _preview(chunk.text),
        }
        for chunk in contexts
    ]
    return FinalResponse(
        answer=answer.strip(),
        citations=citations,
        confidence=_confidence_label(scores, evidence_ok=True),
        rejected=False,
        reason="ok",
    )


def _is_answer_supported(answer: str, contexts: list[Chunk]) -> bool:
    answer_tokens = _tokens(answer)
    if len(answer_tokens) < 4:
        return False
    answer_counter = Counter(answer_tokens)

    for chunk in contexts:
        ctx_counter = Counter(_tokens(chunk.text))
        common = sum((answer_counter & ctx_counter).values())
        overlap_ratio = common / max(1, len(answer_tokens))
        if overlap_ratio >= config.MIN_EVIDENCE_OVERLAP:
            return True
    return False


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _confidence_label(scores: list[float], evidence_ok: bool) -> str:
    top = max(scores) if scores else 0.0
    spread = top - (scores[1] if len(scores) > 1 else 0.0)
    if top >= 0.75 and spread >= 0.1 and evidence_ok:
        return "High"
    if top >= 0.5 and evidence_ok:
        return "Medium"
    return "Low"


def _preview(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _reject(reason: str, scores: list[float] | None = None) -> FinalResponse:
    return FinalResponse(
        answer=NO_INFO_TEXT,
        citations=[],
        confidence=_confidence_label(scores or [0.0], evidence_ok=False),
        rejected=True,
        reason=reason,
    )

