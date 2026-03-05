from src.guardrails import NO_INFO_TEXT, validate
from src.types import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="c1",
        doc_id="d1",
        filename="f.pdf",
        page_num=1,
        text=text,
        token_count=len(text.split()),
        char_start=0,
        char_end=len(text),
    )


def test_guardrails_reject_low_score() -> None:
    chunk = _chunk("neural networks use weighted connections between nodes")
    out = validate("Neural networks use weighted nodes.", [chunk], [0.2])
    assert out.answer == NO_INFO_TEXT
    assert out.rejected is True


def test_guardrails_accept_grounded_answer() -> None:
    chunk = _chunk("Photosynthesis converts light energy into chemical energy in plants.")
    out = validate("Photosynthesis converts light energy into chemical energy in plants.", [chunk], [0.8])
    assert out.rejected is False
    assert out.citations

