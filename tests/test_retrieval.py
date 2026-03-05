from src.retriever import Retriever


def test_hit_to_chunk_parsing() -> None:
    hit = {
        "chunk_id": "x1",
        "text": "sample chunk text",
        "metadata": {
            "doc_id": "docA",
            "filename": "a.pdf",
            "page_num": 2,
            "token_count": 3,
            "char_start": 10,
            "char_end": 26,
            "section": "intro",
        },
    }
    chunk = Retriever._hit_to_chunk(hit)
    assert chunk.doc_id == "docA"
    assert chunk.filename == "a.pdf"
    assert chunk.page_num == 2
    assert chunk.metadata["section"] == "intro"

