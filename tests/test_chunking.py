from src.chunker import make_chunks
from src.types import Document, PageText


def test_chunking_produces_overlap_and_metadata() -> None:
    text = " ".join(f"word{i}" for i in range(120))
    doc = Document(doc_id="doc1", filename="notes.pdf", pages=[PageText(page_num=1, text=text)])
    chunks = make_chunks(doc, chunk_size_words=50, overlap_words=10)

    assert len(chunks) == 3
    assert chunks[0].page_num == 1
    assert chunks[0].doc_id == "doc1"
    assert chunks[1].text.split()[0] == "word40"
    assert chunks[0].char_end > chunks[0].char_start

