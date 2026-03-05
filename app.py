from __future__ import annotations

import streamlit as st

from src import config
from src.answerer import AnswerGenerator, LLMRuntimeError
from src.chunker import make_chunks
from src.embeddings import EmbeddingEncoder
from src.guardrails import validate
from src.ingest_pdf import process_pdf
from src.memory import ConversationMemory
from src.ocr import OCRDependencyError
from src.reporting import generate_pdf_report
from src.retriever import Retriever
from src.types import Document, FinalResponse, PageText
from src.utils import ensure_data_dirs, load_json, make_doc_id, save_json
from src.vector_store import VectorStore


st.set_page_config(page_title="Offline Notes QA", layout="wide")

PERFORMANCE_PRESETS = {
    "Fast": {"dpi": 170, "top_k": 4, "num_predict": 180},
    "Balanced": {"dpi": 200, "top_k": 6, "num_predict": 260},
    "Accurate": {"dpi": 240, "top_k": 8, "num_predict": 350},
}


@st.cache_resource(show_spinner=False)
def get_encoder() -> EmbeddingEncoder:
    return EmbeddingEncoder()


@st.cache_resource(show_spinner=False)
def get_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner=False)
def get_answerer(timeout_s: int) -> AnswerGenerator:
    return AnswerGenerator(timeout_s=timeout_s)


def initialize_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []
    if "indexed_docs" not in st.session_state:
        st.session_state.indexed_docs = set()
    if "recent_sources" not in st.session_state:
        st.session_state.recent_sources = []
    if "qa_log" not in st.session_state:
        st.session_state.qa_log = []


def main() -> None:
    ensure_data_dirs()
    initialize_state()

    encoder = get_encoder()
    store = get_store()
    _inject_styles()

    st.sidebar.title("Control Panel")
    profile = st.sidebar.selectbox("Performance profile", list(PERFORMANCE_PRESETS.keys()), index=1)
    preset = PERFORMANCE_PRESETS[profile]
    ocr_dpi = preset["dpi"]
    top_k = preset["top_k"]
    num_predict = preset["num_predict"]
    timeout_s = config.OLLAMA_TIMEOUT_S

    if st.sidebar.button("Clear chat"):
        st.session_state.memory.clear()
        st.session_state.qa_log = []
        st.success("Chat cleared.")

    report_bytes = generate_pdf_report(
        qa_log=st.session_state.qa_log,
        model_name=config.OLLAMA_MODEL_NAME,
        indexed_docs_count=len(st.session_state.indexed_docs),
    )
    st.sidebar.download_button(
        "Download PDF Report",
        data=report_bytes,
        file_name="notespilot_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    answerer = get_answerer(timeout_s=int(timeout_s))
    retriever = Retriever(store=store, encoder=encoder)

    st.title("NotesPilot: Offline Handwritten Notes QA")
    st.caption("Upload notes, build index, ask questions, and always get source-grounded answers.")

    left, center, right = st.columns([1.25, 2.2, 1.4], gap="large")

    with left:
        st.markdown("### Upload & Index")
        uploads = st.file_uploader("Add one or more PDF notes", type=["pdf"], accept_multiple_files=True)
        if st.button("Process PDFs", type="primary", use_container_width=True):
            if not uploads:
                st.warning("Upload at least one PDF.")
            else:
                process_uploads(
                    uploads=uploads,
                    encoder=encoder,
                    store=store,
                    ocr_dpi=int(ocr_dpi),
                    max_pages=None,
                )
                st.success("Processing complete.")

    with center:
        st.markdown("### Ask Questions")
        for msg in st.session_state.memory.get():
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask from your uploaded notes only...")
        if question:
            st.session_state.memory.add_user(question)
            with st.chat_message("user"):
                st.write(question)

            with st.spinner("Retrieving evidence and generating answer..."):
                chunks, scores = retriever.search(question, top_k=top_k)
                st.session_state.last_contexts = chunks
                try:
                    raw = answerer.generate_answer(
                        question,
                        chunks,
                        st.session_state.memory.get(),
                        num_predict=num_predict,
                    )
                    final = validate(raw.answer, chunks, scores)
                except LLMRuntimeError as exc:
                    final = FinalResponse(
                        answer=(
                            "I couldn't generate the answer because local model memory is full. "
                            "Please use a smaller model or close other GPU-heavy apps."
                        ),
                        citations=[],
                        confidence="Low",
                        rejected=True,
                        reason=str(exc),
                    )
                st.session_state.memory.add_assistant(final.answer)
                st.session_state.recent_sources = final.citations
                st.session_state.qa_log.append(
                    {
                        "question": question,
                        "answer": final.answer,
                        "confidence": final.confidence,
                        "citations": final.citations,
                    }
                )

            with st.chat_message("assistant"):
                st.write(final.answer)
                st.caption(f"Confidence: {final.confidence}")
                if final.citations:
                    st.markdown("**Citations:**")
                    for cite in final.citations[:4]:
                        st.write(f"- {cite['filename']} | page {cite['page_num']} | {cite['chunk_id']}")
                else:
                    st.caption(f"Reason: {final.reason}")

    with right:
        st.markdown("### Evidence Viewer")
        contexts = st.session_state.last_contexts
        if not contexts:
            st.info("Ask a question to inspect supporting chunks.")
        else:
            for idx, chunk in enumerate(contexts, start=1):
                with st.expander(f"{idx}. {chunk.filename} | page {chunk.page_num}", expanded=False):
                    st.code(chunk.text[:900], language="text")
                    st.caption(chunk.chunk_id)

        st.markdown("### Latest Sources")
        if not st.session_state.recent_sources:
            st.caption("No sources yet.")
        else:
            for source in st.session_state.recent_sources[:4]:
                st.write(f"- {source['filename']} | p.{source['page_num']}")

    st.caption(
        f"Offline target mode enabled. Model: {config.OLLAMA_MODEL_NAME} | "
        f"Index DB: {config.VECTOR_DB_PATH}"
    )


def process_uploads(
    uploads,
    encoder: EmbeddingEncoder,
    store: VectorStore,
    ocr_dpi: int,
    max_pages: int | None,
) -> None:
    progress = st.progress(0)
    status = st.empty()
    total = len(uploads)

    for i, upload in enumerate(uploads, start=1):
        raw_bytes = upload.getvalue()
        path = config.RAW_DIR / upload.name
        path.write_bytes(raw_bytes)
        status.write(f"Processing {upload.name} ({i}/{total})...")

        try:
            doc_id = make_doc_id(upload.name, raw_bytes)
            cached_path = config.PROCESSED_DIR / f"{doc_id}.json"
            if cached_path.exists():
                payload = load_json(cached_path)
                document = Document(
                    doc_id=payload["doc_id"],
                    filename=payload["filename"],
                    pages=[PageText(page_num=p["page_num"], text=p["text"]) for p in payload["pages"]],
                )
            else:
                document = process_pdf(
                    path,
                    dpi=ocr_dpi,
                    max_pages=max_pages,
                    progress_cb=lambda page_i, page_total: status.write(
                        f"OCR {upload.name}: page {page_i}/{page_total}"
                    ),
                )
        except OCRDependencyError as exc:
            status.error(str(exc))
            st.error(
                "OCR setup needed. Install Tesseract and restart Streamlit. "
                "If already installed, set env var `TESSERACT_CMD` to the full path of `tesseract.exe`."
            )
            return

        chunks = make_chunks(document)
        vectors = encoder.encode_chunks(chunks)
        store.upsert(chunks, vectors)

        serializable = {
            "doc_id": document.doc_id,
            "filename": document.filename,
            "pages": [{"page_num": p.page_num, "text": p.text} for p in document.pages],
            "chunk_count": len(chunks),
        }
        save_json(config.PROCESSED_DIR / f"{document.doc_id}.json", serializable)
        st.session_state.indexed_docs.add(document.doc_id)
        progress.progress(i / total)

    status.success("All PDFs processed and indexed.")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --brand-ink: #12343b;
          --brand-bg: #f4efe7;
          --brand-card: #ffffff;
          --brand-accent: #1e6f5c;
        }
        .stApp {
          background: radial-gradient(circle at 10% 10%, #f7f4ee 0%, #efe6d8 50%, #eadfcf 100%);
          color: var(--brand-ink);
        }
        .stChatMessage {
          border: 1px solid #dfd3c3;
          border-radius: 14px;
          background: var(--brand-card);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
