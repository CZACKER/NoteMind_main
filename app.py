from __future__ import annotations

import html
import fitz
import re
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


st.set_page_config(page_title="NotesPilot", layout="wide")

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
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""


def main() -> None:
    ensure_data_dirs()
    initialize_state()
    _inject_styles()

    encoder = get_encoder()
    store = get_store()

    profile, preset = _render_sidebar()
    ocr_dpi = preset["dpi"]
    top_k = preset["top_k"]
    num_predict = preset["num_predict"]
    timeout_s = config.OLLAMA_TIMEOUT_S

    answerer = get_answerer(timeout_s=int(timeout_s))
    retriever = Retriever(store=store, encoder=encoder)

    _render_header(profile=profile, store=store)

    left, center, right = st.columns([1.15, 2.15, 1.45], gap="large")
    with left:
        uploads = _render_upload_panel(encoder=encoder, store=store, ocr_dpi=ocr_dpi)
    with center:
        _render_chat_panel(
            retriever=retriever,
            answerer=answerer,
            top_k=top_k,
            num_predict=num_predict,
        )
    with right:
        _render_evidence_panel()

    st.markdown(
        f"""
        <div class="np-footer">
          <span>Offline mode</span>
          <span>Model: <code>{html.escape(config.OLLAMA_MODEL_NAME)}</code></span>
          <span>Index DB: <code>{html.escape(str(config.VECTOR_DB_PATH))}</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _ = uploads  # suppress unused variable warning semantics


def _render_sidebar() -> tuple[str, dict]:
    st.sidebar.markdown(
        """
        <div class="np-sidebrand">
          <div class="np-sidebrand-icon">NP</div>
          <div>
            <div class="np-sidebrand-title">NotesPilot</div>
            <div class="np-sidebrand-sub">Offline Handwritten QA</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Workspace")
    profile = st.sidebar.selectbox("Performance profile", list(PERFORMANCE_PRESETS.keys()), index=1)
    preset = PERFORMANCE_PRESETS[profile]

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.qa_log = []
        st.session_state.last_contexts = []
        st.session_state.recent_sources = []
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
    return profile, preset


def _render_header(profile: str, store: VectorStore) -> None:
    st.markdown(
        """
        <div class="np-header">
          <div class="np-header-title">NotesPilot</div>
          <div class="np-header-sub">Modern offline AI assistant for handwritten and diagram-rich notes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(_metric_card("Profile", profile), unsafe_allow_html=True)
    m2.markdown(_metric_card("Indexed Chunks", str(store.count())), unsafe_allow_html=True)
    m3.markdown(_metric_card("Documents", str(len(st.session_state.indexed_docs))), unsafe_allow_html=True)
    m4.markdown(_metric_card("Status", "Ready"), unsafe_allow_html=True)


def _metric_card(label: str, value: str) -> str:
    return (
        '<div class="np-metric">'
        f'<div class="np-metric-value">{html.escape(value)}</div>'
        f'<div class="np-metric-label">{html.escape(label)}</div>'
        "</div>"
    )


def _render_upload_panel(encoder: EmbeddingEncoder, store: VectorStore, ocr_dpi: int):
    st.markdown("### Upload & Index")
    st.markdown('<div class="np-section-sub">Drag PDFs and build your local search index.</div>', unsafe_allow_html=True)
    uploads = st.file_uploader("Add one or more PDF notes", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")

    if uploads:
        st.markdown('<div class="np-file-grid">', unsafe_allow_html=True)
        for up in uploads:
            size_mb = up.size / (1024 * 1024)
            st.markdown(
                f"""
                <div class="np-file-card">
                  <div class="np-file-icon">PDF</div>
                  <div class="np-file-meta">
                    <div class="np-file-name">{html.escape(up.name)}</div>
                    <div class="np-file-size">{size_mb:.2f} MB</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    disabled = not bool(uploads)
    if st.button("Process PDFs", type="primary", use_container_width=True, disabled=disabled):
        process_uploads(
            uploads=uploads,
            encoder=encoder,
            store=store,
            ocr_dpi=int(ocr_dpi),
            max_pages=None,
        )
        st.success("Processing complete.")
    return uploads


def _render_chat_panel(retriever: Retriever, answerer: AnswerGenerator, top_k: int, num_predict: int) -> None:
    st.markdown("### Ask Questions")
    st.markdown('<div class="np-section-sub">Chat with your notes. Every answer is evidence-grounded.</div>', unsafe_allow_html=True)

    chat_shell = st.container()
    with chat_shell:
        for msg in st.session_state.memory.get():
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    question = st.chat_input("Ask from your uploaded notes only...")
    if question:
        st.session_state.last_question = question
        st.session_state.memory.add_user(question)
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Retrieving evidence and generating answer..."):
            effective_top_k = _get_query_top_k(question, top_k)
            chunks, scores = retriever.search(question, top_k=effective_top_k)
            chunks, scores = _rerank_chunks_for_query(chunks, scores, question)
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
            st.markdown(
                f'<div class="np-confidence">Confidence: <strong>{html.escape(final.confidence)}</strong></div>',
                unsafe_allow_html=True,
            )
            if _is_diagram_query(question):
                _show_cited_diagram_pages(
                    citations=final.citations,
                    query=question,
                    contexts=chunks,
                )
            if final.citations:
                st.markdown("**Sources**")
                for cite in final.citations[:4]:
                    st.markdown(
                        f'<div class="np-source-chip">{html.escape(cite["filename"])} • page {cite["page_num"]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption(f"Reason: {final.reason}")


def _render_evidence_panel() -> None:
    st.markdown("### Evidence Viewer")
    st.markdown('<div class="np-section-sub">Retrieved chunks used to ground the answer.</div>', unsafe_allow_html=True)
    contexts = st.session_state.last_contexts
    query = st.session_state.last_question
    if not contexts:
        st.info("Ask a question to inspect supporting chunks.")
    else:
        for idx, chunk in enumerate(contexts, start=1):
            preview = _highlight_query_terms((chunk.text[:900] or ""), query)
            st.markdown(
                f"""
                <div class="np-evidence-card">
                  <div class="np-evidence-top">
                    <span class="np-evidence-index">#{idx}</span>
                    <span class="np-evidence-meta">{html.escape(chunk.filename)} • page {chunk.page_num}</span>
                  </div>
                  <div class="np-evidence-body">{preview}</div>
                  <div class="np-evidence-id">{html.escape(chunk.chunk_id)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Latest Sources")
    if not st.session_state.recent_sources:
        st.caption("No sources yet.")
    else:
        for src in st.session_state.recent_sources[:6]:
            st.markdown(
                f'<div class="np-latest-item">{html.escape(src["filename"])} • page {src["page_num"]}</div>',
                unsafe_allow_html=True,
            )


def _highlight_query_terms(text: str, query: str) -> str:
    escaped = html.escape(text)
    terms = [t for t in _keyword_tokens(query) if len(t) > 3][:8]
    if not terms:
        return escaped.replace("\n", "<br>")
    for term in terms:
        pattern = re.compile(rf"(?i)\b({re.escape(term)})\b")
        escaped = pattern.sub(r"<mark>\1</mark>", escaped)
    return escaped.replace("\n", "<br>")


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
          --bg: #f7f9fc;
          --card: #ffffff;
          --ink: #0f172a;
          --muted: #64748b;
          --line: #e2e8f0;
          --accent: #2563eb;
          --accent-2: #22c55e;
          --danger: #ef4444;
          --shadow: 0 10px 28px rgba(2, 6, 23, 0.06);
          --radius: 16px;
        }
        .stApp {
          background: radial-gradient(circle at 0% 0%, #fdfefe, #f3f7fb 45%, #eef3f9 100%);
          color: var(--ink);
        }
        .main .block-container {
          max-width: 1420px;
          padding-top: 1rem;
          padding-bottom: 1.8rem;
          animation: fadeIn .35s ease-out;
        }
        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, #0f172a, #111827);
          color: #f8fafc !important;
          border-right: 1px solid #1f2937;
        }
        [data-testid="stSidebar"] * {
          color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div {
          background: #1f2937 !important;
          border: 1px solid #334155 !important;
        }
        .np-sidebrand {
          display: flex; gap: .7rem; align-items: center; margin-bottom: 1rem;
          padding: .4rem .2rem;
        }
        .np-sidebrand-icon {
          width: 34px; height: 34px; border-radius: 10px;
          display: grid; place-items: center; font-weight: 800;
          background: linear-gradient(135deg, #2563eb, #06b6d4); color: #fff;
        }
        .np-sidebrand-title { font-weight: 700; font-size: 1.02rem; color: #f8fafc; }
        .np-sidebrand-sub { font-size: .78rem; color: #94a3b8; }
        .np-header {
          margin-bottom: .6rem;
          padding: .65rem .2rem .2rem .2rem;
        }
        .np-header-title {
          font-size: clamp(1.8rem, 2.6vw, 2.8rem);
          font-weight: 800;
          letter-spacing: -0.02em;
          color: #0f172a;
        }
        .np-header-sub {
          color: var(--muted);
          margin-top: .28rem;
        }
        .np-metric {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 14px;
          box-shadow: var(--shadow);
          padding: .72rem .82rem;
          margin-bottom: .65rem;
        }
        .np-metric-value {
          font-weight: 800;
          color: #0f172a;
          font-size: 1rem;
        }
        .np-metric-label {
          color: #64748b;
          font-size: .78rem;
          margin-top: .1rem;
        }
        [data-testid="column"] > div {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: var(--radius);
          box-shadow: var(--shadow);
          padding: .95rem 1rem;
          height: 100%;
        }
        h3 {
          font-size: 1.1rem !important;
          font-weight: 700 !important;
          margin-bottom: .35rem !important;
        }
        .np-section-sub {
          color: #64748b;
          font-size: .84rem;
          margin-bottom: .5rem;
        }
        .np-file-grid { display: grid; gap: .5rem; margin: .4rem 0 .8rem 0; }
        .np-file-card {
          display: grid; grid-template-columns: 46px 1fr; gap: .58rem;
          padding: .58rem; border: 1px solid var(--line); border-radius: 12px;
          background: #f8fafc;
          transition: all .2s ease;
        }
        .np-file-card:hover { transform: translateY(-1px); box-shadow: 0 8px 20px rgba(2,6,23,.08); }
        .np-file-icon {
          width: 46px; height: 46px; border-radius: 10px; display: grid; place-items: center;
          font-size: .72rem; font-weight: 700; color: #fff;
          background: linear-gradient(135deg, #ef4444, #f97316);
        }
        .np-file-name { font-weight: 600; color: #0f172a; }
        .np-file-size { color: #64748b; font-size: .78rem; }
        .stButton > button, .stDownloadButton > button {
          border-radius: 12px !important;
          font-weight: 700 !important;
          transition: all .2s ease !important;
        }
        .stButton > button[kind="primary"] {
          background: linear-gradient(90deg, #2563eb, #4f46e5) !important;
          border: 0 !important;
          color: #fff !important;
          box-shadow: 0 10px 22px rgba(37,99,235,.28);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
          transform: translateY(-1px);
          filter: brightness(1.03);
        }
        .stButton > button:disabled {
          opacity: .55 !important;
          cursor: not-allowed !important;
        }
        .stDownloadButton > button {
          background: linear-gradient(90deg, #0f766e, #16a34a) !important;
          color: #fff !important;
          border: 0 !important;
        }
        .stChatInputContainer {
          border-radius: 14px;
          border: 1px solid var(--line);
          background: #fff;
          box-shadow: var(--shadow);
        }
        .stChatMessage {
          background: #fff;
          border: 1px solid var(--line);
          border-radius: 14px;
          box-shadow: 0 6px 16px rgba(2,6,23,.05);
          animation: slideUp .22s ease-out;
        }
        .np-confidence {
          margin-top: .3rem; margin-bottom: .35rem;
          color: #334155; font-size: .9rem;
        }
        .np-source-chip {
          display: inline-flex; align-items: center;
          margin: .14rem .3rem .14rem 0; padding: .22rem .56rem;
          border-radius: 999px; border: 1px solid #bfdbfe;
          background: #eff6ff; color: #1e3a8a;
          font-size: .78rem; font-weight: 600;
        }
        .np-evidence-card {
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #fff;
          padding: .65rem .72rem;
          margin-bottom: .58rem;
          transition: all .18s ease;
        }
        .np-evidence-card:hover {
          box-shadow: var(--shadow);
          transform: translateY(-1px);
        }
        .np-evidence-top {
          display: flex; justify-content: space-between; gap: .5rem;
          margin-bottom: .38rem;
        }
        .np-evidence-index {
          font-size: .72rem; font-weight: 700;
          color: #1d4ed8; background: #dbeafe;
          border-radius: 999px; padding: .12rem .45rem;
        }
        .np-evidence-meta { color: #334155; font-size: .78rem; font-weight: 600; }
        .np-evidence-body { color: #334155; font-size: .84rem; line-height: 1.42; }
        .np-evidence-body mark {
          background: #fef08a;
          border-radius: 4px;
          padding: 0 .08rem;
        }
        .np-evidence-id { margin-top: .35rem; color: #94a3b8; font-size: .72rem; }
        .np-latest-item {
          border: 1px solid var(--line);
          border-radius: 10px;
          padding: .34rem .5rem;
          margin-bottom: .32rem;
          font-size: .8rem;
          color: #334155;
          background: #f8fafc;
        }
        .np-footer {
          display: flex; gap: .8rem; flex-wrap: wrap;
          margin-top: .9rem; padding: .62rem .7rem;
          border: 1px solid var(--line); border-radius: 12px; background: #fff;
          color: #64748b; font-size: .8rem;
        }
        @media (max-width: 1100px) {
          .main .block-container { padding-left: .8rem; padding-right: .8rem; }
        }
        @media (max-width: 900px) {
          [data-testid="column"] > div { padding: .74rem .78rem; }
          .np-header-title { font-size: 1.7rem; }
          .np-footer { font-size: .74rem; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _is_diagram_query(query: str) -> bool:
    q = query.lower()
    return any(token in q for token in ["diagram", "flowchart", "draw", "model", "network"])


def _show_cited_diagram_pages(citations: list[dict], query: str, contexts: list) -> None:
    st.markdown("**Diagram from notes (original PDF page):**")
    ranked_keys = _rank_cited_pages(query=query, citations=citations, contexts=contexts)
    preferred = _best_page_from_processed(query=query, citations=citations)
    if preferred is not None:
        ranked_keys = [preferred] + [k for k in ranked_keys if k != preferred]
    if not ranked_keys:
        ranked_keys = [
            (str(c.get("filename", "")), int(c.get("page_num", 0)))
            for c in citations
            if c.get("filename") and int(c.get("page_num", 0)) > 0
        ]
    if not ranked_keys and preferred is not None:
        ranked_keys = [preferred]
    if not ranked_keys:
        st.info("No matching diagram page found in processed notes for this query.")
        return

    shown = 0
    seen: set[tuple[str, int]] = set()
    for filename, page_num in ranked_keys:
        key = (filename, page_num)
        if page_num <= 0 or key in seen:
            continue
        image = _render_pdf_page_png(filename, page_num)
        if image is None:
            continue
        st.image(image, caption=f"{filename} - page {page_num}", use_container_width=True)
        seen.add(key)
        shown += 1
        if shown >= 2:
            break
    if shown == 0:
        st.info("Original diagram page could not be loaded from local PDF. Using text-based explanation only.")


def _rank_cited_pages(query: str, citations: list[dict], contexts: list) -> list[tuple[str, int]]:
    query_tokens = set(_keyword_tokens(query))
    candidate_pages: set[tuple[str, int]] = set()
    for c in citations:
        filename = str(c.get("filename", ""))
        page_num = int(c.get("page_num", 0))
        if filename and page_num > 0:
            candidate_pages.add((filename, page_num))

    if not candidate_pages:
        return []

    scores: dict[tuple[str, int], float] = {k: 0.0 for k in candidate_pages}
    for chunk in contexts:
        key = (getattr(chunk, "filename", ""), int(getattr(chunk, "page_num", 0)))
        if key not in scores:
            continue
        text = (getattr(chunk, "text", "") or "").lower()
        tokens = set(_keyword_tokens(text))
        overlap = len(query_tokens & tokens)
        score = float(overlap)

        if "precedence network" in query.lower():
            if "precedence network" in text:
                score += 6.0
            if "activity relationships" in text:
                score += 3.0
            if "program test" in text:
                score += 2.0
            if "code" in text and "data take-on" in text:
                score += 2.0

        if "fig." in text or "figure" in text or "[diagram notes]" in text:
            score += 1.5
        scores[key] += score

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked]


def _keyword_tokens(text: str) -> list[str]:
    stop = {
        "the", "a", "an", "is", "are", "of", "in", "to", "for", "and", "with",
        "give", "neat", "diagram", "explain", "detail", "details",
    }
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in stop and len(t) > 2]


def _get_query_top_k(query: str, base_top_k: int) -> int:
    if _is_diagram_query(query):
        return min(12, base_top_k + 4)
    return base_top_k


def _rerank_chunks_for_query(chunks: list, scores: list[float], query: str) -> tuple[list, list[float]]:
    if not chunks:
        return chunks, scores
    q_tokens = set(_keyword_tokens(query))
    enriched = []
    for chunk, sim in zip(chunks, scores, strict=False):
        text = (getattr(chunk, "text", "") or "").lower()
        c_tokens = set(_keyword_tokens(text))
        overlap = len(q_tokens & c_tokens)
        bonus = 0.12 * overlap
        if "overall structure" in query.lower() and "overall structure" in text:
            bonus += 1.8
        if "precedence network" in query.lower() and "precedence network" in text:
            bonus += 1.8
        if "[diagram notes]" in text or "fig." in text or "figure" in text:
            bonus += 0.4
        enriched.append((chunk, float(sim) + bonus))
    enriched.sort(key=lambda x: x[1], reverse=True)
    out_chunks = [x[0] for x in enriched]
    out_scores = [x[1] for x in enriched]
    return out_chunks, out_scores


@st.cache_data(show_spinner=False)
def _load_processed_pages() -> list[dict]:
    pages: list[dict] = []
    for path in config.PROCESSED_DIR.glob("*.json"):
        try:
            payload = load_json(path)
        except Exception:
            continue
        filename = str(payload.get("filename", ""))
        for page in payload.get("pages", []):
            pages.append(
                {
                    "filename": filename,
                    "page_num": int(page.get("page_num", 0)),
                    "text": str(page.get("text", "")),
                }
            )
    return pages


def _best_page_from_processed(query: str, citations: list[dict]) -> tuple[str, int] | None:
    pages = _load_processed_pages()
    if not pages:
        return None

    cited_files = {str(c.get("filename", "")) for c in citations if c.get("filename")}
    q = query.lower()
    q_tokens = set(_keyword_tokens(query))
    required_terms = _required_terms_for_query(q)

    best_key: tuple[str, int] | None = None
    best_score = 0.0
    for row in pages:
        filename = row["filename"]
        if cited_files and filename not in cited_files:
            continue
        text = row["text"].lower()
        if required_terms and not _required_terms_present(required_terms, text):
            continue
        tokens = set(_keyword_tokens(text))
        score = float(len(q_tokens & tokens))
        if "overall structure" in q and "overall structure" in text:
            score += 10.0
        if "dbms" in q and "dbms" in text:
            score += 3.0
        if "precedence network" in q and "precedence network" in text:
            score += 10.0
        if "network planning model" in q and "network planning model" in text:
            score += 10.0
        if "[diagram notes]" in text or "fig." in text or "figure" in text:
            score += 1.2
        if "dbms" in q and "dbms" in filename.lower():
            score += 2.0
        if "precedence" in q and "notes" in filename.lower():
            score += 1.0

        if score > best_score:
            best_score = score
            best_key = (filename, int(row["page_num"]))

    if best_key is None or best_score < 2.0:
        return None
    return best_key


def _required_terms_for_query(q: str) -> list[str]:
    must: list[str] = []
    if "overall structure" in q:
        must.extend(["overall", "structure"])
    if "dbms" in q:
        must.append("dbms")
    if "precedence network" in q:
        must.extend(["precedence", "network"])
    if "network planning model" in q:
        must.extend(["network", "planning", "model"])
    if "er model" in q or "e-r model" in q:
        must.extend(["entity", "relationship"])
    seen = set()
    out: list[str] = []
    for t in must:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _required_terms_present(required_terms: list[str], page_text: str) -> bool:
    text_tokens = set(_keyword_tokens(page_text))
    matched = sum(1 for term in required_terms if term in text_tokens or term in page_text)
    if len(required_terms) <= 2:
        return matched >= len(required_terms)
    return matched >= max(2, len(required_terms) - 1)


@st.cache_data(show_spinner=False)
def _render_pdf_page_png(filename: str, page_num: int, dpi: int = 220) -> bytes | None:
    pdf_path = config.RAW_DIR / filename
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(str(pdf_path))
        index = page_num - 1
        if index < 0 or index >= len(doc):
            doc.close()
            return None
        pix = doc[index].get_pixmap(dpi=dpi, alpha=False)
        image_bytes = pix.tobytes("png")
        doc.close()
        return image_bytes
    except Exception:
        return None


if __name__ == "__main__":
    main()

