from __future__ import annotations

from datetime import datetime
import re

from fpdf import FPDF


def generate_pdf_report(
    qa_log: list[dict],
    model_name: str,
    indexed_docs_count: int,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "NotesPilot Session Report", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Model: {model_name}", ln=True)
    pdf.cell(0, 8, f"Indexed docs (session): {indexed_docs_count}", ln=True)
    pdf.cell(0, 8, f"Q&A turns: {len(qa_log)}", ln=True)

    for i, item in enumerate(qa_log, start=1):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        _write_multiline(pdf, 7, f"Q{i}: {_safe_pdf_text(item.get('question', ''))}")

        pdf.set_font("Helvetica", "", 11)
        _write_multiline(pdf, 6, f"Answer: {_safe_pdf_text(item.get('answer', ''))}")
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, f"Confidence: {_safe_pdf_text(item.get('confidence', 'Unknown'))}", ln=True)

        citations = item.get("citations", []) or []
        if citations:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 6, "Sources:", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for cite in citations[:5]:
                line = f"- {cite.get('filename', 'unknown')} | page {cite.get('page_num', '?')} | {cite.get('chunk_id', '')}"
                _write_multiline(pdf, 5, _safe_pdf_text(line))
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 6, "No citations (insufficient evidence).", ln=True)

    output = pdf.output(dest="S")
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return output.encode("latin-1")


def _write_multiline(pdf: FPDF, height: float, text: str) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, height, text)


def _safe_pdf_text(value: str) -> str:
    text = (value or "").replace("\r\n", "\n").replace("\r", "\n")
    # Remove characters unsupported by built-in PDF fonts.
    text = "".join(ch for ch in text if (ch == "\n" or 32 <= ord(ch) <= 126))
    # Break very long tokens that can cause line-wrap failures.
    text = re.sub(r"([^\s]{40})(?=[^\s])", r"\1 ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()
