from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

import cv2
import fitz
import numpy as np

from .clean_text import clean_ocr_text
from .config import OCR_DPI
from .diagram_reader import extract_diagram_notes
from .types import Document, PageText
from .utils import make_doc_id
from .ocr import extract_page_text


def process_pdf(
    path: str | Path,
    dpi: int = OCR_DPI,
    max_pages: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Document:
    pdf_path = Path(path)
    content = pdf_path.read_bytes()
    doc_id = make_doc_id(pdf_path.name, content)

    pdf = fitz.open(pdf_path)
    pages: list[PageText] = []
    total_pages = len(pdf)
    final_pages = total_pages if max_pages is None else min(total_pages, max_pages)

    for i in range(final_pages):
        page = pdf[i]
        image = _render_page_image(page=page, dpi=dpi)
        native_text = (page.get_text("text") or "").strip()
        ocr_text = extract_page_text(image)

        # Use the better text source; some PDFs already contain clean text.
        native_score = _text_quality_score(native_text)
        ocr_score = _text_quality_score(ocr_text)
        raw_text = native_text if native_score >= ocr_score else ocr_text

        cleaned = clean_ocr_text(raw_text)
        diagram_notes = extract_diagram_notes(image)
        if diagram_notes:
            cleaned = f"{cleaned}\n\n[Diagram Notes]\n{diagram_notes}".strip()
        pages.append(PageText(page_num=i + 1, text=cleaned))
        if progress_cb is not None:
            progress_cb(i + 1, final_pages)

    pdf.close()
    return Document(doc_id=doc_id, filename=pdf_path.name, pages=pages)


def _render_page_image(page: fitz.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _text_quality_score(text: str) -> float:
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z]{2,}", text)
    alpha = sum(ch.isalpha() for ch in text)
    total = max(1, len(text))
    alpha_ratio = alpha / total
    return (0.02 * len(words)) + alpha_ratio
