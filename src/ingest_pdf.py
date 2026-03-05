from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
from typing import Callable

import cv2
import fitz
import numpy as np

from .clean_text import clean_ocr_text
from .config import NATIVE_TEXT_SCORE_THRESHOLD, OCR_DPI, OCR_MAX_WORKERS
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
    total_pages = len(pdf)
    pdf.close()
    final_pages = total_pages if max_pages is None else min(total_pages, max_pages)
    pages: list[PageText] = [PageText(page_num=i + 1, text="") for i in range(final_pages)]

    worker_count = max(1, min(OCR_MAX_WORKERS, final_pages))
    if worker_count == 1:
        for i in range(final_pages):
            page_text = _process_page(pdf_path=pdf_path, page_index=i, dpi=dpi)
            pages[i] = page_text
            if progress_cb is not None:
                progress_cb(i + 1, final_pages)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            future_map = {
                ex.submit(_process_page, pdf_path=pdf_path, page_index=i, dpi=dpi): i
                for i in range(final_pages)
            }
            for future in as_completed(future_map):
                i = future_map[future]
                pages[i] = future.result()
                completed += 1
                if progress_cb is not None:
                    progress_cb(completed, final_pages)

    return Document(doc_id=doc_id, filename=pdf_path.name, pages=pages)


def _process_page(pdf_path: Path, page_index: int, dpi: int) -> PageText:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        native_text = (page.get_text("text") or "").strip()
        native_score = _text_quality_score(native_text)

        raw_text = native_text
        diagram_notes = ""

        if native_score < NATIVE_TEXT_SCORE_THRESHOLD:
            image = _render_page_image(page=page, dpi=dpi)
            ocr_text = extract_page_text(image)
            ocr_score = _text_quality_score(ocr_text)
            raw_text = native_text if native_score >= ocr_score else ocr_text
            diagram_notes = extract_diagram_notes(image)
        else:
            # Lightweight diagram detection for native-text pages.
            if page.get_drawings():
                image = _render_page_image(page=page, dpi=min(dpi, 170))
                diagram_notes = extract_diagram_notes(image)

        cleaned = clean_ocr_text(raw_text)
        if diagram_notes:
            cleaned = f"{cleaned}\n\n[Diagram Notes]\n{diagram_notes}".strip()
        return PageText(page_num=page_index + 1, text=cleaned)
    finally:
        doc.close()


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
