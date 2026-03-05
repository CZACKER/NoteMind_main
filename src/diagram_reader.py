from __future__ import annotations

import re

import cv2
import numpy as np
import pytesseract


def extract_diagram_notes(image_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 170)
    edge_ratio = float(np.count_nonzero(edges)) / max(1.0, float(edges.size))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(20, image_bgr.shape[1] // 16),
        maxLineGap=10,
    )
    line_count = 0 if lines is None else len(lines)
    diagram_likely = (line_count >= 20) or (edge_ratio >= 0.06)
    if not diagram_likely:
        return ""

    # OCR around line-heavy visuals usually captures labels/boxes/arrows text.
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    labels = pytesseract.image_to_string(thresh, config="--oem 3 --psm 11", lang="eng")
    labels = _compact_labels(labels)
    if not labels:
        return f"Diagram-like content detected (lines={line_count}, edge_ratio={edge_ratio:.2f})."

    return (
        f"Diagram-like content detected (lines={line_count}, edge_ratio={edge_ratio:.2f}). "
        f"Possible labels/text in diagram: {labels}"
    )


def _compact_labels(text: str, max_words: int = 60) -> str:
    words = re.findall(r"[A-Za-z0-9_./-]+", text)
    if not words:
        return ""
    compact = " ".join(words[:max_words])
    return compact.strip()

