from __future__ import annotations

import re


HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.IGNORECASE),
]


def clean_ocr_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    lines = [line.strip() for line in text.splitlines()]

    filtered: list[str] = []
    for line in lines:
        if not line:
            continue
        if any(pattern.match(line) for pattern in HEADER_FOOTER_PATTERNS):
            continue
        filtered.append(_fix_hyphen_breaks(line))

    joined = "\n".join(filtered)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    joined = re.sub(r"[ \t]+", " ", joined)
    return joined.strip()


def _fix_hyphen_breaks(line: str) -> str:
    # Rejoins OCR split words like "informa- tion" -> "information".
    return re.sub(r"([A-Za-z])-\s+([A-Za-z])", r"\1\2", line)

