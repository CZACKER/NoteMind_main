from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import config


def ensure_data_dirs() -> None:
    for path in [
        config.DATA_DIR,
        config.RAW_DIR,
        config.PROCESSED_DIR,
        config.INDEX_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def make_doc_id(filename: str, content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()[:12]
    stem = Path(filename).stem.replace(" ", "_")
    return f"{stem}_{digest}"


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())
