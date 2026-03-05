from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

COLLECTION_NAME = "notes_chunks"
VECTOR_DB_PATH = INDEX_DIR / "vectors.sqlite"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:3b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "600"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1536"))
OLLAMA_FALLBACK_MODEL = os.getenv("OLLAMA_FALLBACK_MODEL", "qwen2.5:3b-instruct")

CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS", "380"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "80"))
TOP_K = int(os.getenv("TOP_K", "6"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))
MIN_EVIDENCE_OVERLAP = float(os.getenv("MIN_EVIDENCE_OVERLAP", "0.12"))
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
TROCR_ENABLED = os.getenv("TROCR_ENABLED", "1") == "1"
TROCR_MODEL_NAME = os.getenv("TROCR_MODEL_NAME", "microsoft/trocr-base-handwritten")
TROCR_TRIGGER_CONFIDENCE = float(os.getenv("TROCR_TRIGGER_CONFIDENCE", "55.0"))
NATIVE_TEXT_SCORE_THRESHOLD = float(os.getenv("NATIVE_TEXT_SCORE_THRESHOLD", "2.2"))
OCR_MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", str(min(8, os.cpu_count() or 4))))
