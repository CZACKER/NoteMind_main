# Offline Handwritten Notes QA (Hackathon 2026)

Local-first RAG system for handwritten PDF notes:
- OCR handwritten pages locally
- Chunk + embed + index in local SQLite vector DB
- Answer with local Ollama model
- Show citations (file, page, chunk)
- Reject weak answers with: "I don't have enough information in the provided notes."
- Download a session PDF report with Q/A, confidence, and sources
- Extract diagram hints (labels/structure) and support neat diagram answers

## Stack
- Streamlit UI
- PyMuPDF + OpenCV + Tesseract OCR
- SentenceTransformers embeddings
- SQLite vector store (persistent)
- Ollama local LLM (7B/8B)

## Project Layout
```text
app.py
requirements.txt
src/
tests/
data/raw
data/processed
data/index
```

## Prerequisites (Windows)
1. Install Python 3.10+
2. Install Tesseract OCR and add to PATH
   - Recommended installer: https://github.com/UB-Mannheim/tesseract/wiki
   - Typical path: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
   - Verify:
     - `where tesseract`
     - `tesseract --version`
   - If PATH is not picked up, set env var before running app:
     - `set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
3. Install Ollama and pull a local model:
   - `ollama pull qwen2.5:7b-instruct`

## Setup
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run App
```bash
streamlit run app.py
```

## How to Use
1. Upload one or more PDFs.
2. Click `Process PDFs`.
3. Ask questions in chat.
4. Check citations and evidence panel.
5. You can ask: "Give neat diagram of ...". The model can return Mermaid diagram text.
5. Use sidebar `Performance profile`:
   - `Fast`: quickest response, lower OCR quality and shorter generation.
   - `Balanced`: recommended default.
   - `Accurate`: better quality, slower response.
6. Click `Download PDF Report` in sidebar to export the session summary.

## Optional Config
Environment variables:
- `OLLAMA_MODEL_NAME` (default: `qwen2.5:7b-instruct`)
- `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)
- `OLLAMA_TIMEOUT_S` (default: `600`)
- `OLLAMA_NUM_CTX` (default: `1536`) lower this to reduce GPU memory usage
- `OLLAMA_FALLBACK_MODEL` (default: `qwen2.5:3b-instruct`) used automatically on OOM
- `EMBED_MODEL_NAME` (default: `BAAI/bge-small-en-v1.5`)
- `TOP_K` (default: `6`)
- `MIN_RETRIEVAL_SCORE` (default: `0.35`)
- `OCR_DPI` (default: `200`)
- `TROCR_ENABLED` (default: `1`) enables handwritten-cursive fallback OCR
- `TROCR_MODEL_NAME` (default: `microsoft/trocr-base-handwritten`)
- `TROCR_TRIGGER_CONFIDENCE` (default: `55.0`) runs TrOCR when Tesseract confidence is low

Note: TrOCR model is downloaded on first use. For fully offline demos, run one warm-up ingestion while internet is available so the model is cached locally.

## Evaluation Script
Create JSON like:
```json
{
  "questions": [
    {
      "question": "What is photosynthesis?",
      "expected_keywords": ["light", "energy", "plants"]
    }
  ]
}
```

Run:
```bash
python -m src.evaluator --questions eval_questions.json
```

## Tests
```bash
pytest -q
```

## Notes
- No cloud LLM is used.
- OCR is local-only.
- The app is designed for deterministic retrieval and conservative answering.
- If you upgraded from an older version, delete files in `data/processed/` and re-process PDFs to index new diagram notes.
- If Ollama shows CUDA out-of-memory, pull/use a smaller model:
  - `ollama pull qwen2.5:3b-instruct`
