from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pytesseract import Output
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from . import config


class OCRDependencyError(RuntimeError):
    pass


def preprocess_image(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=15)

    # Adaptive thresholding handles uneven lighting in photographed notes.
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    angle = _estimate_skew_angle(thresh)
    deskewed = _rotate_image(thresh, angle)
    return deskewed


def extract_page_text(image_bgr: np.ndarray) -> str:
    ensure_tesseract_available()
    processed = preprocess_image(image_bgr)
    candidates = [
        (processed, "--oem 3 --psm 6"),
        (processed, "--oem 3 --psm 11"),
        (_upscale(processed, 1.5), "--oem 3 --psm 6"),
        (_enhance_contrast(processed), "--oem 3 --psm 4"),
    ]

    best_text = ""
    best_score = -1.0
    best_confidence = 0.0
    for image, cfg in candidates:
        text, confidence = _ocr_with_confidence(image, cfg)
        score = confidence + (0.003 * len(text))
        if score > best_score:
            best_text = text
            best_score = score
            best_confidence = confidence

    if config.TROCR_ENABLED and best_confidence < config.TROCR_TRIGGER_CONFIDENCE:
        trocr_text = _trocr_extract(processed)
        if _text_quality_score(trocr_text) > _text_quality_score(best_text):
            best_text = trocr_text

    return best_text.strip()


def _estimate_skew_angle(binary_image: np.ndarray) -> float:
    coords = np.column_stack(np.where(binary_image < 128))
    if len(coords) < 50:
        return 0.0
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return -angle


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        m,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _enhance_contrast(image: np.ndarray) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=1.35, beta=0)


def _upscale(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def _ocr_with_confidence(image: np.ndarray, config: str) -> tuple[str, float]:
    text = pytesseract.image_to_string(image, config=config, lang="eng")
    data = pytesseract.image_to_data(image, config=config, lang="eng", output_type=Output.DICT)
    conf_values: list[float] = []
    for raw_conf, token in zip(data.get("conf", []), data.get("text", []), strict=False):
        token = (token or "").strip()
        if not token:
            continue
        try:
            c = float(raw_conf)
        except ValueError:
            continue
        if c >= 0:
            conf_values.append(c)
    avg_conf = float(np.mean(conf_values)) if conf_values else 0.0
    return text, avg_conf


def _text_quality_score(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(ch.isalpha() for ch in text)
    words = sum(1 for token in text.split() if any(c.isalpha() for c in token))
    return (alpha / max(1, len(text))) + (0.01 * words)


@lru_cache(maxsize=1)
def _load_trocr() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    processor = TrOCRProcessor.from_pretrained(config.TROCR_MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(config.TROCR_MODEL_NAME)
    model.eval()
    return processor, model


def _trocr_extract(image: np.ndarray) -> str:
    try:
        processor, model = _load_trocr()
    except Exception:
        return ""

    try:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=300)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()
    except Exception:
        return ""


def ensure_tesseract_available() -> None:
    configured = os.getenv("TESSERACT_CMD", "").strip()
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    else:
        fallback = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if fallback.exists():
            pytesseract.pytesseract.tesseract_cmd = str(fallback)

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise OCRDependencyError(
            "Tesseract OCR is not installed or not in PATH. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki, "
            "or set TESSERACT_CMD to full tesseract.exe path."
        ) from exc
