from __future__ import annotations

from typing import Tuple, Dict, Any
import re
import cv2
import numpy as np

from app.utils.helpers import save_debug_image
from app.ocr.utils_image import detect_table_region
from app.services.google_vision import document_text_from_bytes

# ============================================================
# CONFIG — tuned for PTA forms
# ============================================================

COMMENT_TOP_PAD_PX = 12
COMMENT_EXTRA_HEIGHT_RATIO = 0.30
COMMENT_MAX_BOTTOM_RATIO = 0.85
X_MARGIN_RATIO = 0.03


# ============================================================
# COMMENT ROI (VISUAL, NOT TEXT-BASED)
# ============================================================

def _find_comment_roi(page_gray: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = page_gray.shape[:2]
    _, _, _, table_bottom = detect_table_region(page_gray)

    x1 = int(X_MARGIN_RATIO * w)
    x2 = int((1.0 - X_MARGIN_RATIO) * w)

    y1 = min(h - 1, table_bottom + COMMENT_TOP_PAD_PX)
    y2 = min(
        int(y1 + COMMENT_EXTRA_HEIGHT_RATIO * h),
        int(COMMENT_MAX_BOTTOM_RATIO * h),
    )

    if y2 - y1 < 50:
        y2 = min(h, y1 + 80)

    return x1, y1, x2, y2


# ============================================================
# COMMENT SANITIZER (PRIVACY-FIRST)
# ============================================================

LABEL_PATTERNS = [
    r"\bname\s*:",
    r"\bcontact\s*number\s*:",
    r"\bward.?s\s*name\s*:",
    r"\bdepartment.*graduation\s*:",
    r"\bparent.?s\s*signature",
    r"\bdate\s*:",
]

PRINTED_PHRASES = [
    "please make any additional comments",
    "comments or suggestions",
    "strengthen our programmes",
    "students overall holistic development",
    "parent feedback form",
]


def sanitize_comment_for_llm(raw: str) -> str:
    """
    Privacy-first sanitizer.
    Removes labels + PII, keeps opinion text.
    """

    if not raw:
        return ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    output = []

    label_hits = 0

    for ln in lines:
        low = ln.lower()

        # printed boilerplate
        if any(p in low for p in PRINTED_PHRASES):
            continue

        # label headers
        if any(re.search(p, low) for p in LABEL_PATTERNS):
            label_hits += 1
            continue

        # phone numbers
        if re.search(r"\b\d{8,12}\b", low.replace(" ", "")):
            label_hits += 1
            continue

        # years
        if re.search(r"\b20\d{2}\b", low):
            label_hits += 1
            continue

        # name-like short lines
        if len(low.split()) <= 4 and low.replace(" ", "").isalpha():
            label_hits += 1
            continue

        # once labels dominate → stop (prevents leakage)
        if label_hits >= 2:
            break

        output.append(ln)

    return "\n".join(output).strip()


# ============================================================
# MAIN API — COMMENT EXTRACTION
# ============================================================

def extract_comments_only(
    page_bgr: np.ndarray,
    debug_dir: str | None = None,
) -> Dict[str, Any]:
    """
    FINAL comment extraction:
      - visual ROI
      - single OCR call
      - aggressive PII sanitization
      - LLM-safe output
    """

    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = _find_comment_roi(gray)

    crop = page_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return {"value": "", "confidence": 0.0, "source": "pta-comment-only"}

    if debug_dir:
        save_debug_image(crop, f"{debug_dir}/comment_crop.png")

    png = cv2.imencode(".png", crop)[1].tobytes()
    resp = document_text_from_bytes(png)

    if not resp or not resp.full_text_annotation:
        return {"value": "", "confidence": 0.0, "source": "pta-comment-only"}

    raw_text = resp.full_text_annotation.text.strip()
    cleaned = sanitize_comment_for_llm(raw_text)

    return {
        "value": cleaned,
        "confidence": 0.9 if cleaned else 0.0,
        "source": "pta-comment-only",
    }
