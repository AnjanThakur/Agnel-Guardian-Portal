# app/ocr/pta_free_logic.py
from __future__ import annotations

import base64
import datetime as dt
import re
from typing import Dict, Any, List

import cv2
import numpy as np
from app.ocr.grammar_fix import grammar_fix
from app.ocr.text_normalizer import normalize_ocr_text
from app.core.config import DEBUG_ROOT, bump_and_check_limit
from app.core.logger import get_logger
from app.models.constants import PTA_QUESTION_KEYS
from app.models.schemas import OCRRequest
from app.services.google_vision import (
    document_text_from_bytes,
    vision_response_to_lines,
)
from app.utils.helpers import ensure_dir, save_debug_image
from app.ocr.utils_image import (
    detect_table_region,
    segment_uniform_grid,
    tick_score_strong,
    crop_cell,
)

logger = get_logger("pta_free")


# ---------------------------------------------------------------------------
# Fallback comments extractor (text-only, no bbox)
# ---------------------------------------------------------------------------
def extract_comments_block(all_text: str) -> str:
    """
    Fallback extractor: tries to pull ONLY handwritten comments between:
    - question 10 area
    - parent’s signature block
    Works even when OCR reorders lines, but may be noisy.
    """
    lines = [ln.strip() for ln in all_text.split("\n") if ln.strip()]
    comments_started = False
    collected: List[str] = []

    for ln in lines:
        low = ln.lower()

        # Start after question 10
        if "10." in low or low.startswith("10 "):
            comments_started = True
            continue

        if comments_started:
            # Stop at signature block
            if "parent" in low and "signature" in low:
                break

            # Skip obvious printed / boilerplate phrases
            skip_keywords = [
                "institute",
                "criterion",
                "counselling",
                "resources",
                "holistic development",
                "overall holistic",
                "sr.no",
                "details",
                "infrastructure",
                "internet",
                "computing",
                "excellent",
                "unsatisfactory",
                "fair",
                "good",
                "library",
                "placements",
                "support facilities",
            ]
            if any(k in low for k in skip_keywords):
                continue

            if len(ln) > 2:
                collected.append(ln)

    return "\n".join(collected).strip()


# ---------------------------------------------------------------------------
# Debug directory helper
# ---------------------------------------------------------------------------
def _debug_dir_for_request() -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ensure_dir(f"{DEBUG_ROOT}/{stamp}_pta_free")


# ---------------------------------------------------------------------------
# Table / rating detector (kept for later use; not wired into response yet)
# ---------------------------------------------------------------------------
def _extract_ratings_from_table(
    gray_page: np.ndarray, debug_dir: str | None = None
) -> Dict[str, Dict[str, Any]]:
    """
    Core table logic:
    - find table region (largest rectangle / fallback bottom region)
    - binarize
    - split into 10x4 uniform grid
    - score ticks by ink density
    """
    x1, y1, x2, y2 = detect_table_region(gray_page)
    table = gray_page[y1:y2, x1:x2].copy()
    if debug_dir:
        save_debug_image(table, f"{debug_dir}/10_table_region.png")

    g_blur = cv2.GaussianBlur(table, (3, 3), 0)
    _, bw = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_inv = cv2.bitwise_not(bw)

    if debug_dir:
        save_debug_image(bw_inv, f"{debug_dir}/11_table_bw_inv.png")

    N_ROWS = 10
    N_COLS = 4
    row_bands, col_bands = segment_uniform_grid(
        bw_inv, N_ROWS, N_COLS, y_margin=4, x_margin=6
    )

    ratings: List[tuple[int | None, float]] = []

    for r_idx, (ry1, ry2) in enumerate(row_bands):
        scores: List[float] = []
        for c_idx, (cx1, cx2) in enumerate(col_bands):
            cell = crop_cell(bw_inv, cx1, ry1, cx2, ry2)
            score = tick_score_strong(cell)
            scores.append(score)

            if debug_dir:
                vis = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
                cv2.putText(
                    vis,
                    f"{score:.3f}",
                    (2, min(16, vis.shape[0] - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                save_debug_image(
                    vis, f"{debug_dir}/cell_r{r_idx+1}_c{c_idx+1}.png"
                )

        if not scores:
            ratings.append((None, 0.0))
            continue

        scores_np = np.array(scores, dtype="float32")
        best_idx = int(np.argmax(scores_np))
        smax = float(scores_np[best_idx])
        sorted_scores = np.sort(scores_np)
        ssec = float(sorted_scores[-2]) if len(sorted_scores) >= 2 else 0.0
        mean = float(scores_np.mean())
        std = float(scores_np.std() + 1e-6)
        z = (smax - mean) / std if std > 1e-6 else 0.0
        margin = smax - ssec
        ratio = smax / max(ssec, 1e-6)

        Z_THR, M_THR, R_THR, BASE_THR = 0.9, 0.07, 1.20, 0.16

        pick: int | None = None
        if (z >= Z_THR and margin >= M_THR) or (ratio >= R_THR) or (smax >= BASE_THR):
            pick = best_idx + 1

        conf = float(max(0.0, min(0.95, smax)))
        if pick is None:
            conf = 0.0

        ratings.append((pick, conf))

    result: Dict[str, Dict[str, Any]] = {}
    for idx, key in enumerate(PTA_QUESTION_KEYS):
        if idx < len(ratings):
            val, conf = ratings[idx]
        else:
            val, conf = (None, 0.0)

        result[key] = {
            "value": int(val) if val in (1, 2, 3, 4) else None,
            "confidence": round(float(conf), 3) if val in (1, 2, 3, 4) else 0.0,
            "source": "pta-grid-even",
        }

    return result


# ---------------------------------------------------------------------------
# Robust bbox-based text field extraction
# ---------------------------------------------------------------------------
def _extract_text_fields(lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Robust bbox-based extraction.
    PRIMARY GOAL: comments must be correct (even if long).
    """

    if not lines:
        return {
            "parent_name": {"value": "", "confidence": 0.0, "source": "pta-text"},
            "ward_name": {"value": "", "confidence": 0.0, "source": "pta-text"},
            "contact_number": {"value": "", "confidence": 0.0, "source": "pta-text"},
            "department_year": {"value": "", "confidence": 0.0, "source": "pta-text"},
            "parent_signature": {"value": "", "confidence": 0.0, "source": "pta-text"},
            "signature_date": {"value": "", "confidence": 0.0, "source": "pta-text"},
            "comments": {"value": "", "confidence": 0.0, "source": "pta-text"},
        }

    all_text = "\n".join(ln.get("text", "") for ln in lines)

    # ---------------- helpers ----------------
    def y_center(ln):
        _, y1, _, y2 = ln.get("box", [0, 0, 0, 0])
        return (y1 + y2) / 2

    def is_printed_text(text: str) -> bool:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return True
        lower_ratio = sum(c.islower() for c in letters) / len(letters)
        return lower_ratio < 0.15  # ONLY rule

    def looks_like_label(text: str) -> bool:
        low = text.lower()
        return any(k in low for k in [
            "name", "contact", "ward", "department", "graduation",
            "parent", "signature", "please make", "suggestions",
            "student overall", "strengthen"
        ])

    def looks_like_info_value(text: str) -> bool:
        t = text.lower().strip()
        return (
            ":" in text
            or re.fullmatch(r"\d{8,12}", t)
            or re.search(r"(b|m)[-\s]?tech|\d{4}", t)
            or re.search(r"\bcomp[-\s]?[a-z]\b", t)
        )

    # ---------------- geometry ----------------
    ys = [y_center(ln) for ln in lines]
    page_top, page_bottom = min(ys), max(ys)
    page_height = max(1.0, page_bottom - page_top)

    comments_label_bottom = None
    info_block_top = None

    for ln in lines:
        txt = ln["text"].lower()
        y1, y2 = ln["box"][1], ln["box"][3]

        if "please" in txt and "comment" in txt:
            comments_label_bottom = y2

        if any(k in txt for k in [
            "name:", "contact number", "ward's name",
            "department", "year of graduation", "parent's signature"
        ]):
            info_block_top = y1 if info_block_top is None else min(info_block_top, y1)

    if comments_label_bottom is None:
        comments_label_bottom = page_top + 0.55 * page_height

    if info_block_top is None:
        info_block_top = page_bottom - 0.12 * page_height

    comment_y1 = comments_label_bottom + 8
    comment_y2 = info_block_top - 8

    # ---------------- collect comments ----------------
    comment_items = []

    for ln in lines:
        yc = y_center(ln)
        if not (comment_y1 <= yc <= comment_y2):
            continue

        text = ln["text"].strip()
        if not text:
            continue

        if looks_like_label(text):
            continue
        if looks_like_info_value(text):
            continue
        if is_printed_text(text):
            continue
        if len(text) < 3:
            continue

        x1, y1, x2, y2 = ln["box"]
        comment_items.append({
            "text": text,
            "y": yc,
            "x": x1,
            "h": max(1, y2 - y1)
        })

    # ---------------- ordering ----------------
    comments = ""
    if comment_items:
        comment_items.sort(key=lambda c: c["y"])

        ordered = []
        row = []
        mean_h = np.mean([c["h"] for c in comment_items])
        row_threshold = max(8, 0.6 * mean_h)

        for c in comment_items:
            if not row or abs(c["y"] - row[-1]["y"]) <= row_threshold:
                row.append(c)
            else:
                row.sort(key=lambda r: r["x"])
                ordered.extend(r["text"] for r in row)
                row = [c]

        if row:
            row.sort(key=lambda r: r["x"])
            ordered.extend(r["text"] for r in row)

        comments = "\n".join(ordered).strip()

    # ---------------- fallback ----------------
    if not comments:
        comments = extract_comments_block(all_text)

    # ---------------- info fields ----------------
    info_lines = [ln for ln in lines if y_center(ln) >= info_block_top - 5]

    def smart_find(pattern, ctx):
        rgx = re.compile(pattern, re.IGNORECASE)
        for i, ln in enumerate(ctx):
            txt = ln["text"].strip()
            if rgx.search(txt):
                if ":" in txt:
                    val = txt.split(":", 1)[1].strip()
                    if val:
                        return val
                if i + 1 < len(ctx):
                    nxt = ctx[i + 1]["text"].strip()
                    if nxt:
                        return nxt
        return ""

    parent_name = smart_find(r"\bname\b", info_lines)
    ward_name = smart_find(r"ward", info_lines)
    contact_number = smart_find(r"contact", info_lines)
    department_year = smart_find(r"depart|graduat|year", info_lines)

    # ---------------- date ----------------
    date_re = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
    signature_date = ""

    for ln in sorted(lines, key=y_center, reverse=True):
        m = date_re.search(ln["text"])
        if m:
            signature_date = m.group(0)
            break

    def conf(v): return 0.9 if v else 0.0
    normalized = normalize_ocr_text(comments)
    fixed_comments = grammar_fix(normalized)

    return {
        "parent_name": {"value": parent_name, "confidence": conf(parent_name), "source": "pta-text"},
        "ward_name": {"value": ward_name, "confidence": conf(ward_name), "source": "pta-text"},
        "contact_number": {"value": contact_number, "confidence": conf(contact_number), "source": "pta-text"},
        "department_year": {"value": department_year, "confidence": conf(department_year), "source": "pta-text"},
        "parent_signature": {"value": "", "confidence": 0.0, "source": "pta-text"},
        "signature_date": {"value": signature_date, "confidence": conf(signature_date), "source": "pta-text"},
        "comments": {"value": fixed_comments,"confidence": 0.85 if fixed_comments else 0.0,"source": "pta-text",},
    }

# ---------------------------------------------------------------------------
# Main entry point used by the FastAPI route
# ---------------------------------------------------------------------------
def run_pta_free(req: OCRRequest) -> Dict[str, Any]:
    ok, usage = bump_and_check_limit()
    if not ok:
        return {"error": "Limit exceeded", "usage": usage}

    # Decode base64 image
    image_b64 = req.imageBase64.split(",")[-1]
    image_bytes = base64.b64decode(image_b64)

    # Call Vision API (1 call per page)
    response = document_text_from_bytes(image_bytes)

    # Convert to flat text + structured line objects with boxes
    full_text, lines = vision_response_to_lines(response)

    # Extract text fields (bbox-aware)
    text_fields = _extract_text_fields(lines)

    return {
        "fields": text_fields,
        "usage": usage,
        "raw_text": full_text,
        "debug_lines": lines,
    }
