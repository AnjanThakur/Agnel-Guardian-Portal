# app/ocr/table_extractor.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

from app.models.constants import PTA_QUESTION_KEYS
from app.utils.helpers import save_debug_image
from app.ocr.utils_image import detect_table_region


# ============================================================
# CONFIG
# ============================================================

N_ROWS = 10
ROW_MIN_HEIGHT_PX = 14
HEADER_SKIP_RATIO = 0.10   # top % of rating strip reserved for header
HLINE_KERNEL_RATIO = 0.55


# ============================================================
# RATING BLOCK DETECTION (X RANGE ONLY)
# ============================================================

def _find_rating_block_x(ocr_lines: List[Dict[str, Any]], table_x1: int, table_x2: int) -> Tuple[int, int]:
    digit_centers: List[int] = []

    for ln in ocr_lines or []:
        if ln.get("text", "").strip() in {"1", "2", "3", "4"} and "box" in ln:
            x1, _, x2, _ = ln["box"]
            digit_centers.append((x1 + x2) // 2)

    table_w = max(1, table_x2 - table_x1)

    if digit_centers:
        anchor_x = max(digit_centers)
        block_w = int(0.28 * table_w)
        x2 = min(table_x2, anchor_x + int(0.05 * table_w))
        x1 = max(table_x1, x2 - block_w)
        if x2 > x1:
            return x1, x2

    # safe fallback: rightmost area
    return table_x1 + int(0.72 * table_w), table_x2


# ============================================================
# GRID-BASED ROW DETECTION
# ============================================================

def _cluster_indices(idxs: np.ndarray, gap: int = 2) -> List[int]:
    if idxs.size == 0:
        return []
    idxs = np.sort(idxs)
    groups = []
    s = idxs[0]
    p = idxs[0]
    for v in idxs[1:]:
        if v <= p + gap:
            p = v
        else:
            groups.append((s + p) // 2)
            s = p = v
    groups.append((s + p) // 2)
    return groups


def _extract_row_bands(rating_body: np.ndarray, debug_dir: str | None = None) -> List[Tuple[int, int]]:
    """
    Finds horizontal grid lines and returns consecutive bands between them.
    """
    h, w = rating_body.shape[:2]
    if h < 30 or w < 30:
        return []

    blur = cv2.GaussianBlur(rating_body, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    kernel_w = max(60, int(w * HLINE_KERNEL_RATIO))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    hlines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

    if debug_dir:
        save_debug_image(hlines, f"{debug_dir}/_dbg_hlines.png")

    proj = hlines.sum(axis=1).astype(np.float32)
    if proj.max() < 1e-6:
        return []

    proj /= proj.max()
    ys = np.where(proj > 0.35)[0]
    lines = _cluster_indices(ys)

    if len(lines) < 2:
        return []

    bands = [(lines[i], lines[i + 1]) for i in range(len(lines) - 1)]
    # filter too thin
    bands = [(a, b) for a, b in bands if (b - a) >= ROW_MIN_HEIGHT_PX]
    return bands


# ============================================================
# MAIN API
# ============================================================

def extract_table_ratings(
    gray_page: np.ndarray,
    ocr_lines: List[Dict[str, Any]] | None,
    debug_dir: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Produces:
      - table_crop.png
      - rating_strip_full.png
      - _dbg_rating_body.png
      - row_1_rating_strip.png ... row_10_rating_strip.png
    Returns dict with value=None (ML will decide).
    """
    results: Dict[str, Dict[str, Any]] = {}

    if gray_page is None or getattr(gray_page, "size", 0) == 0:
        for k in PTA_QUESTION_KEYS:
            results[k] = {"value": None, "confidence": 0.0, "source": "pta-table"}
        return results

    if gray_page.ndim == 3:
        gray_page = cv2.cvtColor(gray_page, cv2.COLOR_BGR2GRAY)

    # light denoise is fine (keeps grid)
    gray_page = cv2.fastNlMeansDenoising(gray_page, None, 15, 7, 21)

    # ---- table ----
    x1, y1, x2, y2 = detect_table_region(gray_page)
    table = gray_page[y1:y2, x1:x2]

    if table.size == 0:
        for k in PTA_QUESTION_KEYS:
            results[k] = {"value": None, "confidence": 0.0, "source": "pta-table"}
        return results

    if debug_dir:
        save_debug_image(table, f"{debug_dir}/table_crop.png")

    # ---- rating strip (right side) ----
    rx1, rx2 = _find_rating_block_x(ocr_lines or [], x1, x2)
    rating_strip = gray_page[y1:y2, rx1:rx2]

    if debug_dir:
        save_debug_image(rating_strip, f"{debug_dir}/rating_strip_full.png")

    h = rating_strip.shape[0]
    header_cut = int(h * HEADER_SKIP_RATIO)
    body = rating_strip[header_cut:, :]

    if debug_dir:
        save_debug_image(body, f"{debug_dir}/_dbg_rating_body.png")

    # ---- detect row bands ----
    bands = _extract_row_bands(body, debug_dir=debug_dir)

    # fallback to uniform split if needed
    if len(bands) < N_ROWS:
        step = body.shape[0] / N_ROWS
        bands = [(int(i * step), int((i + 1) * step)) for i in range(N_ROWS)]

    # normalize to exactly 10
    if len(bands) > N_ROWS:
        bands = bands[:N_ROWS]

    # if less than 10 even after fallback, pad
    while len(bands) < N_ROWS:
        last_end = bands[-1][1] if bands else 0
        step = max(ROW_MIN_HEIGHT_PX, int(body.shape[0] / N_ROWS))
        bands.append((last_end, min(body.shape[0], last_end + step)))

    # ---- save rows ----
    for i, key in enumerate(PTA_QUESTION_KEYS):
        y1b, y2b = bands[i]
        y2b = max(y2b, y1b + ROW_MIN_HEIGHT_PX)
        y1b = max(0, y1b)
        y2b = min(body.shape[0], y2b)

        row = body[y1b:y2b, :]
        if debug_dir:
            save_debug_image(row, f"{debug_dir}/row_{i+1}_rating_strip.png")

        results[key] = {"value": None, "confidence": 0.0, "source": "pta-table"}

    return results
