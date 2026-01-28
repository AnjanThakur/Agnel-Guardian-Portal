from typing import Tuple, Optional
import cv2
import numpy as np

from app.ocr.grid_detector import detect_grid_lines


# ============================================================
# ROBUST TABLE REGION DETECTION (STRUCTURE-BASED)
# ============================================================

def detect_table_region(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detects PTA table region using GRID STRUCTURE, not area.
    Prevents page-level rectangle selection when image is zoomed out.
    Always returns a valid region (fallback-safe).
    """

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    # --- binarize ---
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9
    )

    contours, _ = cv2.findContours(
        bw,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_score = -1
    best_rect: Optional[Tuple[int, int, int, int]] = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, cw, ch = cv2.boundingRect(approx)

        # ------------------ HARD REJECTIONS ------------------

        # reject page-sized rectangles
        if cw > 0.95 * w and ch > 0.95 * h:
            continue

        # reject too small regions
        if cw < 0.35 * w or ch < 0.25 * h:
            continue

        crop = gray[y:y + ch, x:x + cw]
        if crop.size == 0:
            continue

        # ------------------ STRUCTURE SCORE ------------------

        grid = detect_grid_lines(crop)

        v_count = len(grid.v_lines)
        h_count = len(grid.h_lines)

        # PTA tables have many verticals (rating columns)
        score = (v_count * 2) + h_count

        if score > best_score:
            best_score = score
            best_rect = (x, y, x + cw, y + ch)

    # ------------------ SAFE FALLBACK ------------------
    if best_rect is None:
        return (
            int(0.08 * w),
            int(0.20 * h),
            int(0.96 * w),
            int(0.88 * h),
        )

    # slight padding
    x1, y1, x2, y2 = best_rect
    pad_x = int((x2 - x1) * 0.03)
    pad_y = int((y2 - y1) * 0.03)

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )
