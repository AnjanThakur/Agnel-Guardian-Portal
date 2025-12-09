# app/ocr/utils_image.py
from typing import Tuple

import cv2
import numpy as np


def tick_score_strong(cell_gray: np.ndarray) -> float:
    """
    Heuristic 'ink density' score for a rating cell.
    Higher = more likely to contain a tick / cross / mark.
    """
    if cell_gray is None or cell_gray.size == 0:
        return 0.0
    # normalize
    g = cell_gray.astype("float32") / 255.0
    # darker pixels contribute more
    score = float(np.mean(1.0 - g))
    return score


def segment_uniform_grid(
    table_gray: np.ndarray,
    n_rows: int,
    n_cols: int,
    y_margin: int = 2,
    x_margin: int = 2,
) -> Tuple[list[Tuple[int, int]], list[Tuple[int, int]]]:
    """
    Very simple even grid split:
    - returns row bands [(y1,y2), ...]
    - returns col bands [(x1,x2), ...]
    """
    h, w = table_gray.shape[:2]
    row_bounds = [int(i * h / n_rows) for i in range(n_rows + 1)]
    col_bounds = [int(j * w / n_cols) for j in range(n_cols + 1)]

    row_bands = []
    for i in range(n_rows):
        y1, y2 = row_bounds[i], row_bounds[i + 1]
        row_bands.append((max(0, y1 + y_margin), min(h, y2 - y_margin)))

    col_bands = []
    for j in range(n_cols):
        x1, x2 = col_bounds[j], col_bounds[j + 1]
        col_bands.append((max(0, x1 + x_margin), min(w, x2 - x_margin)))

    return row_bands, col_bands


def crop_cell(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = gray.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return gray[0:0, 0:0]
    return gray[y1:y2, x1:x2]


def detect_table_region(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Use a slightly more robust version of 'largest rectangle' trick
    to find the PTA table part of the page.
    """
    import app.utils.preprocess as prep  # avoid circular import

    bw_inv = prep.binarize_inv(gray)
    contour = prep.largest_rect_contour(bw_inv)
    h, w = gray.shape[:2]

    if contour is None:
        # fallback: bottom 2/3 of the page where the PTA table usually is
        return int(w * 0.05), int(h * 0.25), int(w * 0.95), int(h * 0.95)

    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    # Slight padding
    pad_x = max(5, int((x2 - x1) * 0.03))
    pad_y = max(5, int((y2 - y1) * 0.03))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return x1, y1, x2, y2
