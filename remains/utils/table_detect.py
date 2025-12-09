# utils/table_detect.py
from typing import Tuple, List

import numpy as np
import cv2


def find_table_region(gray: np.ndarray) -> Tuple[int, int, int, int]:
    H, W = gray.shape

    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 9)

    kx = max(15, W // 40)
    ky = max(15, H // 40)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)), iterations=1)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)), iterations=1)
    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = float(w * h)
        if w < W * 0.30 or h < H * 0.25:
            continue

        x_center = x + w * 0.5
        y_center = y + h * 0.5

        right_bias = (x_center / W)
        mid_bias = 1.0 - abs((y_center / H) - 0.55)
        score = area * (1.0 + 0.6 * right_bias + 0.3 * mid_bias)

        if score > best_score:
            best_score = score
            best = (x, y, w, h)

    if best is None:
        x1 = int(W * 0.60); x2 = int(W * 0.95)
        y1 = int(H * 0.24); y2 = int(H * 0.86)
        return (x1, y1, x2, y2)

    x, y, w, h = best
    pad = max(10, int(0.012 * max(W, H)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W - 1, x + w + pad)
    y2 = min(H - 1, y + h + pad)

    if (y2 - y1) < H * 0.18 or (x2 - x1) < W * 0.28:
        fx1 = int(W * 0.60); fx2 = int(W * 0.95)
        fy1 = int(H * 0.24); fy2 = int(H * 0.86)
        return (fx1, fy1, fx2, fy2)

    return (x1, y1, x2, y2)


def split_rows_cols(gray: np.ndarray, box, n_rows=10, n_cols=4):
    x1, y1, x2, y2 = box
    H = max(1, y2 - y1)
    W = max(1, x2 - x1)
    row_bands = [(y1 + (H * i) // n_rows, y1 + (H * (i + 1)) // n_rows) for i in range(n_rows)]
    col_bands = [(x1 + (W * j) // n_cols, x1 + (W * (j + 1)) // n_cols) for j in range(n_cols)]
    return row_bands, col_bands


def cluster_1d(values: List[int], expected: int, tol: int) -> List[int]:
    if not values:
        return []
    vals = sorted(values)
    clusters: List[List[int]] = [[vals[0]]]
    for v in vals[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    centers = [int(sum(c) / len(c)) for c in clusters]
    centers.sort()
    while len(centers) > expected and len(centers) > 2:
        gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
        gaps.sort(key=lambda t: t[0])
        _, idx = gaps[0]
        a, b = centers[idx], centers[idx + 1]
        merged = int((a + b) // 2)
        centers = centers[:idx] + [merged] + centers[idx + 2:]
    return centers


def bands_from_boundaries(bounds: List[int]):
    bounds = sorted(list(set(bounds)))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
