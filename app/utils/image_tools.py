# app/utils/image_tools.py
from typing import List, Tuple

import numpy as np
import cv2


def crop_xyxy(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    H, W = img.shape[:2]
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=img.dtype)

    return img[y1:y2, x1:x2]


def find_table_region(gray: np.ndarray) -> Tuple[int, int, int, int]:
    """Heuristic PTA grid region finder (mostly right side of page)."""
    H, W = gray.shape

    g = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 9
    )

    kx = max(15, W // 40)
    ky = max(15, H // 40)
    hor = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)), iterations=1
    )
    ver = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)), iterations=1
    )
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


def tick_score_strong(cell_gray: np.ndarray) -> float:
    """Stronger tick detector using lines + X-patterns."""
    g = cell_gray
    if g is None or g.size < 25:
        return 0.0
    g_blur = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1
    )

    h, w = bw.shape
    edges = cv2.Canny(bw, 60, 140, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=10,
        minLineLength=max(6, min(h, w) // 4),
        maxLineGap=2
    )

    diag_len = 0.0
    diag_cnt = 0
    hv_len = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = (x2 - x1), (y2 - y1)
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            seg = float(np.hypot(dx, dy))
            if (20 <= ang <= 70) or (110 <= ang <= 160):
                diag_len += seg
                diag_cnt += 1
            if (ang <= 10) or (80 <= ang <= 100) or (ang >= 170):
                hv_len += seg

    k1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    k2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], np.float32)
    binf = (bw > 0).astype(np.float32)
    resp1 = cv2.filter2D(binf, -1, k1)
    resp2 = cv2.filter2D(binf, -1, k2)
    x_resp = float(np.max(resp1) + np.max(resp2)) / 3.0

    ink = (bw > 0).mean()
    area = h * w
    norm = max(80.0, area / 2.5)

    s_raw = (
        0.55 * (diag_cnt + 0.8 * diag_len / 12.0)
        + 0.35 * x_resp
        + 0.10 * ink * 10.0
    ) / (norm / 80.0)

    s = max(0.0, s_raw - 0.25 * (hv_len / 20.0))
    return float(s)


def cluster_1d(values: List[int], expected: int, tol: int) -> List[int]:
    """Cluster line coordinates and return cluster centers."""
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


def bands_from_boundaries(bounds: List[int]) -> List[Tuple[int, int]]:
    """Turn N+1 boundaries into N bands [(b0,b1), (b1,b2), ...]."""
    bounds = sorted(list(set(bounds)))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
