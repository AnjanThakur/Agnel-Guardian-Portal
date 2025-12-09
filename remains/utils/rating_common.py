# utils/rating_common.py
import math
from typing import List

import numpy as np
import cv2


def tick_score_strong(cell_gray: np.ndarray) -> float:
    """Enhanced tick detection using multiple heuristics."""
    g = cell_gray
    if g.size < 25:
        return 0.0
    g_blur = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)

    h, w = bw.shape
    edges = cv2.Canny(bw, 60, 140, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10,
                            minLineLength=max(6, min(h, w) // 4), maxLineGap=2)
    diag_len = 0.0; diag_cnt = 0; hv_len = 0.0
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = (x2 - x1), (y2 - y1)
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            seg = float(math.hypot(dx, dy))
            if (20 <= ang <= 70) or (110 <= ang <= 160):
                diag_len += seg; diag_cnt += 1
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

    s_raw = (0.55 * (diag_cnt + 0.8 * diag_len / 12.0) +
             0.35 * x_resp +
             0.10 * ink * 10.0) / (norm / 80.0)

    s = max(0.0, s_raw - 0.25 * (hv_len / 20.0))
    return float(s)
