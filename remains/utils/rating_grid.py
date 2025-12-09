# utils/rating_grid.py
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

from .io_utils import cvt_gray, crop_xyxy, save_dbg
from .table_detect import find_table_region, cluster_1d, bands_from_boundaries
from .rating_common import tick_score_strong


_QPAT = [
    ("q1_teaching_learning_environment", "teaching[-\\s]?learning\\s+environment"),
    ("q2_monitoring_students_progress", "monitoring\\s+students?\\s+progress"),
    ("q3_faculty_involvement", "involvement\\s+of\\s+faculty"),
    ("q4_infrastructure_facilities", "infrastructure\\s+facilities"),
    ("q5_learning_resources", "learning\\s+resources.*(library|internet|computing)"),
    ("q6_study_environment_and_discipline", "study\\s+environment.*discipline"),
    ("q7_counselling_and_placements", "counsell?ing\\s+and\\s+placements"),
    ("q8_support_facilities", "support\\s+facilities"),
    ("q9_parental_perception", "parental\\s+perception"),
    ("q10_holistic_development", "holistic\\s+development"),
]


def _assign_rows_to_questions(lines: List[Dict[str, Any]],
                              row_bands: List[Tuple[int, int]],
                              bbox) -> List[Optional[str]]:
    import re
    q_ycenters = {}
    for key, pat in _QPAT:
        rgx = re.compile(pat, re.I)
        hit = None
        for ln in lines:
            if rgx.search(ln["text"]):
                _, y1l, _, y2l = ln["box"]
                hit = int((y1l + y2l) // 2)
                break
        if hit is not None:
            q_ycenters[key] = hit

    row_to_key: List[Optional[str]] = []
    for (ry1, ry2) in row_bands:
        rymid = int((ry1 + ry2) // 2)
        best_k, best_d = None, 999999
        for k, yc in q_ycenters.items():
            d = abs(yc - rymid)
            if d < best_d:
                best_k, best_d = k, d
        row_to_key.append(best_k)
    return row_to_key


def extract_ratings_via_dynamic_grid(
    pil: Image.Image,
    lines: List[Dict[str, Any]],
    debug_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    N_ROWS = 10
    N_COLS = 4

    gray_full = cvt_gray(pil)
    x1, y1, x2, y2 = find_table_region(gray_full)
    roi = gray_full[y1:y2, x1:x2].copy()
    H, W = roi.shape[:2]
    canon_keys = [k for k, _ in _QPAT]

    if H < 40 or W < 40:
        return {k: {"value": None, "confidence": 0.0, "source": "pta-free-grid-dyn"} for k in canon_keys}

    g = cv2.GaussianBlur(roi, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 9)

    kx = max(25, W // 14)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)), iterations=1)
    ky = max(25, H // 10)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)), iterations=1)

    y_lines = []
    x_lines = []

    cnts_h, _ = cv2.findContours(hor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_h:
        x, y, w, h = cv2.boundingRect(c)
        if w >= int(W * 0.45) and h <= 8:
            y_lines.append(int(y + h // 2))

    cnts_v, _ = cv2.findContours(ver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_v:
        x, y, w, h = cv2.boundingRect(c)
        if h >= int(H * 0.45) and w <= 8:
            x_lines.append(int(x + w // 2))

    y_tol = max(8, H // 60)
    x_tol = max(8, W // 60)
    y_centers = cluster_1d(y_lines, expected=N_ROWS + 1, tol=y_tol)
    x_centers = cluster_1d(x_lines, expected=N_COLS + 1, tol=x_tol)

    if len(y_centers) < (N_ROWS + 1):
        y_centers = [int(t) for t in np.linspace(0, H - 1, num=N_ROWS + 1)]
    if len(x_centers) < (N_COLS + 1):
        left = int(W * 0.68)
        right = W - 1
        x_centers = [int(t) for t in np.linspace(left, right, num=N_COLS + 1)]

    y_centers = sorted(set([max(0, min(H - 1, yy)) for yy in y_centers]))
    x_centers = sorted(set([max(0, min(W - 1, xx)) for xx in x_centers]))

    row_bands_local = bands_from_boundaries(y_centers)
    col_bands_local = bands_from_boundaries(x_centers)

    def _expand_band(b, lim, pad):
        a, z = b
        return (max(0, a - pad), min(lim, z + pad))

    row_bands_local = [_expand_band(b, H, 2) for b in row_bands_local]
    col_bands_local = [_expand_band(b, W, 2) for b in col_bands_local]

    if len(row_bands_local) != N_ROWS:
        yb = [int(t) for t in np.linspace(0, H, num=N_ROWS + 1)]
        row_bands_local = list(zip(yb[:-1], yb[1:]))
    if len(col_bands_local) != N_COLS:
        xb = [int(t) for t in np.linspace(int(W * 0.68), W, num=N_COLS + 1)]
        col_bands_local = list(zip(xb[:-1], xb[1:]))

    row_scores: List[List[float]] = []
    for (ry1, ry2) in row_bands_local:
        scores = []
        for (cx1, cx2) in col_bands_local:
            cell = crop_xyxy(gray_full, x1 + cx1 + 2, y1 + ry1 + 2, x1 + cx2 - 2, y1 + ry2 - 2)
            s = tick_score_strong(cell)
            scores.append(float(s))
        row_scores.append(scores)

    ratings_local: List[Tuple[Optional[int], float]] = []
    for scores in row_scores:
        if not scores:
            ratings_local.append((None, 0.0))
            continue
        smax = max(scores)
        sorted_sc = sorted(scores)
        ssec = sorted_sc[-2] if len(sorted_sc) >= 2 else 0.0
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        z = (smax - mean) / (std + 1e-6)
        margin = smax - ssec
        ratio = smax / max(ssec, 1e-6)

        Z_THR, M_THR, R_THR, BASE_THR = 1.1, 0.12, 1.28, 0.20
        best_idx = int(np.argmax(scores))
        pick = best_idx + 1 if ((z >= Z_THR and margin >= M_THR) or ratio >= R_THR or smax >= BASE_THR) else None
        conf = float(max(0.0, min(0.95, smax)))
        ratings_local.append((pick, conf if pick else 0.0))

    row_bands_full = [(y1 + a, y1 + b) for (a, b) in row_bands_local]
    row_keys = _assign_rows_to_questions(lines, row_bands_full, (x1, y1, x2, y2))

    if len(row_keys) != len(ratings_local):
        row_keys = (row_keys + [None] * len(ratings_local))[:len(ratings_local)]

    stable_keys: List[str] = []
    for i in range(N_ROWS):
        rk = row_keys[i] if i < len(row_keys) else None
        key = rk if rk in canon_keys else canon_keys[i]
        stable_keys.append(key)
    seen = set()
    for i, key in enumerate(stable_keys):
        if key in seen:
            stable_keys[i] = canon_keys[i]
        seen.add(stable_keys[i])

    res: Dict[str, Dict[str, Any]] = {}
    for i, key in enumerate(stable_keys):
        v, c = ratings_local[i]
        res[key] = {"value": v if v in (1, 2, 3, 4) else None,
                    "confidence": round(float(c), 3) if v in (1, 2, 3, 4) else 0.0,
                    "source": "pta-free-grid-dyn"}

    for key in canon_keys:
        res.setdefault(key, {"value": None, "confidence": 0.0, "source": "pta-free-grid-dyn"})

    if debug_dir:
        dbg = cv2.cvtColor(gray_full.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (a, b) in row_bands_full:
            cv2.line(dbg, (x1, a), (x2, a), (255, 0, 0), 1)
            cv2.line(dbg, (x1, b), (x2, b), (255, 0, 0), 1)
        for (cx1l, cx2l) in col_bands_local:
            cv2.line(dbg, (x1 + cx1l, y1), (x1 + cx1l, y2), (0, 0, 255), 1)
            cv2.line(dbg, (x1 + cx2l, y1), (x1 + cx2l, y2), (0, 0, 255), 1)
        for i, (pick, _) in enumerate(ratings_local):
            if pick:
                ry1f, ry2f = row_bands_full[i]
                cx1l, cx2l = col_bands_local[pick - 1]
                cv2.rectangle(dbg, (x1 + cx1l, ry1f), (x1 + cx2l, ry2f), (0, 200, 255), 2)
        save_dbg(dbg, f"{debug_dir}/14_grid_dynamic.png")

    return res
