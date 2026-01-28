# app/ocr/grid_detector.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import cv2


@dataclass
class GridLines:
    v_lines: List[int]   # x positions
    h_lines: List[int]   # y positions


def _group_1d_indices(idxs: np.ndarray, gap: int = 2) -> List[int]:
    """Group consecutive indices into line centers."""
    if idxs.size == 0:
        return []
    idxs = np.sort(idxs)
    groups = []
    start = idxs[0]
    prev = idxs[0]
    for v in idxs[1:]:
        if v <= prev + gap:
            prev = v
        else:
            groups.append(int((start + prev) // 2))
            start = v
            prev = v
    groups.append(int((start + prev) // 2))
    return groups


def _merge_close(lines: List[Tuple[int, float]], gap: int = 12) -> List[int]:
    """
    Merge multiple close-by line candidates by keeping the strongest
    within each neighborhood.
    """
    if not lines:
        return []
    lines = sorted(lines, key=lambda t: t[0])
    out = []
    i = 0
    while i < len(lines):
        y0, _ = lines[i]
        group = [lines[i]]
        j = i + 1
        while j < len(lines) and (lines[j][0] - y0) <= gap:
            group.append(lines[j])
            j += 1
        best = max(group, key=lambda t: t[1])
        out.append(int(best[0]))
        i = j
    return sorted(out)


def detect_grid_lines(table_gray: np.ndarray) -> GridLines:
    """
    Detect table grid lines using morphology:
      - adaptive threshold to get ink/grid
      - vertical open to isolate vertical lines
      - horizontal open to isolate horizontal lines
      - 1D projection to find line positions
    """
    gray = table_gray
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31, 9
    )

    # -------------------- vertical lines --------------------
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 15)))
    v_img = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)

    v_proj = v_img.sum(axis=0).astype(np.float32)
    v_proj /= (v_proj.max() + 1e-6)

    v_idxs = np.where(v_proj > 0.5)[0]
    v_lines = _group_1d_indices(v_idxs, gap=2)

    # -------------------- horizontal lines --------------------
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, w // 20), 1))
    h_img = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)

    h_proj = h_img.sum(axis=1).astype(np.float32)
    h_proj /= (h_proj.max() + 1e-6)

    # lower threshold because some lines are faint
    h_idxs = np.where(h_proj > 0.2)[0]
    h_candidates = _group_1d_indices(h_idxs, gap=2)
    h_lines_scored = [(y, float(h_proj[y])) for y in h_candidates]
    h_lines = _merge_close(h_lines_scored, gap=12)

    return GridLines(v_lines=v_lines, h_lines=h_lines)
