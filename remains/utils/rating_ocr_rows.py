# utils/rrating_ocr_rows.py
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2

from .io_utils import cvt_gray, crop_xyxy
from .table_detect import find_table_region, split_rows_cols
from .rating_common import tick_score_strong

_QPAT = [
    ("q1_teaching_learning_environment",          r"teaching[-\s]?learning\s+environment"),
    ("q2_monitoring_students_progress",           r"monitoring\s+students?\s+progress"),
    ("q3_faculty_involvement",                    r"involvement\s+of\s+faculty"),
    ("q4_infrastructure_facilities",              r"infrastructure\s+facilities"),
    ("q5_learning_resources",                     r"learning\s+resources.*(library|internet|computing)"),
    ("q6_study_environment_and_discipline",       r"study\s+environment.*discipline"),
    ("q7_counselling_and_placements",             r"counsell?ing\s+and\s+placements"),
    ("q8_support_facilities",                     r"support\s+facilities"),
    ("q9_parental_perception",                    r"parental\s+perception"),
    ("q10_holistic_development",                  r"holistic\s+development"),
]


def extract_ratings_via_ocr_rows(
    pil,
    lines: List[Dict[str, Any]],
    debug_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:

    gray_full = cvt_gray(pil)
    box = find_table_region(gray_full)
    x1, y1, x2, y2 = box

    H, W = y2 - y1, x2 - x1
    if H < 40 or W < 40:
        return {k: {"value": None, "confidence": 0.0, "source": "ocr-row"} for k, _ in _QPAT}

    row_bands, col_bands = split_rows_cols(gray_full, box, n_rows=10, n_cols=4)

    row_scores = []
    for ry1, ry2 in row_bands:
        scores = []
        for cx1, cx2 in col_bands:
            cell = crop_xyxy(gray_full, cx1, ry1, cx2, ry2)
            scores.append(tick_score_strong(cell))
        row_scores.append(scores)

    result = {}
    canon_keys = [k for k, _ in _QPAT]

    for i, scores in enumerate(row_scores):
        if not scores:
            result[canon_keys[i]] = {"value": None, "confidence": 0.0, "source": "ocr-row"}
            continue

        best_idx = int(np.argmax(scores))
        best_val = best_idx + 1 if scores[best_idx] > 0.10 else None
        conf = float(min(0.95, scores[best_idx]))

        result[canon_keys[i]] = {
            "value": best_val,
            "confidence": conf if best_val else 0.0,
            "source": "ocr-row"
        }

    return result
