# utils/rating_label.py
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from .io_utils import cvt_gray, crop_xyxy
from .rating_common import tick_score_strong
from .table_detect import find_table_region, split_rows_cols

_QPAT = [
    ("q1_teaching_learning_environment",          r"teaching[-\\s]?learning\\s+environment"),
    ("q2_monitoring_students_progress",           r"monitoring\\s+students?\\s+progress"),
    ("q3_faculty_involvement",                    r"involvement\\s+of\\s+faculty"),
    ("q4_infrastructure_facilities",              r"infrastructure\\s+facilities"),
    ("q5_learning_resources",                     r"learning\\s+resources.*(library|internet|computing)"),
    ("q6_study_environment_and_discipline",       r"study\\s+environment.*discipline"),
    ("q7_counselling_and_placements",             r"counsell?ing\\s+and\\s+placements"),
    ("q8_support_facilities",                     r"support\\s+facilities"),
    ("q9_parental_perception",                    r"parental\\s+perception"),
    ("q10_holistic_development",                  r"holistic\\s+development"),
]


def extract_ratings_via_label_aligned(
    pil,
    lines: List[Dict[str, Any]],
    debug_dir: Optional[str] = None
):
    gray = cvt_gray(pil)
    x1, y1, x2, y2 = find_table_region(gray)

    row_bands, col_bands = split_rows_cols(gray, (x1, y1, x2, y2), 10, 4)

    results = {}
    canon_keys = [k for k, _ in _QPAT]

    for i, (ry1, ry2) in enumerate(row_bands):
        scores = []
        for cx1, cx2 in col_bands:
            c = crop_xyxy(gray, cx1, ry1, cx2, ry2)
            scores.append(tick_score_strong(c))

        if not scores:
            results[canon_keys[i]] = {"value": None, "confidence": 0.0, "source": "label-aligned"}
            continue

        best_idx = int(np.argmax(scores))
        best_val = best_idx + 1 if scores[best_idx] > 0.10 else None

        results[canon_keys[i]] = {
            "value": best_val,
            "confidence": float(min(0.95, scores[best_idx])) if best_val else 0.0,
            "source": "label-aligned",
        }

    return results
