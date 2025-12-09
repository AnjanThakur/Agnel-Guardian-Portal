# services/ocr_service.py
from typing import Dict, Any, Optional

import cv2
from PIL import Image

from config.quota import get_usage
from utils.io_utils import b64_to_pil, ensure_dir
from utils.preprocess import preprocess_page
from utils.vision_api import vision_document_text
from utils.lines import line_boxes_from_vision
from utils.rating_semantic import extract_ratings_from_text_blocks
from utils.rating_grid import extract_ratings_via_dynamic_grid
# If you wire them in:
# from utils.rating_ocr_rows import extract_ratings_via_ocr_rows
# from utils.rating_label import extract_ratings_via_label_aligned
from utils.io_utils import cvt_gray
from utils.table_detect import find_table_region, split_rows_cols
from utils.field_utils import (
    read_right_or_below,
    normalize_phone,
    phone_re,
    extract_comments,
    fix_common_comment_typos,
    split_department_and_year,
    extract_signature_and_date,
    reassign_fields_by_schema,
)


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


def _vision_page_once(pil: Image.Image, debug_dir: Optional[str]):
    png = preprocess_page(pil, upscale=1.35, denoise=True, debug_dir=None)
    resp = vision_document_text(png)
    lines = line_boxes_from_vision(resp)
    if debug_dir:
        with open(f"{debug_dir}/00_raw_text.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ln["text"] for ln in lines))
    return resp, lines


def _combine_rating_sources(
    base: Dict[str, Dict[str, Any]],
    *others: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Simple combinator:
    base -> then overlay others in order when they have higher confidence.
    """
    final = dict(base)
    for src in others:
        if not src:
            continue
        for k, v in src.items():
            val = v.get("value")
            conf = float(v.get("confidence", 0.0))
            if val not in (1, 2, 3, 4):
                continue
            cur = final.get(k, {"value": None, "confidence": 0.0, "source": ""})
            if cur["value"] not in (1, 2, 3, 4) or conf >= float(cur.get("confidence", 0.0)) + 0.05:
                final[k] = {"value": val, "confidence": round(conf, 3), "source": v.get("source", "pta-free")}
    # ensure all present
    canon_keys = [k for k, _ in _QPAT]
    for k in canon_keys:
        final.setdefault(k, {"value": None, "confidence": 0.0, "source": "pta-free"})
    return final


def run_pta_free_ocr(image_b64: str, debug_dir: Optional[str]) -> Dict[str, Any]:
    pil = b64_to_pil(image_b64)
    resp, lines = _vision_page_once(pil, debug_dir)
    full_text = resp.full_text_annotation.text if resp and resp.full_text_annotation else ""

    # 1) Ratings — multiple strategies
    rating_text = extract_ratings_from_text_blocks(lines)
    rating_dyn_grid = extract_ratings_via_dynamic_grid(pil, lines, debug_dir)

    # You can plug these in when you've moved the code:
    rating_rows = {}
    rating_label = {}
    # rating_rows = extract_ratings_via_ocr_rows(pil, lines, debug_dir)
    # rating_label = extract_ratings_via_label_aligned(pil, lines, debug_dir)

    ratings_combined = _combine_rating_sources(rating_dyn_grid, rating_rows, rating_label, rating_text)

    # 2) Text fields
    parent_name = read_right_or_below(lines, r"^\s*(name|mamo)\s*:")
    contact_num = read_right_or_below(lines, r"^\s*contact\s*number\s*:", is_phone=True)
    ward_name = read_right_or_below(lines, r"^\s*ward'?s?\s*name\s*:")
    dept_year = read_right_or_below(lines, r"^\s*department\s*and\s*year\s*of\s*graduation\s*:")

    if contact_num:
        m = phone_re.search(contact_num)
        contact_num = normalize_phone(m.group(0)) if m else normalize_phone(contact_num)

    comments_raw = extract_comments(lines)
    comments = fix_common_comment_typos(comments_raw)

    dep_val, year_val = split_department_and_year(dept_year)
    sig_text, sig_date, sig_conf = extract_signature_and_date(lines)

    mapped = dict(ratings_combined)

    mapped["parent_signature"] = {
        "value": sig_text,
        "confidence": sig_conf if sig_text else 0.0,
        "source": "pta-free"
    }
    mapped["signature_date"] = {
        "value": sig_date,
        "confidence": sig_conf if sig_date else 0.0,
        "source": "pta-free"
    }

    mapped["parent_name"] = {"value": parent_name, "confidence": 0.9 if parent_name else 0.0, "source": "pta-free"}
    mapped["contact_number"] = {"value": contact_num, "confidence": 0.9 if contact_num else 0.0, "source": "pta-free"}
    mapped["ward_name"] = {"value": ward_name, "confidence": 0.9 if ward_name else 0.0, "source": "pta-free"}

    mapped["department"] = {"value": dep_val, "confidence": 0.9 if dep_val else 0.0, "source": "pta-free"}
    mapped["year_of_graduation"] = {"value": year_val, "confidence": 0.9 if year_val else 0.0, "source": "pta-free"}

    mapped["department_year"] = {
        "value": dept_year,
        "confidence": 0.9 if dept_year else 0.0,
        "source": "pta-free"
    }

    legacy = " ".join([p for p in [sig_text, sig_date] if p]).strip()
    mapped["parent_signature_and_date"] = {
        "value": legacy,
        "confidence": sig_conf if legacy else 0.0,
        "source": "pta-free"
    }

    mapped["comments"] = {
        "value": comments,
        "confidence": 0.9 if comments else 0.0,
        "source": "pta-free"
    }

    mapped = reassign_fields_by_schema(mapped)

    return {
        "engine": "vision",
        "mode": "pta-free",
        "fields": mapped,
        "text": full_text,
        "usage": get_usage(),
        "debug_dir": debug_dir
    }
