# app/ocr/pta_free_logic.py
from __future__ import annotations
import json
import os
import base64
import datetime as dt
from typing import Dict, Any, List

import cv2
import numpy as np
from pdf2image import convert_from_bytes

from app.core.config import DEBUG_ROOT, bump_and_check_limit
from app.core.logger import get_logger
from app.models.constants import PTA_QUESTION_KEYS
from app.models.schemas import OCRRequest

from app.ocr.preprocess_image import preprocess_document_image
from app.ocr.table_extractor import extract_table_ratings
from app.ml.pta_rating_infer import infer_rating_rows

from app.services.google_vision import document_text_from_bytes, vision_response_to_lines
from app.utils.helpers import ensure_dir, save_debug_image

from app.analysis.run_full_analysis import run_full_feedback_analysis

logger = get_logger("pta_free")

# ============================================================
# EXECUTION MODES
# ============================================================
DATASET_MODE = False
ENABLE_ML = True


# ============================================================
# DEBUG DIR
# ============================================================
def _debug_dir_for_request() -> str:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return ensure_dir(f"{DEBUG_ROOT}/{stamp}_pta_free")


# ============================================================
# COMMENT EXTRACTION
# ============================================================
def extract_comments_only(
    page_bgr: np.ndarray,
    table_bottom_y: int,
    debug_dir: str | None = None,
) -> str:
    h, w = page_bgr.shape[:2]

    x1 = int(0.03 * w)
    x2 = int(0.97 * w)
    y1 = min(h - 1, int(table_bottom_y) + 10)
    y2 = int(0.92 * h)

    if y2 <= y1:
        return ""

    crop = page_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return ""

    if debug_dir:
        save_debug_image(crop, f"{debug_dir}/comment_crop.png")

    png_bytes = cv2.imencode(".png", crop)[1].tobytes()
    resp = document_text_from_bytes(png_bytes)

    raw = ""
    if resp and resp.full_text_annotation:
        raw = resp.full_text_annotation.text or ""

    return _clean_comment_text(raw)


def _clean_comment_text(text: str) -> str:
    if not text:
        return ""

    banned = [
        "please make any additional comments",
        "comments or suggestions",
        "strengthen our programmes",
        "students overall holistic development",
        "parent feedback form",
    ]

    out: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if any(b in s.lower() for b in banned):
            continue
        out.append(s)

    return "\n".join(out).strip()


# ============================================================
# PAGE PROCESSOR
# ============================================================
def _process_single_page(img_bgr: np.ndarray, debug_dir: str) -> Dict[str, Any]:
    save_debug_image(img_bgr, f"{debug_dir}/orig_bgr.png")

    pre_gray = preprocess_document_image(img_bgr)
    save_debug_image(pre_gray, f"{debug_dir}/preprocessed_gray.png")

    raw_text = ""
    ocr_lines: List[Dict[str, Any]] = []

    if not DATASET_MODE:
        png = cv2.imencode(".png", img_bgr)[1].tobytes()
        resp = document_text_from_bytes(png)
        raw_text, ocr_lines = vision_response_to_lines(resp)

    ratings = extract_table_ratings(
        pre_gray,
        ocr_lines=ocr_lines,
        debug_dir=debug_dir,
    )

    from app.ocr.utils_image import detect_table_region
    _, _, _, table_bottom = detect_table_region(pre_gray)

    comments = ""
    if not DATASET_MODE:
        comments = extract_comments_only(
            page_bgr=img_bgr,
            table_bottom_y=table_bottom,
            debug_dir=debug_dir,
        )

    if ENABLE_ML and not DATASET_MODE:
        row_images: List[np.ndarray] = []
        for i in range(1, 11):
            p = f"{debug_dir}/row_{i}_rating_strip.png"
            row = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            row_images.append(row if row is not None else np.zeros((96, 384), np.uint8))

        preds = infer_rating_rows(row_images)
        for key, pred in zip(PTA_QUESTION_KEYS, preds):
            ratings[key] = {
                "value": pred.get("value"),
                "confidence": float(pred.get("confidence", 0.0)),
                "status": pred.get("status", "empty_or_noise"),
                "source": "pta-ml",
            }

    return {
        "ratings": ratings,
        "comments": comments,
    }


# ============================================================
# MAIN ENTRY (IMAGE OR PDF)
# ============================================================
def run_pta_free(req: OCRRequest) -> Dict[str, Any]:
    ok, usage = bump_and_check_limit()
    if not ok:
        return {"error": "Limit exceeded", "usage": usage}

    file_bytes = base64.b64decode(req.imageBase64.split(",")[-1])
    debug_root = _debug_dir_for_request()

    is_pdf = file_bytes[:4] == b"%PDF"

    # ---------------- PDF ----------------
    if is_pdf:
        import traceback

        pages = convert_from_bytes(file_bytes, dpi=300, fmt="png")
        page_images = [
            cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages
        ]

        page_results = []

        for idx, img_bgr in enumerate(page_images, start=1):
            page_dir = os.path.join(debug_root, f"page_{idx:02d}")
            os.makedirs(page_dir, exist_ok=True)

            try:
                page_data = _process_single_page(img_bgr, page_dir)
                page_data["page"] = idx
                page_results.append(page_data)

                json_path = os.path.join(page_dir, f"page_{idx:02d}_results.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(page_data, f, indent=2, ensure_ascii=False)

                print(f"[OK] JSON written → {json_path}")

            except Exception:
                err_path = os.path.join(page_dir, "ERROR.txt")
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())

                print(f"[ERROR] Page {idx} failed — see {err_path}")

        # -------- REPORT --------
        report = None
        if not DATASET_MODE:
            all_comments = [p["comments"] for p in page_results if p.get("comments")]
            if all_comments:
                report_dir = ensure_dir(f"{debug_root}/report")
                report = run_full_feedback_analysis(all_comments, report_dir)

        return {
            "pages": page_results,
            "report": report,
            "usage": usage,
            "debug_root": debug_root,
        }

    # ---------------- IMAGE ----------------
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Invalid image", "usage": usage}

    page_dir = ensure_dir(f"{debug_root}/page_01")
    result = _process_single_page(img, page_dir)

    report = None
    if not DATASET_MODE and result.get("comments"):
        report_dir = ensure_dir(f"{debug_root}/report")
        report = run_full_feedback_analysis([result["comments"]], report_dir)

    return {
        **result,
        "report": report,
        "usage": usage,
        "debug_root": debug_root,
    }
