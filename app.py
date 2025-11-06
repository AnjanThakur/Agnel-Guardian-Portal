# app.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Tuple, Optional
import base64, io, os, re, yaml, json, threading, time, math
from PIL import Image
import numpy as np
import cv2

# Primary OCR: Google Cloud Vision
from google.cloud import vision
from google.api_core.exceptions import ServiceUnavailable

# Optional local fallback OCR (free)
try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

import datetime as dt

# -------------------------------
# Directories
# -------------------------------
DEBUG_ROOT = "debug_out"
os.makedirs(DEBUG_ROOT, exist_ok=True)

TEMPLATE_DIR = "templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app = FastAPI(title="Agnel OCR — Google Cloud Vision (with free fallback)")

# -------------------------------
# CORS (tighten for prod)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Config / Quota guard
# -------------------------------
CONFIG = {
    "vision_safety_limit": 880,
    "max_width": 1400,
    "max_height": 1800,
    "bw_otsu": True,
    "pta_free_rows": 10,
    "pta_free_cols": 4,
    "min_tick_area_ratio": 0.003,
}

COUNTER_FILE = "usage_counter.json"
LOCK = threading.Lock()

def _load_counter():
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    return {"month": dt.date.today().strftime("%Y-%m"), "count": 0}

def _save_counter(data):
    with open(COUNTER_FILE, "w") as f:
        json.dump(data, f)

def get_usage():
    d = _load_counter()
    month = dt.date.today().strftime("%Y-%m")
    if d.get("month") != month:
        d = {"month": month, "count": 0}
        _save_counter(d)
    return d

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()

_LEADING_LABEL = re.compile(r"^\s*([A-Za-z][A-Za-z \t#\.]{0,40}?):\s*", re.I)

def strip_leading_label(val: str) -> str:
    return _LEADING_LABEL.sub("", val or "").strip()

_phone_re = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")

def normalize_phone(s: str) -> str:
    digits = "".join(ch for ch in (s or "") if ch.isdigit() or ch == "+")
    if digits and not digits.startswith("+") and len(digits) >= 10:
        digits = "+91" + digits[-10:]
    return digits

def _bump_and_check_limit(limit=1000):
    with LOCK:
        d = get_usage()
        if d["count"] >= limit:
            return False, d
        d["count"] += 1
        _save_counter(d)
        return True, d

def _should_use_vision():
    d = get_usage()
    return d["count"] < CONFIG["vision_safety_limit"]

# -------------------------------
# Models
# -------------------------------
class OCRReq(BaseModel):
    imageBase64: str
    template: Optional[str] = None
    debug: Optional[bool] = False

# -------------------------------
# Utilities
# -------------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _save_dbg(img: np.ndarray, path: str):
    try:
        cv2.imwrite(path, img)
    except Exception:
        pass

def b64_to_pil(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64.split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")

def pil_to_bytes(pil: Image.Image, fmt=".png") -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG" if fmt == ".png" else "JPEG")
    return buf.getvalue()

def safe_crop_pil(pil: Image.Image, box_xyxy: Tuple[int,int,int,int]) -> Optional[Image.Image]:
    W, H = pil.size
    x1,y1,x2,y2 = box_xyxy
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W,   x2))
    y2 = max(0, min(H,   y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return pil.crop((x1,y1,x2,y2))

def _resize_bound(pil: Image.Image, max_w: int, max_h: int) -> Image.Image:
    W, H = pil.size
    scale = min(max_w / W, max_h / H, 1.0)
    if scale < 1.0:
        pil = pil.resize((int(W*scale), int(H*scale)), Image.LANCZOS)
    return pil

# -------------------------------
# Preprocess helpers
# -------------------------------
def _edge_energy_horizontal(gray: np.ndarray) -> float:
    gy = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=3)
    return float(np.abs(gy).mean())

def _rotate90(gray: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:  return gray
    if k == 1:  return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if k == 2:  return cv2.rotate(gray, cv2.ROTATE_180)
    return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

def _auto_orient_90s(gray: np.ndarray, debug_dir: Optional[str] = None) -> Tuple[np.ndarray, int]:
    best_k, best_e, best_img = 0, -1.0, gray
    for k in (0, 1, 2, 3):
        test = _rotate90(gray, k)
        e = _edge_energy_horizontal(test)
        if debug_dir: 
            _save_dbg(test, os.path.join(debug_dir, f"00_orient_{k*90}.png"))
        if e > best_e:
            best_e, best_k, best_img = e, k, test
    return best_img, best_k

def _deskew_small(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    coords = np.column_stack(np.where((255 - gray) > 0))
    angle = 0.0
    if coords.size:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        angle = max(-12.0, min(12.0, angle))
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    fixed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return fixed, angle

def _maybe_invert(bw: np.ndarray) -> np.ndarray:
    white_ratio = (bw == 255).mean()
    return 255 - bw if white_ratio < 0.5 else bw

def _pick_best_bw(cands: List[np.ndarray]) -> np.ndarray:
    scores = [_edge_energy_horizontal(c) for c in cands]
    return cands[int(np.argmax(scores))]

def _to_gray(img_rgb: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)

def preprocess_page(pil: Image.Image, upscale=1.35, denoise=True, debug_dir=None) -> bytes:
    """Legacy preprocessing function - kept for compatibility."""
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)
    if denoise:
        g = cv2.fastNlMeansDenoising(g, h=12)
    if upscale and upscale != 1.0:
        g = cv2.resize(g, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, 8)
    white_ratio = (thr == 255).mean()
    if white_ratio < 0.5:
        thr = 255 - thr
    ok, png = cv2.imencode(".png", thr)
    return png.tobytes()

def preprocess_page_for_vision(pil: Image.Image, upscale=1.4, denoise=True, do_bw=True, debug_dir: Optional[str] = None) -> bytes:
    """Advanced preprocessing with orientation correction and deskewing."""
    pil = _resize_bound(pil, CONFIG["max_width"], CONFIG["max_height"])
    gray0 = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)

    gray1, _ = _auto_orient_90s(gray0, debug_dir)
    if debug_dir: 
        _save_dbg(gray1, os.path.join(debug_dir, "01_oriented.png"))

    pad = int(0.04 * max(gray1.shape))
    gpad = cv2.copyMakeBorder(gray1, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    gray2p, _ = _deskew_small(gpad)
    keep = int(0.015 * max(gray2p.shape))
    gray2 = gray2p if min(gray2p.shape) <= 2 * keep else gray2p[keep:-keep, keep:-keep]
    if debug_dir: 
        _save_dbg(gray2, os.path.join(debug_dir, "02_deskew.png"))

    clahe = cv2.createCLAHE(clipLimit=3.6, tileGridSize=(8, 8))
    gray3 = clahe.apply(gray2)
    if denoise:
        gray3 = cv2.fastNlMeansDenoising(gray3, h=10)
    if upscale and upscale != 1.0:
        gray3 = cv2.resize(gray3, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    if not do_bw:
        ok, png = cv2.imencode(".png", gray3)
        return png.tobytes()

    thr_adapt = cv2.adaptiveThreshold(gray3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 8)
    _, thr_otsu = cv2.threshold(gray3, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    best = _maybe_invert(_pick_best_bw([thr_adapt, thr_otsu]))
    if debug_dir:
        _save_dbg(best, os.path.join(debug_dir, "03_best_bw.png"))
    ok, png = cv2.imencode(".png", best)
    return png.tobytes()

# -------------------------------
# Vision API
# -------------------------------
def get_vision_client():
    return vision.ImageAnnotatorClient()

def vision_document_text(png_bytes: bytes, retries: int = 2, delay: float = 0.6, *, return_usage: bool = False):
    client = get_vision_client()
    image = vision.Image(content=png_bytes)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = client.document_text_detection(image=image)
            if return_usage:
                return resp, 1
            return resp
        except ServiceUnavailable as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay * (2 ** attempt))
                continue
            break
        except Exception as e:
            raise
    raise RuntimeError(f"Vision service unavailable after {retries+1} attempts: {last_exc!r}")

def tesseract_document_text(pil: Image.Image) -> str:
    if not HAVE_TESS:
        return ""
    g = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        return pytesseract.image_to_string(bw, config="--psm 6").strip()
    except Exception:
        return pytesseract.image_to_string(pil, config="--psm 6").strip()

# -------------------------------
# Text structuring helpers
# -------------------------------
def line_boxes_from_vision(resp) -> List[Dict[str, Any]]:
    if not resp or not resp.full_text_annotation:
        return []
    words = []
    page_h = 0
    for page in resp.full_text_annotation.pages:
        page_h = max(page_h, page.height or 0)
        for block in page.blocks:
            for para in block.paragraphs:
                for w in para.words:
                    txt = "".join(s.text for s in w.symbols)
                    conf = float(w.confidence or 0.0)
                    v = w.bounding_box.vertices
                    xs = [v[i].x for i in range(4)]
                    ys = [v[i].y for i in range(4)]
                    words.append({"text": txt, "conf": conf, "box": [min(xs), min(ys), max(xs), max(ys)]})
    if not words:
        return []
    words.sort(key=lambda b: (b["box"][1], b["box"][0]))

    ytol = max(8, int(0.012 * (page_h or 1200)))
    rows: List[List[Dict[str, Any]]] = []
    for w in words:
        placed = False
        for row in rows:
            if abs(row[0]["box"][1] - w["box"][1]) <= ytol:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])

    line_objs: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda t: t["box"][0])

        segs: List[List[Dict[str, Any]]] = [[]]
        for i, w in enumerate(row):
            if i == 0:
                segs[-1].append(w)
                continue
            prev = row[i - 1]
            gap = w["box"][0] - prev["box"][2]
            line_height = max(10, prev["box"][3] - prev["box"][1])
            if gap > line_height * 2.2:
                segs.append([w])
            else:
                segs[-1].append(w)

        for seg in segs:
            if not seg:
                continue
            text = " ".join(t["text"] for t in seg).strip()
            conf = float(np.mean([t["conf"] for t in seg]))
            x1 = min(t["box"][0] for t in seg)
            y1 = min(t["box"][1] for t in seg)
            x2 = max(t["box"][2] for t in seg)
            y2 = max(t["box"][3] for t in seg)
            line_objs.append({"text": text, "conf": conf, "box": [x1, y1, x2, y2]})
    return line_objs

# -------------------------------
# NEW: Semantic+layout rating extractor (adds no deletions)
# -------------------------------
def extract_ratings_from_text_blocks(lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Maps each numbered row label (1..10) to the nearest rating digit (1..4),
    BUT strictly within that question's y-window (from its line down to the
    next question's line). Also ignores the '1 2 3 4' header row.
    """
    questions_by_num = {
        1:  "q1_teaching_learning_environment",
        2:  "q2_monitoring_students_progress",
        3:  "q3_faculty_involvement",
        4:  "q4_infrastructure_facilities",
        5:  "q5_learning_resources",
        6:  "q6_study_environment_and_discipline",
        7:  "q7_counselling_and_placements",
        8:  "q8_support_facilities",
        9:  "q9_parental_perception",
        10: "q10_holistic_development",
    }

    # Normalize: (y_mid, text, conf, box)
    clean = []
    for ln in lines or []:
        t = (ln.get("text") or "").strip()
        if not t:
            continue
        x1, y1, x2, y2 = (ln.get("box") or [0, 0, 0, 0])[:4]
        y_mid = int((y1 + y2) // 2)
        clean.append({"y": y_mid, "text": t, "conf": float(ln.get("conf") or 0.0), "box": [x1, y1, x2, y2]})
    if not clean:
        return {k: {"value": None, "confidence": 0.0, "source": "pta-free-rating"} for k in questions_by_num.values()}
    clean.sort(key=lambda r: r["y"])

    # Detect question headers like "1.", "2)", "10 -", etc.
    q_rows = []
    q_header_rx = re.compile(r"^\s*(\d{1,2})\s*[.)\-:–]?\s*(.*)$")
    for r in clean:
        m = q_header_rx.match(r["text"])
        if not m:
            continue
        qnum = int(m.group(1))
        if 1 <= qnum <= 10:
            q_rows.append({"qnum": qnum, "y": r["y"], "text": r["text"]})

    # Fallback: if Vision dropped the leading numbers, assume vertical order from long-ish lines
    if len(q_rows) < 10:
        seen = {qr["qnum"] for qr in q_rows}
        next_needed = 1
        for r in clean:
            if next_needed > 10:
                break
            if next_needed in seen:
                next_needed += 1
                continue
            if len(r["text"]) >= 18 or len(r["text"].split()) >= 3:
                q_rows.append({"qnum": next_needed, "y": r["y"], "text": r["text"]})
                next_needed += 1
        q_rows.sort(key=lambda d: d["qnum"])

    # Collect candidate rating digits (1..4), ignoring header-like lines
    digit_rx = re.compile(r"(?<!\d)([1-4])(?!\d)")
    def _looks_like_header(t: str) -> bool:
        s = t.replace(" ", "")
        if "1234" in s:
            return True
        nums = digit_rx.findall(t)
        return len(set(nums)) >= 3  # "1 2 3" or more on same line → header-ish

    rating_lines = []
    for r in clean:
        if _looks_like_header(r["text"]):
            continue
        nums = [int(n) for n in digit_rx.findall(r["text"])]
        if nums:
            rating_lines.append({"y": r["y"], "nums": nums, "text": r["text"]})

    # Window each question from its y to next question's y; pick nearest digit line inside window
    result: Dict[str, Dict[str, Any]] = {}

    def pick_value(nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        # Prefer higher ratings if multiple digits occur on same line to avoid picking stray "1"
        for v in (4, 3, 2):
            if v in nums:
                return v
        return nums[-1]

    if q_rows:
        for i, q in enumerate(q_rows):
            qnum = q["qnum"]
            qkey = questions_by_num.get(qnum)
            if not qkey:
                continue

            y_start = q["y"]
            y_end = q_rows[i + 1]["y"] if i + 1 < len(q_rows) else y_start + 99999

            # Only consider ratings on the same line or *below* this question, up to the next question
            cands = [rl for rl in rating_lines if (y_start <= rl["y"] <= y_end)]
            nearest, best_dy = None, 1e9
            for rl in cands:
                dy = abs(rl["y"] - y_start)
                if dy < best_dy:
                    best_dy = dy
                    nearest = rl

            val, conf = None, 0.0
            if nearest:
                val = pick_value(nearest["nums"])
                # Confidence scales down slightly if further from the label line
                conf = float(max(0.20, min(0.85, 0.85 - min(best_dy, 80) / 110.0)))

            result[qkey] = {
                "value": val if val in (1, 2, 3, 4) else None,
                "confidence": conf if val in (1, 2, 3, 4) else 0.0,
                "source": "pta-free-rating"
            }

    # Ensure all present / clamp
    for qkey in questions_by_num.values():
        if qkey not in result:
            result[qkey] = {"value": None, "confidence": 0.0, "source": "pta-free-rating"}
        if result[qkey]["value"] not in (1, 2, 3, 4, None):
            result[qkey]["value"] = None
            result[qkey]["confidence"] = 0.0

    return result

    """
    Maps each numbered row label (1..10) to the nearest rating digit (1..4)
    using a semantic + Y-position heuristic on Vision line fragments.

    Input:  lines = [{"text": str, "conf": float, "box":[x1,y1,x2,y2]}, ...]
    Output: { "q1_teaching_learning_environment": {"value": int|None, "confidence": float, "source": "pta-free-rating"}, ...}
    """
    # Canonical keys must match your _QPAT/canon order
    questions_by_num = {
        1:  "q1_teaching_learning_environment",
        2:  "q2_monitoring_students_progress",
        3:  "q3_faculty_involvement",
        4:  "q4_infrastructure_facilities",
        5:  "q5_learning_resources",
        6:  "q6_study_environment_and_discipline",
        7:  "q7_counselling_and_placements",
        8:  "q8_support_facilities",
        9:  "q9_parental_perception",
        10: "q10_holistic_development",
    }

    # Normalize: (y_mid, text, conf)
    clean = []
    for ln in lines or []:
        t = (ln.get("text") or "").strip()
        if not t:
            continue
        x1, y1, x2, y2 = (ln.get("box") or [0, 0, 0, 0])[:4]
        y_mid = int((y1 + y2) // 2)
        clean.append({"y": y_mid, "text": t, "conf": float(ln.get("conf") or 0.0)})
    if not clean:
        return {k: {"value": None, "confidence": 0.0, "source": "pta-free-rating"} for k in questions_by_num.values()}
    clean.sort(key=lambda r: r["y"])

    # Detect question headers like "1.", "2)", "10 -", etc.
    q_rows = []
    q_header_rx = re.compile(r"^\s*(\d{1,2})\s*[.)\-:–]?\s*(.*)$")
    for r in clean:
        m = q_header_rx.match(r["text"])
        if not m:
            continue
        qnum = int(m.group(1))
        if 1 <= qnum <= 10:
            q_rows.append({"qnum": qnum, "y": r["y"], "text": r["text"]})

    # Fallback: if Vision dropped the leading numbers, assume vertical order
    if len(q_rows) < 10:
        seen_qnums = {qr["qnum"] for qr in q_rows}
        next_needed = 1
        for r in clean:
            if next_needed > 10:
                break
            if next_needed in seen_qnums:
                next_needed += 1
                continue
            # Heuristic: question labels tend to be longer
            if len(r["text"]) >= 18 or len(r["text"].split()) >= 3:
                q_rows.append({"qnum": next_needed, "y": r["y"], "text": r["text"]})
                next_needed += 1
        q_rows.sort(key=lambda d: d["qnum"])

    # Collect candidate rating digits (1..4) by line
    digit_rx = re.compile(r"(?<!\d)([1-4])(?!\d)")
    rating_lines = []
    for r in clean:
        nums = [int(n) for n in digit_rx.findall(r["text"])]
        if nums:
            rating_lines.append({"y": r["y"], "nums": nums, "text": r["text"]})

    # Window each question from its y to next question's y; pick nearest digit line
    result: Dict[str, Dict[str, Any]] = {}

    def pick_value(nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        # Prefer higher ratings if multiple digits occur on same line to avoid bullet "1" noise
        for v in (4, 3, 2):
            if v in nums:
                return v
        return nums[-1]

    if q_rows:
        for i, q in enumerate(q_rows):
            qnum = q["qnum"]
            qkey = questions_by_num.get(qnum)
            if not qkey:
                continue

            y_start = q["y"]
            y_end = q_rows[i + 1]["y"] if i + 1 < len(q_rows) else y_start + 99999

            TOL = 6
            cands = [rl for rl in rating_lines if (y_start - TOL) <= rl["y"] <= y_end]
            nearest, best_dy = None, 1e9
            for rl in cands:
                dy = abs(rl["y"] - y_start)
                if dy < best_dy:
                    best_dy = dy
                    nearest = rl

            val, conf = None, 0.0
            if nearest:
                val = pick_value(nearest["nums"])
                conf = float(max(0.15, min(0.85, 0.85 - min(best_dy, 60) / 90.0)))

            result[qkey] = {
                "value": val if val in (1, 2, 3, 4) else None,
                "confidence": conf if val in (1, 2, 3, 4) else 0.0,
                "source": "pta-free-rating"
            }

    # Ensure all questions exist
    for qkey in questions_by_num.values():
        if qkey not in result:
            result[qkey] = {"value": None, "confidence": 0.0, "source": "pta-free-rating"}

    # Clamp to {1..4}
    for k, v in result.items():
        if v["value"] not in (1, 2, 3, 4, None):
            v["value"] = None
            v["confidence"] = 0.0

    return result

# -------------------------------
# PTA FREE — grid detection
# -------------------------------
def _find_table_region(gray: np.ndarray) -> Tuple[int,int,int,int]:
    H, W = gray.shape

    g = cv2.GaussianBlur(gray, (3,3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 9)

    kx = max(15, W // 40)
    ky = max(15, H // 40)
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)), iterations=1)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)), iterations=1)
    grid = cv2.bitwise_or(hor, ver)
    grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=1)

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
        mid_bias   = 1.0 - abs((y_center / H) - 0.55)
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

def _split_rows_cols(gray: np.ndarray, box, n_rows=10, n_cols=4):
    x1,y1,x2,y2 = box
    H = max(1, y2-y1); W = max(1, x2-x1)
    row_bands = [(y1 + (H*i)//n_rows, y1 + (H*(i+1))//n_rows) for i in range(n_rows)]
    col_bands = [(x1 + (W*j)//n_cols, x1 + (W*(j+1))//n_cols) for j in range(n_cols)]
    return row_bands, col_bands

def _prep_cell(gray_cell: np.ndarray) -> np.ndarray:
    if gray_cell.size == 0:
        return np.zeros((1,1), np.uint8)
    bw = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = bw.shape[:2]
    th = max(1, h // 12)
    tw = max(1, w // 12)
    bw = bw[th:h-th or None, tw:w-tw or None]
    if bw.size == 0:
        return np.zeros((1,1), np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    return bw

def _line_angles_from_bw(bw: np.ndarray) -> list:
    if bw.size == 0:
        return []
    edges = cv2.Canny(bw, 50, 150, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=8,
                           minLineLength=max(5, bw.shape[1]//8),
                           maxLineGap=8)
    if lines is None:
        return []
    angs = []
    for x1,y1,x2,y2 in lines[:,0,:]:
        dx = x2 - x1; dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if ang > 90: ang -= 180
        if ang < -90: ang += 180
        angs.append(float(ang))
    return angs

def _tick_score_from_angles(angles: list) -> float:
    if not angles:
        return 0.0
    pos = [a for a in angles if a >= 0]
    neg = [a for a in angles if a < 0]

    def in_range(v, lo, hi): return lo <= v <= hi

    pos_good = sum(in_range(a, 35, 80) for a in pos)
    neg_shallow = sum(in_range(a, -70, -10) for a in neg)
    neg_steep = sum(in_range(a, -80, -35) for a in neg)

    score = 0.0
    if pos_good >= 1 and (neg_shallow >= 1 or neg_steep >= 1):
        score = 0.9
    elif pos_good >= 1 and neg_steep >= 1:
        score = 0.7
    elif pos_good >= 1 or neg_shallow >= 1 or neg_steep >= 1:
        score = 0.45

    score += 0.05 * min(4, len(angles))
    return float(max(0.0, min(1.0, score)))

def _tick_score_strong(cell_gray: np.ndarray) -> float:
    """Enhanced tick detection using multiple heuristics."""
    g = cell_gray
    if g.size < 25:
        return 0.0
    g_blur = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # (fix) this was wrong: cv2.morphologyEx(bw, np.uint8(np.ones((2,2))), ...)
    # If you want a no-op, skip it; if you want a light open, set iterations=1.
    # We'll keep the original intention (close to connect ticks).
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)

    h, w = bw.shape
    edges = cv2.Canny(bw, 60, 140, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10,
                            minLineLength=max(6, min(h,w)//4), maxLineGap=2)
    diag_len = 0.0; diag_cnt = 0; hv_len = 0.0
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            dx, dy = (x2-x1), (y2-y1)
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            seg = float(math.hypot(dx, dy))
            if (20 <= ang <= 70) or (110 <= ang <= 160):
                diag_len += seg; diag_cnt += 1
            if (ang <= 10) or (80 <= ang <= 100) or (ang >= 170):
                hv_len += seg

    k1 = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float32)
    k2 = np.array([[0,0,1],[0,1,0],[1,0,0]], np.float32)
    binf = (bw > 0).astype(np.float32)
    resp1 = cv2.filter2D(binf, -1, k1)
    resp2 = cv2.filter2D(binf, -1, k2)
    x_resp = float(np.max(resp1) + np.max(resp2)) / 3.0

    ink = (bw > 0).mean()
    area = h*w
    norm = max(80.0, area/2.5)

    s_raw = (0.55*(diag_cnt + 0.8*diag_len/12.0) +
             0.35*x_resp +
             0.10*ink*10.0) / (norm/80.0)

    s = max(0.0, s_raw - 0.25*(hv_len/20.0))
    return float(s)

    """Enhanced tick detection using multiple heuristics."""
    g = cell_gray
    if g.size < 25:
        return 0.0
    g_blur = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, np.uint8(np.ones((2,2))), np.ones((2,2), np.uint8), iterations=0)  # no-op but preserved
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8), iterations=1)

    h, w = bw.shape
    edges = cv2.Canny(bw, 60, 140, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10,
                            minLineLength=max(6, min(h,w)//4), maxLineGap=2)
    diag_len = 0.0; diag_cnt = 0; hv_len = 0.0
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            dx, dy = (x2-x1), (y2-y1)
            ang = abs(np.degrees(np.arctan2(dy, dx)))
            seg = float(math.hypot(dx, dy))
            if (20 <= ang <= 70) or (110 <= ang <= 160):
                diag_len += seg; diag_cnt += 1
            if (ang <= 10) or (80 <= ang <= 100) or (ang >= 170):
                hv_len += seg

    k1 = np.array([[1,0,0],[0,1,0],[0,0,1]], np.float32)
    k2 = np.array([[0,0,1],[0,1,0],[1,0,0]], np.float32)
    binf = (bw > 0).astype(np.float32)
    resp1 = cv2.filter2D(binf, -1, k1)
    resp2 = cv2.filter2D(binf, -1, k2)
    x_resp = float(np.max(resp1) + np.max(resp2)) / 3.0

    ink = (bw > 0).mean()
    area = h*w
    norm = max(80.0, area/2.5)

    s_raw = (0.55*(diag_cnt + 0.8*diag_len/12.0) +
             0.35*x_resp +
             0.10*ink*10.0) / (norm/80.0)

    s = max(0.0, s_raw - 0.25*(hv_len/20.0))
    return float(s)

def _crop_xyxy(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((1,1), dtype=np.uint8)

    H, W = img.shape[:2]
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1,1), dtype=img.dtype)

    return img[y1:y2, x1:x2]

# -------------------------------
# Question patterns and field extraction
# -------------------------------
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

def _assign_rows_to_questions(lines: List[Dict[str,Any]],
                              row_bands: List[Tuple[int,int]],
                              bbox: Tuple[int,int,int,int]) -> List[Optional[str]]:
    q_ycenters: Dict[str,int] = {}
    for key, pat in _QPAT:
        rgx = re.compile(pat, re.I)
        hit = None
        for ln in lines:
            if rgx.search(ln["text"]):
                x1,y1l,x2,y2l = ln["box"]
                hit = int((y1l+y2l)//2); break
        if hit is not None:
            q_ycenters[key] = hit

    row_to_key: List[Optional[str]] = []
    for (ry1,ry2) in row_bands:
        rymid = int((ry1+ry2)//2)
        best_k, best_d = None, 999999
        for k, yc in q_ycenters.items():
            d = abs(yc - rymid)
            if d < best_d:
                best_k, best_d = k, d
        row_to_key.append(best_k)
    return row_to_key

def _read_right_or_below(lines, label_rgx: str, *, below_rows=1, is_phone=False):
    """Extract field value from right of label or below it."""
    lab = re.compile(label_rgx, re.I)
    for i, ln in enumerate(lines):
        if not lab.search(ln["text"]):
            continue
        y0 = ln["box"][1]
        
        # Try same line
        row = " ".join(m["text"] for m in lines if abs(m["box"][1] - y0) <= 18)
        m = re.search(r":\s*(.+)$", row)
        if m:
            val = m.group(1).strip()
            val = re.split(r"\s+[A-Za-z][^:]{1,40}:\s*", val)[0].strip()
            if is_phone:
                digits = re.sub(r"\D+", "", val)
                if len(digits) >= 10:
                    return digits
            if val:
                return val

        # Check above line (some forms print name above "Name:")
        if i > 0:
            prev_text = lines[i - 1]["text"].strip()
            if prev_text and len(prev_text.split()) <= 4 and not re.search(r":", prev_text):
                if is_phone:
                    digits = re.sub(r"\D+", "", prev_text)
                    if len(digits) >= 10:
                        return digits
                return prev_text

        # Check below lines
        taken = []
        for mline in lines[i + 1:i + 1 + max(1, below_rows)]:
            t = mline["text"].strip()
            if re.match(r"^\s*[A-Za-z].{1,40}:\s*", t):
                break
            taken.append(t)
        val = " ".join(taken).strip()
        if is_phone:
            digits = re.sub(r"\D+", "", val)
            if len(digits) >= 10:
                return digits
        return val
    return ""

def _clean_spell(s: str) -> str:
    """Fix common OCR spelling mistakes."""
    repl = {
        "charitles":"charities",
        "leaming":"learning",
        "interet":"internet","intenret":"internet",
        "strenghthen":"strengthen","strongthon":"strengthen",
        "holstic":"holistic",
    }
    t = s
    for k,v in repl.items():
        t = re.sub(rf"\b{k}\b", v, t, flags=re.I)
    return t

def _extract_comments(lines: list) -> str:
    """Extract comments section from text lines."""
    hint_re  = re.compile(r"additional\s+comments|suggestions", re.I)
    label_re = re.compile(r"^\s*(name|mamo|contact\s*number|ward'?s?\s*name|department\s*and\s*year\s*of\s*graduation)\s*:", re.I)

    hint_i = None
    for i, ln in enumerate(lines):
        if hint_re.search(_norm(ln["text"])):
            hint_i = i
            break
    if hint_i is None:
        return ""

    stop_i = None
    for j in range(hint_i + 1, len(lines)):
        if label_re.search(lines[j]["text"]):
            stop_i = j
            break

    take = lines[hint_i+1: stop_i] if stop_i is not None else lines[hint_i+1: hint_i+8]
    text = "\n".join(l["text"].strip() for l in take if l["text"].strip())
    text = re.sub(r"^\s*strengthen our programmes\s*\.\s*\n?", "", text, flags=re.I)
    return _clean_spell(text.strip())

def _fix_common_comment_typos(text: str) -> str:
    """Fix common typos in comment text."""
    if not text:
        return text
    corrections = {
        r"\bweed to improve\b": "need to improve",
        r"\bWeed\b": "Need",
        r"\bholp\b": "help",
        r"\bstrenghten\b": "strengthen",
        r"\bthenry\b": "theory",      # OCR: "thenry" -> "theory"
        r"\bthoery\b": "theory",      # common swap -> "theory"
        r"\brather\s+then\b": "rather than",  # grammar
    }
    out = text
    for patt, repl in corrections.items():
        out = re.sub(patt, repl, out, flags=re.I)
    return out.strip()

    """Fix common typos in comment text."""
    if not text:
        return text
    corrections = {
        r"\bweed to improve\b": "need to improve",
        r"\bWeed\b": "Need",
        r"\bholp\b": "help",
        r"\bstrenghten\b": "strengthen",
    }
    out = text
    for patt, repl in corrections.items():
        out = re.sub(patt, repl, out, flags=re.I)
    return out.strip()

def _cvt_gray(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)

def _vision_page_once(pil: Image.Image, debug_dir: Optional[str]) -> Tuple[Any, List[Dict[str,Any]]]:
    """Run Vision OCR once and return response + structured lines."""
    png = preprocess_page(pil, upscale=1.35, denoise=True, debug_dir=None)
    resp = vision_document_text(png)
    lines = line_boxes_from_vision(resp)
    if debug_dir:
        with open(os.path.join(debug_dir, "00_raw_text.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(ln["text"] for ln in lines))
    return resp, lines

def _split_department_and_year(val: str) -> Tuple[str, str]:
    """
    Given OCR like 'B-Tech-2027' or 'BTECH 2027', return ('BTECH'/'B-Tech', '2027').
    Keeps original spacing/dashes for dept; prefers a 4-digit year 20xx.
    """
    s = (val or "").strip()
    if not s:
        return "", ""
    # Find a 4-digit year first
    m = re.search(r"\b(20\d{2})\b", s)
    year = m.group(1) if m else ""
    dept = s
    if year:
        dept = s[:m.start()].strip(" -:/.,")
    # Light normalize department (collapse spaces around dashes)
    dept = re.sub(r"\s*-\s*", "-", dept).strip()
    # If still empty, try to extract letters before any 2-digit year
    if not year:
        m2 = re.search(r"\b(\d{2})\b", s)
        if m2:
            yy = int(m2.group(1))
            # heuristic: treat 24..39 as 2024..2039
            if 24 <= yy <= 39:
                year = f"20{yy:02d}"
                dept = s[:m2.start()].strip(" -:/.,")
    return dept, year


# -------------------------------
# Field validation and reassignment
# -------------------------------
FIELD_SCHEMAS = {
    "parent_name": {
        "pattern": r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}$",
        "type": "name",
    },
    "contact_number": {
        "pattern": r"\+?\d[\d\s()\-]{7,}\d",
        "type": "phone",
    },
    "ward_name": {
        "pattern": r"^[A-Za-z0-9\- ]{2,20}$",
        "type": "text",
    },
    "department_year": {
        "pattern": r"(B[\s\-]*Tech|BE|B\.?Sc).*20\d{2}",
        "type": "text",
    },
    "comments": {
        "pattern": r".{10,}",
        "type": "paragraph",
        "multi_line": True,
    },
        "department": {
        "pattern": r"^[A-Za-z][A-Za-z \-\.]{1,40}$",
        "type": "text",
    },
    "year_of_graduation": {
        "pattern": r"\b20\d{2}\b",
        "type": "text",
    },
}

def _validate_field(field_name: str, value: str) -> bool:
    """Return True if value matches field's expected pattern."""
    schema = FIELD_SCHEMAS.get(field_name)
    if not schema or not value:
        return False
    return bool(re.search(schema["pattern"], value.strip(), re.I))

def reassign_fields_by_schema(fields: dict) -> dict:
    """
    Cleans up OCR mix-ups by comparing each field's value
    with expected regex patterns and redistributing if needed.
    """
    out = dict(fields)

    def _move_value(src, dst):
        if not out.get(src) or not out.get(dst):
            return
        val = out[src].get("value", "").strip()
        if val:
            prev = out[dst].get("value", "").strip()
            out[dst]["value"] = (prev + "\n" + val).strip() if prev else val
            out[src]["value"] = ""
            out[src]["confidence"] = 0.0
            out[dst]["confidence"] = max(out[dst].get("confidence", 0), 0.9)

    # Name accidentally inside comments
    cmnt = out.get("comments", {}).get("value", "") or ""
    pname = out.get("parent_name", {}).get("value", "") or ""
    if cmnt:
        lines = [ln.strip() for ln in cmnt.splitlines() if ln.strip()]
        if lines and _validate_field("parent_name", lines[-1]) and not pname:
            out["parent_name"]["value"] = lines[-1]
            out["parent_name"]["confidence"] = 0.9
            lines.pop(-1)
            out["comments"]["value"] = "\n".join(lines).strip()

    # Name looks like a full sentence → belongs to comments
    pname = out.get("parent_name", {}).get("value", "") or ""
    if pname and not _validate_field("parent_name", pname):
        _move_value("parent_name", "comments")

    # Phone number in wrong field → move to contact_number
    for key in ["comments", "parent_name"]:
        val = out.get(key, {}).get("value", "")
        if val and re.search(FIELD_SCHEMAS["contact_number"]["pattern"], val):
            _move_value(key, "contact_number")

    # Ward/Dept swap (if year pattern found in ward field)
    ward = out.get("ward_name", {}).get("value", "") or ""
    dept = out.get("department_year", {}).get("value", "") or ""
    if ward and _validate_field("department_year", ward) and not dept:
        _move_value("ward_name", "department_year")

    return out

def _cluster_1d(values: List[int], expected: int, tol: int) -> List[int]:
    """
    1D positional clustering for line coordinates.
    - values: raw x or y positions (ints)
    - expected: expected cluster count (e.g., rows+1 or cols+1)
    - tol: max gap inside a cluster
    Returns cluster centers (sorted).
    """
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
    # If we got too many centers, greedily merge nearest until expected-ish
    while len(centers) > expected and len(centers) > 2:
        gaps = [(centers[i+1] - centers[i], i) for i in range(len(centers)-1)]
        # merge the SMALLEST gap pair
        gaps.sort(key=lambda t: t[0])
        _, idx = gaps[0]
        a, b = centers[idx], centers[idx+1]
        merged = int((a+b)//2)
        centers = centers[:idx] + [merged] + centers[idx+2:]
    return centers

def _bands_from_boundaries(bounds: List[int]) -> List[Tuple[int, int]]:
    """
    Turn N+1 boundaries into N bands [(b0,b1), (b1,b2), ...].
    """
    bounds = sorted(list(set(bounds)))
    return [(bounds[i], bounds[i+1]) for i in range(len(bounds)-1)]

def extract_ratings_via_dynamic_grid(pil: Image.Image,
                                     lines: List[Dict[str, Any]],
                                     debug_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Dynamic grid detector:
    - auto-finds horizontal & vertical table lines
    - clusters them into 10 row bands x 4 column bands
    - uses _tick_score_strong on each cell to pick rating 1..4
    - maps rows to question keys using your existing regex matcher

    Returns: dict like {"q1_teaching_learning_environment": {"value": 3, "confidence": 0.78, "source": "pta-free-grid-dyn"}, ...}
    """
    N_ROWS = 10
    N_COLS = 4

    gray_full = _cvt_gray(pil)
    # limit to coarse table region first (your existing heuristic)
    x1, y1, x2, y2 = _find_table_region(gray_full)
    roi = gray_full[y1:y2, x1:x2].copy()
    H, W = roi.shape[:2]
    if H < 40 or W < 40:
        # fallback empty
        keys = [k for k, _ in _QPAT]
        return {k: {"value": None, "confidence": 0.0, "source": "pta-free-grid-dyn"} for k in keys}

    # --- binarize & line extraction
    g = cv2.GaussianBlur(roi, (3, 3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 9)

    # horizontal lines (long wide kernel)
    kx = max(25, W // 14)  # robust across zoom
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)), iterations=1)

    # vertical lines (long tall kernel)
    ky = max(25, H // 10)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky)), iterations=1)

    # collect line positions as centers
    y_lines: List[int] = []
    x_lines: List[int] = []

    # horizontal contours → y centers
    cnts_h, _ = cv2.findContours(hor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_h:
        x, y, w, h = cv2.boundingRect(c)
        if w >= int(W * 0.45) and h <= 8:  # long-ish and thin
            y_lines.append(int(y + h // 2))

    # vertical contours → x centers
    cnts_v, _ = cv2.findContours(ver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_v:
        x, y, w, h = cv2.boundingRect(c)
        if h >= int(H * 0.45) and w <= 8:
            x_lines.append(int(x + w // 2))

    # cluster to N_ROWS+1 & N_COLS+1 boundaries
    y_tol = max(8, H // 60)
    x_tol = max(8, W // 60)
    y_centers = _cluster_1d(y_lines, expected=N_ROWS + 1, tol=y_tol)
    x_centers = _cluster_1d(x_lines, expected=N_COLS + 1, tol=x_tol)

    # if missing, synthesize evenly
    if len(y_centers) < (N_ROWS + 1):
        y_centers = [int(t) for t in np.linspace(0, H - 1, num=N_ROWS + 1)]
    if len(x_centers) < (N_COLS + 1):
        # push grid to the right-most 28-30% where rating boxes usually live
        left = int(W * 0.68)
        right = W - 1
        x_centers = [int(t) for t in np.linspace(left, right, num=N_COLS + 1)]

    y_centers = sorted(set([max(0, min(H - 1, yy)) for yy in y_centers]))
    x_centers = sorted(set([max(0, min(W - 1, xx)) for xx in x_centers]))

    row_bands_local = _bands_from_boundaries(y_centers)
    col_bands_local = _bands_from_boundaries(x_centers)

    # expand slightly to include ticks inside borders
    def _expand_band(b: Tuple[int, int], lim: int, pad: int) -> Tuple[int, int]:
        a, z = b
        return (max(0, a - pad), min(lim, z + pad))

    row_bands_local = [_expand_band(b, H, 2) for b in row_bands_local]
    col_bands_local = [_expand_band(b, W, 2) for b in col_bands_local]

    # if we didn't get exactly 10x4, synthesize bands evenly
    if len(row_bands_local) != N_ROWS:
        yb = [int(t) for t in np.linspace(0, H, num=N_ROWS + 1)]
        row_bands_local = list(zip(yb[:-1], yb[1:]))
    if len(col_bands_local) != N_COLS:
        xb = [int(t) for t in np.linspace(int(W * 0.68), W, num=N_COLS + 1)]
        col_bands_local = list(zip(xb[:-1], xb[1:]))

    # score cells and pick column per row
    row_scores: List[List[float]] = []
    for (ry1, ry2) in row_bands_local:
        scores = []
        for (cx1, cx2) in col_bands_local:
            # crop in full-image coordinates
            cell = _crop_xyxy(gray_full, x1 + cx1 + 2, y1 + ry1 + 2, x1 + cx2 - 2, y1 + ry2 - 2)
            s = _tick_score_strong(cell)
            scores.append(float(s))
        row_scores.append(scores)

    # compute picks with confidence from margin/ratio
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

    # map rows -> canonical keys using your label regex matcher (re-uses existing code)
    # build fake "row bands in full image coords" for _assign_rows_to_questions
    row_bands_full = [(y1 + a, y1 + b) for (a, b) in row_bands_local]
    row_keys = _assign_rows_to_questions(lines, row_bands_full, (x1, y1, x2, y2))
    canon_keys = [k for k, _ in _QPAT]

    # sanitize keys length
    if len(row_keys) != len(ratings_local):
        row_keys = (row_keys + [None] * len(ratings_local))[:len(ratings_local)]

    # ensure stable mapping to canonical keys (dedupe / fallback by index)
    stable_keys: List[str] = []
    for i in range(N_ROWS):
        rk = row_keys[i] if i < len(row_keys) else None
        key = rk if rk in canon_keys else canon_keys[i]
        stable_keys.append(key)
    # de-dup same key collisions
    seen = set()
    for i, key in enumerate(stable_keys):
        if key in seen:
            stable_keys[i] = canon_keys[i]
        seen.add(stable_keys[i])

    # build result
    res: Dict[str, Dict[str, Any]] = {}
    for i, key in enumerate(stable_keys):
        v, c = ratings_local[i]
        res[key] = {"value": v if v in (1, 2, 3, 4) else None,
                    "confidence": round(float(c), 3) if v in (1, 2, 3, 4) else 0.0,
                    "source": "pta-free-grid-dyn"}

    # ensure all present
    for key in canon_keys:
        res.setdefault(key, {"value": None, "confidence": 0.0, "source": "pta-free-grid-dyn"})

    # optional debug overlay
    if debug_dir:
        dbg = cv2.cvtColor(gray_full.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (a, b) in row_bands_full:
            cv2.line(dbg, (x1, a), (x2, a), (255, 0, 0), 1)
            cv2.line(dbg, (x1, b), (x2, b), (255, 0, 0), 1)
        for (cx1l, cx2l) in col_bands_local:
            X = (x1 + cx1l, x1 + cx1l)
            cv2.line(dbg, (X[0], y1), (X[1], y2), (0, 0, 255), 1)
            X2 = (x1 + cx2l, x1 + cx2l)
            cv2.line(dbg, (X2[0], y1), (X2[1], y2), (0, 0, 255), 1)
        # mark picks
        for i, (pick, _) in enumerate(ratings_local):
            if pick:
                ry1f, ry2f = row_bands_full[i]
                cx1l, cx2l = col_bands_local[pick-1]
                cv2.rectangle(dbg,
                              (x1 + cx1l, ry1f), (x1 + cx2l, ry2f),
                              (0, 200, 255), 2)
        _save_dbg(dbg, os.path.join(debug_dir, "14_grid_dynamic.png"))

    return res

def extract_ratings_via_ocr_rows(pil: Image.Image,
                                 lines: List[Dict[str, Any]],
                                 debug_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Rating extractor that defines row windows purely from OCR-detected question
    numbers/labels (semantic), *not* from grid lines. Robust when grids are skewed
    or when line detection fails.

    Returns: {canon_key: {"value": 1..4|None, "confidence": float, "source": "pta-free-ocr-rows"}}
    """
    N_ROWS = 10
    N_COLS = 4
    canon_keys = [
        "q1_teaching_learning_environment",
        "q2_monitoring_students_progress",
        "q3_faculty_involvement",
        "q4_infrastructure_facilities",
        "q5_learning_resources",
        "q6_study_environment_and_discipline",
        "q7_counselling_and_placements",
        "q8_support_facilities",
        "q9_parental_perception",
        "q10_holistic_development",
    ]

    gray = _cvt_gray(pil)
    # Use your coarse table finder only to limit X range; rows come from OCR text
    x1, y1, x2, y2 = _find_table_region(gray)
    H, W = gray.shape[:2]
    roiH, roiW = (y2 - y1), (x2 - x1)

    # Rating area: rightmost band
    RIGHT_FRACTION = 0.72   # a tad wider than before
    rating_x1 = x1 + int(max(0, roiW * RIGHT_FRACTION))
    min_w = max(48, int(roiW * 0.22))
    if (x2 - rating_x1) < min_w:
        rating_x1 = max(x1, x2 - min_w)
    rating_w = max(8, x2 - rating_x1)

    # ---- 1) Collect Y centers for rows from OCR numbers "1.", "2)", etc.
    num_rx = re.compile(r"^\s*(\d{1,2})\s*[.)\-:–]?\s*", re.I)
    y_by_qnum: Dict[int, int] = {}
    for ln in (lines or []):
        t = (ln.get("text") or "").strip()
        if not t:
            continue
        m = num_rx.match(t)
        if not m:
            continue
        qn = int(m.group(1))
        if 1 <= qn <= 10:
            x1l, y1l, x2l, y2l = (ln.get("box") or [0, 0, 0, 0])[:4]
            y_mid = int((y1l + y2l) // 2)
            # Keep the first seen (top-most) for each qn
            if qn not in y_by_qnum:
                y_by_qnum[qn] = y_mid

    # ---- 2) Fill gaps using label patterns (_QPAT) if numbers missing
    if len(y_by_qnum) < 10:
        for idx, (key, pat) in enumerate(_QPAT, start=1):
            if idx in y_by_qnum:
                continue
            rgx = re.compile(pat, re.I)
            hit = None
            for ln in lines or []:
                if rgx.search(ln.get("text") or ""):
                    _, y1l, _, y2l = (ln.get("box") or [0, 0, 0, 0])[:4]
                    hit = int((y1l + y2l) // 2)
                    break
            if hit is not None:
                y_by_qnum[idx] = hit

    # ---- 3) Interpolate any remaining gaps using neighbors
    ys = [y_by_qnum.get(i) for i in range(1, N_ROWS + 1)]
    # If we have at least first and last, linear interpolate the missing ones
    if ys[0] is not None and ys[-1] is not None:
        first_y, last_y = ys[0], ys[-1]
        for i in range(N_ROWS):
            if ys[i] is None:
                # proportion along the sequence
                ys[i] = int(first_y + (last_y - first_y) * (i / (N_ROWS - 1)))
    # If still None anywhere, try to fill from nearest known neighbor
    for i in range(N_ROWS):
        if ys[i] is None:
            # search forward
            f = next((ys[j] for j in range(i + 1, N_ROWS) if ys[j] is not None), None)
            # search backward
            b = next((ys[j] for j in range(i - 1, -1, -1) if ys[j] is not None), None)
            ys[i] = f if f is not None else (b if b is not None else (y1 + int((i + 0.5) * roiH / N_ROWS)))

    # ---- 4) Convert Y centers to row windows (midpoints between neighbors)
    bounds = []
    for i in range(N_ROWS + 1):
        if i == 0:
            b = max(y1, ys[0] - int(0.5 * (ys[1] - ys[0]) if N_ROWS > 1 else roiH / N_ROWS))
        elif i == N_ROWS:
            b = min(y2, ys[-1] + int(0.5 * (ys[-1] - ys[-2]) if N_ROWS > 1 else roiH / N_ROWS))
        else:
            b = int((ys[i - 1] + ys[i]) // 2)
        bounds.append(int(b))
    row_bands = [(bounds[i], bounds[i + 1]) for i in range(N_ROWS)]

    # ---- 5) Build 4 equal rating columns in the rating area (robust enough with small skew)
    col_bounds = [rating_x1 + int(round(rating_w * j / N_COLS)) for j in range(N_COLS + 1)]
    col_bands = [(col_bounds[j], col_bounds[j + 1]) for j in range(N_COLS)]

    # ---- 6) Score (with small vertical shifts) and pick per row
    def score_cell(cx1, cx2, ry1, ry2):
        inset_x, inset_y = 2, 2
        c = _crop_xyxy(gray, max(0, cx1 + inset_x), max(0, ry1 + inset_y),
                             min(W,  cx2 - inset_x), max(ry1 + inset_y + 1, ry2 - inset_y))
        return float(_tick_score_strong(c))

    ratings: List[Tuple[Optional[int], float]] = []
    for (ry1b, ry2b) in row_bands:
        best_pick, best_score = None, -1.0
        # small vertical sweeps for robustness
        for dy in (-6, -3, 0, 3, 6):
            ry1 = max(0, ry1b + dy)
            ry2 = min(H - 1, ry2b + dy)
            # score the four columns
            scs = [score_cell(cx1, cx2, ry1, ry2) for (cx1, cx2) in col_bands]
            smax = max(scs) if scs else 0.0
            idx = int(np.argmax(scs)) if scs else -1
            if smax > best_score:
                best_score = smax
                best_pick = (idx + 1) if idx >= 0 else None
        conf = float(max(0.0, min(0.95, best_score)))
        ratings.append((best_pick, conf if best_pick else 0.0))

    # ---- 7) Build result dict
    res: Dict[str, Dict[str, Any]] = {}
    for i, key in enumerate(canon_keys):
        v, c = ratings[i]
        res[key] = {
            "value": v if v in (1, 2, 3, 4) else None,
            "confidence": round(c, 3) if v in (1, 2, 3, 4) else 0.0,
            "source": "pta-free-ocr-rows",
        }

    # Optional debug overlay
    if debug_dir:
        dbg = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # row windows
        for (a, b) in row_bands:
            cv2.line(dbg, (x1, a), (x2, a), (255, 0, 0), 1)
            cv2.line(dbg, (x1, b), (x2, b), (255, 0, 0), 1)
        # columns
        for (cx1b, cx2b) in col_bands:
            cv2.line(dbg, (cx1b, y1), (cx1b, y2), (0, 0, 255), 1)
            cv2.line(dbg, (cx2b, y1), (cx2b, y2), (0, 0, 255), 1)
        # marks
        for i, (pick, _) in enumerate(ratings):
            if pick:
                (ra, rb) = row_bands[i]
                (cx1b, cx2b) = col_bands[pick - 1]
                cv2.rectangle(dbg, (cx1b, ra), (cx2b, rb), (0, 200, 255), 2)
        _save_dbg(dbg, os.path.join(debug_dir, "15_grid_ocr_rows.png"))

    return res

# --- NEW: tiny helpers to extract "Parents Signature and date" ---

_DATE_RX = re.compile(
    r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b"  # 08/10/25, 8-10-2025, 08.10.2025
)

def _normalize_dmy_date(d: str) -> str:
    """
    Normalize D/M/Y or DD/MM/YY to ISO YYYY-MM-DD (Indian DMY assumption).
    If year has 2 digits, assumes 20YY.
    Returns the normalized string, or the original if it can't be parsed.
    """
    m = _DATE_RX.search(d or "")
    if not m:
        return d or ""
    dd, mm, yy = m.groups()
    try:
        d_i = int(dd)
        m_i = int(mm)
        y_i = int(yy)
        if y_i < 100:
            y_i += 2000
        # basic sanity
        if not (1 <= d_i <= 31 and 1 <= m_i <= 12 and 2000 <= y_i <= 2100):
            return d or ""
        return f"{y_i:04d}-{m_i:02d}-{d_i:02d}"
    except Exception:
        return d or ""

def _find_label_index(lines: List[Dict[str, Any]], label_rx: str) -> Optional[int]:
    rx = re.compile(label_rx, re.I)
    for i, ln in enumerate(lines or []):
        if rx.search(ln.get("text") or ""):
            return i
    return None

def extract_signature_and_date(lines: List[Dict[str, Any]]) -> Tuple[str, str, float]:
    """
    Looks for a label like 'Parents Signature and date:' (variants allowed),
    then searches the following few lines for a date. The handwritten signature
    typically won't OCR well, so we return an empty signature by default.
    Returns: (signature_text, date_iso_or_raw, confidence)
    """
    # Common label variants (OCR can mangle a bit)
    label_rx = r"parents?\s*signature.*date|signature\s*and\s*date|parents?\s*signature"
    idx = _find_label_index(lines, label_rx)

    search_block: List[str] = []
    if idx is not None:
        # collect up to next 6 lines after the label (skip other labels)
        stop_rx = re.compile(r":\s*$|^\s*[A-Za-z].{1,40}:\s*", re.I)
        for ln in lines[idx+1: idx+1+6]:
            t = (ln.get("text") or "").strip()
            if stop_rx.search(t) and "date" not in t.lower():
                break
            if t:
                search_block.append(t)

    # If label not found, scan the last ~10 lines near the bottom of the page
    if not search_block:
        tail = [ (ln.get("text") or "").strip() for ln in (lines[-12:] if lines else []) ]
        search_block = [t for t in tail if t]

    raw_date = ""
    for t in search_block:
        m = _DATE_RX.search(t)
        if m:
            raw_date = m.group(0)
            break

    date_norm = _normalize_dmy_date(raw_date) if raw_date else ""
    conf = 0.9 if date_norm or raw_date else 0.0
    # Signature: OCR usually blank/garbled; leave empty string but keep field present.
    return "", (date_norm or raw_date), conf

def extract_ratings_via_label_aligned(pil: Image.Image,
                                      lines: List[Dict[str, Any]],
                                      debug_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Label-aligned rater:
    - Finds the '1 2 3 4' header (or '1234') in OCR near the right side of the table
    - Uses those digit x-centers to define the 4 rating columns (image-wide)
    - Defines row windows from OCR question indices/labels (like OCR-rows)
    - Scores each (row, col) with _tick_score_strong

    Returns: {canon_key: {"value": 1..4|None, "confidence": float, "source": "pta-free-label-align"}}
    """
    N_ROWS = 10
    N_COLS = 4
    canon_keys = [
        "q1_teaching_learning_environment",
        "q2_monitoring_students_progress",
        "q3_faculty_involvement",
        "q4_infrastructure_facilities",
        "q5_learning_resources",
        "q6_study_environment_and_discipline",
        "q7_counselling_and_placements",
        "q8_support_facilities",
        "q9_parental_perception",
        "q10_holistic_development",
    ]

    gray = _cvt_gray(pil)
    X1, Y1, X2, Y2 = _find_table_region(gray)
    H, W = gray.shape[:2]
    roiW = X2 - X1

    # ---- 1) Find header digits "1 2 3 4" (or "1234") and collect their x-centers
    digit_rx = re.compile(r"(?<!\d)([1-4])(?!\d)")
    header_candidates = []
    for ln in (lines or []):
        t = (ln.get("text") or "").strip()
        if not t:
            continue
        x1, y1, x2, y2 = (ln.get("box") or [0, 0, 0, 0])[:4]
        # focus on the table area and right side
        if y1 < Y1 or y2 > Y2:
            continue
        if x1 < X1 + int(roiW * 0.55):  # header typically far-right
            continue
        nums = digit_rx.findall(t)
        if not nums:
            continue
        # require at least 3 distinct digits among 1..4 on this line OR '1234' contiguous
        distinct = len(set(nums))
        good = (distinct >= 3) or ("1234" in t.replace(" ", "")) or ("1 2 3 4" in t)
        if good:
            header_candidates.append({"text": t, "box": (x1, y1, x2, y2)})

    # from candidates, determine four x-centers
    x_centers: List[int] = []
    if header_candidates:
        # choose the top-most candidate (closest to Y1)
        header_candidates.sort(key=lambda r: r["box"][1])
        hx1, hy1, hx2, hy2 = header_candidates[0]["box"]
        # collect per-word boxes containing a single digit 1..4 in that y band
        band_mid = int((hy1 + hy2) // 2)
        band_tol = max(6, (hy2 - hy1) // 2 + 4)
        per_word = []
        for ln in (lines or []):
            t = (ln.get("text") or "").strip()
            if not t:
                continue
            xx1, yy1, xx2, yy2 = (ln.get("box") or [0, 0, 0, 0])[:4]
            ymid = int((yy1 + yy2) // 2)
            if abs(ymid - band_mid) <= band_tol:
                m = digit_rx.fullmatch(t)  # a single digit on this word
                if m:
                    xc = int((xx1 + xx2) // 2)
                    d = int(m.group(1))
                    per_word.append((d, xc))
        # if we got some, compute center per digit (1..4)
        by_digit: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
        for d, xc in per_word:
            if d in by_digit:
                by_digit[d].append(xc)
        for d in (1, 2, 3, 4):
            if by_digit[d]:
                x_centers.append(int(sum(by_digit[d]) / len(by_digit[d])))

        # If still not 4 centers, try to infer evenly from header box
        if len(x_centers) < 4:
            xs = [int(hx1 + j * (hx2 - hx1) / 3.0) for j in range(4)]
            x_centers = xs

    # If none found, synthesize from rightmost area (fallback)
    if not x_centers:
        left = X1 + int(roiW * 0.70)
        right = X2 - 1
        x_centers = [int(t) for t in np.linspace(left, right, num=N_COLS)]

    x_centers = sorted([max(X1, min(X2 - 1, xc)) for xc in x_centers])[:4]
    # Convert centers to bands via midpoints
    col_bounds = [X1] + [int((x_centers[i] + x_centers[i + 1]) // 2) for i in range(len(x_centers) - 1)] + [X2]
    # keep only 4 bands around the 4 centers
    if len(col_bounds) >= 5:
        # choose the 4 bands whose mids are closest to the 4 x_centers
        bands = [(col_bounds[i], col_bounds[i + 1]) for i in range(len(col_bounds) - 1)]
        mids = [int((a + b) // 2) for (a, b) in bands]
        # map each center to closest band (unique)
        taken = set()
        col_bands = []
        for xc in x_centers:
            idx = int(np.argmin([abs(m - xc) if i not in taken else 10**9 for i, m in enumerate(mids)]))
            taken.add(idx)
            col_bands.append(bands[idx])
        col_bands.sort(key=lambda b: (b[0] + b[1]) // 2)
    else:
        # even split as last resort
        cb = [X1 + int(j * (X2 - X1) / N_COLS) for j in range(N_COLS + 1)]
        col_bands = list(zip(cb[:-1], cb[1:]))

    # ---- 2) Build row windows from OCR numbers/labels (same idea as OCR-rows)
    num_rx = re.compile(r"^\s*(\d{1,2})\s*[.)\-:–]?\s*", re.I)
    y_by_qnum: Dict[int, int] = {}
    for ln in (lines or []):
        t = (ln.get("text") or "").strip()
        if not t:
            continue
        m = num_rx.match(t)
        if not m:
            continue
        qn = int(m.group(1))
        if 1 <= qn <= 10:
            _, yy1l, _, yy2l = (ln.get("box") or [0, 0, 0, 0])[:4]
            y_mid = int((yy1l + yy2l) // 2)
            if y_mid >= Y1 and y_mid <= Y2 and qn not in y_by_qnum:
                y_by_qnum[qn] = y_mid

    # fill via _QPAT if needed
    if len(y_by_qnum) < 10:
        for idx, (key, pat) in enumerate(_QPAT, start=1):
            if idx in y_by_qnum:
                continue
            rgx = re.compile(pat, re.I)
            hit = None
            for ln in (lines or []):
                if rgx.search(ln.get("text") or ""):
                    _, yy1l, _, yy2l = (ln.get("box") or [0, 0, 0, 0])[:4]
                    hit = int((yy1l + yy2l) // 2)
                    break
            if hit is not None:
                y_by_qnum[idx] = hit

    ys = [y_by_qnum.get(i) for i in range(1, N_ROWS + 1)]
    if ys[0] is not None and ys[-1] is not None:
        first_y, last_y = ys[0], ys[-1]
        for i in range(N_ROWS):
            if ys[i] is None:
                ys[i] = int(first_y + (last_y - first_y) * (i / (N_ROWS - 1)))
    for i in range(N_ROWS):
        if ys[i] is None:
            # fallback to even spacing inside table region
            ys[i] = Y1 + int((i + 0.5) * (Y2 - Y1) / N_ROWS)

    # Convert y centers to bands (midpoints)
    bounds = []
    for i in range(N_ROWS + 1):
        if i == 0:
            b = max(Y1, ys[0] - int(0.5 * (ys[1] - ys[0]) if N_ROWS > 1 else (Y2 - Y1) / N_ROWS))
        elif i == N_ROWS:
            b = min(Y2, ys[-1] + int(0.5 * (ys[-1] - ys[-2]) if N_ROWS > 1 else (Y2 - Y1) / N_ROWS))
        else:
            b = int((ys[i - 1] + ys[i]) // 2)
        bounds.append(int(b))
    row_bands = [(bounds[i], bounds[i + 1]) for i in range(N_ROWS)]

    # ---- 3) Score and pick per row/col
    def score_cell(cx1, cx2, ry1, ry2):
        inset_x, inset_y = 2, 2
        c = _crop_xyxy(gray,
                       max(0, cx1 + inset_x), max(0, ry1 + inset_y),
                       min(W, cx2 - inset_x), max(ry1 + inset_y + 1, ry2 - inset_y))
        return float(_tick_score_strong(c))

    ratings: List[Tuple[Optional[int], float]] = []
    for (ry1b, ry2b) in row_bands:
        best_pick, best_score = None, -1.0
        for dy in (-6, -3, 0, 3, 6):
            ry1 = max(0, ry1b + dy)
            ry2 = min(H - 1, ry2b + dy)
            scs = [score_cell(cx1, cx2, ry1, ry2) for (cx1, cx2) in col_bands]
            if not scs:
                continue
            idx = int(np.argmax(scs))
            smax = float(scs[idx])
            if smax > best_score:
                best_score = smax
                best_pick = idx + 1
        conf = float(max(0.0, min(0.95, best_score)))
        ratings.append((best_pick, conf if best_pick else 0.0))

    # ---- 4) Build result
    res: Dict[str, Dict[str, Any]] = {}
    for i, key in enumerate(canon_keys):
        v, c = ratings[i]
        res[key] = {
            "value": v if v in (1, 2, 3, 4) else None,
            "confidence": round(c, 3) if v in (1, 2, 3, 4) else 0.0,
            "source": "pta-free-label-align",
        }

    # Debug overlay
    if debug_dir:
        dbg = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dbg, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
        for (a, b) in row_bands:
            cv2.line(dbg, (X1, a), (X2, a), (255, 0, 0), 1)
            cv2.line(dbg, (X1, b), (X2, b), 1, 1)
        for (cx1b, cx2b) in col_bands:
            cv2.line(dbg, (cx1b, Y1), (cx1b, Y2), (0, 0, 255), 1)
            cv2.line(dbg, (cx2b, Y1), (cx2b, Y2), (0, 0, 255), 1)
        for i, (pick, _) in enumerate(ratings):
            if pick:
                (ra, rb) = row_bands[i]
                (cx1b, cx2b) = col_bands[pick - 1]
                cv2.rectangle(dbg, (cx1b, ra), (cx2b, rb), (0, 200, 255), 2)
        _save_dbg(dbg, os.path.join(debug_dir, "16_label_aligned.png"))

    return res


# -------------------------------
# PTA FREE ENDPOINT
# -------------------------------
@app.post("/ocr/pta_free")
def ocr_pta_free(req: OCRReq):
    ok, usage = _bump_and_check_limit(limit=1000)
    if not ok:
        return JSONResponse(status_code=429, content={"error": "Monthly free-tier limit reached.", "usage": usage})

    debug_dir = None
    if req.debug:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        debug_dir = _ensure_dir(os.path.join(DEBUG_ROOT, f"{stamp}_pta_free"))

    try:
        pil = b64_to_pil(req.imageBase64)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Bad image: {e}"})

    try:
        resp, lines = _vision_page_once(pil, debug_dir)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Vision failed: {e}"})

    full_text = resp.full_text_annotation.text if resp and resp.full_text_annotation else ""
    full_text = _clean_spell(full_text)

    # --- 1) semantic+layout text-based rating extraction (you already added)
    rating_fields_text = extract_ratings_from_text_blocks(lines)

    # --- 2) your existing grid-based (coarse) detector
    gray = _cvt_gray(pil)
    bbox = _find_table_region(gray)
    row_bands_tmp, _ = _split_rows_cols(gray, bbox, n_rows=10, n_cols=4)
    row_bands_rc = [(rb[1], rb[3]) if len(rb) >= 4 else tuple(rb) for rb in row_bands_tmp]
    row_bands_rc = [
        (max(0, ry1 - (10 if i == 0 else 2)), min(gray.shape[0], ry2 + 2))
        for i, (ry1, ry2) in enumerate(row_bands_rc)
    ]
    x1, y1, x2, y2 = bbox
    W = x2 - x1
    RIGHT_FRACTION = 0.74
    rating_x1 = x1 + int(W * RIGHT_FRACTION)
    min_w = max(48, int(W * 0.22))
    if (x2 - rating_x1) < min_w:
        rating_x1 = max(x1, x2 - min_w)
    rating_w = max(8, x2 - rating_x1)
    col_bands_rc = []
    for j in range(4):
        cx1 = rating_x1 + (rating_w * j) // 4
        cx2 = rating_x1 + (rating_w * (j + 1)) // 4
        col_bands_rc.append((cx1, cx2))
    col_bands_rc = [
        (max(0, cx1 - 2), min(gray.shape[1], cx2 + 3))
        for (cx1, cx2) in col_bands_rc
    ]
    if debug_dir:
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
        for (ry1,ry2) in row_bands_rc:
            cv2.line(dbg,(x1,ry1),(x2,ry1),(255,0,0),1)
            cv2.line(dbg,(x1,ry2),(x2,ry2),(255,0,0),1)
        for (cx1,cx2) in col_bands_rc:
            cv2.line(dbg,(cx1,y1),(cx1,y2),(0,0,255),1)
            cv2.line(dbg,(cx2,y1),(cx2,y2),(0,0,255),1)
        _save_dbg(dbg, os.path.join(debug_dir,"13_grid_simple.png"))

    row_keys = _assign_rows_to_questions(lines, row_bands_rc, bbox)
    canon_keys = [k for k, _ in _QPAT]
    if len(row_keys) != len(row_bands_rc):
        row_keys = (row_keys + [None] * len(row_bands_rc))[:len(row_bands_rc)]
    row_keys = [
        (rk if rk else (canon_keys[i] if i < len(canon_keys) else f"q{i+1}_unknown"))
        for i, rk in enumerate(row_keys)
    ]

    ratings = []
    def _best_row_shift(ry1:int, ry2:int, col_bands_rc):
        SHIFTS = (-6, -4, -2, 0, 2, 4, 6)
        best_shift = 0
        best_peak  = -1.0
        for dy in SHIFTS:
            y1s = max(0, ry1 + dy)
            y2s = min(gray.shape[0], ry2 + dy)
            if y2s - y1s < 6:
                continue
            row_peak = -1.0
            for (cx1, cx2) in col_bands_rc:
                inset_x, inset_y = 2, 2
                cell_in = _crop_xyxy(gray, cx1 + inset_x, y1s + inset_y, cx2 - inset_x, y2s - inset_y)
                s_in = _tick_score_strong(cell_in)
                pad_x, pad_y = 3, 2
                cell_pad = _crop_xyxy(gray,
                                      max(0, cx1 - pad_x), max(0, y1s + pad_y),
                                      min(gray.shape[1], cx2 + pad_x), max(y1s + pad_y + 1, y2s - pad_y))
                s_pad = _tick_score_strong(cell_pad)
                s = max(float(s_in), 0.85 * float(s_pad))
                if s > row_peak:
                    row_peak = s
            if row_peak > best_peak:
                best_peak = row_peak
                best_shift = dy
        return best_shift

    for rid,(ry1,ry2) in enumerate(row_bands_rc, start=1):
        dy = _best_row_shift(ry1, ry2, col_bands_rc)
        ry1s = max(0, ry1 + dy)
        ry2s = min(gray.shape[0], ry2 + dy)
        scores = []
        for cid,(cx1b,cx2b) in enumerate(col_bands_rc, start=1):
            inset_x, inset_y = 2, 2
            cell_in = _crop_xyxy(gray, cx1b + inset_x, ry1s + inset_y, cx2b - inset_x, ry2s - inset_y)
            s_in = _tick_score_strong(cell_in)
            pad_x, pad_y = 3, 2
            cell_pad = _crop_xyxy(
                gray,
                max(0, cx1b - pad_x),
                max(0, ry1s + pad_y),
                min(gray.shape[1], cx2b + pad_x),
                max(ry1s + pad_y + 1, ry2s - pad_y)
            )
            s_pad = _tick_score_strong(cell_pad)
            s = max(float(s_in), 0.85 * float(s_pad))
            scores.append(s)
            if debug_dir:
                vis = cv2.cvtColor(cell_in, cv2.COLOR_GRAY2BGR)
                cv2.putText(vis, f"{s:.3f}", (2, min(14, vis.shape[0]-2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
                _save_dbg(vis, os.path.join(debug_dir, f"cell_r{rid}_c{cid}.png"))
        if not scores:
            ratings.append((None, 0.0))
            continue
        smax = max(scores)
        ssec = sorted(scores)[-2] if len(scores) >= 2 else 0.0
        mean = float(np.mean(scores))
        std  = float(np.std(scores))
        z    = (smax - mean) / (std + 1e-6)
        margin = smax - ssec
        ratio  = smax / max(ssec, 1e-6)
        Z_THR, M_THR, R_THR, BASE_THR = 1.25, 0.15, 1.35, 0.22
        best_idx = int(np.argmax(scores))
        confident = (z >= Z_THR and margin >= M_THR) or (ratio >= R_THR)
        conf = float(min(1.0, max(0.0, smax)))
        pick = (best_idx+1) if confident else None
        if pick is None:
            borderline = (smax >= 0.16) and (margin >= 0.02 or ratio >= 1.04)
            if borderline:
                pick = best_idx + 1
                conf = max(conf, 0.45 + 0.35 * smax)
        if pick is None and smax >= BASE_THR:
            pick = best_idx + 1
            conf = max(conf, smax)
        conf = float(max(0.0, min(1.0, conf)))
        ratings.append((pick, conf))

    # Map to keys
    mapped: Dict[str, Dict[str, Any]] = {}
    stable_keys: List[str] = []
    for i in range(len(row_bands_rc)):
        rk = row_keys[i]
        key = rk if rk in canon_keys else (canon_keys[i] if i < len(canon_keys) else f"q{i+1}_unknown")
        stable_keys.append(key)
    seen = set()
    for i, key in enumerate(stable_keys):
        if key in seen:
            key = canon_keys[i] if i < len(canon_keys) else f"q{i+1}_unknown"
            stable_keys[i] = key
        seen.add(key)
    if len(ratings) < len(row_bands_rc):
        ratings = ratings + [(None, 0.0)] * (len(row_bands_rc) - len(ratings))
    for i in range(len(canon_keys)):
        key = canon_keys[i]
        try:
            row_idx = stable_keys.index(key)
            choice, conf = ratings[row_idx]
        except ValueError:
            choice, conf = (None, 0.0)
        mapped[key] = {
            "value": choice,
            "confidence": round(float(conf), 3),
            "source": "pta-free-rating"
        }

    # --- 3) merge text-based
    for k, v in (rating_fields_text or {}).items():
        tv, tc = v.get("value"), float(v.get("confidence", 0.0))
        gv = mapped.get(k, {}).get("value")
        gc = float(mapped.get(k, {}).get("confidence", 0.0))
        if tv in (1, 2, 3, 4):
            if gv not in (1, 2, 3, 4) or tc >= gc + 0.10:
                mapped[k] = {"value": tv, "confidence": round(tc, 3), "source": "pta-free-rating-text"}
            else:
                mapped[k]["source"] = mapped[k].get("source", "") + "+grid"
        else:
            mapped.setdefault(k, {"value": gv if gv in (1,2,3,4) else None,
                                  "confidence": gc if gv in (1,2,3,4) else 0.0,
                                  "source": mapped.get(k, {}).get("source", "pta-free-rating")})

    # --- 4) dynamic grid (you added)
    rating_fields_dyn = {}
    try:
        if 'extract_ratings_via_dynamic_grid' in globals():
            rating_fields_dyn = extract_ratings_via_dynamic_grid(pil, lines, debug_dir)
    except Exception:
        rating_fields_dyn = {}
    for k, v in (rating_fields_dyn or {}).items():
        dv, dc = v.get("value"), float(v.get("confidence", 0.0))
        cv = mapped.get(k, {}).get("value")
        cc = float(mapped.get(k, {}).get("confidence", 0.0))
        if dv in (1, 2, 3, 4) and (cv not in (1, 2, 3, 4) or dc >= cc + 0.05):
            mapped[k] = {"value": dv, "confidence": round(dc, 3), "source": "pta-free-grid-dyn"}
        else:
            if k in mapped:
                mapped[k]["source"] = (mapped[k].get("source", "") + "+dyn").strip("+")

    # --- 5) OCR-rows (you added)
    rating_fields_rows = {}
    try:
        if 'extract_ratings_via_ocr_rows' in globals():
            rating_fields_rows = extract_ratings_via_ocr_rows(pil, lines, debug_dir)
    except Exception:
        rating_fields_rows = {}
    for k, v in (rating_fields_rows or {}).items():
        rv, rc = v.get("value"), float(v.get("confidence", 0.0))
        cv = mapped.get(k, {}).get("value")
        cc = float(mapped.get(k, {}).get("confidence", 0.0))
        if rv in (1, 2, 3, 4) and (cv not in (1, 2, 3, 4) or rc >= cc + 0.05):
            mapped[k] = {"value": rv, "confidence": round(rc, 3), "source": (v.get("source") or "pta-free-ocr-rows")}
        else:
            if k in mapped:
                mapped[k]["source"] = (mapped[k].get("source", "") + "+rows").strip("+")
        # --- NEW: Label-aligned extractor (uses header '1 2 3 4' to lock columns)
    rating_fields_label = {}
    try:
        if 'extract_ratings_via_label_aligned' in globals():
            rating_fields_label = extract_ratings_via_label_aligned(pil, lines, debug_dir)
    except Exception:
        rating_fields_label = {}

    for k, v in (rating_fields_label or {}).items():
        lv, lc = v.get("value"), float(v.get("confidence", 0.0))
        cv = mapped.get(k, {}).get("value")
        cc = float(mapped.get(k, {}).get("confidence", 0.0))
        # prefer label-aligned if current missing/weak, or a bit more confident
        if lv in (1, 2, 3, 4) and (cv not in (1, 2, 3, 4) or lc >= cc + 0.05):
            mapped[k] = {"value": lv, "confidence": round(lc, 3), "source": "pta-free-label-align"}
        else:
            if k in mapped:
                mapped[k]["source"] = (mapped[k].get("source", "") + "+label").strip("+")

    # --- 6) NEW: label-aligned scan (per-label Y band)
    rating_fields_labely = {}
    try:
        if 'extract_ratings_via_label_aligned' in globals():
            rating_fields_labely = extract_ratings_via_label_aligned(pil, lines, debug_dir)
    except Exception:
        rating_fields_labely = {}
    for k, v in (rating_fields_labely or {}).items():
        lv, lc = v.get("value"), float(v.get("confidence", 0.0))
        cv = mapped.get(k, {}).get("value")
        cc = float(mapped.get(k, {}).get("confidence", 0.0))
        # Give label-aligned a slight priority (it keys off the actual label line)
        if lv in (1, 2, 3, 4) and (cv not in (1, 2, 3, 4) or lc >= cc + 0.03):
            mapped[k] = {"value": lv, "confidence": round(lc, 3), "source": "pta-free-label-y"}
        else:
            if k in mapped:
                mapped[k]["source"] = (mapped[k].get("source", "") + "+labelY").strip("+")

    # --- 7) final tie-breaker: if multiple sources produced the same value, keep the highest confidence and append sources
    for k in [kk for kk, _ in _QPAT]:
        if k not in mapped:
            continue
        # nothing to do here — sources already merged with confidence comparison above

    # ---- Text fields (unchanged)
    parent_name  = _read_right_or_below(lines, r"^\s*(name|mamo)\s*:")
    contact_num  = _read_right_or_below(lines, r"^\s*contact\s*number\s*:", is_phone=True)
    ward_name    = _read_right_or_below(lines, r"^\s*ward'?s?\s*name\s*:")
    dept_year    = _read_right_or_below(lines, r"^\s*department\s*and\s*year\s*of\s*graduation\s*:")

    if contact_num:
        m = _phone_re.search(contact_num)
        contact_num = normalize_phone(m.group(0)) if m else normalize_phone(contact_num)

    comments = _fix_common_comment_typos(_extract_comments(lines))
    if comments:
        lines_only = [ln.strip() for ln in comments.splitlines() if ln.strip()]
        while lines_only and re.search(r"^\s*(strengthen\s+our\s+programme?s?\s*\.*|please\s+make\s+any\s+additional)", lines_only[0], re.I):
            lines_only.pop(0)
        lines_only = [ln for ln in lines_only
                      if not re.match(r"^\s*(name|mamo|contact\s*number|ward'?s?\s*name|department)", ln, re.I)]
        comments = "\n".join(lines_only)
        comments = _clean_spell(comments.strip())
            # --- NEW: Parents Signature + Date (separate fields)

        # Split department/year into separate fields (keep original combined for compat)
    dep_val, year_val = _split_department_and_year(dept_year)
        # --- Parents Signature + Date (separate fields)
    sig_text, sig_date, sig_conf = extract_signature_and_date(lines)

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


    mapped["parent_name"]    = {"value": parent_name, "confidence": 0.9 if parent_name else 0.0, "source": "pta-free"}
    mapped["contact_number"] = {"value": contact_num, "confidence": 0.9 if contact_num else 0.0, "source": "pta-free"}
    mapped["ward_name"]      = {"value": ward_name, "confidence": 0.9 if ward_name else 0.0, "source": "pta-free"}

    # NEW separate fields
    mapped["department"]          = {"value": dep_val,  "confidence": 0.9 if dep_val  else 0.0, "source": "pta-free"}
    mapped["year_of_graduation"]  = {"value": year_val, "confidence": 0.9 if year_val else 0.0, "source": "pta-free"}

    mapped["department_year"] = {
        "value": dept_year,
        "confidence": 0.9 if dept_year else 0.0,
        "source": "pta-free"
    }

    # --- Parents Signature + Date
    sig_text, sig_date, sig_conf = extract_signature_and_date(lines)
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

    # (Optional) keep legacy combined field for old clients
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


    # remove trailing parent name in comments if duplicated
    if mapped["comments"]["value"] and mapped["parent_name"]["value"]:
        comment_lines = [ln.strip() for ln in mapped["comments"]["value"].splitlines() if ln.strip()]
        pname_norm = _norm(mapped["parent_name"]["value"])
        while comment_lines and _norm(comment_lines[-1]) == pname_norm:
            comment_lines.pop(-1)
        mapped["comments"]["value"] = "\n".join(comment_lines).strip()
        if not mapped["comments"]["value"]:
            mapped["comments"]["confidence"] = 0.0

    mapped = reassign_fields_by_schema(mapped)

    return {
        "engine": "vision",
        "mode": "pta-free",
        "fields": mapped,
        "text": full_text,
        "usage": get_usage(),
        "debug_dir": debug_dir
    }

    ok, usage = _bump_and_check_limit(limit=1000)
    if not ok:
        return JSONResponse(status_code=429, content={"error": "Monthly free-tier limit reached.", "usage": usage})

    debug_dir = None
    if req.debug:
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        debug_dir = _ensure_dir(os.path.join(DEBUG_ROOT, f"{stamp}_pta_free"))

    try:
        pil = b64_to_pil(req.imageBase64)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Bad image: {e}"})

    try:
        resp, lines = _vision_page_once(pil, debug_dir)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Vision failed: {e}"})

    full_text = resp.full_text_annotation.text if resp and resp.full_text_annotation else ""
    full_text = _clean_spell(full_text)

    # --- NEW: semantic+layout text-based rating extraction (no deletions)
    rating_fields_text = extract_ratings_from_text_blocks(lines)

    gray = _cvt_gray(pil)
    bbox = _find_table_region(gray)
    
    row_bands_tmp, _ = _split_rows_cols(gray, bbox, n_rows=10, n_cols=4)
    row_bands_rc = [(rb[1], rb[3]) if len(rb) >= 4 else tuple(rb) for rb in row_bands_tmp]
    
    row_bands_rc = [
        (max(0, ry1 - (10 if i == 0 else 2)), min(gray.shape[0], ry2 + 2))
        for i, (ry1, ry2) in enumerate(row_bands_rc)
    ]

    # Build rating column bands properly
    x1, y1, x2, y2 = bbox
    W = x2 - x1
    RIGHT_FRACTION = 0.74
    rating_x1 = x1 + int(W * RIGHT_FRACTION)
    
    min_w = max(48, int(W * 0.22))
    if (x2 - rating_x1) < min_w:
        rating_x1 = max(x1, x2 - min_w)
    
    rating_w = max(8, x2 - rating_x1)
    
    # Create 4 equal columns in the rating area
    col_bands_rc = []
    for j in range(4):
        cx1 = rating_x1 + (rating_w * j) // 4
        cx2 = rating_x1 + (rating_w * (j + 1)) // 4
        col_bands_rc.append((cx1, cx2))
    
    # Expand with padding
    col_bands_rc = [
        (max(0, cx1 - 2), min(gray.shape[1], cx2 + 3))
        for (cx1, cx2) in col_bands_rc
    ]

    if debug_dir:
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dbg, (x1,y1), (x2,y2), (0,255,0), 2)
        for (ry1,ry2) in row_bands_rc:
            cv2.line(dbg,(x1,ry1),(x2,ry1),(255,0,0),1)
            cv2.line(dbg,(x1,ry2),(x2,ry2),(255,0,0),1)
        for (cx1,cx2) in col_bands_rc:
            cv2.line(dbg,(cx1,y1),(cx1,y2),(0,0,255),1)
            cv2.line(dbg,(cx2,y1),(cx2,y2),(0,0,255),1)
        _save_dbg(dbg, os.path.join(debug_dir,"13_grid_simple.png"))

    row_keys = _assign_rows_to_questions(lines, row_bands_rc, bbox)
    canon_keys = [k for k, _ in _QPAT]
    if len(row_keys) != len(row_bands_rc):
        row_keys = (row_keys + [None] * len(row_bands_rc))[:len(row_bands_rc)]
    row_keys = [
        (rk if rk else (canon_keys[i] if i < len(canon_keys) else f"q{i+1}_unknown"))
        for i, rk in enumerate(row_keys)
    ]

    ratings = []

    def _best_row_shift(ry1:int, ry2:int, col_bands_rc):
        """Find optimal vertical shift for row to maximize tick detection."""
        SHIFTS = (-6, -4, -2, 0, 2, 4, 6)
        best_shift = 0
        best_peak  = -1.0
        for dy in SHIFTS:
            y1s = max(0, ry1 + dy)
            y2s = min(gray.shape[0], ry2 + dy)
            if y2s - y1s < 6:
                continue
            row_peak = -1.0
            for (cx1, cx2) in col_bands_rc:
                inset_x, inset_y = 2, 2
                cell_in = _crop_xyxy(gray, cx1 + inset_x, y1s + inset_y, cx2 - inset_x, y2s - inset_y)
                s_in = _tick_score_strong(cell_in)
                pad_x, pad_y = 3, 2
                cell_pad = _crop_xyxy(gray,
                                      max(0, cx1 - pad_x), max(0, y1s + pad_y),
                                      min(gray.shape[1], cx2 + pad_x), max(y1s + pad_y + 1, y2s - pad_y))
                s_pad = _tick_score_strong(cell_pad)
                s = max(float(s_in), 0.85 * float(s_pad))
                if s > row_peak:
                    row_peak = s
            if row_peak > best_peak:
                best_peak = row_peak
                best_shift = dy
        return best_shift

    for rid,(ry1,ry2) in enumerate(row_bands_rc, start=1):
        dy = _best_row_shift(ry1, ry2, col_bands_rc)
        ry1s = max(0, ry1 + dy)
        ry2s = min(gray.shape[0], ry2 + dy)

        scores = []
        for cid,(cx1,cx2) in enumerate(col_bands_rc, start=1):
            inset_x, inset_y = 2, 2
            cell_in = _crop_xyxy(gray, cx1 + inset_x, ry1s + inset_y, cx2 - inset_x, ry2s - inset_y)
            s_in = _tick_score_strong(cell_in)

            pad_x, pad_y = 3, 2
            cell_pad = _crop_xyxy(
                gray,
                max(0, cx1 - pad_x),
                max(0, ry1s + pad_y),
                min(gray.shape[1], cx2 + pad_x),
                max(ry1s + pad_y + 1, ry2s - pad_y)
            )
            s_pad = _tick_score_strong(cell_pad)

            s = max(float(s_in), 0.85 * float(s_pad))
            scores.append(s)

            if debug_dir:
                vis = cv2.cvtColor(cell_in, cv2.COLOR_GRAY2BGR)
                cv2.putText(vis, f"{s:.3f}", (2, min(14, vis.shape[0]-2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
                _save_dbg(vis, os.path.join(debug_dir, f"cell_r{rid}_c{cid}.png"))

        if not scores:
            ratings.append((None, 0.0))
            continue

        smax = max(scores)
        ssec = sorted(scores)[-2] if len(scores) >= 2 else 0.0
        mean = float(np.mean(scores))
        std  = float(np.std(scores))
        z    = (smax - mean) / (std + 1e-6)
        margin = smax - ssec
        ratio  = smax / max(ssec, 1e-6)

        Z_THR, M_THR, R_THR, BASE_THR = 1.25, 0.15, 1.35, 0.22

        best_idx = int(np.argmax(scores))
        confident = (z >= Z_THR and margin >= M_THR) or (ratio >= R_THR)

        conf = float(min(1.0, max(0.0, smax)))
        pick = (best_idx+1) if confident else None

        if pick is None:
            borderline = (smax >= 0.16) and (margin >= 0.02 or ratio >= 1.04)
            if borderline:
                pick = best_idx + 1
                conf = max(conf, 0.45 + 0.35 * smax)

        if pick is None and smax >= BASE_THR:
            pick = best_idx + 1
            conf = max(conf, smax)

        conf = float(max(0.0, min(1.0, conf)))
        ratings.append((pick, conf))

    # Map ratings to question keys
    mapped: Dict[str, Dict[str, Any]] = {}

    stable_keys: List[str] = []
    for i in range(len(row_bands_rc)):
        rk = row_keys[i]
        key = rk if rk in canon_keys else (canon_keys[i] if i < len(canon_keys) else f"q{i+1}_unknown")
        stable_keys.append(key)

    seen = set()
    for i, key in enumerate(stable_keys):
        if key in seen:
            key = canon_keys[i] if i < len(canon_keys) else f"q{i+1}_unknown"
            stable_keys[i] = key
        seen.add(key)

    if len(ratings) < len(row_bands_rc):
        ratings = ratings + [(None, 0.0)] * (len(row_bands_rc) - len(ratings))

    for i in range(len(canon_keys)):
        key = canon_keys[i]
        try:
            row_idx = stable_keys.index(key)
            choice, conf = ratings[row_idx]
        except ValueError:
            choice, conf = (None, 0.0)
        mapped[key] = {
            "value": choice,
            "confidence": round(float(conf), 3),
            "source": "pta-free-rating"
        }

    # --- NEW: Hybrid merge — prefer semantic/text rating when it's clearly better
    for k, v in (rating_fields_text or {}).items():
        tv, tc = v.get("value"), float(v.get("confidence", 0.0))
        gv = mapped.get(k, {}).get("value")
        gc = float(mapped.get(k, {}).get("confidence", 0.0))
        if tv in (1, 2, 3, 4):
            # prefer text if grid missing/weak, or text is >=0.10 more confident
            if gv not in (1, 2, 3, 4) or tc >= gc + 0.10:
                mapped[k] = {"value": tv, "confidence": round(tc, 3), "source": "pta-free-rating-text"}
            else:
                mapped[k]["source"] = mapped[k].get("source", "") + "+grid"
        else:
            mapped.setdefault(k, {"value": gv if gv in (1,2,3,4) else None,
                                  "confidence": gc if gv in (1,2,3,4) else 0.0,
                                  "source": mapped.get(k, {}).get("source", "pta-free-rating")})

    # --- NEW: Dynamic grid detector (auto line-clustered) — optional & safe-guarded
    rating_fields_dyn = {}
    try:
        if 'extract_ratings_via_dynamic_grid' in globals():
            rating_fields_dyn = extract_ratings_via_dynamic_grid(pil, lines, debug_dir)
    except Exception:
        rating_fields_dyn = {}

    for k, v in (rating_fields_dyn or {}).items():
        dv, dc = v.get("value"), float(v.get("confidence", 0.0))
        cv = mapped.get(k, {}).get("value")
        cc = float(mapped.get(k, {}).get("confidence", 0.0))
        # prefer dynamic grid if current missing/weak, or confidence better by 0.05
        if dv in (1, 2, 3, 4) and (cv not in (1, 2, 3, 4) or dc >= cc + 0.05):
            mapped[k] = {"value": dv, "confidence": round(dc, 3), "source": "pta-free-grid-dyn"}
        else:
            if k in mapped:
                mapped[k]["source"] = (mapped[k].get("source", "") + "+dyn").strip("+")
    
        # --- NEW: OCR-row windows rater (semantic rows; no line dependence)
    rating_fields_rows = {}
    try:
        if 'extract_ratings_via_ocr_rows' in globals():
            rating_fields_rows = extract_ratings_via_ocr_rows(pil, lines, debug_dir)
    except Exception:
        rating_fields_rows = {}

    for k, v in (rating_fields_rows or {}).items():
        rv, rc = v.get("value"), float(v.get("confidence", 0.0))
        cv = mapped.get(k, {}).get("value")
        cc = float(mapped.get(k, {}).get("confidence", 0.0))
        # prefer OCR-rows when current missing/weak, or confidence better by 0.05
        if rv in (1, 2, 3, 4) and (cv not in (1, 2, 3, 4) or rc >= cc + 0.05):
            mapped[k] = {"value": rv, "confidence": round(rc, 3), "source": (v.get("source") or "pta-free-ocr-rows")}
        else:
            if k in mapped:
                mapped[k]["source"] = (mapped[k].get("source", "") + "+rows").strip("+")


    # Extract text fields
    parent_name  = _read_right_or_below(lines, r"^\s*(name|mamo)\s*:")
    contact_num  = _read_right_or_below(lines, r"^\s*contact\s*number\s*:", is_phone=True)
    ward_name    = _read_right_or_below(lines, r"^\s*ward'?s?\s*name\s*:")
    dept_year    = _read_right_or_below(lines, r"^\s*department\s*and\s*year\s*of\s*graduation\s*:")

    if contact_num:
        m = _phone_re.search(contact_num)
        contact_num = normalize_phone(m.group(0)) if m else normalize_phone(contact_num)

    comments = _fix_common_comment_typos(_extract_comments(lines))

    if comments:
        lines_only = [ln.strip() for ln in comments.splitlines() if ln.strip()]
        while lines_only and re.search(r"^\s*(strengthen\s+our\s+programme?s?\s*\.*|please\s+make\s+any\s+additional)", lines_only[0], re.I):
            lines_only.pop(0)
        lines_only = [ln for ln in lines_only
                      if not re.match(r"^\s*(name|mamo|contact\s*number|ward'?s?\s*name|department)", ln, re.I)]
        comments = "\n".join(lines_only)
        comments = _clean_spell(comments.strip())

    mapped["parent_name"]               = {"value": parent_name, "confidence": 0.9 if parent_name else 0.0, "source": "pta-free"}
    mapped["contact_number"]            = {"value": contact_num, "confidence": 0.9 if contact_num else 0.0, "source": "pta-free"}
    mapped["ward_name"]                 = {"value": ward_name, "confidence": 0.9 if ward_name else 0.0, "source": "pta-free"}
    mapped["department_year"]           = {"value": dept_year, "confidence": 0.9 if dept_year else 0.0, "source": "pta-free"}
    mapped["parent_signature_and_date"] = {"value": "", "confidence": 0.0, "source": "pta-free"}
    mapped["comments"]                  = {"value": comments, "confidence": 0.9 if comments else 0.0, "source": "pta-free"}

    # Clean up comments: remove parent name if it appears at the end
    if mapped["comments"]["value"] and mapped["parent_name"]["value"]:
        comment_lines = [ln.strip() for ln in mapped["comments"]["value"].splitlines() if ln.strip()]
        pname_norm = _norm(mapped["parent_name"]["value"])
        
        # Remove parent name from end of comments if it matches
        while comment_lines and _norm(comment_lines[-1]) == pname_norm:
            comment_lines.pop(-1)
        
        mapped["comments"]["value"] = "\n".join(comment_lines).strip()
        if not mapped["comments"]["value"]:
            mapped["comments"]["confidence"] = 0.0

    # Apply field validation and reassignment
    mapped = reassign_fields_by_schema(mapped)

    return {
        "engine": "vision",
        "mode": "pta-free",
        "fields": mapped,
        "text": full_text,
        "usage": get_usage(),
        "debug_dir": debug_dir
    }


# -------------------------------
# Simple UI
# -------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"/><title>Agnel OCR — Vision</title>
<style>
body{font-family:system-ui,Arial;margin:20px}
.card{max-width:980px;margin:auto;border:1px solid #ddd;border-radius:12px;padding:16px}
.row{display:flex;gap:16px}
img{max-height:420px;border:1px solid #ddd;border-radius:8px}
.btn{background:#2563eb;color:#fff;border:0;padding:8px 12px;border-radius:8px;cursor:pointer}
pre{white-space:pre-wrap;word-break:break-word}
label{font-weight:600}
small{color:#555}
</style></head>
<body>
<div class="card">
  <h2>Agnel OCR — Google Vision <small>(with free fallback)</small></h2>
  <div>
    <label>Mode</label>
    <select id="mode">
      <option value="auto">Intelligent (no template)</option>
      <option value="pta">Template (YAML)</option>
      <option value="pta_free" selected>PTA (label-based)</option>
    </select>
    <input id="template" placeholder="template name (e.g. pta_parent_feedback_v1)" style="display:none;margin-left:8px;"/>
  </div>
  <div style="margin-top:8px;">
    <label><input type="checkbox" id="debugFlag"/> Save debug images</label>
  </div>
  <div style="margin-top:8px;">
    <input type="file" id="file" accept="image/*,pdf"/>
  </div>
  <div style="margin-top:12px;">
    <button class="btn" onclick="run()">Run OCR</button>
  </div>
  <div id="preview" style="margin-top:16px;"></div>
  <h3>Fields</h3><pre id="fields"></pre>
  <h3>Usage</h3><pre id="conf"></pre>
  <h3>Raw text</h3><pre id="text"></pre>
</div>
<script>
  const modeSel = document.getElementById('mode');
  const tpl = document.getElementById('template');
  modeSel.addEventListener('change',()=>{ tpl.style.display = modeSel.value==='pta' ? '' : 'none'; });

  function fileToDataURL(f){
    return new Promise(res=>{
      const r=new FileReader();
      r.onload=()=>res(r.result);
      r.readAsDataURL(f);
    });
  }

  async function run(){
    const f = document.getElementById('file').files[0];
    if(!f){ alert('Choose an image'); return; }
    const b64 = await fileToDataURL(f);
    document.getElementById('preview').innerHTML = '<img src="'+b64+'"/>';

    const mode = modeSel.value;
    const dbg = document.getElementById('debugFlag').checked;
    const url =
      mode==='pta' ? '/ocr/pta' :
      mode==='pta_free' ? '/ocr/pta_free' :
      '/ocr/auto';

    const payload = { imageBase64: b64, template: tpl.value || null, debug: dbg };
    const resp = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
    const out = await resp.json();

    document.getElementById('fields').textContent = JSON.stringify(out.fields || out.error || {}, null, 2);
    document.getElementById('conf').textContent   = JSON.stringify(out.usage || {}, null, 2);
    document.getElementById('text').textContent   = out.text || '';
  }
</script>
</body></html>
"""
    return HTMLResponse(HTML_PAGE)