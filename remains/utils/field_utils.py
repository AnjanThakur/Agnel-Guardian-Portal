# utils/field_utils.py
import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# ----------------------
# TEXT NORMALIZATION
# ----------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ----------------------
# PHONE NORMALIZATION
# ----------------------

phone_re = re.compile(r"\b(\+91[- ]?)?[6-9]\d{9}\b")

def normalize_phone(s: str) -> str:
    s = s.replace(" ", "").replace("-", "")
    if s.startswith("+91"):
        s = s[3:]
    if len(s) == 10:
        return s
    return s[-10:] if len(s) > 10 else s


# ----------------------
# READING LABELS RIGHT/BELOW
# ----------------------

def read_right_or_below(lines: List[Dict[str, Any]], regex, is_phone=False):
    rgx = re.compile(regex, re.I)
    hits = []
    for ln in lines:
        t = ln["text"]
        m = rgx.search(t)
        if m:
            x1, y1, x2, y2 = ln["box"]
            hits.append(((x1, y1, x2, y2), t))

    if not hits:
        return None

    (x1, y1, x2, y2), label_text = hits[0]

    best = None
    best_d = 1e9

    for ln in lines:
        xt, yt, x3, y3 = ln["box"]
        if ln["text"] == label_text:
            continue

        # right
        if xt > x2 and abs(yt - y1) < 25:
            d = abs(xt - x2)
            if d < best_d:
                best = ln["text"]
                best_d = d

        # below
        if yt > y2 and abs(xt - x1) < 60:
            d = abs(yt - y2)
            if d < best_d:
                best = ln["text"]
                best_d = d

    if best and is_phone:
        m = phone_re.search(best)
        if m:
            return normalize_phone(m.group(0))
    return _norm(best) if best else None


# ----------------------
# COMMENTS EXTRACTION
# ----------------------

def extract_comments(lines: List[Dict[str, Any]]) -> Optional[str]:
    comment_rx = re.compile(r"comments?", re.I)
    start_y = None
    for ln in lines:
        if comment_rx.search(ln["text"]):
            start_y = ln["box"][1]
            break

    if start_y is None:
        return None

    texts = []
    for ln in lines:
        if ln["box"][1] > start_y + 15:
            t = ln["text"]
            if len(t.split()) >= 2:
                texts.append(t)

    return _norm(" ".join(texts)) if texts else None


def fix_common_comment_typos(t: Optional[str]) -> Optional[str]:
    if not t:
        return t
    t = t.replace("facultu", "faculty")
    t = re.sub(r"\bco[ -]?curricular\b", "co-curricular", t, flags=re.I)
    t = re.sub(r"\bbehav(?:i)?our?\b", "behaviour", t, flags=re.I)
    return t


# ----------------------
# DEPARTMENT + YEAR
# ----------------------

def split_department_and_year(text: Optional[str]):
    if not text:
        return None, None
    text = _norm(text)

    year_rx = re.compile(r"\b(20\d{2})\b")
    m = year_rx.search(text)
    year = m.group(1) if m else None

    dept = text.replace(year or "", "").strip(" ,:-") if year else text
    dept = dept if len(dept) > 2 else None

    return dept, year


# ----------------------
# SIGNATURE + DATE
# ----------------------

date_re = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")

def extract_signature_and_date(lines: List[Dict[str, Any]]):
    sig_text = None
    sig_date = None
    conf = 0.0

    for ln in lines:
        t = ln["text"]
        if "signature" in t.lower():
            sig_text = t
            conf += 0.5

        m = date_re.search(t)
        if m:
            sig_date = m.group(1)
            conf += 0.5

    return sig_text, sig_date, min(conf, 1.0)


# ----------------------
# SCHEMA + VALIDATION
# ----------------------

FIELD_SCHEMAS = {
    "parent_name": str,
    "contact_number": str,
    "ward_name": str,
    "department_year": str,
    "department": str,
    "year_of_graduation": str,
    "comments": str,
    "parent_signature": str,
    "signature_date": str,
}

def _validate_field(key, value):
    if key not in FIELD_SCHEMAS:
        return value
    typ = FIELD_SCHEMAS[key]
    if value is None:
        return None
    try:
        return typ(value)
    except Exception:
        return None


def reassign_fields_by_schema(fields: Dict[str, Any]):
    out = {}
    for k, v in fields.items():
        val = v.get("value")
        good = _validate_field(k, val)
        out[k] = {**v, "value": good}
    return out
