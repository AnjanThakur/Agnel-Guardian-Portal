# app/analysis/sanatize_text.py
from __future__ import annotations
import re
from typing import List

# Labels that must NEVER reach NLP/LLM
_LABELS = [
    r"name",
    r"contact\s*number",
    r"ward'?s?\s*name",
    r"department",
    r"graduation",
    r"parent'?s?\s*signature",
    r"signature",
    r"date",
]

# Printed boilerplate that can leak into comment ROI
_PRINTED_PHRASES = [
    "please make any additional comments",
    "comments or suggestions",
    "strengthen our programmes",
    "students overall holistic development",
    "parent feedback form",
    "fr. c. rodrigues",
    "agnel charities",
    "please rate on the following",
    "the information provided in this form",
]

# PII patterns
_RE_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-]{7,}\d)(?!\d)")
_RE_EMAIL = re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.I)

# Dates: 29/10/25, 29-10-2025, July 29 2025, 1. 08/10/25 etc.
_RE_DATE_NUM = re.compile(r"\b\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2,4}\b")
_RE_DATE_WORD = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b.*\b\d{2,4}\b",
    re.I
)

# Years / batches like 2026, 2062 etc.
_RE_YEAR = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")

# “Dept + year” noisy OCR patterns
_RE_DEPT_YEAR = re.compile(r"\b(b\.?\s*tech|m\.?\s*tech|btech|mtech|comp|computer|it|aids|aiml)\b.*\b(19\d{2}|20\d{2}|21\d{2})\b", re.I)

# A “label line” if contains label keyword OR looks like “Label:”
def _is_label_line(line: str) -> bool:
    low = line.lower()
    if ":" in low:
        left = low.split(":", 1)[0]
        if any(re.search(rf"\b{lab}\b", left) for lab in _LABELS):
            return True
    if any(re.search(rf"\b{lab}\b", low) for lab in _LABELS):
        # even without colon OCR sometimes drops it
        return True
    return False


def _is_printed_boilerplate(line: str) -> bool:
    low = " ".join(line.lower().split())
    return any(p in low for p in _PRINTED_PHRASES)


def _looks_like_pii_value(line: str) -> bool:
    low = line.strip().lower()
    if _RE_PHONE.search(low):
        return True
    if _RE_EMAIL.search(low):
        return True
    if _RE_DATE_NUM.search(low) or _RE_DATE_WORD.search(low):
        return True
    if _RE_YEAR.search(low):
        return True
    if _RE_DEPT_YEAR.search(low):
        return True
    # short “name-like” lines (OCR)
    toks = [t for t in re.split(r"\s+", line.strip()) if t]
    if 1 <= len(toks) <= 4 and all(t[:1].isalpha() for t in toks) and not any(ch.isdigit() for ch in line):
        # risk: could delete short legit comment; we only apply this in label-zone, not globally.
        return False
    return False


def sanitize_for_analysis(raw: str) -> str:
    """
    Sanitizes comment text for NLP/LLM:
      - removes boilerplate printed phrases
      - removes label headers AND their following values
      - removes PII (phone/email/date/year)
      - keeps free-form feedback lines
    """
    if not raw:
        return ""

    # normalize whitespace + split
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return ""

    cleaned: List[str] = []
    in_label_zone = False
    label_zone_budget = 8  # how many subsequent lines we can skip after labels before exiting

    for ln in lines:
        low = ln.lower().strip()

        # drop boilerplate
        if _is_printed_boilerplate(ln):
            continue

        # detect label start
        if _is_label_line(ln):
            in_label_zone = True
            label_zone_budget = 8
            continue

        # within label zone: drop likely label values
        if in_label_zone:
            label_zone_budget -= 1

            # end label zone if we hit a very “sentence-like” line
            # (helps if parent wrote next to labels)
            if len(ln) >= 25 and (" " in ln) and not _RE_PHONE.search(ln) and not _RE_DATE_NUM.search(ln):
                in_label_zone = False
                cleaned.append(ln)
                continue

            # remove common PII/value lines
            if _looks_like_pii_value(ln) or _RE_PHONE.search(ln) or _RE_EMAIL.search(ln) or _RE_DATE_NUM.search(ln) or _RE_DATE_WORD.search(ln) or _RE_YEAR.search(ln):
                continue

            # If we’ve skipped enough, exit label zone
            if label_zone_budget <= 0:
                in_label_zone = False

            # otherwise skip short label-ish fragments
            continue

        # global PII removal (even if not in label zone)
        if _RE_PHONE.search(ln) or _RE_EMAIL.search(ln) or _RE_DATE_NUM.search(ln) or _RE_DATE_WORD.search(ln):
            continue

        # keep actual feedback
        cleaned.append(ln)

    # join as paragraph
    out = " ".join(cleaned).strip()

    # final cleanup: collapse repeated spaces
    out = re.sub(r"\s+", " ", out).strip()

    return out
