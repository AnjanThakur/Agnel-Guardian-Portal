"""
OCR text normalization utilities.

Purpose:
- Fix OCR line-break word fragmentation
- Fix missing spaces between words
- Prepare text for grammar correction

Does NOT:
- Change meaning
- Rewrite sentences
- Perform grammar correction
"""

import re


def normalize_ocr_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    lines = [ln.strip() for ln in text.splitlines()]

    normalized = []
    buffer = ""

    for line in lines:
        if not line:
            if buffer:
                normalized.append(buffer)
                buffer = ""
            normalized.append("")
            continue

        # If previous line ends mid-word, join
        if buffer:
            if _looks_like_word_fragment(buffer, line):
                buffer = buffer + line
            else:
                normalized.append(buffer)
                buffer = line
        else:
            buffer = line

    if buffer:
        normalized.append(buffer)

    result = "\n".join(normalized)

    # Fix missing spaces between lowercase+uppercase words
    result = re.sub(r"([a-z])([A-Z])", r"\1 \2", result)

    # Fix missing spaces between words like "thanjust"
    result = re.sub(r"([a-zA-Z])([A-Z][a-z])", r"\1 \2", result)

    return result


def _looks_like_word_fragment(prev: str, curr: str) -> bool:
    """
    Detects OCR word splits:
    - Givestudents + more → Givestudentsmore
    - hand + Experience → handExperience
    - hands + on → hands on
    """
    if not prev or not curr:
        return False

    # classic mid-word join
    if prev[-1].islower() and curr[0].islower():
        return True

    # broken phrase: short word + Capitalized continuation
    if (
        len(prev) <= 5
        and prev.isalpha()
        and curr[0].isupper()
        and curr[1:].islower()
    ):
        return True

    return False

