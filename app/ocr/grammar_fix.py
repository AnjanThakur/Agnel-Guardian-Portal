"""
Grammar + spacing correction utility for OCR-extracted text.

Pipeline:
1. Fix missing spaces using word segmentation
2. Fix grammar, punctuation, capitalization using LanguageTool

Does NOT:
- rewrite meaning
- hallucinate content
"""

from __future__ import annotations
from typing import Optional, List

import language_tool_python
from wordfreq import zipf_frequency
from wordsegment import load as ws_load, segment as ws_segment

# load wordsegment once
ws_load()

_tool: Optional[language_tool_python.LanguageTool] = None


def _get_tool() -> language_tool_python.LanguageTool:
    global _tool
    if _tool is None:
        _tool = language_tool_python.LanguageTool("en-US")
    return _tool


def _needs_segmentation(token: str) -> bool:
    """
    Decide if a token looks like glued words.
    """
    if len(token) < 12:
        return False

    # if it’s a known word, don’t touch it
    if zipf_frequency(token.lower(), "en") > 3:
        return False

    # long alphabetic token with low frequency → likely glued
    return token.isalpha()


def _segment_line(line: str) -> str:
    words: List[str] = []

    for token in line.split():
        if _needs_segmentation(token):
            parts = ws_segment(token.lower())
            words.append(" ".join(parts))
        else:
            words.append(token)

    return " ".join(words)


def grammar_fix(text: str) -> str:
    """
    Safe grammar + spacing correction.

    - Preserves line breaks
    - Fixes missing spaces
    - Fixes grammar
    """
    if not text or not text.strip():
        return ""

    tool = _get_tool()
    fixed_lines: List[str] = []

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            fixed_lines.append("")
            continue

        try:
            # 1️⃣ spacing fix
            spaced = _segment_line(raw)

            # 2️⃣ grammar fix
            corrected = tool.correct(spaced)

            fixed_lines.append(corrected)
        except Exception:
            fixed_lines.append(raw)

    return "\n".join(fixed_lines)
