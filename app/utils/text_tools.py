# app/utils/text_tools.py
import re
from typing import Dict, List, Tuple


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def build_line_y_order(lines: List[Dict]) -> List[Dict]:
    """
    Given Vision lines [{text, box}], return sorted by Y center.
    """
    def y_center(ln):
        box = ln.get("box") or [0, 0, 0, 0]
        return (box[1] + box[3]) / 2.0

    return sorted(lines, key=y_center)


def extract_field_by_label(
    lines: List[Dict],
    label_rx: str,
    read_below: bool = True,
    max_y_gap: int = 60,
) -> str:
    """
    Very simple heuristic:
    - find a line that matches label_rx
    - then read the text of the *next* line or the same line after ':'.
    """
    rx = re.compile(label_rx, re.I)
    ordered = build_line_y_order(lines)

    for i, ln in enumerate(ordered):
        txt = (ln.get("text") or "").strip()
        m = rx.search(txt)
        if not m:
            continue

        # Try "right of colon" on same line
        colon_pos = txt.find(":")
        if colon_pos >= 0 and colon_pos < len(txt) - 1:
            val = txt[colon_pos + 1 :].strip()
            if val:
                return normalize_space(val)

        # Try next line below
        if read_below and i + 1 < len(ordered):
            below = ordered[i + 1]
            y_here = (ln["box"][1] + ln["box"][3]) / 2
            y_below = (below["box"][1] + below["box"][3]) / 2
            if abs(y_below - y_here) <= max_y_gap:
                return normalize_space(below.get("text") or "")

    return ""
