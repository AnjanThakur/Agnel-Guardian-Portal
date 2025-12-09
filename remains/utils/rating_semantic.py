# utils/rating_semantic.py
import re
from typing import List, Dict, Any

import numpy as np


def extract_ratings_from_text_blocks(lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Semantic + layout mapping of question numbers 1..10 to ratings 1..4 using Vision line fragments.
    """
    questions_by_num = {
        1: "q1_teaching_learning_environment",
        2: "q2_monitoring_students_progress",
        3: "q3_faculty_involvement",
        4: "q4_infrastructure_facilities",
        5: "q5_learning_resources",
        6: "q6_study_environment_and_discipline",
        7: "q7_counselling_and_placements",
        8: "q8_support_facilities",
        9: "q9_parental_perception",
        10: "q10_holistic_development",
    }

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

    q_rows = []
    q_header_rx = re.compile(r"^\s*(\d{1,2})\s*[.)\-:–]?\s*(.*)$")
    for r in clean:
        m = q_header_rx.match(r["text"])
        if not m:
            continue
        qnum = int(m.group(1))
        if 1 <= qnum <= 10:
            q_rows.append({"qnum": qnum, "y": r["y"], "text": r["text"]})

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

    digit_rx = re.compile(r"(?<!\d)([1-4])(?!\d)")

    def _looks_like_header(t: str) -> bool:
        s = t.replace(" ", "")
        if "1234" in s:
            return True
        nums = digit_rx.findall(t)
        return len(set(nums)) >= 3

    rating_lines = []
    for r in clean:
        if _looks_like_header(r["text"]):
            continue
        nums = [int(n) for n in digit_rx.findall(r["text"])]
        if nums:
            rating_lines.append({"y": r["y"], "nums": nums, "text": r["text"]})

    result: Dict[str, Dict[str, Any]] = {}

    def pick_value(nums):
        if len(nums) == 1:
            return nums[0]
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
                conf = float(max(0.20, min(0.85, 0.85 - min(best_dy, 80) / 110.0)))
            result[qkey] = {
                "value": val if val in (1, 2, 3, 4) else None,
                "confidence": conf if val in (1, 2, 3, 4) else 0.0,
                "source": "pta-free-rating",
            }

    for qkey in questions_by_num.values():
        if qkey not in result:
            result[qkey] = {"value": None, "confidence": 0.0, "source": "pta-free-rating"}

    return result
