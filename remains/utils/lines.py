# utils/lines.py
from typing import List, Dict, Any

import numpy as np


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
