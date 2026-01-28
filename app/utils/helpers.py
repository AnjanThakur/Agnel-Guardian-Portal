# app/utils/helpers.py
import base64
from io import BytesIO
from typing import Tuple

from PIL import Image
# app/utils/helpers.py
import cv2
import os

def b64_to_bytes(b64: str) -> bytes:
    """
    Accepts either:
      - raw base64 string
      - or 'data:image/png;base64,...'
    """
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def b64_to_pil(b64: str) -> Image.Image:
    data = b64_to_bytes(b64)
    return Image.open(BytesIO(data)).convert("RGB")


def pil_to_cv2(pil_img: Image.Image):
    """
    Convert RGB PIL -> BGR numpy (OpenCV).
    """
    import cv2
    import numpy as np

    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def ensure_dir(path: str) -> str:
    from pathlib import Path

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)




def save_debug_image(img, path):
    if img is None:
        return
    if not hasattr(img, "size") or img.size == 0:
        return
    cv2.imwrite(path, img)


