# utils/io_utils.py
import base64
import io
import os
from typing import Tuple, Optional

import numpy as np
import cv2
from PIL import Image

from config.settings import DEBUG_ROOT


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_dbg(img: np.ndarray, path: str) -> None:
    try:
        cv2.imwrite(path, img)
    except Exception:
        pass


def b64_to_pil(image_b64: str) -> Image.Image:
    raw = base64.b64decode(image_b64.split(",")[-1])
    return Image.open(io.BytesIO(raw)).convert("RGB")


def pil_to_bytes(pil: Image.Image, fmt: str = ".png") -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG" if fmt == ".png" else "JPEG")
    return buf.getvalue()


def safe_crop_pil(pil: Image.Image, box_xyxy: Tuple[int, int, int, int]) -> Optional[Image.Image]:
    W, H = pil.size
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return pil.crop((x1, y1, x2, y2))


def resize_bound(pil: Image.Image, max_w: int, max_h: int) -> Image.Image:
    W, H = pil.size
    scale = min(max_w / W, max_h / H, 1.0)
    if scale < 1.0:
        pil = pil.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
    return pil


def cvt_gray(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)


def crop_xyxy(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    H, W = img.shape[:2]
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=img.dtype)

    return img[y1:y2, x1:x2]
