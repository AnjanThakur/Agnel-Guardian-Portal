# utils/vision_api.py
from typing import Any, Tuple

from google.cloud import vision
from google.api_core.exceptions import ServiceUnavailable

try:
    import pytesseract
    HAVE_TESS = True
except Exception:
    HAVE_TESS = False

import cv2
import numpy as np
from PIL import Image


def get_vision_client():
    return vision.ImageAnnotatorClient()


def vision_document_text(png_bytes: bytes, retries: int = 2, delay: float = 0.6, *, return_usage: bool = False):
    client = get_vision_client()
    image = vision.Image(content=png_bytes)
    last_exc = None
    import time

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
