# app/services/ocr_service.py
import time
from typing import Any, Tuple, List

from google.cloud import vision
from google.api_core.exceptions import ServiceUnavailable

from PIL import Image

from app.utils.preprocess import preprocess_page
from app.utils.text_tools import line_boxes_from_vision


def get_vision_client():
    return vision.ImageAnnotatorClient()


def vision_document_text(png_bytes: bytes, retries: int = 2, delay: float = 0.6) -> Any:
    """Call Cloud Vision document_text_detection with retry on ServiceUnavailable."""
    client = get_vision_client()
    image = vision.Image(content=png_bytes)
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return client.document_text_detection(image=image)
        except ServiceUnavailable as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay * (2**attempt))
                continue
            break
        except Exception:
            raise
    raise RuntimeError(f"Vision service unavailable after {retries + 1} attempts: {last_exc!r}")


def vision_page_once(pil: Image.Image, debug_dir: str | None) -> Tuple[Any, List[dict]]:
    """Run Vision OCR once and return (response, structured lines)."""
    png = preprocess_page(pil, upscale=1.35, denoise=True, debug_dir=debug_dir)
    resp = vision_document_text(png)
    lines = line_boxes_from_vision(resp)
    if debug_dir:
        with open(f"{debug_dir}/00_raw_text.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ln["text"] for ln in lines))
    return resp, lines
