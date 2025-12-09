import cv2
import numpy as np
from app.utils.helpers import save_debug_image

def preprocess_page(image_bytes: bytes, debug_dir: str | None = None):
    """
    Minimal preprocessing:
    - decode bytes to image
    - convert to grayscale
    - light denoise
    - resize too-large images
    """
    # Decode
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image bytes")

    # Resize if too large
    MAX_SIDE = 2000
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Light blur to remove noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Save debug output
    if debug_dir:
        save_debug_image(gray, f"{debug_dir}/01_preprocessed_gray.png")

    return gray, img  # gray for grid, img for OCR (color)
