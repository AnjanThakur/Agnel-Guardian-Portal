from typing import List
from pdf2image import convert_from_bytes
import numpy as np
import cv2


def pdf_bytes_to_images(pdf_bytes: bytes) -> List[np.ndarray]:
    """
    Converts PDF bytes into a list of OpenCV BGR images (one per page).
    """
    pil_pages = convert_from_bytes(
        pdf_bytes,
        dpi=300,          # important for tick visibility
        fmt="png"
    )

    images = []
    for pil_img in pil_pages:
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        images.append(bgr)

    return images
