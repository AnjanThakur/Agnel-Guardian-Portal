from __future__ import annotations
from typing import List
import fitz  # PyMuPDF
import numpy as np
import cv2

def pdf_bytes_to_bgr_pages(pdf_bytes: bytes, dpi: int = 200) -> List[np.ndarray]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in range(len(doc)):
        pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # pixmap is RGB; convert to BGR for cv2 consistency
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pages.append(img)

    return pages
