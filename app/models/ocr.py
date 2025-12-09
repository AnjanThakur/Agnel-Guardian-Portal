# app/models/ocr.py
from typing import List, Dict, Any
from pydantic import BaseModel


class OCRLine(BaseModel):
    text: str
    box: List[int]  # [x1, y1, x2, y2]


class OCRPage(BaseModel):
    lines: List[OCRLine]
    width: int
    height: int


class OCRResult(BaseModel):
    text: str
    pages: List[OCRPage]
    raw: Dict[str, Any] | None = None
