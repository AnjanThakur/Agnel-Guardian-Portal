# models/schemas.py
from pydantic import BaseModel
from typing import Optional


class OCRReq(BaseModel):
    imageBase64: str
    template: Optional[str] = None
    debug: Optional[bool] = False
