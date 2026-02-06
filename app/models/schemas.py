# app/models/schemas.py
from typing import Optional, Dict, Any
from pydantic import BaseModel


class OCRRequest(BaseModel):
    imageBase64: str
    template: str | None = None
    saveDebug: bool = False
    department: str = "General"
    class_name: str = "Unknown"



class PTAFields(BaseModel):
    """Loose structure; 'fields' itself is a free dict of field -> {value, confidence, source}."""
    fields: Dict[str, Dict[str, Any]]
    text: str
    engine: str = "vision"
    mode: str = "pta-free"
    usage: Dict[str, Any]
    debug_dir: Optional[str] = None
