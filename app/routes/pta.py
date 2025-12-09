# app/routes/pta_free.py
from fastapi import APIRouter

from app.models.schemas import OCRRequest
from app.ocr.pta_free_logic import run_pta_free

router = APIRouter(prefix="/ocr", tags=["pta_free"])


@router.post("/pta_free")
def ocr_pta_free(req: OCRRequest):
    """
    PTA-free endpoint:
    - 1 Google Vision DOCUMENT_TEXT_DETECTION call per page
    - Handwritten tick detection using OpenCV grid logic
    """
    return run_pta_free(req)
