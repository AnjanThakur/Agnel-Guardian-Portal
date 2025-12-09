# app/routes/auto.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.ocr import OCRReq

router = APIRouter()


@router.post("/ocr/auto")
def ocr_auto(req: OCRReq):
    # TODO: implement auto-mode OCR
    return JSONResponse(
        status_code=501,
        content={"error": "Automatic OCR mode (/ocr/auto) not implemented yet."},
    )
