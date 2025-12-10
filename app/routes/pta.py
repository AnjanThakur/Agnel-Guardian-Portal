# app/routes/pta.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.ocr import OCRReq

router = APIRouter()


@router.post("/ocr/pta")
def ocr_pta(req: OCRReq):
    # TODO: implement YAML-template-based PTA OCR
    return JSONResponse(
        status_code=501,
        content={"error": "Template-based PTA OCR (/ocr/pta) not implemented yet."},
    )
