from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.models.ocr import OCRReq
from app.ocr.pta_free_logic import run_pta_free_ocr

router = APIRouter()


@router.post("/ocr/pta_free")
def ocr_pta_free(req: OCRReq):
    """
    Thin wrapper around the PTA-free OCR engine.
    All heavy logic is in app.ocr.pta_free_logic.run_pta_free_ocr.
    """
    try:
        result = run_pta_free_ocr(req.imageBase64, debug=bool(req.debug))
        return JSONResponse(content=result)
    except HTTPException as e:
        # Let FastAPI propagate structured errors
        raise e
    except Exception as e:
        # Catch any unexpected error to avoid 500 with no message
        raise HTTPException(status_code=500, detail=f"Internal error in PTA-free OCR: {e}")
