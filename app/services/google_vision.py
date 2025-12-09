# app/services/google_vision.py
from google.cloud import vision
from google.oauth2 import service_account
from pathlib import Path
import base64


# Load credentials inside the project (no env var required)
SERVICE_KEY_PATH = Path(__file__).resolve().parent.parent / "service_account.json"

credentials = service_account.Credentials.from_service_account_file(
    str(SERVICE_KEY_PATH)
)

client = vision.ImageAnnotatorClient(credentials=credentials)


def document_text_from_bytes(image_bytes: bytes):
    """
    Runs DOCUMENT_TEXT_DETECTION on raw bytes and returns the response object.
    """
    image = vision.Image(content=image_bytes)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise Exception(response.error.message)

    return response


def document_text_from_b64(image_b64: str):
    """
    Accepts base64 string (with or without prefix), decodes it,
    and performs document text detection.
    """
    b64_data = image_b64.split(",")[-1]
    img_bytes = base64.b64decode(b64_data)
    return document_text_from_bytes(img_bytes)


def vision_response_to_lines(response):
    """
    Convert Vision API response into a list of dicts:
    { "text": "...", "box": [x1,y1,x2,y2] }
    """

    lines = []
    full_text = response.full_text_annotation.text or ""

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:

                # compute bounding box of paragraph
                vertices = para.bounding_box.vertices
                x1 = vertices[0].x or 0
                y1 = vertices[0].y or 0
                x2 = vertices[2].x or 0
                y2 = vertices[2].y or 0

                # extract paragraph text
                text = "".join(
                    symbol.text for word in para.words for symbol in word.symbols
                ).strip()

                if text:
                    lines.append({
                        "text": text,
                        "box": [x1, y1, x2, y2]
                    })

    return full_text, lines

