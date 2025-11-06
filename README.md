# Agnel OCR — Vision Service

This service uses **Google Cloud Vision API** (with an optional free Tesseract fallback) to extract structured data from scanned PTA/feedback forms, including **ratings**, **text fields**, and **signature date**.

---

## Requirements

- **Python 3.10+**
- A **Google Cloud** project with **Vision API** enabled
- A **Service Account** key (JSON) with the role `Cloud Vision API User`
- (Optional) **Tesseract OCR** installed on the host for free/local fallback

> Windows (choco): `choco install tesseract`
>
> macOS (brew): `brew install tesseract`
>
> Linux (apt): `sudo apt-get install tesseract-ocr`

---

## Setup

```bash
# 1) Clone / open the project folder
cd ocr-vision-service

# 2) Create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# 3) Upgrade pip and install deps
pip install --upgrade pip
pip install -r requirements.txt
```

# Windows cmd

set GOOGLE_APPLICATION_CREDENTIALS=C:\agnel-guardian-portal\cloudvision-ocr-service\cloud-vision-key\agnel-guardian-portal-4e0f0f5eeb52.json

# Run server

uvicorn app:app --reload --port 5001
