# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.pta_free import router as pta_free_router

app = FastAPI(
    title="Agnel OCR – Vision PTA",
    version="0.1.0",
)

# CORS (relaxed; tighten if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static UI (simple HTML)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def root_index():
    return {"message": "Agnel OCR – PTA Free Endpoint is running", "ui": "/static/index.html"}


# PTA free endpoint (single Vision call + OpenCV table)
app.include_router(pta_free_router)
