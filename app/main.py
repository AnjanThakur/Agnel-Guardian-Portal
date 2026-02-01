from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import mimetypes
import os

# Create a custom StaticFiles to force correct Content-Type for JS modules on Windows
class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith(".js") or path.endswith(".mjs"):
            response.headers["Content-Type"] = "application/javascript"
        return response

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

# Use custom SPAStaticFiles for assets
app.mount("/assets", SPAStaticFiles(directory="app/static/assets"), name="assets")
app.mount("/static", SPAStaticFiles(directory="app/static"), name="static")

from fastapi.responses import FileResponse

@app.get("/")
def root_index():
    # Serve the React App Entry Point
    return FileResponse("app/static/index.html")

# PTA free endpoint (single Vision call + OpenCV table)
app.include_router(pta_free_router)
