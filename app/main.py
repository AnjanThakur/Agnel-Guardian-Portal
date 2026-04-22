from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.core.security_headers import SecurityHeadersMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import mimetypes
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Create a custom StaticFiles to force correct Content-Type for JS modules on Windows
class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if path.endswith(".js") or path.endswith(".mjs"):
            response.headers["Content-Type"] = "application/javascript"
        return response

from app.routes.pta_free import router as pta_free_router
from app.routes.analysis_routes import router as analysis_router

app = FastAPI(
    title="Agnel OCR – Vision PTA",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Apply global security headers middleware (Cyber Security Enhancement)
app.add_middleware(SecurityHeadersMiddleware)

# Use custom SPAStaticFiles for assets
app.mount("/assets", SPAStaticFiles(directory="app/static/assets"), name="assets")
app.mount("/static", SPAStaticFiles(directory="app/static"), name="static")

from fastapi.responses import FileResponse

@app.get("/")
def root_index():
    # Serve the React App Entry Point
    return FileResponse("app/static/index.html")

# API Routers
app.include_router(pta_free_router)
app.include_router(analysis_router)

from app.routes.analytics import router as analytics_router
app.include_router(analytics_router)

from app.routes.student_routes import router as student_router
app.include_router(student_router)

from app.routes.auth_routes import router as auth_router
app.include_router(auth_router)

from app.routes.message_routes import router as message_router
app.include_router(message_router)