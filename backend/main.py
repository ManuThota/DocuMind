"""
DocuMind — FastAPI Application Entry Point
"""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api_routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_DIR     = BASE_DIR / "database" / "vector_db"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting DocuMind...")

    # Clear stale uploads
    cleared = sum(1 for f in UPLOAD_DIR.iterdir()
                  if f.is_file() and not f.unlink())
    logger.info("🗑️  Cleared %d stale upload(s)", cleared)

    # Clear vector DB
    if DB_DIR.exists():
        try:
            shutil.rmtree(DB_DIR)
            DB_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("🗑️  Cleared vector DB")
        except Exception as e:
            logger.warning("Could not clear vector DB: %s", e)

    # Pre-warm models — verify + cache model names before first request
    try:
        from backend.api_routes import prewarm_models
        await prewarm_models()
        logger.info("✅ Models ready")
    except Exception as e:
        logger.warning("⚠️  Pre-warm failed (will retry on first request): %s", e)

    yield
    logger.info("👋 Shutting down DocuMind...")


app = FastAPI(
    title="DocuMind",
    description="RAG system for PDF and Image Q&A",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow large uploads (200 MB)
class LargeUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request._body_max_size = 200 * 1024 * 1024
        return await call_next(request)

app.add_middleware(LargeUploadMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Static files + templates
_frontend = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(_frontend / "static")), name="static")
templates = Jinja2Templates(directory=str(_frontend / "templates"))

app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "DocuMind"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000,
                reload=True, timeout_keep_alive=300)