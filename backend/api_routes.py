"""
API Routes — DocuMind
"""

import logging
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.services.pdf_parser   import PDFParser
from backend.services.ocr_parser   import OCRParser
from backend.services.text_chunker import TextChunker
from backend.services.embedding   import EmbeddingService
from backend.services.vector_store import VectorStore
from backend.services.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

_BASE = Path(__file__).parent
UPLOAD_DIR = _BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Services — chunk_size=400 for faster embedding, overlap=40
pdf_parser       = PDFParser()
ocr_parser       = OCRParser()
chunker          = TextChunker(chunk_size=600, chunk_overlap=60)
embedding_service = EmbeddingService()
vector_store     = VectorStore(persist_directory=str(_BASE / "database" / "vector_db"))
rag_pipeline     = RAGPipeline(embedding_service=embedding_service, vector_store=vector_store)


async def prewarm_models():
    """Pre-load embedding model at startup so first upload is instant."""
    try:
        await embedding_service.verify_model()
        logger.info("Embedding model pre-warmed")
    except Exception as e:
        logger.warning("Pre-warm failed (non-fatal): %s", e)


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5


class DocumentChunk(BaseModel):
    text: str
    source: str
    chunk_id: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: List[DocumentChunk]
    question: str


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed_types = {
        "application/pdf": "pdf",
        "image/jpeg": "image", "image/jpg": "image", "image/png": "image",
        "image/tiff": "image", "image/bmp": "image",
        "image/gif": "image",  "image/webp": "image",
    }
    content_type = file.content_type or ""
    file_ext     = Path(file.filename or "").suffix.lower()

    if content_type in allowed_types:
        file_type = allowed_types[content_type]
    elif file_ext == ".pdf":
        file_type = "pdf"
    elif file_ext in {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".webp"}:
        file_type = "image"
    else:
        raise HTTPException(400,
            f"Unsupported file type: {file.filename}. Supported: PDF, JPEG, PNG, TIFF, BMP, GIF, WEBP")

    # Save file
    file_id      = str(uuid.uuid4())
    safe_name    = f"{file_id}_{Path(file.filename or 'upload').name}"
    file_path    = UPLOAD_DIR / safe_name
    try:
        content = await file.read()
        file_path.write_bytes(content)
        logger.info("Saved %s (%d bytes)", safe_name, len(content))
    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")

    # Extract text
    try:
        if file_type == "pdf":
            pages    = pdf_parser.extract_text(str(file_path))
            raw_text = "\n\n".join(p["text"] for p in pages if p["text"].strip())
        else:
            pages    = None
            raw_text = ocr_parser.extract_text(str(file_path))

        if not raw_text.strip():
            raise HTTPException(422, "No text could be extracted from the document.")
        logger.info("Extracted %d chars from %s", len(raw_text), file.filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Text extraction failed: {e}")

    # Chunk
    try:
        chunks = chunker.chunk_text(
            raw_text, source=file.filename or safe_name,
            pages=pages if file_type == "pdf" else None)
        logger.info("Created %d chunks from %s", len(chunks), file.filename)
    except Exception as e:
        raise HTTPException(500, f"Chunking failed: {e}")

    # Embed + store
    try:
        doc_ids = await embedding_service.embed_and_store(chunks, vector_store)
        logger.info("Stored %d embeddings for %s", len(doc_ids), file.filename)
    except Exception as e:
        raise HTTPException(500, f"Embedding/storage failed: {e}")

    return JSONResponse(content={
        "message":             "Document processed successfully",
        "filename":            file.filename,
        "file_id":             file_id,
        "chunks_created":      len(chunks),
        "characters_extracted": len(raw_text),
        "file_type":           file_type,
    })


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    try:
        result = await rag_pipeline.answer(question=request.question, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error("RAG error: %s", str(e), exc_info=True)
        raise HTTPException(500, f"Failed to generate answer: {str(e)}")


@router.get("/documents")
async def list_documents():
    try:
        docs         = vector_store.list_sources()
        total_chunks = vector_store.get_document_count()
        return {"documents": docs, "count": len(docs), "total_chunks": total_chunks}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/documents")
async def clear_documents():
    try:
        vector_store.clear()
        deleted = sum(1 for f in UPLOAD_DIR.iterdir()
                      if f.is_file() and not f.unlink())
        return {"message": "All documents cleared", "files_deleted": deleted}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/documents/{source:path}")
async def delete_source(source: str):
    try:
        deleted = vector_store.delete_source(source)
        for f in UPLOAD_DIR.iterdir():
            if f.is_file() and f.name.endswith(source):
                try: f.unlink()
                except: pass
        return {"message": f"Deleted {deleted} chunks for '{source}'", "chunks_deleted": deleted}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/reset")
async def reset_session():
    try:
        deleted = 0
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                try: f.unlink(); deleted += 1
                except: pass
        vector_store.clear()
        logger.info("Session reset: cleared %d file(s)", deleted)
        return {"message": "Session reset", "files_deleted": deleted}
    except Exception as e:
        logger.warning("Reset error (non-fatal): %s", e)
        return {"message": "Reset attempted", "error": str(e)}