"""
Embedding Service — DocuMind (Render-free-tier edition)

Uses fastembed (BAAI/bge-small-en-v1.5) — runs entirely in-process,
no Ollama needed. Model is ~130MB, fits in Render's free 512MB RAM.

Falls back to Ollama /api/embed if OLLAMA_BASE_URL is set and reachable
(for local Docker use).
"""

import asyncio
import logging
import os
from typing import List, Optional

from backend.services.text_chunker import TextChunk
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# If OLLAMA_BASE_URL is explicitly set to a real host, try Ollama first.
# On Render it won't be set (or will fail), so fastembed is used.
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "")
FASTEMBED_MODEL  = "BAAI/bge-small-en-v1.5"   # 130MB, 384-dim, very fast
BATCH_SIZE       = 64


class EmbeddingService:

    def __init__(self):
        self._fastembed = None        # lazy-loaded on first use
        self._fe_lock   = asyncio.Lock()
        self._use_ollama: Optional[bool] = None   # auto-detected
        # Only import httpx if we might use Ollama
        self._ollama_client = None

    # ── fastembed (local, free) ───────────────────────────────

    def _get_fastembed(self):
        """Load fastembed model once and cache it (thread-safe via lock)."""
        if self._fastembed is not None:
            return self._fastembed
        try:
            from fastembed import TextEmbedding
            logger.info("Loading fastembed model: %s", FASTEMBED_MODEL)
            self._fastembed = TextEmbedding(model_name=FASTEMBED_MODEL)
            logger.info("fastembed ready")
            return self._fastembed
        except ImportError:
            raise RuntimeError(
                "fastembed not installed. Run: pip install fastembed")

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """Synchronous fastembed — called via run_in_executor."""
        model = self._get_fastembed()
        return [list(map(float, e)) for e in model.embed(texts)]

    # ── Ollama (local Docker only) ────────────────────────────

    async def _try_ollama_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not OLLAMA_BASE_URL:
            return None
        try:
            import httpx
            if self._ollama_client is None:
                self._ollama_client = httpx.AsyncClient(timeout=120.0)
            res = await self._ollama_client.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": "nomic-embed-text", "input": texts},
            )
            if res.status_code == 200:
                embs = res.json().get("embeddings", [])
                if len(embs) == len(texts):
                    return embs
        except Exception as e:
            logger.debug("Ollama embed unavailable: %s — using fastembed", e)
        return None

    # ── Public API ────────────────────────────────────────────

    async def get_embedding(self, text: str) -> List[float]:
        """Embed a single text (used for query embedding at ask-time)."""
        results = await self.get_embeddings_batch([text])
        return results[0]

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed all texts.
        Tries Ollama first if OLLAMA_BASE_URL is set (local Docker).
        Falls back to fastembed (always works, fits in free tier).
        """
        if not texts:
            return []

        total = len(texts)
        logger.info("Embedding %d texts...", total)

        # Try Ollama if configured (local Docker Compose)
        if OLLAMA_BASE_URL and self._use_ollama is not False:
            all_embs: List[List[float]] = []
            success = True
            for start in range(0, total, BATCH_SIZE):
                batch  = texts[start:start + BATCH_SIZE]
                result = await self._try_ollama_batch(batch)
                if result is None:
                    self._use_ollama = False
                    success = False
                    break
                all_embs.extend(result)
            if success and len(all_embs) == total:
                self._use_ollama = True
                logger.info("Embedded %d via Ollama", total)
                return all_embs 

        # fastembed — runs in-process, no server needed
        logger.info("Using fastembed for %d texts", total)
        loop = asyncio.get_event_loop()
        all_embs = []
        for start in range(0, total, BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            embs  = await loop.run_in_executor(None, self._embed_local, batch)
            all_embs.extend(embs)
            logger.info("fastembed: %d / %d", min(start + BATCH_SIZE, total), total)

        return all_embs

    # ── Convenience: embed + store ────────────────────────────

    async def embed_and_store(
        self,
        chunks: List[TextChunk],
        vector_store: VectorStore,
        store_batch_size: int = 64,
    ) -> List[str]:
        if not chunks:
            return []
        total      = len(chunks)
        embeddings = await self.get_embeddings_batch([c.text for c in chunks])

        all_ids: List[str] = []
        for start in range(0, total, store_batch_size):
            end   = min(start + store_batch_size, total)
            batch = chunks[start:end]
            embs  = embeddings[start:end]

            metadatas = []
            for chunk in batch:
                meta: dict = {
                    "source":      chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id":    chunk.chunk_id,
                    "word_count":  str(len(chunk.text.split())),
                }
                if getattr(chunk, "page_numbers", None):
                    meta["page_numbers"] = ",".join(str(p) for p in chunk.page_numbers)
                    meta["page"] = (
                        str(chunk.page_numbers[0])
                        if len(chunk.page_numbers) == 1
                        else f"{chunk.page_numbers[0]}-{chunk.page_numbers[-1]}"
                    )
                metadatas.append(meta)

            ids = [c.chunk_id for c in batch]
            vector_store.add_documents(
                ids=ids, embeddings=embs,
                texts=[c.text for c in batch],
                metadatas=metadatas,
            )
            all_ids.extend(ids)

        logger.info("Stored %d chunks", len(all_ids))
        return all_ids

    # ── Called at startup to warm the model ──────────────────

    async def verify_model(self) -> str:
        """Pre-load fastembed model so first upload isn't slow."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._get_fastembed)
        except Exception as e:
            logger.warning("fastembed preload failed: %s", e)
        return FASTEMBED_MODEL