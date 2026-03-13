"""
RAG Pipeline — DocuMind (Render-free-tier edition)

LLM: Groq API (free tier — llama-3.1-8b-instant, very fast)
     Falls back to Ollama if OLLAMA_BASE_URL is set (local Docker).
Embeddings: handled by EmbeddingService (fastembed / Ollama).
"""

import asyncio
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import httpx

from backend.services.embedding    import EmbeddingService
from backend.services.reranker     import BM25Reranker
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── LLM config ────────────────────────────────────────────────
# Groq (free, fast — set GROQ_API_KEY on Render)
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"

# Ollama (local Docker fallback)
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
PREFERRED_MODELS = ["llama3.2:3b", "llama3.2", "llama3:latest", "llama3", "mistral", "phi3"]

SYSTEM_PROMPT = """\
You are DocuMind. Answer ONLY using the context below.
Be direct and complete. List ALL relevant items. Never refuse if the info exists in context.\
"""

RAG_PROMPT = """\
CONTEXT:
{context}

QUESTION: {question}

Answer (use only the context above):\
"""


class RAGPipeline:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store:      VectorStore,
    ):
        self.embedding_service = embedding_service
        self.vector_store      = vector_store
        self.reranker          = BM25Reranker(alpha=0.5)
        self._resolved_model:  Optional[str] = None
        self._resolve_lock     = asyncio.Lock()
        self._client           = httpx.AsyncClient(timeout=120.0)

    # ── Model resolution ──────────────────────────────────────

    async def _resolve_model(self) -> str:
        """Returns ('groq', model) or ('ollama', model)."""
        if self._resolved_model:
            return self._resolved_model

        async with self._resolve_lock:
            if self._resolved_model:
                return self._resolved_model

            # Prefer Groq if API key is set
            if GROQ_API_KEY:
                self._resolved_model = f"groq:{GROQ_MODEL}"
                logger.info("LLM backend: Groq (%s)", GROQ_MODEL)
                return self._resolved_model

            # Fallback: Ollama (local Docker)
            if OLLAMA_BASE_URL:
                try:
                    res = await self._client.get(f"{OLLAMA_BASE_URL}/api/tags")
                    if res.status_code == 200:
                        available = [m["name"] for m in res.json().get("models", [])]
                        for pref in [OLLAMA_LLM_MODEL] + PREFERRED_MODELS:
                            base  = pref.split(":")[0]
                            match = next((m for m in available if base in m), None)
                            if match:
                                self._resolved_model = f"ollama:{match}"
                                logger.info("LLM backend: Ollama (%s)", match)
                                return self._resolved_model
                except Exception as e:
                    logger.warning("Ollama check failed: %s", e)

            raise RuntimeError(
                "No LLM configured. Set GROQ_API_KEY in Space → Settings → Repository Secrets, "
                "or run Ollama locally for Docker Compose."
            )

    # ── Entry point ───────────────────────────────────────────

    async def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        doc_count = self.vector_store.get_document_count()
        if doc_count == 0:
            return {"answer": "No documents uploaded yet.", "sources": [], "question": question}

        original_emb, _ = await asyncio.gather(
            self.embedding_service.get_embedding(question),
            self._resolve_model(),
        )

        expanded    = self._rule_expand(question)
        all_queries = [question] + expanded

        retrieve_k = min(max(top_k * 4, 20), doc_count)
        candidates = await self._multi_retrieve(all_queries, original_emb, retrieve_k)
        if not candidates:
            return {"answer": "I couldn't find relevant information.",
                    "sources": [], "question": question}

        reranked = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=min(top_k * 2, len(candidates)),
        )
        final = self._diversify(reranked, top_k)

        context     = self._build_context(final)
        answer_text = await self._generate(question, context)
        sources     = self._align_evidence(final, answer_text, question)

        return {"answer": answer_text, "sources": sources, "question": question}

    # ── Rule expansion ────────────────────────────────────────

    def _rule_expand(self, question: str) -> List[str]:
        q   = question.strip().rstrip("?")
        kw  = re.sub(
            r'\b(how many|what|which|where|when|who|is|are|the|a|an|in|of|'
            r'from|to|does|did|was|were|do|this|that)\b', '', q, flags=re.I)
        kw  = ' '.join(kw.split())
        out: List[str] = []
        if kw and kw.lower() != q.lower():
            out.append(kw)
        if re.search(r'\b(how many|count|number|list|version|release)\b', q, re.I):
            out.append((kw or q) + " list all")
        return list(dict.fromkeys(out))[:2]

    # ── Multi retrieval ───────────────────────────────────────

    async def _multi_retrieve(
        self, queries: List[str], original_emb: List[float], retrieve_k: int
    ) -> List[Dict]:

        async def _search(emb: List[float]) -> List[Dict]:
            try:
                return self.vector_store.search(query_embedding=emb, top_k=retrieve_k)
            except Exception as e:
                logger.warning("Search failed: %s", e); return []

        async def _embed_search(q: str) -> List[Dict]:
            try:
                emb = await self.embedding_service.get_embedding(q)
                return await _search(emb)
            except Exception as e:
                logger.warning("Embed+search '%s': %s", q[:40], e); return []

        tasks       = [_search(original_emb)] + [_embed_search(q) for q in queries[1:]]
        all_results = await asyncio.gather(*tasks)

        merged: Dict[str, Dict] = {}
        hits:   Dict[str, int]  = {}
        for results in all_results:
            for chunk in results:
                cid       = chunk["id"]
                hits[cid] = hits.get(cid, 0) + 1
                if cid not in merged or chunk["score"] > merged[cid]["score"]:
                    merged[cid] = chunk
        for cid in merged:
            if hits[cid] > 1:
                merged[cid]["score"] = min(merged[cid]["score"] + (hits[cid]-1)*0.05, 1.0)
        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    # ── Diversify ─────────────────────────────────────────────

    def _diversify(self, chunks: List[Dict], top_k: int) -> List[Dict]:
        if not chunks: return []
        by_source: dict = defaultdict(list)
        for c in chunks:
            by_source[c.get("metadata", {}).get("source", "unknown")].append(c)
        sources  = sorted(by_source, key=lambda s: by_source[s][0]["score"], reverse=True)
        selected: List[Dict] = []
        for src in sources:
            if len(selected) >= top_k: break
            selected.append(by_source[src][0])
        if len(selected) < top_k:
            used = {c["id"] for c in selected}
            for c in chunks:
                if c["id"] not in used:
                    selected.append(c)
                    if len(selected) >= top_k: break
        return selected

    # ── Context builder ───────────────────────────────────────

    def _build_context(self, chunks: List[Dict]) -> str:
        by_source: dict    = defaultdict(list)
        order:     List[str] = []
        for c in chunks:
            src  = c.get("metadata", {}).get("source", "Unknown")
            page = c.get("metadata", {}).get("page", "")
            if src not in by_source:
                order.append(src)
            prefix = f"[p.{page}] " if page else ""
            by_source[src].append(prefix + c["text"])
        return "\n\n".join(
            f"[{src}]\n" + "\n\n".join(by_source[src]) for src in order
        )

    # ── Generation ────────────────────────────────────────────

    async def _generate(self, question: str, context: str) -> str:
        backend = await self._resolve_model()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": RAG_PROMPT.format(
                context=context, question=question)},
        ]

        if backend.startswith("groq:"):
            return await self._generate_groq(messages)
        else:
            model = backend.split(":", 1)[1]
            return await self._generate_ollama(messages, model)

    async def _generate_groq(self, messages: List[Dict]) -> str:
        payload = {
            "model":       GROQ_MODEL,
            "messages":    messages,
            "max_tokens":  1024,
            "temperature": 0.1,
        }
        try:
            res = await self._client.post(
                GROQ_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type":  "application/json",
                },
            )
            res.raise_for_status()
            answer = res.json()["choices"][0]["message"]["content"].strip()
            if not answer:
                raise ValueError("Empty response from Groq")
            logger.info("Groq generated %d chars", len(answer))
            return answer
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise RuntimeError("Invalid GROQ_API_KEY. Check your Render environment variables.")
            if e.response.status_code == 429:
                raise RuntimeError("Groq rate limit hit. Wait a moment and try again.")
            raise RuntimeError(f"Groq API error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Groq generation failed: {e}")

    async def _generate_ollama(self, messages: List[Dict], model: str) -> str:
        payload = {
            "model":   model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature":    0.1,
                "num_predict":    512,
                "num_ctx":        4096,
                "repeat_penalty": 1.1,
            },
        }
        try:
            res = await self._client.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            res.raise_for_status()
            answer = res.json().get("message", {}).get("content", "").strip()
            if not answer:
                raise ValueError("Empty response from Ollama")
            logger.info("Ollama generated %d chars", len(answer))
            return answer
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    # ── Evidence alignment ────────────────────────────────────

    def _align_evidence(self, chunks: List[Dict], answer: str, question: str) -> List[Dict]:
        STOP = {
            'a','an','the','and','or','but','in','on','at','to','for','of','with',
            'by','from','is','are','was','were','be','been','it','its','that','this',
            'as','not','what','which','who','how','i','you','we','they','do','does',
            'did','will','would','can','could','should','have','has','had',
        }
        def tok(t: str) -> set:
            return {w for w in re.findall(r'\b[a-z0-9][a-z0-9]*\b', t.lower())
                    if w not in STOP and len(w) > 1}

        answer_tokens = tok(answer)
        if not answer_tokens:
            return self._fmt(chunks)

        scored = []
        for c in chunks:
            ct = tok(c["text"])
            if not ct: continue
            overlap = len(ct & answer_tokens) / len(ct | answer_tokens)
            nc = dict(c)
            nc["score"] = round(overlap, 4)
            scored.append(nc)

        scored.sort(key=lambda x: x["score"], reverse=True)
        relevant = [c for c in scored if c["score"] >= 0.04]
        return self._fmt(relevant if relevant else scored[:1])

    def _fmt(self, chunks: List[Dict]) -> List[Dict]:
        seen: set = set()
        out = []
        for c in chunks:
            cid  = c.get("id", c.get("metadata", {}).get("chunk_id", ""))
            if cid in seen: continue
            seen.add(cid)
            meta = c.get("metadata", {})
            page = meta.get("page") or meta.get("page_numbers", "") or None
            out.append({
                "text":     c["text"],
                "source":   meta.get("source", "Unknown"),
                "chunk_id": meta.get("chunk_id", cid),
                "score":    round(c.get("score", 0.0), 4),
                "page":     page,
            })
        return out