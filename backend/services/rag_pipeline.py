"""
RAG Pipeline — DocuMind
embed → retrieve → rerank → generate → evidence alignment
"""

import asyncio
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import httpx

from backend.services.embeddings import EmbeddingService
from backend.services.reranker   import BM25Reranker
from backend.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL        = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
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
        ollama_url:        str = OLLAMA_BASE_URL,
        llm_model:         str = LLM_MODEL,
    ):
        self.embedding_service = embedding_service
        self.vector_store      = vector_store
        self.ollama_url        = ollama_url
        self.llm_model         = llm_model
        self.reranker          = BM25Reranker(alpha=0.5)
        self._resolved_model:  Optional[str]              = None
        self._resolve_lock     = asyncio.Lock()
        self._client           = httpx.AsyncClient(timeout=600.0)

    # ── Model resolution (once, cached) ──────────────────────

    async def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        async with self._resolve_lock:
            if self._resolved_model:
                return self._resolved_model
            try:
                res = await self._client.get(f"{self.ollama_url}/api/tags")
                if res.status_code == 200:
                    available = [m["name"] for m in res.json().get("models", [])]
                    logger.info("Available LLMs: %s", available)
                    for pref in [self.llm_model] + PREFERRED_MODELS:
                        base  = pref.split(":")[0]
                        match = next((m for m in available if base in m), None)
                        if match:
                            logger.info("Using LLM: %s", match)
                            self._resolved_model = match
                            return match
                    if available:
                        self._resolved_model = available[0]
                        return available[0]
            except Exception as e:
                logger.warning("Model resolve failed: %s", e)
            self._resolved_model = self.llm_model
            return self.llm_model

    # ── Entry point ───────────────────────────────────────────

    async def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        doc_count = self.vector_store.get_document_count()
        if doc_count == 0:
            return {"answer": "No documents uploaded yet.", "sources": [], "question": question}

        # Embed question + resolve model concurrently
        original_emb, _ = await asyncio.gather(
            self.embedding_service.get_embedding(question),
            self._resolve_model(),
        )

        # Rule-based expansion (no extra LLM call)
        expanded    = self._rule_expand(question)
        all_queries = [question] + expanded
        logger.info("Queries: %s", all_queries)

        # Retrieve
        retrieve_k = min(max(top_k * 4, 20), doc_count)
        candidates = await self._multi_retrieve(all_queries, original_emb, retrieve_k)
        if not candidates:
            return {"answer": "I couldn't find relevant information.",
                    "sources": [], "question": question}

        # Rerank + diversify
        reranked = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=min(top_k * 2, len(candidates)),
        )
        final = self._diversify(reranked, top_k)
        logger.info("Final: %d chunks from %d source(s)", len(final),
                    len({c["metadata"].get("source") for c in final}))

        # Build context + generate
        context     = self._build_context(final)
        answer_text = await self._generate(question, context)

        # Align evidence to answer
        sources = self._align_evidence(final, answer_text, question)
        return {"answer": answer_text, "sources": sources, "question": question}

    # ── Rule expansion ────────────────────────────────────────

    def _rule_expand(self, question: str) -> List[str]:
        q       = question.strip().rstrip("?")
        kw      = re.sub(
            r'\b(how many|what|which|where|when|who|is|are|the|a|an|in|of|'
            r'from|to|does|did|was|were|do|this|that)\b', '', q, flags=re.I)
        kw      = ' '.join(kw.split())
        variants: List[str] = []
        if kw and kw.lower() != q.lower():
            variants.append(kw)
        if re.search(r'\b(how many|count|number|list|version|release)\b', q, re.I):
            variants.append((kw or q) + " list all")
        return list(dict.fromkeys(variants))[:2]

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

        tasks      = [_search(original_emb)] + [_embed_search(q) for q in queries[1:]]
        all_results = await asyncio.gather(*tasks)

        merged: Dict[str, Dict] = {}
        hits:   Dict[str, int]  = {}
        for results in all_results:
            for chunk in results:
                cid        = chunk["id"]
                hits[cid]  = hits.get(cid, 0) + 1
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
        """Group by source. Include [p.N] page tag inline so LLM sees location."""
        by_source: dict    = defaultdict(list)
        order:     List[str] = []
        for c in chunks:
            src  = c.get("metadata", {}).get("source", "Unknown")
            page = c.get("metadata", {}).get("page", "")
            if src not in by_source:
                order.append(src)
            prefix = f"[p.{page}] " if page else ""
            by_source[src].append(prefix + c["text"])

        parts = []
        for src in order:
            parts.append(f"[{src}]\n" + "\n\n".join(by_source[src]))
        return "\n\n".join(parts)

    # ── Generation ────────────────────────────────────────────

    async def _generate(self, question: str, context: str) -> str:
        model   = await self._resolve_model()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": RAG_PROMPT.format(
                    context=context, question=question)},
            ],
            "stream": False,
            "options": {
                "temperature":    0.1,
                "num_predict":    512,
                "num_ctx":        4096,
                "repeat_penalty": 1.1,
                "top_k":          20,
                "top_p":          0.9,
            },
        }
        try:
            res    = await self._client.post(f"{self.ollama_url}/api/chat", json=payload)
            res.raise_for_status()
            answer = res.json().get("message", {}).get("content", "").strip()
            if not answer:
                raise ValueError("Empty response from LLM")
            logger.info("Generated %d chars", len(answer))
            return answer
        except httpx.TimeoutException:
            raise RuntimeError("Ollama timed out. Try llama3.2:3b for faster CPU inference.")
        except httpx.ConnectError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.ollama_url}.")
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    # ── Evidence alignment ────────────────────────────────────

    def _align_evidence(self, chunks: List[Dict], answer: str, question: str) -> List[Dict]:
        """
        Score chunks by overlap with the ANSWER only (not retrieval score).
        This ensures evidence cards show text that actually appears in the answer,
        not just text that was semantically close to the question.
        """
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
            if not ct:
                continue
            # Jaccard overlap with answer tokens only
            overlap = len(ct & answer_tokens) / len(ct | answer_tokens)
            nc = dict(c)
            nc["score"] = round(overlap, 4)
            scored.append(nc)

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Keep only chunks with meaningful answer overlap
        relevant = [c for c in scored if c["score"] >= 0.04]
        return self._fmt(relevant if relevant else scored[:1])

    def _fmt(self, chunks: List[Dict]) -> List[Dict]:
        seen: set = set()
        out       = []
        for c in chunks:
            cid  = c.get("id", c.get("metadata", {}).get("chunk_id", ""))
            if cid in seen: continue
            seen.add(cid)
            meta = c.get("metadata", {})
            # page field — set by text_chunker per-page logic, stored in ChromaDB
            page = meta.get("page") or meta.get("page_numbers", "") or None
            out.append({
                "text":     c["text"],
                "source":   meta.get("source", "Unknown"),
                "chunk_id": meta.get("chunk_id", cid),
                "score":    round(c.get("score", 0.0), 4),
                "page":     page,
            })
        return out