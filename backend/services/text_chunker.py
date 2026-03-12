"""
Text Chunker — DocuMind

For PDFs: chunks page-by-page so every chunk has exact page numbers.
Page numbers are assigned at creation time — no text searching, always correct.

Fixes vs previous version:
- carry_words properly reset on force-split (was leaking wrong page into next chunk)  
- Larger chunk_size=600 for large PDFs: fewer chunks = faster embedding + better context
- Minimum word filter raised to 15 to cut more noise from large docs
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TextChunk:
    def __init__(self, text: str, source: str, chunk_index: int,
                 chunk_id: str = None, page_numbers: List[int] = None):
        self.text         = text
        self.source       = source
        self.chunk_index  = chunk_index
        self.chunk_id     = chunk_id or str(uuid.uuid4())
        self.page_numbers = page_numbers or []

    def to_dict(self) -> Dict[str, Any]:
        d = {"text": self.text, "source": self.source,
             "chunk_index": self.chunk_index, "chunk_id": self.chunk_id}
        if self.page_numbers:
            d["page_numbers"] = self.page_numbers
            d["page"] = (str(self.page_numbers[0]) if len(self.page_numbers) == 1
                         else f"{self.page_numbers[0]}-{self.page_numbers[-1]}")
        return d

    def __repr__(self):
        return (f"TextChunk(source={self.source!r}, index={self.chunk_index}, "
                f"pages={self.page_numbers}, words={len(self.text.split())})")


class TextChunker:
    """
    PDF: per-page chunking → exact page numbers, no searching.
    Text/Image: flat chunking.
    chunk_size=600 balances context quality vs embedding cost.
    """

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 60):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info("TextChunker: size=%d overlap=%d", chunk_size, chunk_overlap)

    def chunk_text(
        self,
        text: str,
        source: str,
        pages: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TextChunk]:
        if not text or not text.strip():
            return []
        return self._chunk_pages(pages, source) if pages else self._chunk_flat(text, source)

    # ── PDF: per-page chunking ────────────────────────────────

    def _chunk_pages(self, pages: List[Dict[str, Any]], source: str) -> List[TextChunk]:
        chunks:      List[TextChunk]  = []
        chunk_index: int              = 0
        carry_words: List[str]        = []
        carry_page:  Optional[int]    = None

        for page_data in pages:
            page_num  = page_data["page_number"]
            page_text = page_data.get("text", "").strip()
            if not page_text:
                continue

            sentences = self._split_sentences(page_text)
            # Start with overlap words carried from previous page
            words: List[str] = carry_words[:]

            for sentence in sentences:
                s_words = sentence.split()
                if not s_words:
                    continue

                if words and len(words) + len(s_words) > self.chunk_size:
                    # Include carry_page in tags if the carry words came from a different page
                    page_tags = (sorted({carry_page, page_num})
                                 if carry_page and carry_page != page_num and carry_words
                                 else [page_num])
                    c = self._make(words, source, chunk_index, page_tags)
                    if c:
                        chunks.append(c)
                        chunk_index += 1
                    # Overlap stays on current page; clear carry tracking
                    carry_words = []
                    carry_page  = None
                    words = words[-self.chunk_overlap:] + s_words
                else:
                    words.extend(s_words)

                # Force-split very long accumulations
                while len(words) > self.chunk_size * 2:
                    c = self._make(words[:self.chunk_size], source, chunk_index, [page_num])
                    if c:
                        chunks.append(c)
                        chunk_index += 1
                    words = words[self.chunk_size - self.chunk_overlap:]
                    # After force-split the remaining words are all on current page
                    carry_words = []
                    carry_page  = None

            # Flush remaining words for this page
            if words:
                page_tags = (sorted({carry_page, page_num})
                             if carry_page and carry_page != page_num and carry_words
                             else [page_num])
                c = self._make(words, source, chunk_index, page_tags)
                if c:
                    chunks.append(c)
                    chunk_index += 1
                carry_words = words[-self.chunk_overlap:]
                carry_page  = page_num
            else:
                carry_words = []
                carry_page  = None

        chunks = [c for c in chunks if len(c.text.split()) >= 15]
        logger.info("Chunked '%s': %d chunks (avg %.0f words)", source, len(chunks),
                    sum(len(c.text.split()) for c in chunks) / max(len(chunks), 1))
        return chunks

    # ── Flat chunking (images / plain text) ──────────────────

    def _chunk_flat(self, text: str, source: str) -> List[TextChunk]:
        sentences   = self._split_sentences(text)
        chunks:     List[TextChunk] = []
        words:      List[str]       = []
        chunk_index = 0

        for sentence in sentences:
            s_words = sentence.split()
            if not s_words:
                continue
            if words and len(words) + len(s_words) > self.chunk_size:
                c = self._make(words, source, chunk_index, [])
                if c:
                    chunks.append(c)
                    chunk_index += 1
                words = words[-self.chunk_overlap:] + s_words
            else:
                words.extend(s_words)
            while len(words) > self.chunk_size * 2:
                c = self._make(words[:self.chunk_size], source, chunk_index, [])
                if c:
                    chunks.append(c)
                    chunk_index += 1
                words = words[self.chunk_size - self.chunk_overlap:]

        if words:
            c = self._make(words, source, chunk_index, [])
            if c:
                chunks.append(c)

        chunks = [c for c in chunks if len(c.text.split()) >= 15]
        logger.info("Chunked '%s': %d chunks (avg %.0f words)", source, len(chunks),
                    sum(len(c.text.split()) for c in chunks) / max(len(chunks), 1))
        return chunks

    # ── Helpers ───────────────────────────────────────────────

    def _make(self, words: List[str], source: str, index: int,
              pages: List[int]) -> Optional[TextChunk]:
        body = " ".join(words).strip()
        return TextChunk(text=body, source=source, chunk_index=index,
                         page_numbers=pages) if body else None

    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\s+", " ", text).strip()
        raw  = re.split(
            r"(?<=[.!?])\s+(?=[A-Z\"\'`])|(?<=\n)\s*(?=[A-Z])|\n{2,}", text)
        return [s.strip() for s in raw if len(s.strip()) > 20]