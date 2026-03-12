"""
PDF Parser Service
Extracts text from PDF files using PyMuPDF (fitz)
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Production-grade PDF text extractor using PyMuPDF.
    Handles multi-page PDFs, extracts metadata, and cleans text.
    """

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from each page of a PDF.

        Returns:
            List of dicts with keys: page_number, text, word_count
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF not installed. Run: pip install PyMuPDF")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        pages = []
        try:
            doc = fitz.open(str(path))
            total_pages = len(doc)  # Save count BEFORE closing
            logger.info("Opening PDF: %s (%d pages)", path.name, total_pages)

            for page_num in range(total_pages):
                page = doc[page_num]

                # Extract text with layout preservation
                text = page.get_text("text")

                # Clean text
                cleaned = self._clean_text(text)

                if cleaned.strip():
                    pages.append({
                        "page_number": page_num + 1,
                        "text": cleaned,
                        "word_count": len(cleaned.split()),
                    })

            doc.close()
            logger.info(
                "Extracted text from %d/%d pages of %s",
                len(pages), total_pages, path.name  # Use saved count
            )
        except Exception as e:
            logger.error("PDF extraction error: %s", str(e))
            raise RuntimeError(f"Failed to parse PDF: {str(e)}")

        return pages

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text:
        - Remove excessive whitespace
        - Fix broken lines
        - Remove page headers/footers patterns
        """
        import re

        if not text:
            return ""

        # Replace form feeds and unusual whitespace
        text = text.replace("\f", "\n\n")
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]

        # Remove lines that are just numbers (page numbers)
        lines = [
            line for line in lines
            if not (line.isdigit() and len(line) <= 4)
        ]

        # Rejoin
        text = "\n".join(lines).strip()

        # Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        return text

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF metadata (title, author, etc.)"""
        try:
            import fitz
            doc = fitz.open(file_path)
            meta = doc.metadata
            doc.close()
            return meta or {}
        except Exception as e:
            logger.warning("Could not extract PDF metadata: %s", str(e))
            return {}