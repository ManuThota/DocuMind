"""
OCR Parser Service
Extracts text from images using Tesseract OCR via pytesseract
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class OCRParser:
    """
    Production-grade OCR text extractor using Tesseract.
    Supports preprocessing to improve accuracy on noisy images.
    """

    def __init__(self):
        self._verify_tesseract()

    def _verify_tesseract(self):
        """Verify Tesseract is installed."""
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract version: %s", version)
        except Exception as e:
            logger.warning(
                "Tesseract may not be installed. OCR features will be limited. Error: %s", str(e)
            )

    def extract_text(self, file_path: str, lang: str = "eng") -> str:
        """
        Extract text from an image file using OCR.

        Args:
            file_path: Path to image file
            lang: Tesseract language code (default: English)

        Returns:
            Extracted and cleaned text string
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError(
                "pytesseract and Pillow required. Run: pip install pytesseract Pillow"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")

        try:
            # Open image
            image = Image.open(str(path))
            logger.info(
                "Processing image: %s (%dx%d, mode=%s)",
                path.name, image.width, image.height, image.mode
            )

            # Preprocess for better OCR accuracy
            processed = self._preprocess_image(image)

            # Configure Tesseract for best accuracy
            custom_config = r"--oem 3 --psm 6"

            # Extract text
            text = pytesseract.image_to_string(
                processed,
                lang=lang,
                config=custom_config
            )

            cleaned = self._clean_text(text)
            logger.info(
                "OCR extracted %d characters from %s", len(cleaned), path.name
            )
            return cleaned

        except Exception as e:
            logger.error("OCR extraction error for %s: %s", file_path, str(e))
            raise RuntimeError(f"OCR failed: {str(e)}")

    def _preprocess_image(self, image):
        """
        Preprocess image to improve OCR accuracy:
        - Convert to grayscale
        - Apply adaptive thresholding
        - Denoise
        """
        from PIL import Image, ImageFilter, ImageEnhance
        import io

        # Convert to RGB if needed
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Convert to grayscale
        gray = image.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)

        # Apply sharpening filter
        sharpened = enhanced.filter(ImageFilter.SHARPEN)

        # Try to use OpenCV for advanced preprocessing if available
        try:
            import cv2
            import numpy as np

            # Convert PIL to numpy
            img_array = np.array(sharpened)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)

            # Denoise
            img_array = cv2.fastNlMeansDenoising(img_array, h=10)

            # Convert back to PIL
            return Image.fromarray(img_array)

        except ImportError:
            # OpenCV not available, use basic preprocessing
            logger.debug("OpenCV not available, using basic image preprocessing")
            return sharpened

    def _clean_text(self, text: str) -> str:
        """Clean OCR output text."""
        import re

        if not text:
            return ""

        # Fix common OCR errors
        text = text.replace("|", "I")  # vertical bar → I
        text = text.replace("0", "0")  # normalize zeros

        # Remove excessive whitespace
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove lines with only special characters (OCR artifacts)
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep line if it has enough alphanumeric chars
            alpha_count = sum(1 for c in stripped if c.isalnum())
            if alpha_count >= 2 or not stripped:
                cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines).strip()
