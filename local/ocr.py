"""
Rio Local — OCR Engine (Screen Text Extraction)

Extracts text from screen captures using RapidOCR (ONNX-based).
Used by the struggle detector to hash text content instead of raw pixels,
making "repeated error screen" detection robust against cursor blinks,
clock ticks, and other minor visual noise.

Graceful degradation: if rapidocr-onnxruntime is not installed,
OCREngine.available is False and extract_text() returns None.

Dependencies: rapidocr-onnxruntime, Pillow (already required for screen capture)

Usage::

    ocr = OCREngine()
    if ocr.available:
        text = ocr.extract_text(jpeg_bytes)
        # or from PIL Image:
        text = ocr.extract_text_from_image(pil_image)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import time
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# ── Optional dependency imports (graceful degradation) ────────────────

_rapidocr = None
_pil_image = None


def _ensure_deps() -> bool:
    """Import RapidOCR and Pillow lazily. Returns True if available."""
    global _rapidocr, _pil_image
    if _rapidocr is not None and _pil_image is not None:
        return True
    try:
        from rapidocr_onnxruntime import RapidOCR
        from PIL import Image

        _rapidocr = RapidOCR
        _pil_image = Image
        return True
    except ImportError:
        return False


class OCREngine:
    """Extracts text from screen captures using RapidOCR.

    Features:
      - ONNX-based inference (fast CPU, ~20-50ms per frame)
      - Caches last result: skips OCR if JPEG hash unchanged
      - Async-friendly: extract_text_async() runs in executor
      - Graceful degradation: if deps missing, available=False
    """

    def __init__(self) -> None:
        self._available = _ensure_deps()
        self._engine = None
        self._last_jpeg_hash: Optional[str] = None
        self._last_text: Optional[str] = None

        if self._available:
            try:
                self._engine = _rapidocr()
                log.info("ocr.init", engine="rapidocr-onnxruntime")
            except Exception:
                log.exception("ocr.init_failed")
                self._available = False
        else:
            log.warning(
                "ocr.deps_missing",
                note="Install rapidocr-onnxruntime: pip install rapidocr-onnxruntime",
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if RapidOCR is importable and initialized."""
        return self._available and self._engine is not None

    @property
    def last_text(self) -> Optional[str]:
        """The text from the most recent OCR extraction."""
        return self._last_text

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_text(self, jpeg_bytes: bytes) -> Optional[str]:
        """Extract text from JPEG screenshot bytes.

        Returns the concatenated OCR text, or None if:
          - Dependencies are missing
          - OCR engine not initialized
          - No text detected in the image

        Uses MD5 caching: if the JPEG bytes hash matches the last
        call, returns the cached result without re-running OCR.

        Args:
            jpeg_bytes: Raw JPEG bytes from screen capture.
        """
        if not self.available or jpeg_bytes is None:
            return None

        # Cache check — skip OCR if frame is unchanged
        frame_hash = hashlib.md5(jpeg_bytes).hexdigest()
        if frame_hash == self._last_jpeg_hash:
            return self._last_text

        try:
            t0 = time.monotonic()

            # Decode JPEG → PIL Image → numpy array (RapidOCR input)
            img = _pil_image.open(io.BytesIO(jpeg_bytes))
            import numpy as np
            img_array = np.array(img)

            # Run OCR
            result, _elapse = self._engine(img_array)

            if result is None:
                self._last_jpeg_hash = frame_hash
                self._last_text = None
                return None

            # result is a list of [bbox, text, confidence] entries
            # Extract and join all text lines
            lines = [entry[1] for entry in result if entry[1]]
            text = "\n".join(lines)

            elapsed_ms = (time.monotonic() - t0) * 1000
            log.debug(
                "ocr.extracted",
                lines=len(lines),
                chars=len(text),
                elapsed_ms=round(elapsed_ms, 1),
            )

            # Update cache
            self._last_jpeg_hash = frame_hash
            self._last_text = text if text.strip() else None

            return self._last_text

        except Exception:
            log.exception("ocr.extract_error")
            return None

    def extract_text_from_image(self, pil_image) -> Optional[str]:
        """Extract text from a PIL Image directly.

        Useful when you already have the PIL Image and want to avoid
        JPEG encode → decode round-trip.

        Args:
            pil_image: A PIL.Image.Image object.
        """
        if not self.available or pil_image is None:
            return None

        try:
            t0 = time.monotonic()

            import numpy as np
            img_array = np.array(pil_image)

            result, _elapse = self._engine(img_array)

            if result is None:
                return None

            lines = [entry[1] for entry in result if entry[1]]
            text = "\n".join(lines)

            elapsed_ms = (time.monotonic() - t0) * 1000
            log.debug(
                "ocr.extracted_from_image",
                lines=len(lines),
                chars=len(text),
                elapsed_ms=round(elapsed_ms, 1),
            )

            self._last_text = text if text.strip() else None
            return self._last_text

        except Exception:
            log.exception("ocr.extract_from_image_error")
            return None

    async def extract_text_async(self, jpeg_bytes: bytes) -> Optional[str]:
        """Run OCR extraction in a thread executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.extract_text, jpeg_bytes)

    async def extract_text_from_image_async(self, pil_image) -> Optional[str]:
        """Run image OCR extraction in a thread executor (non-blocking)."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.extract_text_from_image, pil_image)
