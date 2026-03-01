"""
Rio Local -- Screen Capture (Day 6 / L2)

Captures the primary monitor using mss, compresses to JPEG with
configurable quality and resolution, and detects unchanged frames
via MD5 hashing to skip sending duplicates.

Dependencies: mss, Pillow (PIL)

Usage::

    sc = ScreenCapture(fps=0.33, quality=60, resize_factor=0.5)
    if sc.available:
        jpeg = await sc.capture_async()
        if jpeg is not None:
            await ws.send_binary(IMAGE_PREFIX + jpeg)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Wire protocol prefix for image frames
IMAGE_PREFIX = b"\x02"

# ---------------------------------------------------------------------------
# Lazy-import helpers (avoid hard crash if deps missing)
# ---------------------------------------------------------------------------

_mss_mod = None
_pil_image = None


def _ensure_deps() -> bool:
    """Import mss and Pillow lazily. Returns True if available."""
    global _mss_mod, _pil_image
    if _mss_mod is not None and _pil_image is not None:
        return True
    try:
        import mss as _m
        from PIL import Image as _img

        _mss_mod = _m
        _pil_image = _img
        return True
    except ImportError:
        return False


class ScreenCapture:
    """Captures the primary monitor as compressed JPEG bytes.

    Features:
      - mss for fast screen grab (cross-platform)
      - Pillow for JPEG compression with configurable quality
      - Resize to reduce bandwidth (default: 50% = 960x540 from 1080p)
      - MD5 delta detection: skip unchanged frames
      - Async-friendly: capture_async() runs in executor
    """

    def __init__(
        self,
        fps: float = 0.33,
        quality: int = 60,
        resize_factor: float = 0.5,
    ) -> None:
        self._fps = fps
        self._quality = quality
        self._resize_factor = resize_factor
        self._interval = 1.0 / fps if fps > 0 else 3.0
        self._last_hash: Optional[str] = None
        self._available = _ensure_deps()

        if not self._available:
            log.warning(
                "screen_capture.deps_missing",
                note="Install mss and Pillow: pip install mss Pillow",
            )
        else:
            log.info(
                "screen_capture.init",
                fps=fps,
                quality=quality,
                resize_factor=resize_factor,
                interval_s=round(self._interval, 2),
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if mss and Pillow are importable."""
        return self._available

    @property
    def interval(self) -> float:
        """Seconds between periodic captures."""
        return self._interval

    # ------------------------------------------------------------------
    # Core capture
    # ------------------------------------------------------------------

    def capture(self, force: bool = False) -> Optional[bytes]:
        """Capture a screenshot and return JPEG bytes.

        Returns None if:
          - Dependencies are missing
          - The frame is identical to the last one (delta detection)

        Args:
            force: If True, skip delta detection and always return the frame.
        """
        if not self._available or _mss_mod is None or _pil_image is None:
            return None

        try:
            with _mss_mod.mss() as sct:
                try:
                    monitor = sct.monitors[1]  # Primary monitor
                except IndexError:
                    monitor = sct.monitors[0]  # Fallback to full virtual screen
                    log.warning("screen_capture.no_primary_monitor", note="using monitors[0]")
                raw = sct.grab(monitor)

            # Convert to PIL Image (mss provides .rgb for RGB bytes)
            img = _pil_image.frombytes(
                "RGB", (raw.width, raw.height), raw.rgb
            )

            # Resize to reduce size before compression
            if self._resize_factor < 1.0:
                new_w = int(img.width * self._resize_factor)
                new_h = int(img.height * self._resize_factor)
                img = img.resize((new_w, new_h), _pil_image.LANCZOS)

            # Compress to JPEG
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self._quality)
            jpeg_bytes = buf.getvalue()

            # Delta detection via MD5
            if not force:
                frame_hash = hashlib.md5(jpeg_bytes).hexdigest()
                if frame_hash == self._last_hash:
                    log.debug("screen_capture.delta_skip")
                    return None
                self._last_hash = frame_hash

            log.debug(
                "screen_capture.captured",
                size_kb=round(len(jpeg_bytes) / 1024, 1),
                resolution=f"{img.width}x{img.height}",
            )
            return jpeg_bytes

        except Exception:
            log.exception("screen_capture.error")
            return None

    async def capture_async(self, force: bool = False) -> Optional[bytes]:
        """Run capture in a thread executor to avoid blocking the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.capture, force)

    # ------------------------------------------------------------------
    # OCR-enhanced capture
    # ------------------------------------------------------------------

    async def capture_with_text(
        self,
        ocr_engine,
        force: bool = False,
    ) -> tuple[Optional[bytes], Optional[str]]:
        """Capture a screenshot and extract OCR text in one call.

        Returns (jpeg_bytes, ocr_text).  If OCR is unavailable or the
        engine is None, ocr_text will be None (graceful degradation).

        Args:
            ocr_engine: An OCREngine instance, or None to skip OCR.
            force: If True, skip delta detection.
        """
        loop = asyncio.get_running_loop()
        jpeg = await loop.run_in_executor(None, self.capture, force)

        if jpeg is None or ocr_engine is None or not ocr_engine.available:
            return jpeg, None

        ocr_text = await ocr_engine.extract_text_async(jpeg)
        return jpeg, ocr_text

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def force_next_capture(self) -> None:
        """Reset the delta hash so the next capture always sends."""
        self._last_hash = None
