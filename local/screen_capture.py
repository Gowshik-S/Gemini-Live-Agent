"""
Rio Local -- Screen Capture (Day 6 / L2)

Captures the primary monitor using mss, compresses to JPEG with
configurable quality and resolution, and detects unchanged frames
via MD5 hashing to skip sending duplicates.

Dependencies: mss, Pillow (PIL)

Usage::

    sc = ScreenCapture(fps=0.33, quality=85, resize_factor=0.75)
    if sc.available:
        jpeg = await sc.capture_async()
        if jpeg is not None:
            await ws.send_binary(IMAGE_PREFIX + jpeg)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import subprocess
import shutil
from dataclasses import dataclass
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Wire protocol prefix for image frames
IMAGE_PREFIX = b"\x02"

# Circuit breaker: after this many consecutive failures, pause captures
_MAX_CONSECUTIVE_FAILURES = 3
_CIRCUIT_BREAKER_COOLDOWN = 30.0  # seconds to wait before retrying


@dataclass
class CaptureResult:
    """Screenshot result with monitor metadata for coordinate mapping."""
    jpeg: bytes
    original_width: int    # monitor resolution before resize
    original_height: int   # monitor resolution before resize
    resized_width: int     # dimensions after resize
    resized_height: int    # dimensions after resize
    monitor_left: int      # monitor X offset (for multi-monitor)
    monitor_top: int       # monitor Y offset (for multi-monitor)
    resize_factor: float   # the factor used

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


def _detect_wayland() -> bool:
    """Check if the session is running under Wayland."""
    import os
    return bool(os.environ.get("WAYLAND_DISPLAY"))


def _find_wayland_tool() -> Optional[str]:
    """Find a Wayland-compatible screenshot tool."""
    for tool in ("grim", "spectacle", "gnome-screenshot"):
        if shutil.which(tool):
            return tool
    return None


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
        quality: int = 85,
        resize_factor: float = 0.75,
    ) -> None:
        self._fps = fps
        self._quality = quality
        self._resize_factor = resize_factor
        self._interval = 1.0 / fps if fps > 0 else 3.0
        self._last_hash: Optional[str] = None
        self._available = _ensure_deps()

        # Circuit breaker state
        self._consecutive_failures = 0
        self._circuit_open_until: float = 0.0

        # Monitor metadata from last capture (for coordinate mapping)
        self._last_monitor_info: dict = {"left": 0, "top": 0, "width": 0, "height": 0}
        self._last_capture_result: CaptureResult | None = None

        # Backend selection: prefer mss (X11), fallback to CLI tools (Wayland)
        self._use_wayland_fallback = False
        self._wayland_tool: Optional[str] = None

        if not self._available:
            log.warning(
                "screen_capture.deps_missing",
                note="Install mss and Pillow: pip install mss Pillow",
            )
        else:
            # Detect Wayland and pre-select fallback tool
            if _detect_wayland():
                self._wayland_tool = _find_wayland_tool()
                if self._wayland_tool:
                    log.info(
                        "screen_capture.wayland_detected",
                        fallback_tool=self._wayland_tool,
                        note="Will use CLI fallback if mss fails",
                    )
                else:
                    log.warning(
                        "screen_capture.wayland_no_fallback",
                        note="Install 'grim' (Sway) or 'gnome-screenshot' for Wayland support",
                    )

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

    @property
    def resize_factor(self) -> float:
        return self._resize_factor

    def get_last_capture_result(self) -> CaptureResult | None:
        """Return the full CaptureResult from the most recent capture, or None."""
        return self._last_capture_result

    # ------------------------------------------------------------------
    # Core capture
    # ------------------------------------------------------------------

    def capture(self, force: bool = False) -> Optional[bytes]:
        """Capture a screenshot and return JPEG bytes.

        Returns None if:
          - Dependencies are missing
          - The frame is identical to the last one (delta detection)
          - Circuit breaker is open (too many consecutive failures)

        Args:
            force: If True, skip delta detection and always return the frame.
        """
        if not self._available or _pil_image is None:
            return None

        # Circuit breaker: if open, check cooldown
        import time
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            now = time.monotonic()
            if now < self._circuit_open_until:
                return None  # silently skip — circuit is open
            # Cooldown expired — try again
            log.info("screen_capture.circuit_breaker.retry")
            self._consecutive_failures = 0

        try:
            img = None
            monitor_info: dict | None = None

            # Try mss first (fast, no subprocess), unless we know it fails
            if not self._use_wayland_fallback and _mss_mod is not None:
                try:
                    img, monitor_info = self._capture_mss()
                except Exception as exc:
                    # mss failed — if we have a Wayland fallback, switch to it
                    if self._wayland_tool:
                        log.info(
                            "screen_capture.mss_failed_switching_to_wayland",
                            tool=self._wayland_tool,
                            error=str(exc),
                        )
                        self._use_wayland_fallback = True
                    else:
                        raise  # No fallback — let it fail normally

            # Wayland fallback via CLI tool
            if img is None and self._use_wayland_fallback and self._wayland_tool:
                img = self._capture_wayland()

            if img is None:
                self._record_failure()
                return None

            # Store latest monitor metadata for coordinate mapping
            orig_w, orig_h = img.width, img.height
            if monitor_info is not None:
                self._last_monitor_info = monitor_info
            else:
                self._last_monitor_info = {
                    "left": 0, "top": 0,
                    "width": orig_w, "height": orig_h,
                }

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

            # Success — reset failure counter
            self._consecutive_failures = 0

            log.debug(
                "screen_capture.captured",
                size_kb=round(len(jpeg_bytes) / 1024, 1),
                resolution=f"{img.width}x{img.height}",
                backend="wayland" if self._use_wayland_fallback else "mss",
            )

            # Store metadata for get_last_capture_result()
            mi = self._last_monitor_info
            self._last_capture_result = CaptureResult(
                jpeg=jpeg_bytes,
                original_width=orig_w,
                original_height=orig_h,
                resized_width=img.width,
                resized_height=img.height,
                monitor_left=mi.get("left", 0),
                monitor_top=mi.get("top", 0),
                resize_factor=self._resize_factor,
            )

            return jpeg_bytes

        except Exception:
            self._record_failure()
            return None

    def _capture_mss(self):
        """Capture using mss (X11). Returns (PIL Image, monitor_dict) or raises."""
        with _mss_mod.mss() as sct:
            try:
                monitor = sct.monitors[1]  # Primary monitor
            except IndexError:
                monitor = sct.monitors[0]  # Fallback to full virtual screen
                log.warning("screen_capture.no_primary_monitor", note="using monitors[0]")
            raw = sct.grab(monitor)
            monitor_info = dict(monitor)  # copy while sct is open
        return _pil_image.frombytes("RGB", (raw.width, raw.height), raw.rgb), monitor_info

    def _capture_wayland(self):
        """Capture using a Wayland-compatible CLI tool. Returns a PIL Image or None."""
        import tempfile
        import os

        tmp_path = os.path.join(tempfile.gettempdir(), "rio_screenshot.png")

        try:
            if self._wayland_tool == "grim":
                subprocess.run(
                    ["grim", tmp_path],
                    capture_output=True, timeout=5, check=True,
                )
            elif self._wayland_tool == "spectacle":
                subprocess.run(
                    ["spectacle", "-b", "-n", "-f", "-o", tmp_path],
                    capture_output=True, timeout=10, check=True,
                )
            elif self._wayland_tool == "gnome-screenshot":
                subprocess.run(
                    ["gnome-screenshot", "-f", tmp_path],
                    capture_output=True, timeout=5, check=True,
                )
            else:
                return None

            if os.path.exists(tmp_path):
                img = _pil_image.open(tmp_path).convert("RGB")
                os.unlink(tmp_path)
                return img
        except subprocess.TimeoutExpired:
            log.warning("screen_capture.wayland_timeout", tool=self._wayland_tool)
        except subprocess.CalledProcessError as exc:
            log.warning("screen_capture.wayland_failed", tool=self._wayland_tool, error=str(exc))
        except Exception:
            log.exception("screen_capture.wayland_error")
        return None

    def _record_failure(self):
        """Record a capture failure and open circuit breaker if needed."""
        import time
        self._consecutive_failures += 1
        if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
            self._circuit_open_until = time.monotonic() + _CIRCUIT_BREAKER_COOLDOWN
            log.warning(
                "screen_capture.circuit_breaker.open",
                failures=self._consecutive_failures,
                cooldown_s=_CIRCUIT_BREAKER_COOLDOWN,
                note="Screen capture paused — will retry after cooldown",
            )

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
