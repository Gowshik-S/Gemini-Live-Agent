"""
Rio — Centralized Constants

Single source of truth for model names, version strings, and environment
variable names. Every other file should import from here — never hardcode
model strings or version numbers.
"""

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
RIO_VERSION = "0.9.0"

# ---------------------------------------------------------------------------
# Gemini Model Names
# ---------------------------------------------------------------------------
MODEL_FLASH = "gemini-2.5-flash"
MODEL_PRO = "gemini-2.5-pro-preview-03-25"
MODEL_COMPUTER_USE = "gemini-2.5-computer-use-preview"
MODEL_IMAGEN = "imagen-3"

# ---------------------------------------------------------------------------
# API & Environment
# ---------------------------------------------------------------------------
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_GCP_PROJECT_ID = "GCP_PROJECT_ID"
ENV_GCP_REGION = "GCP_REGION"
ENV_RIO_CONFIG = "RIO_CONFIG"

# ---------------------------------------------------------------------------
# Wire Protocol Prefixes
# ---------------------------------------------------------------------------
AUDIO_PREFIX = b"\x01"
IMAGE_PREFIX = b"\x02"

# ---------------------------------------------------------------------------
# Rate Limits
# ---------------------------------------------------------------------------
PRO_RPM_BUDGET = 5       # Max requests/min to Pro model
PRO_RPM_SAFE = 4         # Safe limit (buffer below budget)
FLASH_RPM_BUDGET = 30    # Max requests/min to Flash model

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CLOUD_URL = "ws://localhost:8080/ws/rio/live"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_BLOCK_SIZE = 320
DEFAULT_VISION_FPS = 0.33
DEFAULT_JPEG_QUALITY = 60
DEFAULT_RESIZE_FACTOR = 0.5
