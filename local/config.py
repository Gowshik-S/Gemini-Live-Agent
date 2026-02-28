"""
Rio Local — Configuration Loader

Reads rio/config.yaml and exposes a fully-typed RioConfig dataclass tree.
Every field carries a sensible default so the app can start with a minimal
(or even missing) YAML file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog
import yaml

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Sub-config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    block_size: int = 1_600
    input_device: Optional[str] = None
    output_device: Optional[str] = None


@dataclass
class HotkeyConfig:
    push_to_talk: str = "f2"
    screenshot: str = "f3"
    toggle_proactive: str = "f4"
    screen_mode: str = "f5"


@dataclass
class VadConfig:
    enabled: bool = True
    threshold: float = 0.5


@dataclass
class VisionConfig:
    fps: float = 0.33
    quality: int = 60
    resize_factor: float = 0.5
    default_mode: str = "on_demand"  # on_demand | autonomous


@dataclass
class StruggleConfig:
    enabled: bool = True
    threshold: float = 0.85
    cooldown_seconds: int = 300
    decline_cooldown: int = 600
    demo_mode: bool = False


@dataclass
class MemoryConfig:
    db_path: str = "./rio_memory"
    max_recall: int = 5


@dataclass
class ModelConfig:
    primary: str = "gemini-2.0-flash"
    secondary: str = "gemini-2.5-pro-preview-03-25"
    pro_rpm_budget: int = 5


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class RioConfig:
    """Top-level configuration for the Rio local client."""

    cloud_url: str = "ws://localhost:8080/ws/rio/live"
    session_mode: str = "live"  # "live" for Live API audio, "text" for fallback
    audio: AudioConfig = field(default_factory=AudioConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    struggle: StruggleConfig = field(default_factory=StruggleConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    models: ModelConfig = field(default_factory=ModelConfig)

    # ------------------------------------------------------------------
    # Factory: load from YAML
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "RioConfig":
        """Load configuration from a YAML file.

        Resolution order for *path*:
          1. Explicit argument
          2. ``RIO_CONFIG`` environment variable
          3. ``../config.yaml`` relative to *this* file (i.e. ``rio/config.yaml``)

        Missing file or missing keys gracefully fall back to defaults.
        """
        if path is None:
            path = os.environ.get("RIO_CONFIG")
        if path is None:
            # Default: rio/config.yaml (one level up from local/)
            path = Path(__file__).resolve().parent.parent / "config.yaml"

        path = Path(path)

        if not path.exists():
            log.warning("config.file_not_found", path=str(path), note="using all defaults")
            return cls()

        log.info("config.loading", path=str(path))

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            log.error("config.parse_error", path=str(path), error=str(exc))
            raise SystemExit(f"Failed to parse config: {exc}") from exc

        rio_block: dict = raw.get("rio", raw)  # accept top-level or nested under 'rio:'
        return cls._from_dict(rio_block)

    @classmethod
    def _from_dict(cls, d: dict) -> "RioConfig":
        """Build a RioConfig from a raw dictionary, tolerating missing keys."""
        return cls(
            cloud_url=d.get("cloud_url", cls.cloud_url),
            session_mode=d.get("session_mode", cls.session_mode),
            audio=_build(AudioConfig, d.get("audio")),
            hotkeys=_build(HotkeyConfig, d.get("hotkeys")),
            vad=_build(VadConfig, d.get("vad")),
            vision=_build(VisionConfig, d.get("vision")),
            struggle=_build(StruggleConfig, d.get("struggle")),
            memory=_build(MemoryConfig, d.get("memory")),
            models=_build(ModelConfig, d.get("models")),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError if any required invariants are broken."""
        if not self.cloud_url:
            raise ValueError("cloud_url must be set")
        if not self.cloud_url.startswith(("ws://", "wss://")):
            raise ValueError(f"cloud_url must start with ws:// or wss://, got: {self.cloud_url}")
        if self.audio.sample_rate <= 0:
            raise ValueError(f"audio.sample_rate must be positive, got: {self.audio.sample_rate}")
        if self.audio.block_size <= 0:
            raise ValueError(f"audio.block_size must be positive, got: {self.audio.block_size}")
        if not (0.0 < self.vision.fps <= 30.0):
            raise ValueError(f"vision.fps must be in (0, 30], got: {self.vision.fps}")
        if not (1 <= self.vision.quality <= 100):
            raise ValueError(f"vision.quality must be in [1, 100], got: {self.vision.quality}")
        if not (0.0 < self.vision.resize_factor <= 1.0):
            raise ValueError(f"vision.resize_factor must be in (0, 1], got: {self.vision.resize_factor}")
        if self.struggle.cooldown_seconds < 0:
            raise ValueError(f"struggle.cooldown_seconds must be >= 0, got: {self.struggle.cooldown_seconds}")
        if self.models.pro_rpm_budget < 0:
            raise ValueError(f"models.pro_rpm_budget must be >= 0, got: {self.models.pro_rpm_budget}")
        if not (0.0 < self.vad.threshold <= 1.0):
            raise ValueError(f"vad.threshold must be in (0, 1], got: {self.vad.threshold}")
        log.info("config.validated")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build(klass: type, raw: dict | None):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    if raw is None:
        return klass()
    # Only pass keys that the dataclass actually expects
    valid_keys = {f.name for f in klass.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    return klass(**filtered)
