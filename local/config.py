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
    block_size: int = 320        # 20ms chunks (was 1600/100ms)
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    latency: str = "low"         # "low" for voice assistant, "high" for stability
    use_wasapi: bool = True      # Enable WASAPI low-latency on Windows


@dataclass
class HotkeyConfig:
    push_to_talk: str = "f2"
    screenshot: str = "f3"  # F3 is wired as mute/unmute voice toggle
    toggle_proactive: str = "f4"
    screen_mode: str = "f5"
    live_mode: str = "f6"
    live_translation: str = "f7"
    task_status: str = "f8"


@dataclass
class VadConfig:
    enabled: bool = True
    threshold: float = 0.5


@dataclass
class VisionConfig:
    fps: float = 0.33
    quality: int = 85
    resize_factor: float = 0.75
    default_mode: str = "on_demand"  # on_demand | autonomous
    backend: str = "pyautogui"  # pyautogui | pywin32


@dataclass
class UINavigatorConfig:
    enabled: bool = False
    fps: float = 2.0
    model: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    confidence_threshold: float = 0.85
    analyze_every_n_frames: int = 3
    click_tool: str = "screen_click"  # screen_click (coordinate-grounded)


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
class LoggingConfig:
    log_dir: str = "./logs"
    verbose: bool = False
    max_files: int = 7


@dataclass
class ModelConfig:
    primary: str = "gemini-3-flash-preview"
    live: str = "gemini-2.5-flash-native-audio-preview-12-2025"
    secondary: str = "gemini-3-pro-preview"
    computer_use: str = "gemini-2.5-computer-use-preview-10-2025"
    creative: str = "gemini-3-pro-image-preview"
    imagen: str = "imagen-4"
    pro_rpm_budget: int = 5
    fallback_chain: list = field(default_factory=lambda: ["gemini-2.5-flash"])
    cooldown_seconds: float = 60.0
    timeout_seconds: float = 30.0


@dataclass
class CustomerCareConfig:
    enabled: bool = True
    ticket_dir: str = "./rio_tickets"
    auto_escalate_after: int = 300
    default_priority: str = "medium"


@dataclass
class TutorConfig:
    enabled: bool = True
    progress_dir: str = "./rio_progress"
    default_difficulty: str = "intermediate"
    quiz_num_questions: int = 5
    socratic_mode: bool = True


@dataclass
class SkillsConfig:
    customer_care: CustomerCareConfig = field(default_factory=CustomerCareConfig)
    tutor: TutorConfig = field(default_factory=TutorConfig)


@dataclass
class BrowserConfig:
    default_browser: str = "chromium"  # "chrome" | "chromium" | "edge" | "auto"
    default_profile: str = "Default"  # "Default" or empty for shared Rio profile


@dataclass
class PortalConfig:
    enabled: bool = False
    backend_url: str = "https://riocloud.gowshik.in"
    api_key: str = ""
    validate_on_startup: bool = True
    timeout_seconds: float = 8.0


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
    ui_navigator: UINavigatorConfig = field(default_factory=UINavigatorConfig)
    struggle: StruggleConfig = field(default_factory=StruggleConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    portal: PortalConfig = field(default_factory=PortalConfig)

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
        _defaults = cls()
        return cls(
            cloud_url=d.get("cloud_url", _defaults.cloud_url),
            session_mode=d.get("session_mode", _defaults.session_mode),
            audio=_build(AudioConfig, d.get("audio")),
            hotkeys=_build(HotkeyConfig, d.get("hotkeys")),
            vad=_build(VadConfig, d.get("vad")),
            vision=_build(VisionConfig, d.get("vision")),
            ui_navigator=_build(UINavigatorConfig, d.get("ui_navigator")),
            struggle=_build(StruggleConfig, d.get("struggle")),
            memory=_build(MemoryConfig, d.get("memory")),
            models=_build(ModelConfig, d.get("models")),
            logging=_build(LoggingConfig, d.get("logging")),
            skills=_build_skills(d.get("skills")),
            browser=_build(BrowserConfig, d.get("browser")),
            portal=_build(PortalConfig, d.get("portal")),
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
        if not (0.0 < self.ui_navigator.fps <= 30.0):
            raise ValueError(f"ui_navigator.fps must be in (0, 30], got: {self.ui_navigator.fps}")
        if not (0.0 <= self.ui_navigator.confidence_threshold <= 1.0):
            raise ValueError(
                "ui_navigator.confidence_threshold must be in [0, 1], "
                f"got: {self.ui_navigator.confidence_threshold}"
            )
        if self.ui_navigator.analyze_every_n_frames <= 0:
            raise ValueError(
                "ui_navigator.analyze_every_n_frames must be > 0, "
                f"got: {self.ui_navigator.analyze_every_n_frames}"
            )
        if self.struggle.cooldown_seconds < 0:
            raise ValueError(f"struggle.cooldown_seconds must be >= 0, got: {self.struggle.cooldown_seconds}")
        if self.models.pro_rpm_budget < 0:
            raise ValueError(f"models.pro_rpm_budget must be >= 0, got: {self.models.pro_rpm_budget}")
        if not (0.0 < self.vad.threshold <= 1.0):
            raise ValueError(f"vad.threshold must be in (0, 1], got: {self.vad.threshold}")
        if self.portal.enabled and not self.portal.backend_url.startswith(("http://", "https://")):
            raise ValueError(
                f"portal.backend_url must start with http:// or https://, got: {self.portal.backend_url}"
            )
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


def _build_skills(raw: dict | None) -> SkillsConfig:
    """Build the nested SkillsConfig from a raw dict."""
    if raw is None:
        return SkillsConfig()
    return SkillsConfig(
        customer_care=_build(CustomerCareConfig, raw.get("customer_care")),
        tutor=_build(TutorConfig, raw.get("tutor")),
    )


# ---------------------------------------------------------------------------
# Model resolver: env → yaml → hardcoded default
# ---------------------------------------------------------------------------

# Maps logical model names to (env var, ModelConfig field name)
_MODEL_ENV_MAP = {
    "live":         ("LIVE_MODEL",          "live"),
    "primary":      ("ORCHESTRATOR_MODEL",  "primary"),
    "orchestrator": ("ORCHESTRATOR_MODEL",  "primary"),
    "secondary":    ("RESEARCH_MODEL",      "secondary"),
    "research":     ("RESEARCH_MODEL",      "secondary"),
    "computer_use": ("COMPUTER_USE_MODEL",  "computer_use"),
    "creative":     ("CREATIVE_MODEL",      "creative"),
    "imagen":       ("IMAGEN_MODEL",        "imagen"),
}

# Singleton config — lazily loaded
_cached_config: RioConfig | None = None


def _load_env_file() -> None:
    """Parse rio/cloud/.env and inject missing vars into os.environ."""
    env_file = Path(__file__).resolve().parent.parent / "cloud" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


def get_model(name: str) -> str:
    """Resolve a model name with env → yaml → hardcoded default fallback.

    Usage::

        from rio.local.config import get_model
        cu_model = get_model("computer_use")
        live_model = get_model("live")
    """
    global _cached_config

    # 1. Parse .env file (one-time, fills os.environ for missing vars)
    _load_env_file()

    mapping = _MODEL_ENV_MAP.get(name)
    if mapping is None:
        raise ValueError(f"Unknown model name: {name!r}. Valid: {sorted(_MODEL_ENV_MAP)}")

    env_var, config_field = mapping

    # Priority 1: environment variable (includes .env file values)
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val

    # Priority 2: config.yaml value
    if _cached_config is None:
        _cached_config = RioConfig.load()
    yaml_val = getattr(_cached_config.models, config_field, "").strip()
    if yaml_val:
        return yaml_val

    # Priority 3: hardcoded default from ModelConfig dataclass
    return getattr(ModelConfig(), config_field)


def _load_raw_browser_config() -> dict:
    """Return the raw ``rio.browser`` block from config.yaml (no dataclass)."""
    path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    rio_block = raw.get("rio", raw)
    return rio_block.get("browser", {})
