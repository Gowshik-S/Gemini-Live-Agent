"""Voice plugin runtime loader for Rio Live API.

Custom voice remains optional and plugin-driven. If plugin loading fails,
Rio falls back to default prebuilt voice behavior.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger(__name__)


@dataclass
class VoiceRuntime:
    default_voice: str
    active_voice: str
    plugin_enabled: bool = False
    plugin_name: str | None = None
    plugin: Any | None = None
    plugin_metadata: dict[str, Any] = field(default_factory=dict)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_rio_block() -> dict[str, Any]:
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("voice_plugin.config_parse_failed", error=str(exc))
        return {}
    return raw.get("rio", raw) if isinstance(raw, dict) else {}


def _import_plugin_module(module_path: str):
    errors: list[str] = []
    candidates = [module_path]
    if module_path.startswith("rio.cloud."):
        candidates.append(module_path[len("rio.cloud."):])
    if module_path.startswith("rio."):
        candidates.append(module_path[len("rio."):])

    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise ImportError("; ".join(errors))


def _build_plugin(module_path: str, factory_name: str, plugin_cfg: dict[str, Any]) -> Any:
    module = _import_plugin_module(module_path)
    factory = getattr(module, factory_name, None)
    if callable(factory):
        return factory(plugin_cfg)

    klass = getattr(module, factory_name, None)
    if isinstance(klass, type):
        return klass(plugin_cfg)

    raise AttributeError(
        f"Plugin factory/class '{factory_name}' not found in module '{module_path}'"
    )


def load_voice_runtime(default_voice_env: str) -> VoiceRuntime:
    rio = _load_rio_block()
    voice = rio.get("voice", {}) if isinstance(rio, dict) else {}
    plugin_block = voice.get("plugin", {}) if isinstance(voice, dict) else {}

    default_voice = str(voice.get("default_name", "")).strip() if isinstance(voice, dict) else ""
    if not default_voice:
        default_voice = default_voice_env

    runtime = VoiceRuntime(default_voice=default_voice, active_voice=default_voice)

    plugin_enabled_default = bool(plugin_block.get("enabled", False)) if isinstance(plugin_block, dict) else False
    plugin_enabled = _env_flag("RIO_VOICE_PLUGIN_ENABLED", plugin_enabled_default)
    if not plugin_enabled:
        return runtime

    module_path = os.environ.get("RIO_VOICE_PLUGIN_MODULE", "").strip()
    if not module_path and isinstance(plugin_block, dict):
        module_path = str(plugin_block.get("module", "")).strip()

    factory_name = os.environ.get("RIO_VOICE_PLUGIN_FACTORY", "").strip()
    if not factory_name:
        factory_name = str(plugin_block.get("factory", "create_plugin") if isinstance(plugin_block, dict) else "create_plugin").strip()

    plugin_cfg = plugin_block.get("config", {}) if isinstance(plugin_block, dict) else {}
    if not isinstance(plugin_cfg, dict):
        plugin_cfg = {}

    if not module_path:
        log.warning("voice_plugin.enabled_but_module_missing")
        return runtime

    try:
        plugin = _build_plugin(module_path, factory_name, plugin_cfg)
        runtime.plugin_enabled = True
        runtime.plugin_name = module_path
        runtime.plugin = plugin

        resolve_voice = getattr(plugin, "resolve_voice_name", None)
        if callable(resolve_voice):
            resolved = resolve_voice(default_voice)
            if isinstance(resolved, str) and resolved.strip():
                runtime.active_voice = resolved.strip()

        meta = getattr(plugin, "metadata", None)
        if callable(meta):
            value = meta() or {}
            if isinstance(value, dict):
                runtime.plugin_metadata = value

        log.info(
            "voice_plugin.loaded",
            plugin=runtime.plugin_name,
            active_voice=runtime.active_voice,
        )
    except Exception as exc:
        log.warning(
            "voice_plugin.load_failed",
            module=module_path,
            factory=factory_name,
            error=str(exc),
        )

    return runtime


def apply_voice_plugin(live_config: Any, runtime: VoiceRuntime) -> Any:
    if not runtime.plugin_enabled or runtime.plugin is None:
        return live_config
    patch_fn = getattr(runtime.plugin, "patch_live_config", None)
    if not callable(patch_fn):
        return live_config
    try:
        patched = patch_fn(live_config)
        return patched if patched is not None else live_config
    except Exception as exc:
        log.warning("voice_plugin.patch_failed", error=str(exc))
        return live_config
