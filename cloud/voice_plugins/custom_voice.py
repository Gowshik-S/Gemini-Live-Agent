"""Example custom voice plugin for Rio.

This plugin is intentionally simple: it can override the default prebuilt
Gemini voice name via plugin config.
"""

from __future__ import annotations

from typing import Any


class CustomVoicePlugin:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def resolve_voice_name(self, default_voice: str) -> str | None:
        value = str(self._config.get("voice_name", "")).strip()
        return value or default_voice

    def patch_live_config(self, live_config: Any) -> Any:
        # Hook kept for future provider-specific settings.
        return live_config

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": str(self._config.get("provider", "google-prebuilt")),
            "plugin": "custom_voice",
        }


def create_plugin(config: dict[str, Any] | None = None) -> CustomVoicePlugin:
    return CustomVoicePlugin(config)
