"""
Rio Cloud — Skill Loader (D3)

Loads skill definitions from rio/skills/*.yaml and makes their
tool lists and instruction fragments available to the orchestrator.

Skill format (YAML):
  name: str
  description: str
  tools: list[str]        # tool names this skill provides
  system_instruction_append: str  # added to system instruction
  config: dict            # skill-specific settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def load_skills() -> dict[str, dict[str, Any]]:
    """Load all skill definitions from rio/skills/*.yaml.

    Returns a dict of skill_name → skill_config.
    """
    skills: dict[str, dict[str, Any]] = {}

    if not _SKILLS_DIR.is_dir():
        return skills

    try:
        import yaml
    except ImportError:
        log.warning("skill_loader.yaml_not_available")
        return skills

    for filepath in _SKILLS_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(filepath.read_text(encoding="utf-8")) or {}
            name = data.get("name", filepath.stem)
            skills[name] = {
                "name": name,
                "description": data.get("description", ""),
                "tools": data.get("tools", []),
                "system_instruction_append": data.get("system_instruction_append", ""),
                "config": data.get("config", {}),
            }
            log.info("skill_loader.loaded", skill=name, tools=len(data.get("tools", [])))
        except Exception as exc:
            log.warning("skill_loader.error", file=filepath.name, error=str(exc))

    return skills


def get_skill_instruction_fragment(skills: dict[str, dict]) -> str:
    """Build the combined instruction fragment from all loaded skills."""
    fragments: list[str] = []
    for name, skill in skills.items():
        append = skill.get("system_instruction_append", "").strip()
        if append:
            fragments.append(f"[SKILL: {name}] {append}")
    return "\n".join(fragments)


def get_skill_tool_names(skills: dict[str, dict]) -> set[str]:
    """Get the union of all tool names across loaded skills."""
    tool_names: set[str] = set()
    for skill in skills.values():
        tool_names.update(skill.get("tools", []))
    return tool_names
