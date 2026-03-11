#!/usr/bin/env python3
"""
Rio CLI — Command-line interface for Rio Agent configuration and management.

Inspired by OpenClaw's CLI architecture (openclaw config get/set, openclaw onboard).

Usage::

    rio config get models.primary
    rio config set models.primary gemini-2.5-flash
    rio config show
    rio doctor
    rio run
    rio configure   # Interactive setup wizard

Entry point: `python -m rio.cli` or installed as `rio` command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Resolve paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_RIO_ROOT = _SCRIPT_DIR  # rio/ directory
_LOCAL_DIR = _RIO_ROOT / "local"
_CONFIG_PATH = _RIO_ROOT / "config.yaml"

# Add local/ to path for imports
if str(_LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DIR))


def _load_config():
    """Load Rio configuration."""
    from config import RioConfig
    return RioConfig.load(_CONFIG_PATH)


def _save_config_value(path: str, value: str) -> None:
    """Set a config value by dot-notation path.

    Reads config.yaml, modifies the value, writes back.
    Handles nested paths like 'models.primary' or 'audio.sample_rate'.
    """
    import yaml

    if not _CONFIG_PATH.exists():
        print(f"  [Error] Config file not found: {_CONFIG_PATH}")
        print(f"  Fix: Create one with: rio configure")
        sys.exit(1)

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # Navigate to the right level
    rio_block = raw.get("rio", raw)
    keys = path.split(".")
    target = rio_block

    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]

    # Type inference from value string
    final_key = keys[-1]
    parsed_value = _parse_value(value)
    target[final_key] = parsed_value

    # Write back
    if "rio" in raw:
        raw["rio"] = rio_block
    else:
        raw = rio_block

    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump({"rio": rio_block} if "rio" not in raw else raw, f,
                  default_flow_style=False, allow_unicode=True)

    print(f"  Set {path} = {parsed_value}")


def _parse_value(value: str):
    """Parse a string value into the appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "null" or value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _get_config_value(path: str):
    """Get a config value by dot-notation path."""
    config = _load_config()
    obj = config
    for key in path.split("."):
        if hasattr(obj, key):
            obj = getattr(obj, key)
        elif isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_config_get(args):
    """Handle: rio config get <path>"""
    value = _get_config_value(args.path)
    if value is None:
        print(f"  Key not found: {args.path}")
        sys.exit(1)
    if hasattr(value, "__dataclass_fields__"):
        # Print dataclass as dict
        from dataclasses import asdict
        print(json.dumps(asdict(value), indent=2, default=str))
    else:
        print(f"  {args.path} = {value}")


def cmd_config_set(args):
    """Handle: rio config set <path> <value>"""
    _save_config_value(args.path, args.value)


def cmd_config_show(args):
    """Handle: rio config show"""
    config = _load_config()
    from dataclasses import asdict
    print(json.dumps(asdict(config), indent=2, default=str))


def cmd_doctor(args):
    """Handle: rio doctor — validate config, check deps, test API."""
    print("\n  Rio Doctor — System Diagnostics")
    print("  " + "=" * 40)

    issues = []

    # 1. Platform detection
    try:
        from platform_utils import get_platform, get_missing_dependencies, print_platform_summary
        print_platform_summary()
        missing = get_missing_dependencies()
        issues.extend(missing)
    except ImportError:
        print("  [!] platform_utils not available")

    # 2. Config validation
    print("\n  [Config]")
    try:
        config = _load_config()
        config.validate()
        print(f"  Config: valid ({_CONFIG_PATH})")
        print(f"  Cloud URL: {config.cloud_url}")
        print(f"  Models: primary={config.models.primary}, secondary={config.models.secondary}")
        print(f"  Session mode: {config.session_mode}")
    except Exception as exc:
        print(f"  Config: INVALID — {exc}")
        issues.append({
            "name": "config",
            "purpose": "Configuration file",
            "install_cmd": f"Fix the error in {_CONFIG_PATH}",
            "severity": "critical",
        })

    # 3. API key check
    print("\n  [API]")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    env_file = _RIO_ROOT / "cloud" / ".env"
    if not api_key and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

    if api_key:
        print(f"  API Key: configured ({api_key[:8]}...{api_key[-4:]})")

        # Quick API test
        if args.test_api:
            print("  Testing API connection...")
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Say 'OK' in one word.",
                    config={"max_output_tokens": 5},
                )
                print(f"  API test: OK (Flash responded: {response.text.strip()[:20]})")
            except Exception as exc:
                print(f"  API test: FAILED — {exc}")
                issues.append({
                    "name": "api_key",
                    "purpose": "Gemini API access",
                    "install_cmd": "Check your API key at https://aistudio.google.com/apikey",
                    "severity": "critical",
                })
    else:
        print("  API Key: NOT SET")
        issues.append({
            "name": "GEMINI_API_KEY",
            "purpose": "Gemini API authentication",
            "install_cmd": "Set in rio/cloud/.env or environment",
            "severity": "critical",
        })

    # 4. Model availability
    print("\n  [Models]")
    try:
        config = _load_config()
        from model_fallback import ModelFallbackChain
        chain = ModelFallbackChain(
            primary=config.models.primary,
            fallbacks=[config.models.secondary, "gemini-2.5-flash"],
        )
        models = chain.get_available_models()
        print(f"  Fallback chain: {' → '.join(models)}")
    except ImportError:
        print(f"  Fallback chain: not configured")

    # 5. Summary
    print("\n  " + "=" * 40)
    criticals = [i for i in issues if i.get("severity") == "critical"]
    errors = [i for i in issues if i.get("severity") == "error"]
    warnings = [i for i in issues if i.get("severity") == "warning"]

    if criticals:
        print(f"  [!!!] {len(criticals)} critical issue(s):")
        for i in criticals:
            print(f"        - {i['name']}: {i['purpose']}")
            print(f"          Fix: {i['install_cmd']}")

    if errors:
        print(f"  [!!]  {len(errors)} error(s):")
        for i in errors:
            print(f"        - {i['name']}: {i['purpose']}")
            print(f"          Fix: {i['install_cmd']}")

    if warnings:
        print(f"  [!]   {len(warnings)} warning(s):")
        for i in warnings:
            print(f"        - {i['name']}: {i['purpose']}")

    if not issues:
        print("  All checks passed! Rio is ready to run.")
    print()


def cmd_configure(args):
    """Handle: rio configure — interactive setup wizard."""
    print("\n  Rio Agent — Interactive Setup")
    print("  " + "=" * 40)
    print()

    import yaml

    config_data = {}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            config_data = raw.get("rio", raw)

    # Step 1: API Key
    print("  Step 1/5: Gemini API Key")
    print("  Get your key at: https://aistudio.google.com/apikey")
    current_key = ""
    env_file = _RIO_ROOT / "cloud" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                current_key = line.split("=", 1)[1].strip()
                break

    if current_key:
        print(f"  Current: {current_key[:8]}...{current_key[-4:]}")
        new_key = input("  New API key (Enter to keep current): ").strip()
        if new_key:
            current_key = new_key
    else:
        current_key = input("  API key: ").strip()

    if current_key:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(f"GEMINI_API_KEY={current_key}\n", encoding="utf-8")
        print("  Saved to rio/cloud/.env")
    print()

    # Step 2: Model Selection
    print("  Step 2/5: Model Configuration")
    print("  Available models:")
    print("    1. gemini-2.5-flash (fast, free tier)")
    print("    2. gemini-2.5-pro-preview-03-25 (powerful, requires billing)")
    print("    3. gemini-2.5-computer-use-preview-10-2025 (screen/browser control)")
    current_primary = config_data.get("models", {}).get("primary", "gemini-2.5-flash")
    print(f"  Current primary: {current_primary}")
    choice = input("  Primary model [1/2/3] (Enter for current): ").strip()
    models_map = {
        "1": "gemini-2.5-flash",
        "2": "gemini-2.5-pro-preview-03-25",
        "3": "gemini-2.5-computer-use-preview-10-2025",
    }
    if choice in models_map:
        config_data.setdefault("models", {})["primary"] = models_map[choice]
        print(f"  Set: {models_map[choice]}")
    print()

    # Step 3: Session Mode
    print("  Step 3/5: Session Mode")
    print("    live — Voice + audio (requires microphone)")
    print("    text — Text only (keyboard input)")
    current_mode = config_data.get("session_mode", "live")
    print(f"  Current: {current_mode}")
    mode = input("  Mode [live/text] (Enter for current): ").strip().lower()
    if mode in ("live", "text"):
        config_data["session_mode"] = mode
    print()

    # Step 4: Cloud URL
    print("  Step 4/5: Cloud Server URL")
    current_url = config_data.get("cloud_url", "ws://localhost:8080/ws/rio/live")
    print(f"  Current: {current_url}")
    url = input("  URL (Enter for current): ").strip()
    if url:
        config_data["cloud_url"] = url
    print()

    # Step 5: Vision Settings
    print("  Step 5/5: Vision / Screen Mode")
    print("    on_demand — Capture screen only when asked")
    print("    autonomous — Always watching screen")
    current_vision = config_data.get("vision", {}).get("default_mode", "on_demand")
    print(f"  Current: {current_vision}")
    vmode = input("  Mode [on_demand/autonomous] (Enter for current): ").strip()
    if vmode in ("on_demand", "autonomous"):
        config_data.setdefault("vision", {})["default_mode"] = vmode
    print()

    # Write config
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump({"rio": config_data}, f, default_flow_style=False, allow_unicode=True)

    print("  " + "=" * 40)
    print(f"  Configuration saved to {_CONFIG_PATH}")
    print("  Run 'rio doctor' to verify your setup.")
    print()


def cmd_run(args):
    """Handle: rio run — start the Rio agent."""
    print("  Starting Rio Agent...")
    # Just exec the main.py
    main_py = _LOCAL_DIR / "main.py"
    os.execv(sys.executable, [sys.executable, str(main_py)])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rio",
        description="Rio Agent — AI-powered autonomous assistant",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # rio config ...
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_sub = config_parser.add_subparsers(dest="config_action")

    # rio config get <path>
    get_parser = config_sub.add_parser("get", help="Get a config value")
    get_parser.add_argument("path", help="Dot-notation config path (e.g., models.primary)")

    # rio config set <path> <value>
    set_parser = config_sub.add_parser("set", help="Set a config value")
    set_parser.add_argument("path", help="Dot-notation config path")
    set_parser.add_argument("value", help="Value to set")

    # rio config show
    config_sub.add_parser("show", help="Show full configuration")

    # rio doctor
    doctor_parser = subparsers.add_parser("doctor", help="Run system diagnostics")
    doctor_parser.add_argument("--test-api", action="store_true",
                               help="Test API connectivity (makes a real API call)")

    # rio configure
    subparsers.add_parser("configure", help="Interactive setup wizard")

    # rio run
    subparsers.add_parser("run", help="Start the Rio agent")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "config":
        if args.config_action == "get":
            cmd_config_get(args)
        elif args.config_action == "set":
            cmd_config_set(args)
        elif args.config_action == "show":
            cmd_config_show(args)
        else:
            parser.parse_args(["config", "--help"])
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "configure":
        cmd_configure(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
