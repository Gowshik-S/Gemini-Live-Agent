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
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# Resolve paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_RIO_ROOT = _SCRIPT_DIR  # rio/ directory
_LOCAL_DIR = _RIO_ROOT / "local"
_CLOUD_DIR = _RIO_ROOT / "cloud"
_DEFAULT_CONFIG_PATH = _RIO_ROOT / "config.yaml"
_RIO_HOME = Path(os.environ.get("RIO_HOME", Path.home() / ".rio")).expanduser()
_CONFIG_PATH = Path(os.environ.get("RIO_CONFIG", _RIO_HOME / "config.yaml")).expanduser()
_ENV_PATH = _RIO_HOME / ".env"
_LOG_DIR = _RIO_HOME / "logs"
_PID_DIR = _RIO_HOME / "pids"
_VENV_DIR = _RIO_HOME / "venv"
_RIO_PORTAL_URL = "https://rio.gowshik.in"
_RIO_REGISTER_URL = f"{_RIO_PORTAL_URL}/register"
_REQUIRED_PYTHON_MAJOR = 3
_REQUIRED_PYTHON_MINOR = 11

# Add local/ to path for imports
if str(_LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DIR))


def _ensure_runtime_files() -> None:
    """Create user-writable runtime files for packaged installs."""
    _RIO_HOME.mkdir(parents=True, exist_ok=True)

    if _CONFIG_PATH.exists():
        return

    if _DEFAULT_CONFIG_PATH.exists():
        shutil.copy2(_DEFAULT_CONFIG_PATH, _CONFIG_PATH)
    else:
        _CONFIG_PATH.write_text("rio:\n", encoding="utf-8")


def _ensure_supported_python() -> None:
    if (sys.version_info.major, sys.version_info.minor) == (_REQUIRED_PYTHON_MAJOR, _REQUIRED_PYTHON_MINOR):
        return

    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    required = f"{_REQUIRED_PYTHON_MAJOR}.{_REQUIRED_PYTHON_MINOR}.x"
    print(f"  [Error] Rio requires Python {required}. Detected: Python {current}.")

    target = _find_python_311_launcher()
    if target is None:
        print("  Python 3.11 not found. Attempting automatic installation...")
        _attempt_install_python_311()
        target = _find_python_311_launcher()

    if target is None:
        print("  [Error] Automatic Python 3.11 install did not complete.")
        print("  Fix: install Python 3.11 manually and re-run rio.")
        raise SystemExit(1)

    if not _install_rio_into_target_python(target):
        print("  [Error] Failed to install rio-agent into Python 3.11.")
        print("  Fix: run 'py -3.11 -m pip install --upgrade rio-agent' and retry.")
        raise SystemExit(1)

    cmd = [*target, str(_SCRIPT_DIR / "cli.py"), *sys.argv[1:]]
    print(f"  Relaunching Rio with Python 3.11: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    raise SystemExit(proc.returncode)


def _install_rio_into_target_python(target_python: list[str]) -> bool:
    spec = _rio_install_spec()
    cmd = [*target_python, "-m", "pip", "install", "--upgrade", spec]
    print(f"  Ensuring rio-agent is installed in Python 3.11: {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode == 0


def _rio_install_spec() -> str:
    try:
        version = importlib_metadata.version("rio-agent")
        return f"rio-agent=={version}"
    except Exception:
        pass

    if (_RIO_ROOT / "pyproject.toml").exists():
        return str(_RIO_ROOT)

    return "rio-agent"


def _is_python_311_command(command: list[str]) -> bool:
    try:
        proc = subprocess.run(
            [*command, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return proc.returncode == 0 and proc.stdout.strip() == "3.11"


def _find_python_311_launcher() -> list[str] | None:
    candidates: list[list[str]] = []

    if os.name == "nt" and shutil.which("py"):
        candidates.append(["py", "-3.11"])

    for exe in ("python3.11", "python3", "python"):
        resolved = shutil.which(exe)
        if resolved:
            candidates.append([resolved])

    for candidate in candidates:
        if _is_python_311_command(candidate):
            return candidate
    return None


def _attempt_install_python_311() -> None:
    commands: list[list[str]] = []

    if os.name == "nt":
        if shutil.which("winget"):
            commands.append([
                "winget",
                "install",
                "-e",
                "--id",
                "Python.Python.3.11",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ])
        if shutil.which("choco"):
            commands.append(["choco", "install", "python311", "-y"])
    elif sys.platform == "darwin":
        if shutil.which("brew"):
            commands.append(["brew", "install", "python@3.11"])
    else:
        if shutil.which("apt-get"):
            commands.append(["sudo", "apt-get", "update"])
            commands.append(["sudo", "apt-get", "install", "-y", "python3.11", "python3.11-venv"])
        if shutil.which("dnf"):
            commands.append(["sudo", "dnf", "install", "-y", "python3.11", "python3.11-devel"])
        if shutil.which("yum"):
            commands.append(["sudo", "yum", "install", "-y", "python3.11"])

    for cmd in commands:
        try:
            print(f"  Running: {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=False)
            if proc.returncode == 0 and _find_python_311_launcher() is not None:
                return
        except OSError:
            continue


def _read_api_key_from_env_file() -> str:
    """Read GEMINI_API_KEY from ~/.rio/.env if present."""
    if not _ENV_PATH.exists():
        return ""
    for line in _ENV_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def _ensure_runtime_dirs() -> None:
    _ensure_runtime_files()
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _PID_DIR.mkdir(parents=True, exist_ok=True)
    _VENV_DIR.mkdir(parents=True, exist_ok=True)


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _venv_stamp_path(venv_dir: Path) -> Path:
    return venv_dir / ".rio_requirements_stamp"


def _requirements_stamp(requirements_path: Path) -> str:
    content = requirements_path.read_text(encoding="utf-8", errors="replace")
    payload = f"{sys.version_info.major}.{sys.version_info.minor}|{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_component_venv(component: str, component_dir: Path) -> Path:
    """Create/update a per-component venv under ~/.rio/venv and return its Python path."""
    requirements_path = component_dir / "requirements.txt"
    if not requirements_path.exists():
        return Path(sys.executable)

    venv_dir = _VENV_DIR / component
    venv_python = _venv_python_path(venv_dir)
    expected_stamp = _requirements_stamp(requirements_path)
    stamp_path = _venv_stamp_path(venv_dir)
    force_rebuild = os.environ.get("RIO_REBUILD_VENV", "").strip() in {"1", "true", "True", "yes", "YES"}

    if not venv_python.exists():
        print(f"  Creating {component} runtime venv at {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    current_stamp = ""
    if stamp_path.exists():
        current_stamp = stamp_path.read_text(encoding="utf-8", errors="replace").strip()

    if force_rebuild or current_stamp != expected_stamp:
        print(f"  Installing {component} dependencies...")
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)], check=True)
        stamp_path.write_text(expected_stamp, encoding="utf-8")

    return venv_python


def _bootstrap_runtime_venvs() -> dict[str, Path]:
    auto_venv = os.environ.get("RIO_AUTO_VENV", "1").strip().lower() not in {"0", "false", "no"}
    if not auto_venv:
        return {
            "cloud": Path(sys.executable),
            "local": Path(sys.executable),
        }

    try:
        return {
            "cloud": _ensure_component_venv("cloud", _CLOUD_DIR),
            "local": _ensure_component_venv("local", _LOCAL_DIR),
        }
    except subprocess.CalledProcessError as exc:
        print("  [Error] Failed to initialize runtime virtual environments.")
        print(f"  Command failed with exit code {exc.returncode}.")
        print("  Fix: run setup/setup.sh (Linux/macOS) or setup/setup.bat (Windows), or set RIO_AUTO_VENV=0 to bypass.")
        raise SystemExit(1)


def _pid_path(component: str) -> Path:
    return _PID_DIR / f"{component}.pid"


def _write_pid(component: str, pid: int) -> None:
    _pid_path(component).write_text(str(pid), encoding="utf-8")


def _read_pid(component: str) -> int | None:
    path = _pid_path(component)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _clear_pid(component: str) -> None:
    path = _pid_path(component)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def _stop_pid(pid: int, force: bool = False) -> bool:
    if not _is_pid_running(pid):
        return True

    if os.name == "nt":
        if force:
            proc = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True)
            return proc.returncode == 0
        proc = subprocess.run(["taskkill", "/PID", str(pid), "/T"], capture_output=True, text=True)
        return proc.returncode == 0

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except OSError:
        return False
    time.sleep(0.5)
    return not _is_pid_running(pid)


def _spawn_component(
    component: str,
    cwd: Path,
    env: dict[str, str],
    python_executable: Path,
    background: bool,
) -> int:
    """Start a Rio component and return its process id."""
    log_file = _LOG_DIR / f"{component}.log"
    log_handle = open(log_file, "a", encoding="utf-8", buffering=1)

    cmd = [str(python_executable), "-m", "main"]
    kwargs: dict = {
        "cwd": str(cwd),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "close_fds": True,
    }

    if background:
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **kwargs)
    log_handle.close()
    _write_pid(component, proc.pid)
    return int(proc.pid)


def _load_config():
    """Load Rio configuration."""
    _ensure_runtime_files()
    from config import RioConfig
    return RioConfig.load(_CONFIG_PATH)


def _save_config_value(path: str, value: str) -> None:
    """Set a config value by dot-notation path.

    Reads config.yaml, modifies the value, writes back.
    Handles nested paths like 'models.primary' or 'audio.sample_rate'.
    """
    import yaml

    _ensure_runtime_files()

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


def _dashboard_url_from_cloud_url(cloud_url: str) -> str:
    url = (cloud_url or "ws://localhost:8080/ws/rio/live").strip()
    if url.startswith("wss://"):
        base = "https://" + url[len("wss://"):]
    elif url.startswith("ws://"):
        base = "http://" + url[len("ws://"):]
    elif url.startswith("https://") or url.startswith("http://"):
        base = url
    else:
        base = "http://" + url

    if "/ws/rio/live" in base:
        base = base.split("/ws/rio/live", 1)[0]
    return base.rstrip("/") + "/dashboard"


def _needs_configure() -> bool:
    try:
        cfg = _load_config()
    except Exception:
        return True

    portal_key = (getattr(cfg.portal, "api_key", "") or "").strip()
    if not portal_key or portal_key.lower() in {"your_rio_key_here", "changeme", "todo"}:
        return True
    return False


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
    """Handle: rio doctor — diagnose, auto-fix, then report unresolved issues."""

    def _collect_issues(*, show_details: bool) -> list[dict]:
        issues_local: list[dict] = []

        # 1. Platform detection
        try:
            import platform_utils
            importlib.reload(platform_utils)
            if hasattr(platform_utils, "_platform_info"):
                platform_utils._platform_info = None
            get_missing_dependencies = platform_utils.get_missing_dependencies
            print_platform_summary = platform_utils.print_platform_summary
            if show_details:
                print_platform_summary()
            missing = get_missing_dependencies()
            issues_local.extend(missing)
        except ImportError:
            if show_details:
                print("  [!] platform_utils not available")

        # 2. Config validation
        if show_details:
            print("\n  [Config]")
        try:
            config_local = _load_config()
            config_local.validate()
            if show_details:
                print(f"  Config: valid ({_CONFIG_PATH})")
                print(f"  Cloud URL: {config_local.cloud_url}")
                print(f"  Models: primary={config_local.models.primary}, secondary={config_local.models.secondary}")
                print(f"  Session mode: {config_local.session_mode}")
        except Exception as exc:
            if show_details:
                print(f"  Config: INVALID — {exc}")
            issues_local.append({
                "name": "config",
                "purpose": "Configuration file",
                "install_cmd": f"Fix the error in {_CONFIG_PATH}",
                "severity": "critical",
            })

        # 3. Access
        if show_details:
            print("\n  [Access]")
        try:
            config_local = _load_config()
        except Exception:
            config_local = None

        portal_key = ""
        if config_local is not None:
            portal_key = (getattr(config_local.portal, "api_key", "") or "").strip()
        if not portal_key:
            portal_key = os.environ.get("RIO_PORTAL_API_KEY", "").strip()

        if portal_key:
            masked = f"{portal_key[:6]}...{portal_key[-4:]}" if len(portal_key) >= 10 else "(set)"
            if show_details:
                print(f"  Rio key: configured ({masked})")
        else:
            if show_details:
                print("  Rio key: NOT SET")
                print(f"  Register and get your key: {_RIO_REGISTER_URL}")
            issues_local.append({
                "name": "rio_portal_key",
                "purpose": "Rio portal access",
                "install_cmd": f"Register at {_RIO_REGISTER_URL} and set portal.api_key using 'rio configure'",
                "severity": "critical",
            })

        # 4. Optional API test
        direct_api_key = os.environ.get("GEMINI_API_KEY", "") or _read_api_key_from_env_file()
        if args.test_api:
            if show_details:
                print("\n  [Direct Gemini API Test]")
            if not direct_api_key:
                if show_details:
                    print("  Skipped: GEMINI_API_KEY not set (not required for normal Rio-key flow)")
            else:
                if show_details:
                    print("  Testing direct Gemini API connection...")
                try:
                    from google import genai
                    client = genai.Client(api_key=direct_api_key)
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents="Say 'OK' in one word.",
                        config={"max_output_tokens": 5},
                    )
                    if show_details:
                        print(f"  API test: OK (Flash responded: {response.text.strip()[:20]})")
                except Exception as exc:
                    if show_details:
                        print(f"  API test: FAILED — {exc}")
                    issues_local.append({
                        "name": "gemini_api_optional",
                        "purpose": "Optional direct Gemini API access",
                        "install_cmd": "Set GEMINI_API_KEY only if you need direct local Gemini mode",
                        "severity": "warning",
                    })

        # 5. Model availability (informational)
        if show_details:
            print("\n  [Models]")
        try:
            config_local = _load_config()
            from model_fallback import ModelFallbackChain
            chain = ModelFallbackChain(
                primary=config_local.models.primary,
                fallbacks=[config_local.models.secondary, "gemini-2.5-flash"],
            )
            models = chain.get_available_models()
            if show_details:
                print(f"  Fallback chain: {' → '.join(models)}")
        except ImportError:
            if show_details:
                print("  Fallback chain: not configured")

        return issues_local

    def _is_fixable(issue: dict) -> bool:
        install_cmd = str(issue.get("install_cmd", "")).strip().lower()
        if not install_cmd:
            return False
        return install_cmd.startswith("pip ") or install_cmd.startswith("python -m ")

    def _run_install_segment(segment: str) -> subprocess.CompletedProcess:
        tokens = shlex.split(segment, posix=(os.name != "nt"))
        if not tokens:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        first = tokens[0].lower()
        if first in {"pip", "pip3"}:
            cmd = [sys.executable, "-m", "pip", *tokens[1:]]
        elif first in {"python", "python3", "py"}:
            cmd = [sys.executable, *tokens[1:]]
        else:
            # Fallback only for non-install commands.
            return subprocess.run(segment, shell=True, text=True, capture_output=True)

        return subprocess.run(cmd, shell=False, text=True, capture_output=True)

    def _try_fix(issue: dict) -> bool:
        cmd = str(issue.get("install_cmd", "")).strip()
        if not cmd:
            return False
        print(f"    -> {issue.get('name', 'unknown')}: {cmd}")

        # Support chained installers like: pip install ... && python -m ...
        segments = [seg.strip() for seg in cmd.split("&&") if seg.strip()]
        if not segments:
            print("       [FAILED] empty install command")
            return False

        for segment in segments:
            proc = _run_install_segment(segment)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                detail = err[-1] if err else "unknown error"
                print(f"       [FAILED] {detail}")
                return False

        print("       [OK] fixed")
        return True

    print("\n  Rio Doctor — System Diagnostics")
    print("  " + "=" * 40)

    issues = _collect_issues(show_details=True)

    auto_fix_enabled = not getattr(args, "no_fix", False)
    if auto_fix_enabled and issues:
        fixable = [i for i in issues if _is_fixable(i)]
        if fixable:
            print("\n  [Auto-Fix] Attempting to resolve detected issues...")
            for issue in fixable:
                _try_fix(issue)
            print("\n  [Auto-Fix] Re-running diagnostics...")
            issues = _collect_issues(show_details=False)

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
        print("  ACKNOWLEDGEMENT: All detected errors were resolved. Rio is ready to run.")
    print()


def cmd_configure(args):
    """Handle: rio configure — interactive setup wizard."""
    print("\n  Rio Agent — Interactive Setup")
    print("  " + "=" * 40)
    print()

    import yaml

    config_data = {}
    _ensure_runtime_files()
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            if not isinstance(raw, dict):
                raw = {}
            section = raw.get("rio", raw)
            config_data = section if isinstance(section, dict) else {}

    portal_cfg = config_data.setdefault("portal", {})

    # Step 1: Rio key (primary)
    print("  Step 1/6: Rio Access Key")
    print(f"  Get your key at: {_RIO_REGISTER_URL}")
    current_portal_key = (portal_cfg.get("api_key", "") or "").strip()
    masked_key = f"{current_portal_key[:6]}...{current_portal_key[-4:]}" if len(current_portal_key) >= 10 else "(not set)"
    print(f"  Current RIO key: {masked_key}")
    new_portal_key = input("  RIO key (Enter to keep current): ").strip()
    if new_portal_key:
        portal_cfg["api_key"] = new_portal_key
        portal_cfg["enabled"] = True
        print("  Rio key saved to config.")
    elif not current_portal_key:
        print(f"  No key set yet. Register here: {_RIO_REGISTER_URL}")
    print()

    # Step 2: Model Selection
    print("  Step 2/6: Model Configuration")
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
    print("  Step 3/6: Session Mode")
    print("    live — Voice + audio (requires microphone)")
    print("    text — Text only (keyboard input)")
    current_mode = config_data.get("session_mode", "live")
    print(f"  Current: {current_mode}")
    mode = input("  Mode [live/text] (Enter for current): ").strip().lower()
    if mode in ("live", "text"):
        config_data["session_mode"] = mode
    print()

    # Step 4: Cloud URL
    print("  Step 4/6: Cloud Server URL")
    current_url = config_data.get("cloud_url", "ws://localhost:8080/ws/rio/live")
    print(f"  Current: {current_url}")
    url = input("  URL (Enter for current): ").strip()
    if url:
        config_data["cloud_url"] = url
    print()

    # Step 5: Portal settings (optional)
    print("  Step 5/6: Portal Settings")
    current_portal_enabled = bool(portal_cfg.get("enabled", False))
    current_portal_backend = portal_cfg.get("backend_url", "https://riocloud.gowshik.in")

    print(f"  Current enabled: {current_portal_enabled}")
    enable_answer = input("  Enable portal enforcement? [y/N] (Enter for current): ").strip().lower()
    if enable_answer in {"y", "yes", "true", "1"}:
        portal_cfg["enabled"] = True
    elif enable_answer in {"n", "no", "false", "0"}:
        portal_cfg["enabled"] = False

    print(f"  Current backend: {current_portal_backend}")
    portal_backend = input("  Portal backend URL (Enter for current): ").strip()
    if portal_backend:
        portal_cfg["backend_url"] = portal_backend

    portal_cfg.setdefault("validate_on_startup", True)
    portal_cfg.setdefault("timeout_seconds", 8.0)
    print()

    # Step 6: Vision Settings
    print("  Step 6/6: Vision / Screen Mode")
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
    live_url = _dashboard_url_from_cloud_url(config_data.get("cloud_url", "ws://localhost:8080/ws/rio/live"))
    print(f"  Rio is live at: {live_url}")
    print("  Run 'rio doctor' to verify your setup.")
    print()


def cmd_run(args):
    """Handle: rio run — start cloud + local runtime."""
    print("  Starting Rio Agent...")
    _ensure_runtime_dirs()
    env = os.environ.copy()
    env.setdefault("RIO_CONFIG", str(_CONFIG_PATH))
    if "GEMINI_API_KEY" not in env:
        api_key = _read_api_key_from_env_file()
        if api_key:
            env["GEMINI_API_KEY"] = api_key

    runtimes = _bootstrap_runtime_venvs()

    if not args.skip_configure and _needs_configure():
        print("  Configuration required before run. Redirecting to rio configure...")
        cmd_configure(args)
        env["RIO_CONFIG"] = str(_CONFIG_PATH)

    run_in_background = bool(args.background or (os.name == "nt" and not args.foreground))

    if run_in_background:
        cloud_pid = _spawn_component("cloud", _CLOUD_DIR, env, runtimes["cloud"], background=True)
        time.sleep(1.0)
        local_pid = _spawn_component("local", _LOCAL_DIR, env, runtimes["local"], background=True)
        print("  Rio is running in background mode.")
        print(f"  Cloud PID: {cloud_pid}  |  Log: {_LOG_DIR / 'cloud.log'}")
        print(f"  Local PID: {local_pid}  |  Log: {_LOG_DIR / 'local.log'}")
        print("  Check logs with: rio logs --component both --follow")
        return

    cloud_pid = _spawn_component("cloud", _CLOUD_DIR, env, runtimes["cloud"], background=True)
    print(f"  Cloud started in background (PID {cloud_pid}).")
    print(f"  Cloud log: {_LOG_DIR / 'cloud.log'}")
    print("  Starting local in foreground...")
    os.chdir(_LOCAL_DIR)
    os.environ.update(env)
    local_python = str(runtimes["local"])
    os.execv(local_python, [local_python, "-m", "main"])


def _read_log_lines(path: Path, lines: int = 80) -> list[str]:
    if not path.exists():
        return [f"[missing] {path}\n"]
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        data = handle.readlines()
    return data[-lines:]


def _follow_log(path: Path) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        return

    print(f"  Following {path} (Ctrl+C to stop)")
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                print(line.rstrip())
            else:
                time.sleep(0.4)


def cmd_logs(args):
    """Handle: rio logs — show/tail cloud and local logs."""
    _ensure_runtime_dirs()
    selected = args.component
    paths: list[tuple[str, Path]] = []
    if selected in {"cloud", "both"}:
        paths.append(("cloud", _LOG_DIR / "cloud.log"))
    if selected in {"local", "both"}:
        paths.append(("local", _LOG_DIR / "local.log"))

    if args.follow and selected == "both":
        print("  Follow mode for both components is not supported in one stream.")
        print("  Use one component at a time: rio logs --component cloud --follow")
        print("  or: rio logs --component local --follow")
        return

    if args.follow:
        _follow_log(paths[0][1])
        return

    for name, path in paths:
        print(f"\n===== {name.upper()} LOG ({path}) =====")
        for line in _read_log_lines(path, lines=args.lines):
            print(line.rstrip())


def cmd_status(args):
    """Handle: rio status — show cloud/local process state from pid files."""
    _ensure_runtime_dirs()
    components = ["cloud", "local"] if args.component == "both" else [args.component]
    print("\n  Rio Runtime Status")
    print("  " + "=" * 40)
    all_running = True

    for component in components:
        pid = _read_pid(component)
        log_path = _LOG_DIR / f"{component}.log"
        if pid is None:
            all_running = False
            print(f"  {component.upper()}: STOPPED (no pid file)")
            print(f"    Log: {log_path}")
            continue

        running = _is_pid_running(pid)
        all_running = all_running and running
        state = "RUNNING" if running else "STOPPED"
        print(f"  {component.upper()}: {state} (PID {pid})")
        print(f"    Log: {log_path}")

        if not running:
            _clear_pid(component)

    print("  " + "=" * 40)
    if all_running:
        print("  ACKNOWLEDGEMENT: Requested Rio component(s) are running.")
    else:
        print("  Some requested components are not running.")
    print()


def cmd_stop(args):
    """Handle: rio stop — stop cloud/local process(es) from pid files."""
    _ensure_runtime_dirs()
    components = ["cloud", "local"] if args.component == "both" else [args.component]

    print("\n  Stopping Rio Runtime")
    print("  " + "=" * 40)
    failures: list[str] = []

    for component in components:
        pid = _read_pid(component)
        if pid is None:
            print(f"  {component.upper()}: already stopped (no pid file)")
            continue

        ok = _stop_pid(pid, force=args.force)
        if ok:
            _clear_pid(component)
            print(f"  {component.upper()}: stopped (PID {pid})")
        else:
            failures.append(component)
            print(f"  {component.upper()}: failed to stop (PID {pid})")

    print("  " + "=" * 40)
    if not failures:
        print("  ACKNOWLEDGEMENT: Requested Rio component(s) stopped.")
    else:
        print(f"  Failed to stop: {', '.join(failures)}")
    print()


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
                               help="Test optional direct Gemini API connectivity if GEMINI_API_KEY is set")
    doctor_parser.add_argument("--no-fix", action="store_true",
                               help="Do not auto-attempt fixes; only report issues")

    # rio configure
    subparsers.add_parser("configure", help="Interactive setup wizard")

    # rio run
    run_parser = subparsers.add_parser("run", help="Start Rio cloud + local runtime")
    run_parser.add_argument("--background", action="store_true", help="Run detached in background")
    run_parser.add_argument("--foreground", action="store_true", help="Force foreground mode")
    run_parser.add_argument("--skip-configure", action="store_true", help="Skip configure gate (not recommended)")

    # rio logs
    logs_parser = subparsers.add_parser("logs", help="Show or follow runtime logs")
    logs_parser.add_argument(
        "--component",
        choices=["cloud", "local", "both"],
        default="both",
        help="Which component log to read",
    )
    logs_parser.add_argument("--follow", action="store_true", help="Tail logs continuously")
    logs_parser.add_argument("--lines", type=int, default=80, help="Number of recent lines to show")

    # rio status
    status_parser = subparsers.add_parser("status", help="Show runtime status for cloud/local")
    status_parser.add_argument(
        "--component",
        choices=["cloud", "local", "both"],
        default="both",
        help="Which component status to check",
    )

    # rio stop
    stop_parser = subparsers.add_parser("stop", help="Stop runtime components")
    stop_parser.add_argument(
        "--component",
        choices=["cloud", "local", "both"],
        default="both",
        help="Which component to stop",
    )
    stop_parser.add_argument("--force", action="store_true", help="Force kill process tree")

    return parser


def main():
    _ensure_supported_python()
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
    elif args.command == "logs":
        cmd_logs(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "stop":
        cmd_stop(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
