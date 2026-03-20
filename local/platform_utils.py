"""
Rio — Cross-Platform Utilities

Detects OS platform and provides platform-specific implementations for:
  - Browser launching (system default browser)
  - Screen interaction backends
  - File path resolution
  - Process management
  - Hotkey handling

Inspired by OpenClaw's multi-platform support (macOS/Linux/Windows).

Usage::

    from platform_utils import Platform, get_platform

    plat = get_platform()
    print(plat.name, plat.is_windows, plat.has_accessibility)
    plat.open_url("https://google.com")
    plat.open_file("/path/to/file")
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

log = structlog.get_logger(__name__)


@dataclass
class PlatformInfo:
    """Platform detection results — computed once at startup."""

    name: str = ""                  # "windows", "macos", "linux"
    arch: str = ""                  # "x86_64", "arm64"
    version: str = ""               # OS version string
    is_windows: bool = False
    is_macos: bool = False
    is_linux: bool = False
    is_wsl: bool = False            # Windows Subsystem for Linux
    has_pyautogui: bool = False
    has_playwright: bool = False
    has_pywinauto: bool = False
    has_pynput: bool = False
    has_accessibility: bool = False  # macOS accessibility / Linux AT-SPI
    display_server: str = ""        # "x11", "wayland", "" (Windows/macOS)
    home_dir: str = ""
    config_dir: str = ""            # Platform-specific config directory
    log_dir: str = ""               # Platform-specific log directory
    python_version: str = ""
    node_available: bool = False
    node_version: str = ""

    def to_dict(self) -> dict:
        """Serialize for API/dashboard."""
        return {k: v for k, v in self.__dict__.items()}


def detect_platform() -> PlatformInfo:
    """Detect the current platform and available capabilities.

    Called once at startup. Results cached in the PlatformInfo dataclass.
    """
    info = PlatformInfo()
    system = platform.system().lower()
    info.arch = platform.machine().lower()
    info.version = platform.version()
    info.python_version = platform.python_version()
    info.home_dir = str(Path.home())

    # OS detection
    if system == "windows":
        info.name = "windows"
        info.is_windows = True
        info.config_dir = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        info.log_dir = str(Path(info.config_dir) / "Rio" / "logs")
    elif system == "darwin":
        info.name = "macos"
        info.is_macos = True
        info.config_dir = str(Path.home() / "Library" / "Application Support" / "Rio")
        info.log_dir = str(Path.home() / "Library" / "Logs" / "Rio")
    else:
        info.name = "linux"
        info.is_linux = True
        # WSL detection
        if "microsoft" in platform.release().lower() or "wsl" in platform.release().lower():
            info.is_wsl = True
        info.config_dir = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")) + "/rio"
        info.log_dir = os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local" / "state")) + "/rio/logs"
        # Display server detection
        session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
        if session_type == "wayland":
            info.display_server = "wayland"
        elif session_type == "x11" or os.environ.get("DISPLAY"):
            info.display_server = "x11"

    # Dependency detection
    info.has_pyautogui = _check_import("pyautogui")
    info.has_playwright = _check_import("playwright")
    info.has_pywinauto = _check_import("pywinauto") if info.is_windows else False
    info.has_pynput = _check_import("pynput")

    # Accessibility (macOS)
    if info.is_macos:
        info.has_accessibility = _check_macos_accessibility()

    # Node.js availability
    info.node_available, info.node_version = _check_node()

    return info


def _check_import(module: str) -> bool:
    """Check if a Python module is importable."""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _check_macos_accessibility() -> bool:
    """Check if macOS accessibility permissions are granted."""
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to return name of first process'],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_node() -> tuple[bool, str]:
    """Check if Node.js is available and return version."""
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
    except Exception:
        pass
    return False, ""


# ---------------------------------------------------------------------------
# Platform-specific actions
# ---------------------------------------------------------------------------

_platform_info: Optional[PlatformInfo] = None


def get_platform() -> PlatformInfo:
    """Get cached platform info (detects on first call)."""
    global _platform_info
    if _platform_info is None:
        _platform_info = detect_platform()
    return _platform_info


def open_url(url: str) -> bool:
    """Open a URL in the system default browser."""
    plat = get_platform()
    try:
        if plat.is_windows:
            os.startfile(url)
        elif plat.is_macos:
            subprocess.Popen(["open", url])
        elif plat.is_linux:
            subprocess.Popen(["xdg-open", url])
        else:
            return False
        return True
    except Exception as exc:
        log.warning("platform.open_url_failed", url=url, error=str(exc))
        return False


def open_file(path: str) -> bool:
    """Open a file with the system default application."""
    plat = get_platform()
    try:
        if plat.is_windows:
            os.startfile(path)
        elif plat.is_macos:
            subprocess.Popen(["open", path])
        elif plat.is_linux:
            subprocess.Popen(["xdg-open", path])
        else:
            return False
        return True
    except Exception as exc:
        log.warning("platform.open_file_failed", path=path, error=str(exc))
        return False


def get_browser_launch_command(url: str = "") -> list[str]:
    """Get platform-specific command to launch a browser."""
    plat = get_platform()
    if plat.is_windows:
        return ["cmd", "/c", f"start {url}" if url else "start chrome"]
    elif plat.is_macos:
        return ["open", "-a", "Safari", url] if url else ["open", "-a", "Safari"]
    elif plat.is_linux:
        return ["xdg-open", url] if url else ["xdg-open", "about:blank"]
    return []


def get_screen_interaction_backend() -> str:
    """Determine the best available screen interaction backend.

    Returns: "pyautogui", "applescript", "xdotool", or "none"
    """
    plat = get_platform()

    if plat.has_pyautogui:
        # pyautogui works on all platforms but has Wayland issues
        if plat.is_linux and plat.display_server == "wayland":
            # Check for xdotool as fallback
            if shutil.which("xdotool"):
                return "xdotool"
            log.warning("platform.wayland_no_backend",
                        note="pyautogui may not work on Wayland. Consider X11 or install xdotool.")
        return "pyautogui"

    if plat.is_macos:
        return "applescript"  # Can fall back to AppleScript

    if plat.is_linux and shutil.which("xdotool"):
        return "xdotool"

    return "none"


def get_missing_dependencies() -> list[dict]:
    """Check for missing dependencies and return actionable suggestions.

    Returns list of {name, purpose, install_cmd, severity} dicts.
    """
    plat = get_platform()
    missing = []

    if not plat.has_pyautogui:
        missing.append({
            "name": "pyautogui",
            "purpose": "Screen navigation (click, type, scroll)",
            "install_cmd": "pip install pyautogui>=0.9.54",
            "severity": "error",
        })

    if not plat.has_playwright:
        missing.append({
            "name": "playwright",
            "purpose": "Browser automation (web navigation, form filling)",
            "install_cmd": "pip install playwright && python -m playwright install chromium",
            "severity": "error",
        })

    if plat.is_windows and not plat.has_pywinauto:
        missing.append({
            "name": "pywinauto",
            "purpose": "Windows application automation",
            "install_cmd": "pip install pywinauto",
            "severity": "error",
        })

    if not plat.has_pynput:
        missing.append({
            "name": "pynput",
            "purpose": "Keyboard hotkeys (push-to-talk, shortcuts)",
            "install_cmd": "pip install pynput>=1.7.0",
            "severity": "error",
        })

    return missing


def print_platform_summary() -> None:
    """Print a formatted platform detection summary to console."""
    plat = get_platform()
    missing = get_missing_dependencies()

    print(f"\n  [Platform] {plat.name} {plat.arch} (Python {plat.python_version})")

    if plat.is_wsl:
        print("  [Platform] Running in WSL")

    if plat.is_linux and plat.display_server:
        print(f"  [Platform] Display: {plat.display_server}")

    # Dependency status
    deps = [
        ("pyautogui", plat.has_pyautogui),
        ("playwright", plat.has_playwright),
        ("pynput", plat.has_pynput),
    ]
    if plat.is_windows:
        deps.append(("pywinauto", plat.has_pywinauto))

    for name, available in deps:
        icon = "ready" if available else "MISSING"
        print(f"  [{name}] {icon}")

    if missing:
        print(f"\n  [!] {len(missing)} issue(s) detected:")
        for m in missing:
            severity_icon = {"critical": "!!!", "error": "!!", "warning": "!"}.get(m["severity"], "?")
            print(f"  [{severity_icon}] {m['name']}: {m['purpose']}")
            print(f"       Fix: {m['install_cmd']}")
