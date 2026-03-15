"""
Rio Local — Playwright Browser Tools (E1)

Provides DOM-level browser automation alongside Computer Use vision.
Uses Playwright to manage browser instances with persistent profiles.

Two connection modes:
  1. CDP connect  — if a browser is already running with --remote-debugging-port
  2. launch_persistent_context — Playwright-managed browser (avoids Chrome
     single-instance conflict when Chrome is already open without CDP)

Tools:
  - browser_connect: Connect (or auto-launch) a browser
  - browser_evaluate: Execute JavaScript in the page
  - browser_fill_form: Fill a form field by CSS selector
  - browser_click_element: Click an element by CSS selector
  - browser_extract_text: Extract text from an element
  - browser_wait_for: Wait for a selector to appear
  - browser_screenshot: Take a page screenshot
  - browser_navigate: Navigate to a URL

Profile support:
  pass profile="Default" / "Work" / "Profile 1" etc.
  Each profile gets its own isolated data dir under
  %LOCALAPPDATA%\\Rio\\Profiles\\<profile> (Windows) or
  ~/.local/share/rio/Profiles/<profile> (Linux/Mac).
  This never conflicts with Chrome that is already open.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Globals — single shared Playwright + BrowserContext
# ---------------------------------------------------------------------------
_playwright = None
_context = None   # BrowserContext (from CDP or launch_persistent_context)
_page = None

# ---------------------------------------------------------------------------
# Browser discovery
# ---------------------------------------------------------------------------

_BROWSER_CANDIDATES: dict[str, list[str]] = {
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ],
    "chromium": [
        r"C:\Program Files\Chromium\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Chromium\Application\chrome.exe"),
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ],
    "edge": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
        "/usr/bin/microsoft-edge",
        "/usr/bin/microsoft-edge-stable",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ],
    "brave": [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        "/usr/bin/brave-browser",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ],
}

_AUTO_ORDER = ["chrome", "edge", "chromium", "brave"]


def _find_browser_exe(browser: str) -> str | None:
    """Return the first existing executable path for the given browser name."""
    candidates = _BROWSER_CANDIDATES.get(browser.lower(), [])
    for path in candidates:
        if path and Path(path).exists():
            return path
    return None


def _resolve_browser_exe(browser: str) -> str | None:
    """Resolve browser name to exe path, trying auto-detection order."""
    if browser.lower() == "auto":
        for name in _AUTO_ORDER:
            exe = _find_browser_exe(name)
            if exe:
                return exe
        return None
    return _find_browser_exe(browser)


def _get_persistent_user_data_dir(profile: str) -> Path:
    """Return the Playwright-managed user data dir for the given profile.

    Uses a dedicated Rio directory so it never conflicts with the user's
    normal Chrome instance (which locks its own user-data-dir).
    """
    if sys.platform == "win32":
        base = Path(os.path.expandvars(r"%LOCALAPPDATA%")) / "Rio"
    else:
        base = Path.home() / ".local" / "share" / "rio"
    if profile:
        return base / "Profiles" / profile
    return base / "ChromeAutomation"


def _find_actual_chrome_profile(profile_name: str) -> tuple["Path | None", "str | None"]:
    """Locate the actual Chrome user data dir + profile subfolder for a given display name.

    Reads Chrome's ``Local State`` file to match the profile display name (e.g. "rio")
    to its internal directory name (e.g. "Profile 1").

    Returns:
        (user_data_dir, profile_dir_name) — both are None if Chrome isn't installed.
        profile_dir_name may be None if the profile name wasn't found in Local State.
    """
    import json as _json

    if sys.platform == "win32":
        user_data_root = Path(os.path.expandvars(r"%LOCALAPPDATA%")) / "Google" / "Chrome" / "User Data"
    elif sys.platform == "darwin":
        user_data_root = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
    else:
        user_data_root = Path.home() / ".config" / "google-chrome"

    if not user_data_root.exists():
        return None, None

    local_state_file = user_data_root / "Local State"
    if local_state_file.exists():
        try:
            state = _json.loads(local_state_file.read_text(encoding="utf-8"))
            profiles = state.get("profile", {}).get("info_cache", {})
            for dir_name, info in profiles.items():
                name = info.get("name", "")
                if name.lower() == profile_name.lower():
                    log.info("browser.found_chrome_profile", name=name, dir=dir_name)
                    return user_data_root, dir_name
        except Exception as exc:
            log.debug("browser.chrome_local_state_read_failed", error=str(exc))

    # User data dir exists but couldn't match profile name → return dir anyway
    return user_data_root, None


# ---------------------------------------------------------------------------
# Playwright instance helper
# ---------------------------------------------------------------------------

async def _get_playwright():
    global _playwright
    if _playwright is None:
        from playwright.async_api import async_playwright
        _playwright = await async_playwright().start()
    return _playwright


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

async def _ensure_connection(cdp_url: str = "http://127.0.0.1:9222") -> Any | None:
    """Return the current page if alive, else attempt a CDP connect.

    Returns a Playwright Page on success, None on failure.
    Call browser_connect() to trigger an auto-launch fallback.
    """
    global _context, _page

    # Normalise URL to avoid IPv6 issues on Windows
    cdp_url = cdp_url.replace("localhost", "127.0.0.1")

    # Reuse existing page if still alive
    if _page is not None:
        try:
            await _page.title()
            return _page
        except Exception:
            _page = None

    # Context might still be alive — grab a page from it
    if _context is not None:
        try:
            pages = _context.pages
            if pages:
                _page = pages[0]
                await _page.title()
                return _page
            _page = await _context.new_page()
            return _page
        except Exception:
            _context = None

    # Try CDP connect (for browsers pre-launched with --remote-debugging-port)
    try:
        pw = await _get_playwright()
        browser = await pw.chromium.connect_over_cdp(cdp_url)
        contexts = browser.contexts
        _context = contexts[0] if contexts else await browser.new_context()
        _page = _context.pages[0] if _context.pages else await _context.new_page()
        log.info("browser.connected_cdp", url=cdp_url)
        return _page
    except Exception:
        return None


async def get_browser_context(
    cdp_url: str = "http://127.0.0.1:9222",
    browser: str = "auto",
    profile: str = "",
) -> Any:
    """High-level helper to get a shared BrowserContext.
    
    1. Tries to connect to an existing browser via CDP.
    2. Falls back to launching a persistent context.
    """
    global _context, _page
    
    # Try reusing existing page/context
    page = await _ensure_connection(cdp_url)
    if page is not None:
        return _context

    # Launch new persistent context
    await _launch_with_playwright(browser=browser, profile=profile)
    return _context


async def _launch_with_playwright(
    browser: str = "auto",
    profile: str = "",
) -> Any:
    """Launch a browser via Playwright's launch_persistent_context.

    Strategy:
    1. If ``profile`` is set (or Default), try to find and use the user's actual Chrome
       profile. This gives full access to bookmarks, cookies, and extensions.
    2. If the actual Chrome profile is locked (Chrome already running) or not
       found, fall back to a Playwright-managed Rio profile directory — which
       always works regardless of whether Chrome is running.

    Returns the active Page.
    """
    global _context, _page

    pw = await _get_playwright()
    exe = _resolve_browser_exe(browser)

    base_launch_args = [
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--remote-debugging-port=9222", # Enable CDP for future connections
    ]

    # ── Strategy 1: Try actual Chrome user profile ─────────────────────────
    # If no profile specified, we default to "Default" to use the user's main profile
    target_profile = profile or "Default"
    
    actual_dir, profile_dir_name = _find_actual_chrome_profile(target_profile)
    if actual_dir:
        args = list(base_launch_args)
        if profile_dir_name:
            args.append(f"--profile-directory={profile_dir_name}")

        log.info(
            "browser.launching_actual_chrome_profile",
            browser=browser,
            exe=exe or "playwright-bundled",
            profile=target_profile,
            user_data_dir=str(actual_dir),
            profile_dir=profile_dir_name or "(not found in Local State)",
        )
        try:
            kwargs: dict[str, Any] = {
                "headless": False,
                "no_viewport": True,
                "args": args,
            }
            if exe:
                kwargs["executable_path"] = exe
            _context = await pw.chromium.launch_persistent_context(
                str(actual_dir), **kwargs,
            )
            _page = _context.pages[0] if _context.pages else await _context.new_page()
            log.info("browser.launched_actual_profile", title=await _page.title(), url=_page.url)
            return _page
        except Exception as exc:
            log.warning(
                "browser.actual_profile_failed_fallback",
                error=str(exc),
                note="Chrome may be running and locking its user-data-dir. "
                     "Falling back to Rio-managed profile. "
                     "For best results, launch Chrome with --remote-debugging-port=9222.",
            )
            # Reset context so fallback starts clean
            _context = None
            _page = None

    # ── Strategy 2: Rio-managed profile (always works) ─────────────────────
    user_data_dir = _get_persistent_user_data_dir(profile)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "headless": False,
        "no_viewport": True,
        "args": base_launch_args,
    }
    if exe:
        kwargs["executable_path"] = exe

    log.info(
        "browser.launching_playwright",
        browser=browser,
        exe=exe or "playwright-bundled",
        profile=profile or "default",
        user_data_dir=str(user_data_dir),
    )

    _context = await pw.chromium.launch_persistent_context(
        str(user_data_dir),
        **kwargs,
    )
    _page = _context.pages[0] if _context.pages else await _context.new_page()

    log.info("browser.launched", title=await _page.title(), url=_page.url)
    return _page


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

async def browser_connect(
    cdp_url: str = "http://127.0.0.1:9222",
    browser: str = "auto",
    profile: str = "",
) -> dict[str, Any]:
    """Connect to (or auto-launch) a browser.

    First tries to connect via CDP (for users who manually launched Chrome
    with --remote-debugging-port=9222). If that fails, falls back to
    Playwright's launch_persistent_context which works even when Chrome is
    already running — it opens a separate browser window using a dedicated
    Rio profile directory, avoiding the Chrome single-instance conflict.

    Args:
        cdp_url:  CDP endpoint to try first. Default http://127.0.0.1:9222
        browser:  "auto" | "chrome" | "chromium" | "edge" | "brave"
        profile:  Profile name for the Playwright-managed data dir.
                  e.g. "Default", "Work", "Profile 1". Leave empty for
                  the shared Rio automation profile.
    """
    cdp_url = cdp_url.replace("localhost", "127.0.0.1")

    # Try CDP connect or reuse existing page
    page = await _ensure_connection(cdp_url)
    if page is not None:
        try:
            return {
                "success": True,
                "result": {
                    "title": await page.title(),
                    "url": page.url,
                    "mode": "cdp",
                    "browser": browser,
                    "profile": profile or "default",
                },
            }
        except Exception:
            pass

    # Fall back to Playwright-managed launch (bypasses Chrome single-instance)
    try:
        page = await _launch_with_playwright(browser=browser, profile=profile)
        return {
            "success": True,
            "result": {
                "title": await page.title(),
                "url": page.url,
                "mode": "playwright",
                "browser": browser,
                "profile": profile or "default",
                "data_dir": str(_get_persistent_user_data_dir(profile)),
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "error": (
                f"CDP connect failed ({cdp_url}) and Playwright launch also failed: {exc}\n"
                f"browser={browser!r}, profile={profile!r}"
            ),
        }


async def browser_evaluate(javascript: str, cdp_url: str = "http://127.0.0.1:9222") -> dict[str, Any]:
    """Execute JavaScript in the browser page and return the result."""
    try:
        page = await _ensure_connection(cdp_url)
        if page is None:
            return {"success": False, "error": "No browser connected. Call browser_connect first."}
        result = await page.evaluate(javascript)
        return {
            "success": True,
            "result": str(result)[:5000] if result is not None else None,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_fill_form(
    selector: str,
    value: str,
    cdp_url: str = "http://127.0.0.1:9222",
) -> dict[str, Any]:
    """Fill a form field identified by CSS selector."""
    try:
        page = await _ensure_connection(cdp_url)
        if page is None:
            return {"success": False, "error": "No browser connected. Call browser_connect first."}
        await page.fill(selector, value, timeout=10000)
        return {"success": True, "result": f"Filled '{selector}' with value"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_click_element(
    selector: str,
    cdp_url: str = "http://127.0.0.1:9222",
) -> dict[str, Any]:
    """Click an element identified by CSS selector."""
    try:
        page = await _ensure_connection(cdp_url)
        if page is None:
            return {"success": False, "error": "No browser connected. Call browser_connect first."}
        await page.click(selector, timeout=10000)
        return {"success": True, "result": f"Clicked '{selector}'"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_extract_text(
    selector: str,
    cdp_url: str = "http://127.0.0.1:9222",
) -> dict[str, Any]:
    """Extract text content from an element by CSS selector."""
    try:
        page = await _ensure_connection(cdp_url)
        if page is None:
            return {"success": False, "error": "No browser connected. Call browser_connect first."}
        element = await page.query_selector(selector)
        if element is None:
            return {"success": False, "error": f"Element not found: {selector}"}
        text = await element.text_content()
        return {
            "success": True,
            "result": (text or "")[:5000],
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_wait_for(
    selector: str,
    timeout: int = 30000,
    cdp_url: str = "http://127.0.0.1:9222",
) -> dict[str, Any]:
    """Wait for a CSS selector to appear on the page."""
    try:
        page = await _ensure_connection(cdp_url)
        if page is None:
            return {"success": False, "error": "No browser connected. Call browser_connect first."}
        await page.wait_for_selector(selector, timeout=timeout)
        return {"success": True, "result": f"Element '{selector}' found"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_screenshot(
    cdp_url: str = "http://127.0.0.1:9222",
    full_page: bool = False,
) -> dict[str, Any]:
    """Take a screenshot of the current browser page."""
    try:
        page = await _ensure_connection(cdp_url)
        if page is None:
            return {"success": False, "error": "No browser connected. Call browser_connect first."}
        buf = await page.screenshot(full_page=full_page)
        return {
            "success": True,
            "result": {
                "image_base64": base64.b64encode(buf).decode(),
                "size": len(buf),
                "title": await page.title(),
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_navigate(
    url: str,
    cdp_url: str = "http://127.0.0.1:9222",
) -> dict[str, Any]:
    """Navigate the browser to a URL."""
    try:
        page = await _ensure_connection(cdp_url)
        if page is None:
            return {"success": False, "error": "No browser connected. Call browser_connect first."}
        await page.goto(url, timeout=30000, wait_until="domcontentloaded")
        return {
            "success": True,
            "result": {
                "title": await page.title(),
                "url": page.url,
            },
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def cleanup():
    """Close the Playwright browser context and stop Playwright."""
    global _playwright, _context, _page
    if _context:
        try:
            await _context.close()
        except Exception:
            pass
        _context = None
        _page = None
    if _playwright:
        try:
            await _playwright.stop()
        except Exception:
            pass
        _playwright = None
