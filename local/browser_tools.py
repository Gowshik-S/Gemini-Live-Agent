"""
Rio Local — Playwright CDP Browser Tools (E1)

Provides DOM-level browser automation alongside Computer Use vision.
Uses Playwright's CDP connection to interact with Chromium-based browsers
(Chrome, Edge, Brave) already running on the user's desktop.

Tools:
  - browser_connect: Connect to a running browser via CDP
  - browser_evaluate: Execute JavaScript in the page
  - browser_fill_form: Fill a form field by CSS selector
  - browser_click_element: Click an element by CSS selector
  - browser_extract_text: Extract text from an element
  - browser_wait_for: Wait for a selector to appear
  - browser_screenshot: Take a page screenshot
  - browser_navigate: Navigate to a URL

The browser must be launched with remote debugging enabled:
    chrome.exe --remote-debugging-port=9222
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Lazy-load — playwright is optional
_playwright = None
_browser = None
_page = None


async def _ensure_connection(cdp_url: str = "http://localhost:9222") -> tuple:
    """Connect to a running browser via CDP. Returns (browser, page)."""
    global _playwright, _browser, _page

    if _page is not None:
        try:
            await _page.title()
            return _browser, _page
        except Exception:
            _page = None
            _browser = None

    from playwright.async_api import async_playwright

    if _playwright is None:
        _playwright = await async_playwright().start()

    _browser = await _playwright.chromium.connect_over_cdp(cdp_url)
    contexts = _browser.contexts
    if contexts and contexts[0].pages:
        _page = contexts[0].pages[0]
    else:
        context = contexts[0] if contexts else await _browser.new_context()
        _page = await context.new_page()

    log.info("browser.connected", url=cdp_url, title=await _page.title())
    return _browser, _page


async def browser_connect(cdp_url: str = "http://localhost:9222") -> dict[str, Any]:
    """Connect to a running browser via Chrome DevTools Protocol.

    The browser must be launched with --remote-debugging-port=9222.
    """
    try:
        _, page = await _ensure_connection(cdp_url)
        return {
            "success": True,
            "result": {
                "title": await page.title(),
                "url": page.url,
            },
        }
    except Exception as exc:
        return {"success": False, "error": f"Failed to connect: {exc}"}


async def browser_evaluate(javascript: str, cdp_url: str = "http://localhost:9222") -> dict[str, Any]:
    """Execute JavaScript in the browser page and return the result."""
    try:
        _, page = await _ensure_connection(cdp_url)
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
    cdp_url: str = "http://localhost:9222",
) -> dict[str, Any]:
    """Fill a form field identified by CSS selector."""
    try:
        _, page = await _ensure_connection(cdp_url)
        await page.fill(selector, value, timeout=10000)
        return {"success": True, "result": f"Filled '{selector}' with value"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_click_element(
    selector: str,
    cdp_url: str = "http://localhost:9222",
) -> dict[str, Any]:
    """Click an element identified by CSS selector."""
    try:
        _, page = await _ensure_connection(cdp_url)
        await page.click(selector, timeout=10000)
        return {"success": True, "result": f"Clicked '{selector}'"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_extract_text(
    selector: str,
    cdp_url: str = "http://localhost:9222",
) -> dict[str, Any]:
    """Extract text content from an element by CSS selector."""
    try:
        _, page = await _ensure_connection(cdp_url)
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
    cdp_url: str = "http://localhost:9222",
) -> dict[str, Any]:
    """Wait for a CSS selector to appear on the page."""
    try:
        _, page = await _ensure_connection(cdp_url)
        await page.wait_for_selector(selector, timeout=timeout)
        return {"success": True, "result": f"Element '{selector}' found"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


async def browser_screenshot(
    cdp_url: str = "http://localhost:9222",
    full_page: bool = False,
) -> dict[str, Any]:
    """Take a screenshot of the current browser page."""
    try:
        _, page = await _ensure_connection(cdp_url)
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
    cdp_url: str = "http://localhost:9222",
) -> dict[str, Any]:
    """Navigate the browser to a URL."""
    try:
        _, page = await _ensure_connection(cdp_url)
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
    """Close the Playwright connection."""
    global _playwright, _browser, _page
    if _browser:
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
        _page = None
    if _playwright:
        try:
            await _playwright.stop()
        except Exception:
            pass
        _playwright = None
