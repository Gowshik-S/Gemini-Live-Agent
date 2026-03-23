"""
Rio Local — Browser Agent (Playwright + Gemini Computer Use)

Specialized autonomous agent for web automation with two-tier strategy:
1. Playwright (primary) - Fast DOM-level automation
2. Gemini Computer Use (fallback) - Visual grounding for complex interactions

Uses browser automation tools preferentially, escalates to Computer Use only when needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Optional
import subprocess
import time

import structlog

from constants import MODEL_COMPUTER_USE, MODEL_FLASH

log = structlog.get_logger(__name__)

# Attempt imports — graceful degradation if not installed
try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

try:
    from playwright.async_api import async_playwright, Browser, Page
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


def _load_browser_agent_prompt() -> str:
    """Load browser agent system prompt from markdown file."""
    prompt_file = Path(__file__).parent.parent / "browser_agent_prompt.md"
    if prompt_file.exists():
        try:
            content = prompt_file.read_text(encoding="utf-8")
            # Extract the prompt from the markdown (between triple quotes)
            if '"""' in content:
                parts = content.split('"""')
                if len(parts) >= 3:
                    return parts[1].strip()
            return content
        except Exception as exc:
            log.warning("browser_agent.prompt_load_failed", error=str(exc))
    
    # Fallback to inline prompt
    return """You are Rio's Browser Agent — a specialized autonomous agent for web automation.
You control web browsers using Playwright for DOM-level precision, with fallback
to Gemini Computer Use for visual grounding when Playwright cannot handle the task.

Execute web automation tasks autonomously. Use Playwright tools first (browser_connect,
browser_navigate, browser_click_element, browser_fill_form, browser_extract_text).
If Playwright fails 2x on the same element, escalate to Computer Use for visual grounding.

Always return structured results with success status, result description, and any extracted data."""


BROWSER_AGENT_SYSTEM_PROMPT = _load_browser_agent_prompt()

# Viewport size — the model sees screenshots at this resolution
# and returns coordinates in this space.  Must match context creation.
VIEWPORT_W = 1280
VIEWPORT_H = 720

COMPUTER_USE_SYSTEM = """\
You are a precise browser automation agent using visual grounding.
You see a screenshot of a web browser at {w}x{h} pixels.

Given a goal, analyze the screenshot and return ONE JSON action.

Available actions (coordinates are in screenshot pixel space):
  {{"action": "click", "x": <int>, "y": <int>}}
  {{"action": "double_click", "x": <int>, "y": <int>}}
  {{"action": "right_click", "x": <int>, "y": <int>}}
  {{"action": "type", "text": "..."}}
  {{"action": "press", "key": "Enter"|"Tab"|"Escape"|"Backspace"|...}}
  {{"action": "hotkey", "keys": "Control+a"|"Control+c"|...}}
  {{"action": "scroll", "x": <int>, "y": <int>, "direction": "up"|"down", "amount": 3}}
  {{"action": "goto", "url": "https://..."}}
  {{"action": "wait", "ms": 1000}}
  {{"action": "done", "result": "description of what was achieved"}}
  {{"action": "failed", "error": "description of what went wrong"}}

CRITICAL RULES:
- For click/double_click/right_click: specify EXACT pixel x,y of the element center
- Look at the screenshot carefully — identify buttons, links, inputs by their visual position
- For text input: first click the input field, then use "type" action
- After typing in a search field, use "press" with key "Enter" to submit
- If a page is loading, use "wait"
- When the goal is achieved, use "done"
- If stuck after 3 attempts on same element, use "failed"

Return ONLY the JSON object. No markdown fences, no explanation.
""".format(w=VIEWPORT_W, h=VIEWPORT_H)


class BrowserAgent:
    """Autonomous browser agent using Gemini Computer Use + Playwright.

    Uses coordinate-based visual grounding instead of CSS selectors
    for reliable element targeting across any website.

    Usage::

        agent = BrowserAgent(api_key="...")
        await agent.start()
        result = await agent.execute_step(
            "Go to github.com and search for 'gemini live'",
            expected_outcome="Search results page with relevant repos",
        )
        await agent.stop()
    """

    MAX_ACTION_ROUNDS = 20  # Max model→action→verify cycles per step

    def __init__(
        self,
        api_key: str = "",
        headless: bool = False,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._headless = headless
        self._client: Optional[genai.Client] = None
        self._pw = None
        self._browser: Optional[Browser] = None
        self._context = None
        self._page: Optional[Page] = None
        self._running = False
        self.browser_mode: Optional[str] = None
        self.last_start_error: str = ""
        self._bootstrap_last_attempt: dict[tuple[str, int], float] = {}
        self._log = log.bind(component="browser_agent")

    @property
    def available(self) -> bool:
        return _GENAI_AVAILABLE and _PLAYWRIGHT_AVAILABLE

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_chrome_user_data_dir() -> Path:
        if sys.platform == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
            if not local_appdata:
                raise RuntimeError("LOCALAPPDATA is not set")
            return Path(local_appdata) / "Google" / "Chrome" / "User Data"
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
        return Path.home() / ".config" / "google-chrome"

    @staticmethod
    def _candidate_cdp_urls() -> list[str]:
        """Return CDP endpoints to probe in priority order.

        Supports explicit overrides for demo reliability:
        - RIO_BROWSER_CDP_URLS="http://localhost:9222,http://localhost:9223"
        - RIO_BROWSER_CDP_URL="http://localhost:9222"
        """
        urls: list[str] = []

        env_urls = os.environ.get("RIO_BROWSER_CDP_URLS", "").strip()
        if env_urls:
            for item in env_urls.split(","):
                url = item.strip()
                if url:
                    urls.append(url)

        single_url = os.environ.get("RIO_BROWSER_CDP_URL", "").strip()
        if single_url:
            urls.append(single_url)

        # Demo-friendly defaults for common Chromium-family debug ports.
        if not urls:
            urls = [
                "http://localhost:9222",
                "http://127.0.0.1:9222",
                "http://localhost:9223",
                "http://127.0.0.1:9223",
                "http://localhost:9224",
                "http://127.0.0.1:9224",
            ]

        # Preserve order but remove duplicates.
        seen: set[str] = set()
        deduped: list[str] = []
        for url in urls:
            normalized = url.replace("127.0.0.1", "localhost")
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(url)
        return deduped

    @staticmethod
    def _cdp_only_mode_enabled() -> bool:
        """Return True when BrowserAgent must only attach to existing CDP browsers.

        Default is True for demo/operator safety: do not silently launch a new
        Playwright Chromium profile when existing browser CDP attach fails.
        Set RIO_BROWSER_ALLOW_PERSISTENT_FALLBACK=1 to re-enable fallback.
        """
        allow_fallback = os.environ.get("RIO_BROWSER_ALLOW_PERSISTENT_FALLBACK", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        return not allow_fallback

    @staticmethod
    async def _detect_browser_label(page: Page) -> str:
        """Best-effort browser label from user agent for logging."""
        try:
            ua = await page.evaluate("() => navigator.userAgent || ''")
            ua = str(ua)
            if "Edg/" in ua:
                return "edge"
            if "Brave/" in ua:
                return "brave"
            if "Chrome/" in ua:
                return "chrome"
            if "Chromium/" in ua:
                return "chromium"
        except Exception:
            pass
        return "chromium-family"

    @staticmethod
    def _extract_debug_port(cdp_url: str) -> int:
        try:
            return int(cdp_url.rsplit(":", 1)[-1])
        except Exception:
            return 9222

    @staticmethod
    def _bootstrap_cooldown_seconds() -> float:
        return float(os.environ.get("RIO_BROWSER_CDP_BOOTSTRAP_COOLDOWN_S", "30") or 30)

    @staticmethod
    def _get_bootstrap_user_data_dir(browser_name: str) -> Path:
        if sys.platform == "win32":
            local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
            if local_appdata:
                base = Path(local_appdata) / "Rio" / "CDPBootstrap"
            else:
                base = Path.home() / "AppData" / "Local" / "Rio" / "CDPBootstrap"
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support" / "Rio" / "CDPBootstrap"
        else:
            base = Path.home() / ".local" / "share" / "rio" / "cdp-bootstrap"
        safe_name = (browser_name or "auto").strip().lower() or "auto"
        return base / safe_name

    def _launch_browser_with_cdp(self, port: int, browser_name: str = "auto") -> bool:
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
            os.path.expandvars(r"%ProgramFiles%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        ]

        exe = ""
        for path in candidates:
            if path and os.path.exists(path):
                exe = path
                break
        if not exe:
            exe = "chrome"

        key = (str(exe).lower(), port)
        now = time.monotonic()
        cooldown_s = self._bootstrap_cooldown_seconds()
        last = self._bootstrap_last_attempt.get(key, 0.0)
        if last and (now - last) < cooldown_s:
            self._log.info(
                "browser_agent.cdp_bootstrap_skipped_recent",
                exe=exe,
                port=port,
                retry_after_s=round(cooldown_s - (now - last), 1),
            )
            return True

        self._bootstrap_last_attempt[key] = now

        user_data_dir = self._get_bootstrap_user_data_dir(browser_name)
        user_data_dir.mkdir(parents=True, exist_ok=True)

        args = [
            exe,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={str(user_data_dir)}",
            "--no-first-run",
            "--no-default-browser-check",
            "--new-window",
            "about:blank",
        ]

        profile_name = os.environ.get("CHROME_PROFILE", "").strip()
        if profile_name:
            args.append(f"--profile-directory={profile_name}")

        try:
            kwargs: dict[str, object] = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "stdin": subprocess.DEVNULL,
            }
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            subprocess.Popen(args, **kwargs)
            self._log.info(
                "browser_agent.cdp_bootstrap_started",
                exe=exe,
                port=port,
                user_data_dir=str(user_data_dir),
            )
            return True
        except Exception as exc:
            self._log.warning(
                "browser_agent.cdp_bootstrap_failed",
                exe=exe,
                port=port,
                error=str(exc),
            )
            return False
    async def get_browser_page(self) -> tuple[object, Page]:
        async def _pick_page(context) -> Page:
            if context.pages:
                # Prefer a normal web tab over chrome:// or extension pages.
                for p in context.pages:
                    u = (p.url or "").lower()
                    if u.startswith("http://") or u.startswith("https://"):
                        return p
                return context.pages[0]
            return await context.new_page()

        if self._pw is None:
            self._pw = await async_playwright().start()

        cdp_errors: dict[str, str] = {}
        for cdp_url in self._candidate_cdp_urls():
            try:
                browser = await self._pw.chromium.connect_over_cdp(cdp_url)
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = await _pick_page(context)
                with contextlib.suppress(Exception):
                    await page.bring_to_front()
                self.browser_mode = "cdp"
                browser_label = await self._detect_browser_label(page)
                self._log.info(
                    "browser_agent.browser_mode_selected",
                    mode=self.browser_mode,
                    cdp_url=cdp_url,
                    browser=browser_label,
                )
                return browser, page
            except Exception as exc:
                cdp_errors[cdp_url] = str(exc)

        bootstrap_enabled = os.environ.get("RIO_BROWSER_AUTO_BOOTSTRAP_CDP", "1").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if bootstrap_enabled:
            for cdp_url in self._candidate_cdp_urls():
                port = self._extract_debug_port(cdp_url)
                if not self._launch_browser_with_cdp(port, browser_name="auto"):
                    continue
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.5)
                    try:
                        browser = await self._pw.chromium.connect_over_cdp(cdp_url)
                        context = browser.contexts[0] if browser.contexts else await browser.new_context()
                        page = await _pick_page(context)
                        with contextlib.suppress(Exception):
                            await page.bring_to_front()
                        self.browser_mode = "cdp"
                        browser_label = await self._detect_browser_label(page)
                        self._log.info(
                            "browser_agent.browser_mode_selected",
                            mode=self.browser_mode,
                            cdp_url=cdp_url,
                            browser=browser_label,
                            bootstrapped=True,
                        )
                        return browser, page
                    except Exception:
                        continue

        self._log.warning(
            "browser_agent.cdp_unavailable",
            tried_urls=list(cdp_errors.keys()),
            note=(
                "No CDP endpoint accepted a connection. For demo, launch your target browser "
                "with --remote-debugging-port=9222 (or set RIO_BROWSER_CDP_URLS to the correct port)."
            ),
        )

        cdp_error_summary = " | ".join(f"{url}: {err}" for url, err in cdp_errors.items())

        if self._cdp_only_mode_enabled():
            raise RuntimeError(
                "CDP attach failed for all detected endpoints and CDP-only mode is enabled. "
                "Launch your existing browser with --remote-debugging-port=9222 (or set RIO_BROWSER_CDP_URLS), "
                "then retry. "
                f"CDP errors: {cdp_error_summary}"
            )

        try:
            profile_name = os.environ.get("CHROME_PROFILE", "").strip() or "Default"
            user_data_dir = self._detect_chrome_user_data_dir()

            try:
                context = await self._pw.chromium.launch_persistent_context(
                    str(user_data_dir),
                    headless=False,
                    args=[f"--profile-directory={profile_name}"],
                )
                page = await _pick_page(context)
                with contextlib.suppress(Exception):
                    await page.bring_to_front()
                self.browser_mode = "persistent"
                self._log.info(
                    "browser_agent.browser_mode_selected",
                    mode=self.browser_mode,
                    user_data_dir=str(user_data_dir),
                    profile=profile_name,
                    note="Demo sanity check failed to get CDP mode; running persistent fallback.",
                )
                return context, page
            except Exception as persistent_exc:
                raise RuntimeError(
                    "Browser initialization failed. CDP attach failed and persistent profile launch failed. "
                    "Either launch Chrome with --remote-debugging-port=9222, or close Chrome so the profile lock is released. "
                    f"CDP errors: {cdp_error_summary}; persistent error: {persistent_exc}"
                ) from persistent_exc
        except Exception:
            raise

    async def start(self) -> bool:
        """Launch browser and initialize Gemini client."""
        if not self.available:
            self._log.warning("browser_agent.unavailable",
                              genai=_GENAI_AVAILABLE, playwright=_PLAYWRIGHT_AVAILABLE)
            self.last_start_error = "BrowserAgent dependencies are unavailable"
            return False

        try:
            self.last_start_error = ""
            self._client = genai.Client(api_key=self._api_key)

            browser_or_context, page = await self.get_browser_page()
            self._page = page
            self._context = page.context
            self._browser = browser_or_context if self.browser_mode == "cdp" else None

            await self._page.set_viewport_size({"width": VIEWPORT_W, "height": VIEWPORT_H})

            self._running = True
            self._log.info(
                "browser_agent.started",
                mode=self.browser_mode,
                model=MODEL_COMPUTER_USE,
                viewport=f"{VIEWPORT_W}x{VIEWPORT_H}",
            )
            return True
        except Exception as exc:
            self._log.exception("browser_agent.start_failed")
            self.last_start_error = str(exc)
            await self.stop()
            return False

    async def stop(self) -> None:
        """Shut down browser and clean up resources."""
        self._running = False

        # For CDP mode, do not close browser/context because this agent does not own them.
        # For persistent mode, the launched context is owned by this agent and should be closed.
        if self.browser_mode == "persistent" and self._context is not None:
            try:
                await self._context.close()
            except Exception:
                self._log.debug("browser_agent.context_close_failed")

        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                self._log.debug("browser_agent.playwright_stop_failed")

        self._browser = None
        self._pw = None
        self._page = None
        self._context = None
        self._client = None
        self.browser_mode = None
        self._log.info("browser_agent.stopped")

    # ------------------------------------------------------------------
    # Screenshot helper
    # ------------------------------------------------------------------

    async def _screenshot(self) -> Optional[bytes]:
        """Capture current page as PNG bytes (full viewport)."""
        if self._page is None:
            return None
        try:
            with contextlib.suppress(Exception):
                await self._page.bring_to_front()
            return await self._page.screenshot(type="png", full_page=False)
        except Exception:
            self._log.exception("browser_agent.screenshot_failed")
            return None

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    async def execute_step(
        self,
        goal: str,
        expected_outcome: str = "",
        scratchpad: Optional[dict] = None,
    ) -> dict:
        """Execute a browser step with visual grounding via Computer Use model.

        Returns:
            {"success": bool, "result": str, "error": str, "rounds": int}
        """
        if not self._running or self._page is None or self._client is None:
            return {"success": False, "error": "Browser agent not running", "rounds": 0}

        self._log.info("browser_agent.step_start", goal=goal[:100])

        # Build conversation history for multi-turn context
        history: list[types.Content] = []
        user_context = f"Goal: {goal}"
        if expected_outcome:
            user_context += f"\nExpected outcome: {expected_outcome}"
        if scratchpad:
            # Only include last 3 scratchpad items to avoid token bloat
            recent = dict(list(scratchpad.items())[-3:])
            user_context += f"\nContext from previous steps: {json.dumps(recent)}"

        last_actions: list[str] = []  # Track for stuck detection

        for round_num in range(1, self.MAX_ACTION_ROUNDS + 1):
            # 1. Capture screenshot
            screenshot = await self._screenshot()
            if screenshot is None:
                return {"success": False, "error": "Screenshot capture failed", "rounds": round_num}

            # 2. Build prompt with screenshot + context
            round_context = user_context
            if last_actions:
                round_context += f"\n\nPrevious actions this step: {last_actions[-5:]}"

            parts = [
                types.Part(inline_data=types.Blob(mime_type="image/png", data=screenshot)),
                types.Part(text=round_context),
            ]
            history.append(types.Content(role="user", parts=parts))

            # Keep history bounded (last 6 turns = 12 messages)
            if len(history) > 12:
                history = history[-12:]

            # 3. Send to Computer Use model
            try:
                response = await self._client.aio.models.generate_content(
                    model=MODEL_COMPUTER_USE,
                    contents=history,
                    config=types.GenerateContentConfig(
                        system_instruction=BROWSER_AGENT_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=512,
                    ),
                )
            except Exception:
                self._log.exception("browser_agent.model_error", round=round_num)
                # Fallback: retry with Flash if Computer Use fails
                try:
                    response = await self._client.aio.models.generate_content(
                        model=MODEL_FLASH,
                        contents=history,
                        config=types.GenerateContentConfig(
                            system_instruction=BROWSER_AGENT_SYSTEM_PROMPT,
                            temperature=0.2,
                            max_output_tokens=512,
                        ),
                    )
                except Exception:
                    return {"success": False, "error": "Model call failed", "rounds": round_num}

            # 4. Parse response
            action_json = self._parse_action(response)
            if action_json is None:
                self._log.warning("browser_agent.parse_failed", round=round_num)
                # Add model response to history for context
                if response.text:
                    history.append(types.Content(role="model", parts=[
                        types.Part(text=response.text)
                    ]))
                continue

            # Add model response to history
            history.append(types.Content(role="model", parts=[
                types.Part(text=json.dumps(action_json))
            ]))

            action = action_json.get("action", "")
            self._log.info("browser_agent.action",
                           round=round_num, action=action,
                           x=action_json.get("x"), y=action_json.get("y"))

            # 5. Terminal actions
            if action == "done":
                result = action_json.get("result", "Step completed")
                self._log.info("browser_agent.step_done", rounds=round_num, result=result[:100])
                return {"success": True, "result": result, "rounds": round_num}

            if action == "failed":
                error = action_json.get("error", "Step failed")
                self._log.warning("browser_agent.step_failed", rounds=round_num, error=error[:100])
                return {"success": False, "error": error, "rounds": round_num}

            # 6. Stuck detection — same action 3 times in a row
            action_sig = f"{action}:{action_json.get('x', '')}:{action_json.get('y', '')}:{action_json.get('text', '')}"
            last_actions.append(action_sig)
            if len(last_actions) >= 3 and len(set(last_actions[-3:])) == 1:
                self._log.warning("browser_agent.stuck", action=action_sig)
                return {"success": False, "error": f"Stuck: repeated {action} 3x at same target", "rounds": round_num}

            # 7. Execute the action
            try:
                await self._execute_action(action_json)
            except Exception as exc:
                self._log.warning("browser_agent.action_error", round=round_num, error=str(exc))
                user_context += f"\nRound {round_num} action '{action}' failed: {exc}"
                continue

            # Brief pause for page to settle
            await asyncio.sleep(0.5)

        return {"success": False, "error": "Max action rounds exceeded", "rounds": self.MAX_ACTION_ROUNDS}

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_action(self, response) -> Optional[dict]:
        """Extract JSON action from model response."""
        if not response.candidates:
            return None

        text = response.text or ""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self._log.warning("browser_agent.json_parse_error", text=text[:200])
            return None

    # ------------------------------------------------------------------
    # Playwright action execution (coordinate-based)
    # ------------------------------------------------------------------

    async def _execute_action(self, action_json: dict) -> None:
        """Execute a single Playwright action using pixel coordinates.

        The Computer Use model returns exact pixel coordinates where it
        sees the target element — much more reliable than CSS selectors.
        """
        if self._page is None:
            raise RuntimeError("No active page")

        with contextlib.suppress(Exception):
            await self._page.bring_to_front()

        action = action_json["action"]

        if action == "goto":
            url = action_json["url"]
            await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)

        elif action == "click":
            x = int(action_json["x"])
            y = int(action_json["y"])
            await self._page.mouse.click(x, y)

        elif action == "double_click":
            x = int(action_json["x"])
            y = int(action_json["y"])
            await self._page.mouse.dblclick(x, y)

        elif action == "right_click":
            x = int(action_json["x"])
            y = int(action_json["y"])
            await self._page.mouse.click(x, y, button="right")

        elif action == "type":
            text = action_json["text"]
            await self._page.keyboard.type(text, delay=30)

        elif action == "press":
            key = action_json["key"]
            await self._page.keyboard.press(key)

        elif action == "hotkey":
            keys = action_json["keys"]
            # "Control+a" → press("Control+a") which Playwright handles
            await self._page.keyboard.press(keys)

        elif action == "scroll":
            x = action_json.get("x", VIEWPORT_W // 2)
            y = action_json.get("y", VIEWPORT_H // 2)
            direction = action_json.get("direction", "down")
            amount = action_json.get("amount", 3)
            # Move mouse to position, then scroll
            await self._page.mouse.move(int(x), int(y))
            delta = -amount * 100 if direction == "up" else amount * 100
            await self._page.mouse.wheel(0, delta)

        elif action == "wait":
            ms = min(action_json.get("ms", 1000), 5000)
            await asyncio.sleep(ms / 1000)

        else:
            self._log.warning("browser_agent.unknown_action", action=action)

    # ------------------------------------------------------------------
    # Navigation convenience
    # ------------------------------------------------------------------

    async def goto(self, url: str) -> bool:
        """Navigate to a URL directly."""
        if self._page is None:
            return False
        try:
            await self._page.goto(url, wait_until="domcontentloaded", timeout=15000)
            return True
        except Exception:
            self._log.exception("browser_agent.goto_failed", url=url)
            return False

    async def get_page_title(self) -> str:
        """Return the current page title."""
        if self._page is None:
            return ""
        try:
            return await self._page.title()
        except Exception:
            return ""
