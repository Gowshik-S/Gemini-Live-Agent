"""
Rio Local -- UI Navigator (Live API)

Continuously streams JPEG frames to Gemini Live API and requests structured
UI action proposals. When confidence is high, optionally confirms click
coordinates via the Computer Use model before execution.

Action schema:
  {
    "action": str,
    "element": str,
    "coordinates": {"x": int, "y": int} | null,
    "confidence": float,
    "context": "browser" | "os"
  }
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import structlog

log = structlog.get_logger(__name__)


@dataclass
class NavigatorAction:
    action: str
    element: str
    coordinates: dict[str, int] | None
    confidence: float
    context: str


class UINavigator:
    """Real-time UI navigator powered by Gemini Live API."""

    _RECONNECT_DELAY_SECONDS = 2.0

    def __init__(
        self,
        *,
        tool_executor,
        screen_navigator,
        model: str = "gemini-2.5-flash",
        fps: float = 10.0,
        confidence_threshold: float = 0.85,
        analyze_every_n_frames: int = 3,
        emit_action: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None,
        genai_client=None,
        click_tool: str = "smart_click",
    ) -> None:
        self._tool_executor = tool_executor
        self._screen_navigator = screen_navigator
        self._model = model
        self._fps = max(0.1, float(fps))
        self._confidence_threshold = float(confidence_threshold)
        self._analyze_every_n_frames = max(1, int(analyze_every_n_frames))
        self._emit_action = emit_action
        self._click_tool = click_tool

        self._client = genai_client
        self._task: asyncio.Task | None = None
        self._running = False
        self._frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=4)
        self._mid_response = False

    @property
    def running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    @property
    def fps(self) -> float:
        return self._fps

    def _read_env(self, name: str) -> str:
        value = os.environ.get(name, "").strip()
        if value:
            return value

        env_file = Path(__file__).resolve().parent.parent / "cloud" / ".env"
        if not env_file.is_file():
            return ""

        try:
            for raw_line in env_file.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                if key.strip() != name:
                    continue
                value = raw_value.strip().strip('"').strip("'")
                if value:
                    os.environ[name] = value
                return value
        except OSError:
            pass

        return ""

    def _build_genai_client(self):
        from google import genai

        use_vertex = (self._read_env("GOOGLE_GENAI_USE_VERTEXAI") or "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        project = self._read_env("GOOGLE_CLOUD_PROJECT")
        location = self._read_env("GOOGLE_CLOUD_LOCATION") or "global"

        if use_vertex and project:
            return genai.Client(vertexai=True, project=project, location=location)

        api_key = self._read_env("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for UI Navigator")
        return genai.Client(api_key=api_key)

    async def start(self) -> None:
        if self.running:
            return

        self._client = self._build_genai_client()
        self._running = True
        self._task = asyncio.create_task(self._run(), name="ui_navigator")
        log.info(
            "ui_navigator.started",
            model=self._model,
            confidence_threshold=self._confidence_threshold,
            analyze_every_n_frames=self._analyze_every_n_frames,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._client = None
        self._clear_queue()
        log.info("ui_navigator.stopped")

    def _clear_queue(self) -> None:
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def enqueue_frame(self, jpeg: bytes) -> None:
        if not self.running:
            return
        try:
            self._frame_queue.put_nowait(jpeg)
        except asyncio.QueueFull:
            # Keep latest frame for real-time behavior.
            try:
                _ = self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._frame_queue.put_nowait(jpeg)
            except asyncio.QueueFull:
                pass

    async def _run(self) -> None:
        from google.genai import types

        if self._client is None:
            return

        system_instruction = (
            "You are a UI navigator. Analyze incoming desktop screenshots and emit exactly "
            "one function call to emit_ui_action per analysis request. "
            "Set action to 'none' if there is no clear interaction opportunity. "
            "Detect both browser UI and OS UI such as desktop, taskbar, start menu, and native apps. "
            "Do not output markdown, prose, or extra text."
        )

        action_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "element": {"type": "string"},
                "coordinates": {
                    "type": "object",
                    "nullable": True,
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                },
                "confidence": {"type": "number"},
                "context": {
                    "type": "string",
                    "enum": ["browser", "os"],
                },
            },
            "required": ["action", "element", "coordinates", "confidence", "context"],
        }

        emit_action_fn = types.FunctionDeclaration(
            name="emit_ui_action",
            description=(
                "Emit one UI navigation action candidate for the latest frame."
            ),
            parameters=action_schema,
        )

        live_config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            system_instruction=types.Content(parts=[types.Part.from_text(text=system_instruction)]),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            tools=[types.Tool(function_declarations=[emit_action_fn])],
        )

        while self._running:
            try:
                async with self._client.aio.live.connect(model=self._model, config=live_config) as session:
                    log.info("ui_navigator.live_connected", model=self._model)
                    recv_task = asyncio.create_task(self._receive_loop(session), name="ui_nav_receive")
                    send_task = asyncio.create_task(self._send_loop(session), name="ui_nav_send")
                    done, pending = await asyncio.wait(
                        {recv_task, send_task},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for task in pending:
                        task.cancel()
                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("ui_navigator.live_disconnected", error=str(exc))
                await asyncio.sleep(self._RECONNECT_DELAY_SECONDS)

    async def _send_loop(self, session) -> None:
        from google.genai import types

        frame_count = 0
        while self._running:
            frame = await self._frame_queue.get()
            await session.send_realtime_input(
                video=types.Blob(data=frame, mime_type="image/jpeg"),
            )
            frame_count += 1

            # Ask for action extraction periodically while keeping the frame stream continuous.
            if frame_count % self._analyze_every_n_frames == 0 and not self._mid_response:
                # Mark busy until turn_complete (or disconnect) to avoid overlapping prompts.
                self._mid_response = True
                await session.send_client_content(
                    turns=types.Content(
                        role="user",
                        parts=[types.Part(text="Analyze latest frame and call emit_ui_action exactly once.")],
                    ),
                    turn_complete=True,
                )

    async def _receive_loop(self, session) -> None:
        from google.genai import types

        async for response in session.receive():
            tc = getattr(response, "tool_call", None)
            if tc and tc.function_calls:
                fn_responses = []
                for fc in tc.function_calls:
                    if fc.name != "emit_ui_action":
                        fn_responses.append(
                            types.FunctionResponse(
                                name=fc.name,
                                response={"success": False, "error": "Unknown function"},
                                id=fc.id,
                            )
                        )
                        continue

                    action = self._action_from_payload(dict(fc.args or {}))
                    if action is not None:
                        await self._handle_action(action)
                        fn_responses.append(
                            types.FunctionResponse(
                                name=fc.name,
                                response={"success": True},
                                id=fc.id,
                            )
                        )
                    else:
                        fn_responses.append(
                            types.FunctionResponse(
                                name=fc.name,
                                response={"success": False, "error": "Invalid schema payload"},
                                id=fc.id,
                            )
                        )

                # Ack function calls so the model can continue its turn loop.
                await session.send_tool_response(function_responses=fn_responses)
                continue

            # Backward-compatible fallback: parse plain text JSON if model returns text.
            sc = getattr(response, "server_content", None)
            if sc and getattr(sc, "turn_complete", False):
                self._mid_response = False
            if not sc or not sc.model_turn:
                continue
            self._mid_response = True
            for part in sc.model_turn.parts or []:
                text = getattr(part, "text", None)
                if not text:
                    continue
                action = self._parse_action(text)
                if action is None:
                    continue
                await self._handle_action(action)

    def _action_from_payload(self, payload: dict[str, Any]) -> NavigatorAction | None:
        if not isinstance(payload, dict):
            return None

        action = str(payload.get("action", "none")).strip().lower()
        element = str(payload.get("element", "")).strip()

        coords_raw = payload.get("coordinates")
        coordinates: dict[str, int] | None = None
        if isinstance(coords_raw, dict) and "x" in coords_raw and "y" in coords_raw:
            try:
                coordinates = {"x": int(coords_raw["x"]), "y": int(coords_raw["y"])}
            except Exception:
                coordinates = None

        try:
            confidence = float(payload.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        context = str(payload.get("context", "os")).strip().lower()
        if context not in {"browser", "os"}:
            context = self._fallback_context()

        return NavigatorAction(
            action=action,
            element=element,
            coordinates=coordinates,
            confidence=confidence,
            context=context,
        )

    def _parse_action(self, text: str) -> NavigatorAction | None:
        payload = self._extract_json(text)
        if payload is None:
            return None
        return self._action_from_payload(payload)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None

        # Strip code fences if present.
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def _fallback_context(self) -> str:
        # Conservative fallback when model omits/invalidates context.
        return "os"

    async def _emit(self, payload: dict[str, Any]) -> None:
        if self._emit_action is None:
            return
        try:
            await self._emit_action(payload)
        except Exception:
            log.debug("ui_navigator.emit_failed")

    async def _handle_action(self, action: NavigatorAction) -> None:
        schema = {
            "action": action.action,
            "element": action.element,
            "coordinates": action.coordinates,
            "confidence": round(action.confidence, 4),
            "context": action.context,
        }
        log.info("ui_navigator.action_schema", **schema)
        await self._emit({"type": "ui_navigator_action", **schema})

        if action.action in {"none", "no_action", "wait"}:
            return
        if action.confidence <= self._confidence_threshold:
            return

        result = await self._execute_action(action)
        await self._emit({
            "type": "ui_navigator_execution",
            "action": action.action,
            "context": action.context,
            "success": bool(result.get("success", False)),
            "result": result,
        })

    async def _execute_action(self, action: NavigatorAction) -> dict[str, Any]:
        context = action.context

        # High-confidence clicks are confirmed with Computer Use before execution.
        confirmed_xy: tuple[int, int] | None = None
        if action.action in {"click", "left_click", "double_click", "right_click"}:
            if self._click_tool == "smart_click" and action.element:
                # Delegate entirely to the smart_click tool
                return await self._tool_executor.execute(
                    "smart_click",
                    {
                        "target": action.element,
                        "action": action.action,
                        "clicks": 2 if action.action == "double_click" else 1,
                    }
                )
            elif self._click_tool == "screen_click":
                confirmed_xy = await self._confirm_click_coordinates(action)

        if context == "browser":
            return await self._execute_browser_action(action, confirmed_xy)
        return await self._execute_os_action(action, confirmed_xy)

    async def _confirm_click_coordinates(self, action: NavigatorAction) -> tuple[int, int] | None:
        target = action.element or "target UI element"
        try:
            ground = await self._tool_executor.confirm_click_target_single_snapshot(target)
            if ground.get("success"):
                return int(ground["x"]), int(ground["y"])
        except Exception as exc:
            log.debug("ui_navigator.cu_confirm_failed", error=str(exc))

        if action.coordinates:
            # Fall back to screenshot coordinates if CU confirmation is unavailable.
            return int(action.coordinates["x"]), int(action.coordinates["y"])
        return None

    @staticmethod
    def _looks_like_selector(value: str) -> bool:
        v = value.strip()
        if not v:
            return False
        return v.startswith(("#", ".", "[", "//", "xpath=")) or " " in v and "[" in v

    async def _execute_browser_action(
        self,
        action: NavigatorAction,
        confirmed_xy: tuple[int, int] | None,
    ) -> dict[str, Any]:
        # Ensure Playwright session is available.
        await self._tool_executor.execute("browser_connect", {})

        if action.action in {"navigate", "open", "open_url"} and action.element:
            return await self._tool_executor.execute("browser_navigate", {"url": action.element})

        if action.action in {"click", "left_click", "double_click", "right_click"}:
            if self._looks_like_selector(action.element):
                return await self._tool_executor.execute(
                    "browser_click_element",
                    {"selector": action.element},
                )

            if action.element:
                js = (
                    "(() => {"
                    "const want = " + json.dumps(action.element.lower()) + ";"
                    "const nodes = Array.from(document.querySelectorAll('a,button,[role=button],input,textarea,div,span'));"
                    "const hit = nodes.find(n => (n.innerText || n.value || '').toLowerCase().includes(want));"
                    "if (!hit) return {success:false,error:'element text not found'};"
                    "hit.scrollIntoView({block:'center',inline:'center'});"
                    "hit.click();"
                    "return {success:true,method:'playwright_text_click'};"
                    "})()"
                )
                result = await self._tool_executor.execute("browser_evaluate", {"javascript": js})
                if result.get("success"):
                    return result

            # If browser automation couldn't localize the target, fall back to confirmed OS click.
            if confirmed_xy:
                x, y = confirmed_xy
                return await self._screen_navigator.click_absolute(x, y)

            return {"success": False, "error": "No browser click target could be resolved"}

        if action.action == "type":
            text_value = action.element
            js = (
                "(() => {"
                "const el = document.activeElement;"
                "if (!el) return {success:false,error:'no active element'};"
                "if ('value' in el) { el.value = " + json.dumps(text_value) + ";"
                "el.dispatchEvent(new Event('input', {bubbles:true}));"
                "return {success:true,method:'playwright_set_value'}; }"
                "return {success:false,error:'active element not text-editable'};"
                "})()"
            )
            return await self._tool_executor.execute("browser_evaluate", {"javascript": js})

        if action.action == "scroll":
            dx = 0
            dy = 0
            if action.coordinates:
                dx = int(action.coordinates.get("x", 0))
                dy = int(action.coordinates.get("y", 0))
            # Fall back to element as numeric delta when coordinates are absent.
            elif action.element:
                try:
                    dy = int(action.element)
                except ValueError:
                    dy = 0
            scroll_js = f"window.scrollBy({dx}, {dy}); ({'{'}success:true,method:'window.scrollBy',dx:{dx},dy:{dy}{'}'})"
            return await self._tool_executor.execute("browser_evaluate", {"javascript": scroll_js})

        # Browser hotkeys are still OS-level key events.
        if action.action == "hotkey" and action.element:
            return await self._tool_executor.execute("screen_hotkey", {"keys": action.element})

        return {"success": False, "error": f"Unsupported browser action: {action.action}"}

    async def _execute_os_action(
        self,
        action: NavigatorAction,
        confirmed_xy: tuple[int, int] | None,
    ) -> dict[str, Any]:
        if action.action in {"click", "left_click", "double_click", "right_click"}:
            button = "right" if action.action == "right_click" else "left"
            clicks = 2 if action.action == "double_click" else 1
            if confirmed_xy:
                x, y = confirmed_xy
                return await self._screen_navigator.click_absolute(x, y, button=button, clicks=clicks)
            if action.coordinates:
                return await self._tool_executor.execute(
                    "screen_click",
                    {
                        "x": int(action.coordinates["x"]),
                        "y": int(action.coordinates["y"]),
                        "button": button,
                        "clicks": clicks,
                    },
                )
            return {"success": False, "error": "Missing coordinates for click"}

        if action.action == "move" and action.coordinates:
            return await self._tool_executor.execute(
                "screen_move",
                {"x": int(action.coordinates["x"]), "y": int(action.coordinates["y"])}
            )

        if action.action == "type":
            return await self._tool_executor.execute("screen_type", {"text": action.element})

        if action.action == "hotkey":
            return await self._tool_executor.execute("screen_hotkey", {"keys": action.element})

        if action.action in {"focus", "focus_window"}:
            return await self._tool_executor.execute("focus_window", {"title": action.element})

        return {"success": False, "error": f"Unsupported os action: {action.action}"}

