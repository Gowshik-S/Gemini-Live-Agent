"""
Rio Local — Orchestrator (Autonomous Execution Engine)

The orchestrator is the brain of Rio's autonomy:
  1. Receives a user goal
  2. Calls Gemini Pro to decompose into a step plan
  3. Dispatches each step to the appropriate sub-agent
  4. Verifies each step via screenshot + model
  5. Handles retries, re-planning, and completion notification

Sub-agents:
  - BrowserAgent  — Playwright web automation
  - WindowsAgent  — pywinauto native app control
  - ToolExecutor  — read/write/patch/run_command
  - ScreenNavigator — pyautogui screen interaction
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import structlog

from constants import MODEL_PRO, MODEL_FLASH, MODEL_COMPUTER_USE
from task_state import (
    Task,
    TaskStatus,
    TaskStore,
    Step,
    StepStatus,
    StepType,
    MAX_STEP_RETRIES,
)

# Model fallback
try:
    from model_fallback import (
        ModelFallbackChain,
        classify_error,
        get_diagnostic_message,
        ModelFailoverError,
    )
    _FALLBACK_AVAILABLE = True
except ImportError:
    _FALLBACK_AVAILABLE = False

log = structlog.get_logger(__name__)

# Attempt imports for sub-agents (graceful degradation)
try:
    from browser_agent import BrowserAgent
except ImportError:
    BrowserAgent = None

try:
    from windows_agent import WindowsAgent
except ImportError:
    WindowsAgent = None

try:
    from google import genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

PRO_MODEL = MODEL_PRO
FLASH_MODEL = MODEL_FLASH

# System prompt for vision-guided screen actions
VISION_SCREEN_SYSTEM = """\
You are a precise screen automation agent using visual grounding.
You see a screenshot of a desktop at its actual resolution.

Given a description of what to interact with, analyze the screenshot and
return ONE JSON action with exact pixel coordinates.

Available actions:
  {{"action": "click", "x": <int>, "y": <int>}}
  {{"action": "double_click", "x": <int>, "y": <int>}}
  {{"action": "right_click", "x": <int>, "y": <int>}}
  {{"action": "type", "text": "..."}}
  {{"action": "press", "key": "Enter"|"Tab"|"Escape"|...}}
  {{"action": "hotkey", "keys": "ctrl+s"|"alt+tab"|...}}
  {{"action": "scroll", "x": <int>, "y": <int>, "direction": "up"|"down", "amount": 3}}
  {{"action": "done", "result": "description"}}
  {{"action": "failed", "error": "description"}}

CRITICAL: Specify EXACT pixel coordinates of the target element center.
Look carefully at the screenshot to identify the precise location.

Return ONLY the JSON object. No markdown fences, no explanation.
"""

PLAN_SYSTEM_PROMPT = """\
You are a task planning agent. Given a user's goal, break it down into concrete, 
actionable steps. Return a JSON array of step objects.

Each step object has:
  - "action": Brief description of what to do
  - "step_type": One of "browser", "system", "tool", "creative", "verify"
  - "tool_name": Tool to use (see available tools below)
  - "tool_args": Object with arguments for the tool
  - "expected_outcome": What success looks like

Available tools by step_type:
  browser: browser_goto (url), browser_click, browser_type, browser_search
  system: screen_click (x,y), screen_type (text), screen_hotkey (keys), screen_scroll, launch_app
  tool: run_command (command), read_file (path), write_file (path, content), patch_file
  creative: generate_image, generate_text
  verify: (takes a screenshot and checks if expected outcome matches)

IMPORTANT RULES:
- For ANY web browsing task (searching, visiting websites, navigating pages), use step_type "browser"
- For browser_goto, include "url" in tool_args
- For desktop app interactions, use step_type "system"
- For file/command operations, use step_type "tool"
- Add a "verify" step after critical actions to confirm success
- Keep the plan minimal — 3-8 steps for most tasks. Prefer fewer, larger steps.
- Each step should be self-contained and achievable

Example:
[
  {"action": "Open Google in browser", "step_type": "browser", "tool_name": "browser_goto", "tool_args": {"url": "https://www.google.com"}, "expected_outcome": "Google homepage loads"},
  {"action": "Search for Python tutorials", "step_type": "browser", "tool_name": "browser_type", "tool_args": {"text": "Python tutorials"}, "expected_outcome": "Search query typed in search box"},
  {"action": "Submit search", "step_type": "browser", "tool_name": "browser_click", "tool_args": {}, "expected_outcome": "Search results page appears"},
  {"action": "Verify search results loaded", "step_type": "verify", "tool_name": "", "tool_args": {}, "expected_outcome": "Search results page with Python tutorial links"}
]

Return ONLY the JSON array, no markdown fences or explanation.
"""


class Orchestrator:
    """Autonomous task execution engine.

    Usage::

        orch = Orchestrator(api_key="...", tool_executor=tool_exec)
        orch.set_ws_client(ws_client)
        task = await orch.plan_task("Open Chrome and search for Python tutorials")
        result = await orch.execute_task(task)
    """

    def __init__(
        self,
        api_key: str = "",
        tool_executor=None,
        screen_navigator=None,
        screen_capture=None,
        browser_agent=None,
        windows_agent=None,
    ) -> None:
        self._api_key = api_key
        self._tool_executor = tool_executor
        self._screen_navigator = screen_navigator
        self._screen_capture = screen_capture
        self._browser_agent = browser_agent
        self._windows_agent = windows_agent
        self._creative_agent = None
        self._client: Optional[genai.Client] = None
        self._ws_client = None  # WSClient for sending status updates
        self._task_store = TaskStore()
        self._running_task: Optional[Task] = None
        self._cancelled = False
        self._log = log.bind(component="orchestrator")

        if _GENAI_AVAILABLE and api_key:
            self._client = genai.Client(api_key=api_key)

        # Model fallback chain (Pro → Flash → error with diagnostics)
        self._fallback_chain = None
        if _FALLBACK_AVAILABLE:
            self._fallback_chain = ModelFallbackChain(
                primary=PRO_MODEL,
                fallbacks=[FLASH_MODEL],
            )

    def set_ws_client(self, ws_client) -> None:
        """Attach WebSocket client for sending progress updates."""
        self._ws_client = ws_client

    def set_creative_agent(self, creative_agent) -> None:
        """Attach creative agent for image/text generation steps."""
        self._creative_agent = creative_agent

    @property
    def is_busy(self) -> bool:
        return self._running_task is not None and not self._running_task.is_terminal

    def cancel(self) -> None:
        """Cancel the current task."""
        self._cancelled = True
        if self._running_task is not None:
            self._running_task.mark_cancelled()
            self._task_store.save(self._running_task)
            self._log.info("orchestrator.cancelled", task_id=self._running_task.id)

    def pause(self) -> str:
        """Pause the current task cleanly. Returns a status report.

        The current step finishes (not interrupted mid-action)
        but no new steps will start. The task can be resumed later.
        """
        self._cancelled = True  # Stops the execution loop
        if self._running_task is None:
            return "No active task to pause."

        task = self._running_task
        report = self.get_progress_report()

        # Don't mark cancelled — keep it in RUNNING state for resume
        self._task_store.save(task)
        self._log.info("orchestrator.paused", task_id=task.id)

        return report

    async def resume(self) -> Optional[Task]:
        """Resume the most recent paused/active task.

        Returns the task if resumed, None if nothing to resume.
        """
        # First try the current running task
        task = self._running_task
        if task is not None and task.status == TaskStatus.RUNNING:
            self._cancelled = False
            self._log.info("orchestrator.resume", task_id=task.id)
            return await self.execute_task(task)

        # Otherwise look for active tasks in the store
        active = self._task_store.load_active()
        if not active:
            return None

        task = active[0]  # Most recent active task
        self._log.info("orchestrator.resume_from_store", task_id=task.id)
        self._cancelled = False
        return await self.execute_task(task)

    def get_progress_report(self) -> str:
        """Generate a human-readable progress report for the current task."""
        task = self._running_task
        if task is None:
            return "No active task."

        lines = [
            f"Task: {task.goal}",
            f"Status: {task.status.value}",
            f"Progress: {task.progress}",
            "",
            "Steps:",
        ]

        for i, step in enumerate(task.steps, 1):
            status_icon = {
                "done": "[DONE]",
                "running": "[>>>>]",
                "pending": "[    ]",
                "failed": "[FAIL]",
                "retrying": "[RTRY]",
                "skipped": "[SKIP]",
                "verifying": "[VRFY]",
            }.get(step.status.value, "[????]")

            line = f"  {i}. {status_icon} {step.action}"
            if step.status == StepStatus.DONE and step.result:
                line += f" → {step.result[:60]}"
            elif step.status == StepStatus.FAILED and step.error:
                line += f" ✗ {step.error[:60]}"
            lines.append(line)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    async def plan_task(self, goal: str) -> Task:
        """Use Gemini Pro to decompose a goal into executable steps."""
        task = Task(goal=goal)
        task.mark_running()
        self._task_store.save(task)
        self._log.info("orchestrator.planning", task_id=task.id, goal=goal[:100])

        if self._client is None:
            # No API — create a single generic step
            task.steps = [Step(
                action=goal,
                step_type=StepType.TOOL,
                tool_name="run_command",
                tool_args={"command": "echo 'No Gemini client — manual execution needed'"},
                expected_outcome="User handles manually",
            )]
            task.plan_summary = "Direct execution (no planner available)"
            self._task_store.save(task)
            return task

        try:
            # Use fallback chain if available, otherwise direct call
            if self._fallback_chain is not None:
                async def _generate(model: str = PRO_MODEL, **kw):
                    return await self._client.aio.models.generate_content(
                        model=model,
                        contents=f"User goal: {goal}",
                        config=types.GenerateContentConfig(
                            system_instruction=PLAN_SYSTEM_PROMPT,
                            temperature=0.3,
                            max_output_tokens=2048,
                        ),
                    )
                response = await self._fallback_chain.call_with_fallback(_generate)
            else:
                response = await self._client.aio.models.generate_content(
                    model=PRO_MODEL,
                    contents=f"User goal: {goal}",
                    config=types.GenerateContentConfig(
                        system_instruction=PLAN_SYSTEM_PROMPT,
                        temperature=0.3,
                        max_output_tokens=2048,
                    ),
                )

            text = (response.text or "").strip()
            # Strip markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            steps_data = json.loads(text)
            if not isinstance(steps_data, list):
                steps_data = [steps_data]

            for sd in steps_data:
                step = Step(
                    action=sd.get("action", ""),
                    step_type=StepType(sd.get("step_type", "tool")),
                    tool_name=sd.get("tool_name", ""),
                    tool_args=sd.get("tool_args", {}),
                    expected_outcome=sd.get("expected_outcome", ""),
                )
                task.steps.append(step)

            task.plan_summary = f"{len(task.steps)} steps planned"
            self._log.info("orchestrator.plan_ok",
                           task_id=task.id, steps=len(task.steps))

        except json.JSONDecodeError:
            self._log.warning("orchestrator.plan_parse_error", text=text[:200])
            # Fallback: single step
            task.steps = [Step(
                action=goal,
                step_type=StepType.TOOL,
                expected_outcome="Goal achieved",
            )]
            task.plan_summary = "Single-step fallback (plan parse failed)"

        except Exception as exc:
            self._log.exception("orchestrator.plan_error")
            # Log detailed diagnostics if fallback module available
            if _FALLBACK_AVAILABLE:
                classified = classify_error(exc, model=PRO_MODEL)
                diag = get_diagnostic_message(classified)
                self._log.error("orchestrator.plan_model_error",
                                reason=classified.reason.value,
                                model=classified.model,
                                diagnostic=diag)
                print(f"\n{diag}\n")
            task.steps = [Step(
                action=goal,
                step_type=StepType.TOOL,
                expected_outcome="Goal achieved",
            )]
            task.plan_summary = "Single-step fallback (planner error)"

        self._task_store.save(task)
        return task

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_task(self, task: Task) -> Task:
        """Execute all steps of a task sequentially.

        Returns the completed (or failed) task.
        """
        self._running_task = task
        self._cancelled = False
        self._log.info("orchestrator.execute_start",
                       task_id=task.id, steps=len(task.steps))

        await self._notify(f"Starting task: {task.goal}\nPlan: {task.plan_summary}")

        while not task.is_terminal and not self._cancelled:
            step = task.advance()
            if step is None:
                break  # All steps processed

            self._log.info("orchestrator.step_start",
                           task_id=task.id, step_id=step.id,
                           action=step.action[:80], attempt=step.attempts)

            await self._notify(
                f"Step {task.progress}: {step.action}"
            )
            print(f"  [Step {task.progress}] {step.action}")

            result = await self._execute_step(step, task)

            if step.status == StepStatus.DONE:
                self._log.info("orchestrator.step_done",
                               step_id=step.id, result=step.result[:100])
                print(f"  [Step {step.id}] Done: {step.result[:80]}")
                # Store result in scratchpad for next steps
                task.scratchpad[f"step_{step.id}_result"] = step.result
                # Brief pause between steps for UI to settle
                await asyncio.sleep(0.5)

            elif step.status == StepStatus.RETRYING:
                self._log.warning("orchestrator.step_retry",
                                  step_id=step.id, attempt=step.attempts,
                                  error=step.error[:100])
                print(f"  [Step {step.id}] Retrying ({step.attempts}/{MAX_STEP_RETRIES}): {step.error[:60]}")
                # Reset step for retry
                step.status = StepStatus.PENDING
                await asyncio.sleep(1.5)  # Longer pause before retry

            elif step.status == StepStatus.FAILED:
                self._log.error("orchestrator.step_failed",
                                step_id=step.id, error=step.error[:100])
                print(f"  [Step {step.id}] FAILED: {step.error[:80]}")
                await self._notify(f"Step failed after {step.attempts} attempts: {step.error}")

            self._task_store.save(task)

        # Final status
        self._running_task = None
        self._task_store.save(task)

        status_msg = f"Task {task.status.value}: {task.goal} ({task.progress} steps)"
        self._log.info("orchestrator.execute_done",
                       task_id=task.id, status=task.status.value,
                       progress=task.progress)
        await self._notify(status_msg)

        return task

    async def _execute_step(self, step: Step, task: Task) -> dict:
        """Execute a single step using the appropriate sub-agent."""
        try:
            if step.step_type == StepType.BROWSER:
                return await self._execute_browser_step(step, task)
            elif step.step_type == StepType.SYSTEM:
                return await self._execute_system_step(step, task)
            elif step.step_type == StepType.TOOL:
                return await self._execute_tool_step(step, task)
            elif step.step_type == StepType.CREATIVE:
                return await self._execute_creative_step(step, task)
            elif step.step_type == StepType.VERIFY:
                return await self._execute_verify_step(step, task)
            else:
                # Default: try tool executor
                return await self._execute_tool_step(step, task)
        except Exception as exc:
            step.mark_failed(str(exc))
            return {"success": False, "error": str(exc)}

    async def _execute_browser_step(self, step: Step, task: Task) -> dict:
        """Execute via BrowserAgent (Playwright)."""
        if self._browser_agent is None or not self._browser_agent.available:
            # Fallback: try opening URL via system command if it's a navigation step
            url = step.tool_args.get("url", "")
            if url and self._tool_executor is not None:
                import platform
                if platform.system() == "Windows":
                    cmd = f'start "" "{url}"'
                else:
                    cmd = f'xdg-open "{url}" 2>/dev/null || open "{url}" 2>/dev/null'
                result = await self._tool_executor.execute("run_command", {"command": cmd})
                if result.get("success"):
                    step.mark_done(f"Opened {url} in default browser")
                    await asyncio.sleep(2.0)  # Wait for browser to load
                    return result
            # Fallback to screen navigator for visual interaction
            return await self._execute_system_step(step, task)

        if not self._browser_agent.is_running:
            started = await self._browser_agent.start()
            if not started:
                # Fallback to system step
                return await self._execute_system_step(step, task)

        # If step has a direct URL, navigate first
        url = step.tool_args.get("url", "")
        if url and step.tool_name == "browser_goto":
            await self._browser_agent.goto(url)
            await asyncio.sleep(1.0)

        result = await self._browser_agent.execute_step(
            goal=step.action,
            expected_outcome=step.expected_outcome,
            scratchpad=task.scratchpad,
        )

        if result.get("success"):
            step.mark_done(result.get("result", ""))
        else:
            step.mark_failed(result.get("error", "Browser step failed"))

        return result

    async def _execute_system_step(self, step: Step, task: Task) -> dict:
        """Execute via ScreenNavigator with Computer Use model for precise targeting.

        Instead of using approximate planner coordinates, we capture a screenshot
        and send it to the Computer Use model with the action description to get
        exact pixel coordinates — then execute via screen_navigator.
        """
        # Try WindowsAgent first for structured control
        if self._windows_agent is not None and self._windows_agent.available:
            if step.tool_name in ("focus_window", "find_window", "launch_app", "close_app"):
                result = await self._dispatch_windows_agent(step)
                if result.get("success"):
                    step.mark_done(str(result))
                    return result

        # For screen actions that need coordinates, use vision-guided targeting
        if (self._screen_navigator is not None
                and self._screen_capture is not None
                and self._client is not None
                and step.tool_name in ("screen_click", "screen_scroll", "screen_move", "screen_drag")):
            return await self._execute_vision_guided_step(step, task)

        # For non-coordinate screen actions, execute directly
        if self._screen_navigator is not None and step.tool_name:
            nav = self._screen_navigator
            args = step.tool_args
            result = {"success": False, "error": "Unknown tool"}

            try:
                if step.tool_name == "screen_type":
                    result = await nav.type_text(
                        args.get("text", ""),
                        interval=args.get("interval", 0.02),
                    )
                elif step.tool_name == "screen_hotkey":
                    result = await nav.hotkey(args.get("keys", ""))
                else:
                    # Try tool executor as fallback
                    return await self._execute_tool_step(step, task)

            except Exception as exc:
                result = {"success": False, "error": str(exc)}

            if result.get("success"):
                step.mark_done(f"Executed {step.tool_name}")
                await asyncio.sleep(0.5)
            else:
                step.mark_failed(result.get("error", ""))

            return result

        # No navigator available — try tool executor
        return await self._execute_tool_step(step, task)

    async def _execute_vision_guided_step(self, step: Step, task: Task) -> dict:
        """Use Computer Use model for precise screen coordinate targeting.

        1. Capture screenshot
        2. Send to Computer Use model with action description
        3. Get exact coordinates from model
        4. Execute via screen_navigator
        """
        nav = self._screen_navigator

        # Capture current screenshot
        jpeg = await self._screen_capture.capture_async(force=True)
        if jpeg is None:
            # Fall back to planner coordinates
            return await self._execute_system_step_fallback(step)

        # Build context for the Computer Use model
        cr = self._screen_capture.get_last_capture_result()
        context = (
            f"I need to: {step.action}\n"
            f"Expected outcome: {step.expected_outcome}\n"
        )
        if step.tool_name == "screen_click":
            context += f"Find the exact element to click."
        elif step.tool_name == "screen_scroll":
            context += f"Find the area where I should scroll."
        elif step.tool_name == "screen_drag":
            context += f"Find the start and end points for the drag."

        # Add task scratchpad context
        if task.scratchpad:
            recent = dict(list(task.scratchpad.items())[-3:])
            context += f"\nPrevious context: {json.dumps(recent)}"

        # Use the actual screenshot dimensions for coordinate space
        # The model sees the resized image, so coordinates will be in resized space
        parts = [
            types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=jpeg)),
            types.Part(text=context),
        ]

        try:
            response = await self._client.aio.models.generate_content(
                model=MODEL_COMPUTER_USE,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=VISION_SCREEN_SYSTEM,
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )
        except Exception:
            self._log.exception("orchestrator.vision_model_error")
            return await self._execute_system_step_fallback(step)

        # Parse the model's response
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]

        try:
            action_json = json.loads(text.strip())
        except json.JSONDecodeError:
            self._log.warning("orchestrator.vision_parse_error", text=text[:200])
            return await self._execute_system_step_fallback(step)

        action = action_json.get("action", "")

        # Terminal actions from model
        if action == "done":
            step.mark_done(action_json.get("result", "Step completed"))
            return {"success": True, "result": action_json.get("result", "")}
        if action == "failed":
            step.mark_failed(action_json.get("error", "Step failed"))
            return {"success": False, "error": action_json.get("error", "")}

        # Execute the model's precise action via screen_navigator
        # Coordinates from model are in screenshot (resized) space — screen_navigator handles mapping
        try:
            result = {"success": False, "error": "Unknown action"}

            if action in ("click", "double_click", "right_click"):
                x = int(action_json.get("x", 0))
                y = int(action_json.get("y", 0))
                button = "left" if action == "click" else ("right" if action == "right_click" else "left")
                clicks = 2 if action == "double_click" else 1
                result = await nav.click(x, y, button=button, clicks=clicks)
            elif action == "type":
                result = await nav.type_text(action_json.get("text", ""))
            elif action == "press" or action == "hotkey":
                keys = action_json.get("keys", action_json.get("key", ""))
                result = await nav.hotkey(keys)
            elif action == "scroll":
                x = int(action_json.get("x", 0))
                y = int(action_json.get("y", 0))
                direction = action_json.get("direction", "down")
                amount = action_json.get("amount", 3)
                scroll_clicks = amount if direction == "up" else -amount
                result = await nav.scroll(x, y, scroll_clicks)
            else:
                self._log.warning("orchestrator.unknown_vision_action", action=action)

            if result.get("success"):
                step.mark_done(f"Vision-guided {action} at ({action_json.get('x')}, {action_json.get('y')})")
                await asyncio.sleep(0.5)
            else:
                step.mark_failed(result.get("error", f"Action {action} failed"))

            return result

        except Exception as exc:
            step.mark_failed(str(exc))
            return {"success": False, "error": str(exc)}

    async def _execute_system_step_fallback(self, step: Step) -> dict:
        """Fallback: execute system step using planner-provided coordinates."""
        nav = self._screen_navigator
        if nav is None:
            step.mark_failed("No screen navigator available")
            return {"success": False, "error": "No screen navigator"}

        args = step.tool_args
        result = {"success": False, "error": "Unknown tool"}

        try:
            if step.tool_name == "screen_click":
                result = await nav.click(
                    args.get("x", 0), args.get("y", 0),
                    button=args.get("button", "left"),
                    clicks=args.get("clicks", 1),
                )
            elif step.tool_name == "screen_scroll":
                result = await nav.scroll(
                    args.get("x", 0), args.get("y", 0),
                    args.get("clicks", 3),
                )
            elif step.tool_name == "screen_move":
                result = await nav.move(args.get("x", 0), args.get("y", 0))
            elif step.tool_name == "screen_drag":
                result = await nav.drag(
                    args.get("start_x", 0), args.get("start_y", 0),
                    args.get("end_x", 0), args.get("end_y", 0),
                )
        except Exception as exc:
            result = {"success": False, "error": str(exc)}

        if result.get("success"):
            step.mark_done(f"Executed {step.tool_name} (fallback)")
            await asyncio.sleep(0.5)
        else:
            step.mark_failed(result.get("error", ""))

        return result

    async def _execute_creative_step(self, step: Step, task: Task) -> dict:
        """Execute via CreativeAgent (Imagen 3 / Gemini text generation)."""
        agent = getattr(self, '_creative_agent', None)
        if agent is None:
            # Fallback: treat as a tool step
            return await self._execute_tool_step(step, task)

        result = await agent.execute_step(
            goal=step.action,
            expected_outcome=step.expected_outcome,
            scratchpad=task.scratchpad,
        )

        if result.get("success"):
            step.mark_done(result.get("result", ""))
        else:
            step.mark_failed(result.get("error", "Creative step failed"))

        return result

    async def _execute_tool_step(self, step: Step, task: Task) -> dict:
        """Execute via ToolExecutor (read/write/patch/run_command)."""
        if self._tool_executor is None:
            step.mark_failed("No tool executor available")
            return {"success": False, "error": "No tool executor"}

        tool_name = step.tool_name or "run_command"
        tool_args = step.tool_args or {}

        try:
            result = await self._tool_executor.execute(tool_name, tool_args)
        except Exception as exc:
            result = {"success": False, "error": str(exc)}

        if result.get("success"):
            # Extract meaningful result text
            output = result.get("output", result.get("content", str(result)))
            step.mark_done(str(output)[:500])
        else:
            step.mark_failed(result.get("error", "Tool execution failed"))

        return result

    async def _execute_verify_step(self, step: Step, task: Task) -> dict:
        """Verify by capturing a screenshot and asking Gemini if the action succeeded."""
        if self._screen_capture is None or self._client is None:
            step.mark_done("Verification skipped (no screen capture)")
            return {"success": True, "result": "skipped"}

        try:
            jpeg = await self._screen_capture.capture_async(force=True)
            if jpeg is None:
                step.mark_done("Verification skipped (no frame)")
                return {"success": True, "result": "skipped"}

            response = await self._client.aio.models.generate_content(
                model=FLASH_MODEL,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(inline_data=types.Blob(
                            mime_type="image/jpeg", data=jpeg)),
                        types.Part(text=(
                            f"I just performed this action: {step.action}\n"
                            f"Expected outcome: {step.expected_outcome}\n"
                            "Does the screenshot show the expected outcome? "
                            "Reply with JSON: {\"success\": true/false, \"reason\": \"...\"}"
                        )),
                    ]),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                ),
            )

            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

            try:
                vresult = json.loads(text.strip())
                if vresult.get("success"):
                    step.mark_done(vresult.get("reason", "Verified"))
                else:
                    step.mark_failed(vresult.get("reason", "Verification failed"))
            except json.JSONDecodeError:
                step.mark_done(f"Verification response: {text[:200]}")

            return {"success": step.status == StepStatus.DONE, "result": text[:200]}

        except Exception as exc:
            self._log.warning("orchestrator.verify_error", error=str(exc))
            step.mark_done("Verification skipped (error)")
            return {"success": True, "result": "skipped"}

    async def _dispatch_windows_agent(self, step: Step) -> dict:
        """Dispatch to WindowsAgent for native Windows operations."""
        agent = self._windows_agent
        args = step.tool_args

        if step.tool_name == "focus_window":
            return await agent.focus_app(args.get("title_contains", ""))
        elif step.tool_name == "find_window":
            windows = await agent.list_windows(args.get("title_contains", ""))
            return {"success": True, "windows": windows}
        elif step.tool_name == "launch_app":
            return await agent.launch_app(args.get("path", ""))
        elif step.tool_name == "close_app":
            return await agent.close_app(args.get("title_contains", ""))
        else:
            return {"success": False, "error": f"Unknown windows action: {step.tool_name}"}

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    async def _notify(self, message: str) -> None:
        """Send a status update to the user via WebSocket."""
        self._log.info("orchestrator.notify", msg=message[:100])
        if self._ws_client is not None:
            try:
                await self._ws_client.send_json({
                    "type": "context",
                    "subtype": "task_status",
                    "content": f"[TASK STATUS] {message}",
                })
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Quick run (plan + execute in one call)
    # ------------------------------------------------------------------

    async def run(self, goal: str) -> Task:
        """Plan and execute a task in one call."""
        task = await self.plan_task(goal)
        return await self.execute_task(task)

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def recover_active_tasks(self) -> list[Task]:
        """Load any tasks that were interrupted mid-execution."""
        return self._task_store.load_active()

    def close(self) -> None:
        """Clean up resources."""
        self._task_store.close()
        self._log.info("orchestrator.closed")
