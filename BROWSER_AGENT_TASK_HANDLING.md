# Browser Agent and Task Assignment Flow in Rio

This document explains two things end-to-end:
1. How BrowserAgent works internally.
2. How a task you assign is handled by Rio from detection to completion.

## 1. Runtime Components Involved

1. BrowserAgent
- File: local/browser_agent.py
- Role: Executes browser-focused steps using Playwright + Gemini model-guided actions.

2. Orchestrator
- File: local/orchestrator.py
- Role: Plans tasks into steps and dispatches each step to browser/system/tool/verify handlers.

3. Main runtime loop
- File: local/main.py
- Role: Detects task-like user input, starts autonomous execution, streams status back.

4. Task state persistence
- File: local/task_state.py
- Role: Stores task/step state, retries, and history in SQLite.

5. Cloud-side agent selection (for cloud-routed goals)
- File: cloud/tool_orchestrator.py
- Role: Routes goals to browser_agent or other agents via keyword/semantic selection.

## 2. How BrowserAgent Works (Complete Lifecycle)

### 2.1 Availability and startup

BrowserAgent is considered available only if both are installed:
- google genai SDK
- playwright async API

If unavailable, orchestrator falls back to system/tool execution.

### 2.2 Browser connection strategy

BrowserAgent startup uses a two-tier strategy through get_browser_page():

1. Primary mode: CDP attach
- Attempts playwright chromium connect_over_cdp across candidate endpoints.
- Default probe order includes localhost/127.0.0.1 on ports 9222, 9223, 9224.
- Supports explicit overrides:
  - RIO_BROWSER_CDP_URL
  - RIO_BROWSER_CDP_URLS (comma-separated)
- Optional auto-bootstrap can launch a native browser with remote debugging and retry CDP attach:
  - RIO_BROWSER_AUTO_BOOTSTRAP_CDP=1 (default)
- Reuses first context.
- Chooses first useful page (prefers http/https tab), or creates one.
- Brings page to front.
- Sets browser_mode = cdp.
- Logs browser_mode_selected with detected browser label (chrome/edge/brave/chromium-family).

2. Fallback mode: Persistent profile context
- Default behavior is now CDP-only for existing-browser control.
- Persistent fallback is disabled unless explicitly enabled:
  - RIO_BROWSER_ALLOW_PERSISTENT_FALLBACK=1
- If CDP fails, detects Chrome user data dir by OS:
  - Windows: LOCALAPPDATA/Google/Chrome/User Data
  - macOS: ~/Library/Application Support/Google/Chrome
  - Linux: ~/.config/google-chrome
- Uses profile from CHROME_PROFILE env var, default Default.
- Launches launch_persistent_context with headless=false.
- Selects page similarly and brings it to front.
- Sets browser_mode = persistent.

3. Hard failure behavior
- If both modes fail, raises RuntimeError with guidance:
  - Start Chrome with --remote-debugging-port=9222, or
  - Close Chrome so profile lock is released.

### 2.3 Model setup and viewport

On successful browser page creation:
- Creates Gemini client with GEMINI_API_KEY.
- Forces viewport to 1280x720 for screenshot/action coordinate consistency.
- Marks agent running.

### 2.4 Step execution loop

For each browser step goal:

1. Screenshot
- Captures page screenshot as PNG.
- Brings page to front before screenshot for reliability.

2. Model call
- Sends screenshot + textual goal/expected outcome + scratchpad context.
- Primary model: computer-use model constant.
- Fallback model: flash model constant if primary fails.

3. Parse action JSON
- Expects one JSON action object.
- Supported actions include goto, click, double_click, right_click, type, press, hotkey, scroll, wait, done, failed.

4. Execute action in Playwright
- Goto: page.goto
- Clicks: page.mouse click calls
- Typing/keys: page.keyboard operations
- Scroll: mouse wheel after moving to target point

5. Repeat up to MAX_ACTION_ROUNDS
- Stops early on done or failed action.
- Detects stuck condition when same action repeats 3 times.

### 2.5 Shutdown behavior

On stop:
- If browser_mode is persistent: closes context (owned resource).
- If browser_mode is cdp: does not close attached browser/context (not owned).
- Stops Playwright runtime and clears state.

## 3. How Your Assigned Task Is Handled in Rio

There are two related paths: local runtime tasking and cloud-side agent routing.

### 3.1 Local path: when you type a task in Rio input

1. Task detection
- main.py input loop checks _is_task_request(text).
- Detects explicit task prefixes like task:, do:, execute:, automate:.
- Detects action verbs and task phrases.
- Filters out question/chat starts.

2. Autonomous task start
- If detected and orchestrator is ready, Rio starts _run_autonomous_task(goal, ...).
- Sets task_active and autonomous_mode events.

3. Planning
- Orchestrator plan_task(goal) calls Gemini planner prompt.
- Expects JSON list of step objects with:
  - action
  - step_type (browser/system/tool/creative/verify)
  - tool_name
  - tool_args
  - expected_outcome
- If parse/model fails, falls back to simpler single-step task.

4. Step execution
- execute_task processes steps sequentially.
- Dispatch by step_type:
  - browser -> BrowserAgent path
  - system -> WindowsAgent/ScreenNavigator path
  - tool -> ToolExecutor path
  - creative -> CreativeAgent path
  - verify -> screenshot verification path

5. Browser step handling specifically
- _execute_browser_step ensures BrowserAgent started.
- If tool_name is browser_goto and url exists, performs direct goto first.
- Then BrowserAgent execute_step handles model-guided browser actions.
- On BrowserAgent unavailability/start failure, falls back to system step path.

6. Retry and failure logic
- Each Step tracks attempts.
- mark_failed sets retrying while attempts < MAX_STEP_RETRIES (3).
- Orchestrator retries with delays.
- Task ends as done, partial, failed, or cancelled.

7. User feedback and completion
- Orchestrator sends progress/status messages over websocket.
- _run_autonomous_task sends plan-start and completion context messages.
- Final report summarizes what succeeded or failed.

### 3.2 Cloud path: if routing happens through cloud orchestrator

- cloud/tool_orchestrator.py uses _select_agent(goal, agent_configs, ...).
- Browser-related keywords route to browser_agent first.
- If browser_agent disabled/unavailable, it can fall back to task_executor or other paths.

## 4. What Happens When You Say "Do this task"

Example input:
- "open github and search for rio issues for me"

Flow:
1. input_loop marks it as task request.
2. _run_autonomous_task starts and calls orchestrator.plan_task.
3. Planner emits browser steps and verify step.
4. _execute_browser_step starts BrowserAgent.
5. BrowserAgent opens/attaches browser page and executes model-driven actions.
6. Step results go to task scratchpad.
7. Verify step checks screen outcome.
8. Task report is returned to you in runtime status updates.

## 5. Why You Might See "Opening Chrome" But No Actions

Common causes in this architecture:
1. BrowserAgent start failed and orchestrator fell back silently to system path.
2. BrowserAgent attached to wrong tab/context before action loop.
3. Model returned unparsable action JSON repeatedly.
4. Task was not classified as browser step by planner.
5. BrowserAgent unavailable due to missing Playwright or missing API key.
6. Browser launched without remote debugging port, so CDP probe cannot attach.
7. CDP-only mode is enabled and all endpoints failed, so BrowserAgent intentionally refuses fallback launch.
8. Planner returned non-JSON text and fallback step was malformed (historically this could trigger run_command bad-args retries).

Current fallback behavior:
- If planner output is not valid JSON, orchestrator now uses goal-aware fallback.
- Browser-like goals (for example: open chrome with gmail) are routed to a Browser step directly.
- Non-browser goals use a safe run_command step with an explicit command payload.

## 6. Practical Debug Checklist

1. Confirm BrowserAgent availability printed during startup.
2. Confirm task was detected as autonomous (not sent as normal chat).
3. Check orchestrator step types include browser for web goals.
4. Check browser_agent logs for:
- browser_mode_selected
- browser_agent.cdp_unavailable
- browser_agent.started
- browser_agent.action
- browser_agent.action_error
- browser_agent.step_done or step_failed
5. If CDP mode expected, verify Chrome was launched with --remote-debugging-port=9222.
6. If persistent mode expected, close existing Chrome profile lock conflicts or set CHROME_PROFILE explicitly.

## 7. Key Files to Inspect

- local/browser_agent.py
- local/orchestrator.py
- local/main.py
- local/task_state.py
- cloud/tool_orchestrator.py

## 8. Summary

- BrowserAgent is a model-guided Playwright executor with CDP-first startup and persistent-profile fallback.
- Your assigned task is detected in main input, planned by Orchestrator, dispatched step-by-step, retried on failures, and persisted in TaskStore.
- Browser steps are handled by BrowserAgent first, with fallback paths to system/tool execution when browser automation is unavailable or fails.
