# Rio Build Progress

## Current Layer: L4 — Struggle Detection (Day 9-10) ✅ COMPLETE

### Day 1: Cloud Scaffold — COMPLETE (original agent)
- [x] Cloud scaffold (FastAPI + WS + Gemini session + rate limiter + model router)
- [x] Root config, Dockerfile, service.yaml, deploy.sh

### Day 2: Local Scaffold + Text Round-Trip — COMPLETE
- [x] Local scaffold (entry point + WS client + config + YAML loader)
- [x] Protocol alignment: cloud sends `transcript`, local handles it
- [x] Switched from Live API to standard generate_content_stream (L0 text mode)
- [x] Model: gemini-2.5-flash (gemini-2.0-flash hit free-tier rate limits)
- [x] 429 retry logic with exponential backoff (3 attempts)
- [x] Error feedback: relay sends error frames to client on failure
- [x] Conditional audio_io import (text-only mode when deps missing)
- [x] Heartbeat loop (10s interval, keeps cloud session alive)
- [x] .env created with GEMINI_API_KEY, .gitignore excludes it
- [x] **L0 GATE: "Type locally, see Gemini response" — PASSED**

### Day 3-4 Audio (Live API integration) — COMPLETE
- [x] AudioCapture class (sounddevice, 16kHz, async chunks) — local code exists
- [x] AudioPlayback class (24kHz, queue-based) — local code exists
- [x] Dual-mode GeminiSession: "live" (Live API) + "text" (fallback)
- [x] Model: `gemini-2.5-flash-native-audio-latest` for Live API (confirmed by Day 5 agent)
- [x] `_connect_live()` — opens Live API session with `LiveConnectConfig`:
  - response_modalities=["AUDIO"], voice="Puck", system_instruction, tools
  - Falls back to text mode if Live API connect fails
- [x] Real `send_audio()` — streams PCM via `LiveClientRealtimeInput` + `Blob`
- [x] `send_end_of_turn()` — signals PTT release to Gemini
- [x] `receive_live()` — persistent async generator yielding audio/text/tool_call/turn_complete
  - Parses server_content.model_turn.parts (inline_data for audio, text, function_call)
  - Also handles shorthand response.data / response.text for SDK compatibility
  - Auto-retry on stream errors with 1s backoff
- [x] `send_text()` live branch — injects text into Live session with end_of_turn
- [x] `send_image()` live branch — sends JPEG inline via LiveClientRealtimeInput
- [x] `send_tool_result()` live branch — sends via LiveClientToolResponse
- [x] `close()` — properly closes Live API context manager + session
- [x] `mode` property exposed for relay task selection
- [x] Cloud main.py: `_relay_live_to_client()` persistent relay task
  - Streams audio → binary 0x01 frames to local client
  - Streams text → JSON transcript frames
  - Handles tool_call → waits for result from local → feeds back to Gemini
  - Handles turn_complete and setup_complete control frames
- [x] Cloud main.py: starts persistent relay on connect if live mode
- [x] Cloud main.py: `end_of_speech` control frame → `send_end_of_turn()`
- [x] Cloud main.py: text frames skip relay restart in live mode (persistent relay handles it)
- [x] Cloud main.py: rate limiting skipped for live mode (0 RPM within session)
- [x] SessionManager: `create_session(mode="live")` passes mode through
- [x] All text-mode (L0) code preserved intact as fallback
- [x] All L2 vision code preserved and works in both modes
- [x] All L3 tool code preserved and works in both modes

### Day 5: VAD + Push-to-Talk — COMPLETE
- [x] Silero VAD integration (torch, speech probability detection)
- [x] F2 push-to-talk with pynput
- [x] VAD threshold from config.yaml
- [x] Graceful degradation: ptt+vad / ptt-only / vad-only / always-on
- [ ] Interrupt handling: clear playback buffer when user speaks

### Day 6: Vision / Screen Capture — COMPLETE
- [x] Created `local/screen_capture.py` (mss + Pillow + JPEG compression)
- [x] MD5 delta detection: skip unchanged frames
- [x] Configurable: fps, quality, resize_factor (from config.yaml)
- [x] Async-friendly: capture_async() runs in thread executor
- [x] Periodic capture loop (default: 1 frame / 3s)
- [x] F3 hotkey for on-demand screenshot (forced, no delta skip)
- [x] Cloud: gemini_session.py send_image() with deferred + immediate modes
- [x] Cloud: main.py 0x02 binary frame → gemini.send_image()
- [x] Dashboard broadcast on vision frame receipt
- [x] Updated local/requirements.txt with mss >= 9.0.0, Pillow >= 10.0.0
- [x] Graceful degradation: vision disabled if mss/Pillow missing

### Day 7: MVP Gate — COMPLETE
- [x] All L2 components integrated into main.py orchestrator
- [x] Screen capture + F3 hotkey + periodic loop wired with graceful shutdown
- [x] Version bumped to v0.3.0 (Layer 2: voice + vision)
- [x] Image data flows: local capture → WS binary → cloud relay → Gemini (deferred context)
- [x] F3 screenshot + text question = Gemini sees the screen
- [x] **MVP GATE: Voice + Vision end-to-end architecture — PASSED**

### Day 8: Tools / Function Calling — COMPLETE
- [x] Created `local/tools.py` — ToolExecutor class
  - [x] `read_file(path)` — auto-approve, return contents (truncated at 100K)
  - [x] `write_file(path, content)` — .rio.bak backup, then write
  - [x] `patch_file(path, old_text, new_text)` — find-and-replace with backup
  - [x] `run_command(command)` — subprocess, 30s timeout, blocklist for dangerous commands
  - [x] Safety: blocklist for rm -rf, mkfs, dd, fork bombs, shutdown, etc.
- [x] Cloud: Added `RIO_TOOL_DECLARATIONS` (4 function declarations) to gemini_session.py
- [x] Cloud: `receive()` now yields `str | dict` — text chunks or tool_call dicts
- [x] Cloud: Switched from `generate_content_stream` to `generate_content` (needed for function calling)
- [x] Cloud: Added `send_tool_result(name, result)` for feeding tool results back
- [x] Cloud: Reworked relay in main.py with tool-calling loop (up to 5 rounds)
- [x] Cloud: `tool_result_queue` (asyncio.Queue) bridges main loop ↔ relay task
- [x] Cloud: Added `tool_result` frame handler in main WebSocket loop
- [x] Local: receive_loop handles `tool_call` → execute → send `tool_result` back
- [x] Local: Rich CLI output: shows tool name, args, results, command output
- [x] Local: Graceful degradation if tools.py not available
- [x] Dashboard: broadcasts tool_call and tool_result events
- [x] Version bumped to v0.4.0 (Layer 3: voice + vision + tools)
- [x] System instruction updated to mention tool capabilities

### Day 9-10: Struggle Detection (L4) — COMPLETE
- [x] Created `local/struggle_detector.py` — StruggleDetector class
  - [x] 4 screen-based signals (no pynput keystroke monitoring):
    1. Repeated error — same screen hash 3+ times in 2-min window (weight 0.35)
    2. Long pause on error — screen unchanged >30s after error keywords (weight 0.25)
    3. Rapid small screen changes — 5+ distinct hashes in 60s (weight 0.20)
    4. Stale screen with activity — unchanged >45s but user recently active (weight 0.20)
  - [x] StruggleResult dataclass: confidence, active_signals, should_trigger, reason
  - [x] feed_frame(jpeg_bytes) — stores MD5 hash + timestamp in rolling deque
  - [x] feed_gemini_response(text) — checks for error keywords, primes Signal 2
  - [x] note_user_activity() — updates last activity time for Signal 4
  - [x] evaluate() — computes weighted sum, checks threshold + signal count + cooldown
  - [x] Cooldown: record_trigger() (standard), record_decline() (longer cooldown)
  - [x] Demo mode: threshold=0.4, cooldown=30s, min_signals=1
  - [x] force_trigger() — bypasses all signals for F4 demo key
  - [x] No sklearn dependency — pure Python weighted sum
  - [x] Follows graceful degradation pattern (try/except ImportError in main.py)
  - [x] structlog logging throughout
- [x] Modified `local/main.py`:
  - [x] Added StruggleDetector import guard (try/except)
  - [x] Added struggle detector initialization after tool executor
  - [x] Added `struggle_detection_loop()` — runs every 2s, feeds frames, evaluates, sends context
  - [x] Added `proactive_trigger_loop()` — F4 demo key, force trigger
  - [x] receive_loop: feeds Gemini responses to detector (feed_gemini_response)
  - [x] input_loop: notes user activity (note_user_activity)
  - [x] F4 PushToTalk.create() for demo mode proactive trigger
  - [x] Struggle + proactive trigger tasks added to asyncio task set
  - [x] Shutdown: stops proactive trigger listener
  - [x] Version bumped to v0.5.0 (Layer 4: struggle detection)
- [x] Modified `cloud/main.py`:
  - [x] Replaced context frame stub with full handler
  - [x] Extracts subtype, content, confidence, signals from context frames
  - [x] Calls gemini.send_context() for struggle subtype
  - [x] Broadcasts to dashboard: {type: dashboard, subtype: struggle, confidence, signals}
- [x] Modified `cloud/gemini_session.py`:
  - [x] Added send_context() method — dual-mode (live + text)
  - [x] Live mode: sends to Live session with end_of_turn=True
  - [x] Text mode: appends to history as user message
- [x] Config already wired (config.yaml struggle section + StruggleConfig dataclass)

## Architecture Decisions (Day 9-10 / L4)
- **4 signals (screen-based)**: Deferred pynput keystroke/window-switching signals to polish phase
- **No sklearn**: Pure Python weighted sum, zero extra dependencies
- **F4 manual trigger**: demo_mode repurposes toggle_proactive hotkey as force-trigger
- **send_context() vs send_text()**: Dedicated method keeps struggle injection separate from user messages
- **Evaluation interval**: 2 seconds — fast enough to detect, negligible CPU
- **Frame capture**: force=True in struggle loop to bypass delta detection (needs fresh frames for hash comparison)
- **Error keyword tracking**: Gemini response → feed_gemini_response() → primes Signal 2 (long pause on error)
- **Keystroke signals deferred**: Signals 2/5 from plan (undo/redo, SO switching) moved to L7 polish phase

### Screen Mode Toggle (v0.5.1) — COMPLETE
- [x] Added `default_mode: "on_demand"` to VisionConfig (config.yaml + config.py)
- [x] Added `screen_mode: "f5"` to HotkeyConfig (config.yaml + config.py)
- [x] Added `capture_screen` Gemini tool declaration (gemini_session.py)
- [x] Updated system instruction to explain on-demand screen capture + capture_screen tool
- [x] Gated `screen_capture_loop` with `asyncio.Event(autonomous_mode)` — only sends periodic frames when autonomous
- [x] Added `screen_mode_toggle_loop()` — F5 hotkey toggles on-demand ↔ autonomous
- [x] Intercept `capture_screen` tool_call in `receive_loop` — takes screenshot + sends to cloud
- [x] Initialize autonomous_mode event based on config.vision.default_mode in main()
- [x] Initialize F5 PushToTalk + screen_mode_trigger in main()
- [x] Added screen_mode task to asyncio task set + shutdown cleanup
- [x] Updated receive_loop signature to accept `screen` param for capture_screen
- [x] Version bumped to v0.5.1, banner updated with F5=Screen-mode

## Architecture Decisions
- **L0 (text)**: Standard `generate_content` API (stateful via history) — fallback mode
- **L1 (live)**: Live API (`bidiGenerateContent`) via `aio.live.connect()` — primary mode
- **Model (live)**: `gemini-2.5-flash-native-audio-latest` — audio-native, confirmed working
- **Model (text)**: `gemini-2.5-flash` for text fallback (separate, larger free-tier quota)
- **Relay (live)**: Persistent task — runs for entire WS connection lifetime
- **Relay (text)**: Per-request task — created after each send_text
- **Fallback**: GeminiSession tries live → falls back to text on any connect error
- **Fallback**: AudioCapture/AudioPlayback/VAD/PTT all import-guarded

## Architecture Decisions (Day 8 / L3)
- **Function calling**: Uses Gemini's native function_declarations + FunctionResponse
- **Tool execution**: All tools run locally (not on cloud) — local/tools.py
- **Relay loop**: relay detects tool_call → forwards to local → waits for result via Queue → feeds back → loops
- **Safety**: 5-round max per request, 60s timeout per tool call, command blocklist
- **generate_content vs stream**: Switched to non-streaming generate_content for L3 (function calls don't chunk well in streams)

### Day 12: Dashboard UI Wiring (L6) — COMPLETE
- [x] Created `ui/dashboard/index.html` — single-page layout, dense grid
  - [x] Top bar: brand logo (SVG), connection status pill, latency, model badge, session mode
  - [x] Left column: full-height live transcript panel
  - [x] Right column: struggle gauge + tool execution log + system health grid
  - [x] All icons are inline SVG (no emojis, no external icon libs)
- [x] Created `ui/dashboard/css/style.css` — Obsidian Slate dark theme
  - [x] CSS custom properties (design tokens) for all colors
  - [x] IBM Plex Sans + IBM Plex Mono via Google Fonts
  - [x] Panels, pills, cards, scrollbars, progress bars
  - [x] Responsive: stacks to single column below 860px
  - [x] Pulse animation for connecting state
  - [x] Smooth transitions on all interactive elements
- [x] Created `ui/dashboard/js/websocket.js` — WebSocket connection manager
  - [x] Auto-connect to /ws/dashboard on page load
  - [x] Exponential backoff reconnect (1s up to 15s)
  - [x] Ping keepalive every 10s
  - [x] Event dispatching: routes by type/subtype to other modules
  - [x] Connection status UI updates (pill color changes)
- [x] Created `ui/dashboard/js/transcript.js` — live conversation display
  - [x] SVG avatars for user/rio/system messages
  - [x] Auto-scroll with smart detection (pauses on manual scroll-up)
  - [x] Timestamp per message
  - [x] Prunes at 200 messages
  - [x] Fade-in animation on new messages
- [x] Created `ui/dashboard/js/gauge.js` — struggle confidence meter
  - [x] SVG semicircular arc gauge with gradient (green-yellow-red)
  - [x] Animated needle + arc fill with easing
  - [x] Numeric value display with color coding by threshold
  - [x] 4 signal indicator tiles with active/high states
  - [x] Label updates: No signals / Low activity / Monitoring / Struggling detected
- [x] Created `ui/dashboard/js/toollog.js` — tool execution log
  - [x] Per-tool SVG icons (read_file, write_file, patch_file, run_command, capture_screen)
  - [x] Pending/success/error status icon coloring
  - [x] Argument display (path for file ops, command for shell)
  - [x] Auto-prunes at 50 entries
- [x] Created `ui/dashboard/js/health.js` — system health panel
  - [x] RPM usage with progress bar (green/warning/error states)
  - [x] Session uptime timer
  - [x] Message counter, screenshot counter, struggle trigger counter
  - [x] Rate limiter status (Normal/Caution/Emergency/Critical)
  - [x] Latency polling with color coding
  - [x] Session mode + model badge updates from control events
- [x] Modified `cloud/main.py`:
  - [x] Added `_dashboard_health_broadcast_loop()` — pushes RPM/health every 3s
  - [x] Background task created in lifespan, cancelled on shutdown
  - [x] Broadcasts rate_limit status + health stats
  - [x] Added client connect/disconnect dashboard events
- [x] Version remains v0.5.1 (L6 is UI-only, no local client changes needed)

### Day 13-14: Polish (L7) — COMPLETE
- [x] **Pro model routing** — ModelRouter fully wired
  - [x] `ModelRouter` constructor takes `api_key`, `rate_limiter`, `pro_rpm_budget`
  - [x] `should_use_pro()` — keyword-based trigger detection (PRO_TRIGGER_PHRASES)
  - [x] `call_pro()` — real `generate_content` call to `gemini-2.5-pro-preview-03-25`
  - [x] Pro RPM budget tracking (rolling 60s window, default 5 RPM)
  - [x] Rate limiter integration (BACKGROUND priority for Pro calls)
  - [x] `inject_pro_result()` — feeds Pro analysis into Flash Live session via callback
  - [x] `call_pro_and_inject()` — combined fire-and-forget helper
  - [x] Dashboard model badge updates (Flash ↔ Pro) via broadcast
  - [x] Health endpoint includes Pro RPM stats
- [x] **Degradation ladder** — Rate limiter wired to drop features under load
  - [x] EMERGENCY+: vision frames dropped at cloud (image binary frames rejected)
  - [x] CAUTION+: Pro escalation blocked (Flash handles everything)
  - [x] Dashboard rate_limit broadcast includes vision_active + pro_active flags
- [x] **Reconnect handling** — Gemini session reconnect with relay restart
  - [x] SessionManager flags reconnects via `_reconnect_needed` dict
  - [x] `check_reconnect()` method for main.py to poll and consume
  - [x] `_reconnect_checker` background task in WS handler polls every 5s
  - [x] On reconnect: cancels old relay, swaps `gemini` reference, restarts relay
  - [x] Updates Pro inject callback to use new session
  - [x] Sends `reconnected` control frame to local client
  - [x] Broadcasts reconnect event to dashboard
  - [x] Local client: handles `reconnected` control action in receive_loop
- [x] **Interrupt handling** — Speech-start clears playback buffer
  - [x] VAD speech-start edge detection (vad_was_speaking flag)
  - [x] Playback buffer cleared on first speech frame (not just PTT press)
- [x] **Memory integration** — MemoryStore wired into main loop
  - [x] Memory initialized from config (db_path, max_recall) in main()
  - [x] Tool results stored in memory after successful execution
  - [x] Struggle context queries memory for similar past issues
  - [x] Memory entries injected into struggle context payload
  - [x] Struggle trigger events stored in memory
  - [x] Graceful degradation: memory features skip if chromadb/sentence-transformers missing
- [x] **Error handling hardened**
  - [x] Reconnect checker wrap with try/except (no unhandled crashes)
  - [x] Pro escalation error handling (fire-and-forget with exception logging)
  - [x] `send_json_resilient()` added to WSClient (retry with backoff)
  - [x] Global exception handler in local main entry point
- [x] Version bumped to v0.6.0 (Layer 7: Polish)

## Architecture Decisions (Day 13-14 / L7)
- **Pro routing**: Keyword-based trigger (simple & predictable). Pro is async fire-and-forget — Flash keeps running.
- **Pro injection**: Callback-based — main.py sets the callback, ModelRouter calls it. Decouples routing from session management.
- **Degradation**: Cloud-side only — local client doesn't need to know. Vision frames and Pro calls silently dropped when RPM is high.
- **Reconnect**: Poll-based rather than event-based for simplicity. 5s polling interval is fast enough, negligible CPU.
- **Memory in struggle**: RAG context injected into the struggle prompt so Gemini can reference past similar issues.
- **Interrupt**: VAD edge detection mirrors PTT edge detection pattern — clear playback on speech start, not on every speech frame.

## Upcoming
- L8: Demo + Docs (Days 15-16)

## Lessons
- `gemini-2.0-flash-exp` does NOT exist for bidiGenerateContent (Live API)
- `gemini-2.5-flash-native-audio-latest` works for Live API but ONLY with AUDIO modality
- Free-tier rate limits are per-model; gemini-2.5-flash has separate quota
- For L0 text, standard API is simpler and works; save Live API for L1 audio
- `generate_content_stream` returns an async generator (needs `await` to get it)
- Heredoc writes don't take effect if Python process has the file cached — clear `__pycache__`
- Always clear `__pycache__` before restarting after source changes
- Function calling requires `generate_content` (non-stream) for reliable tool call detection
- FunctionResponse goes in `role="user"` with `function_response` parts
- Tool declarations use dict-based Schema (safer than typed Schema with SDK version variance)
- Pro model calls must be async fire-and-forget — never block the Live relay
- Degradation gating should be cloud-side only (local client doesn't see RPM counters)
- Session reconnect requires swapping the gemini reference AND restarting relay tasks
- VAD interrupt should only fire on speech-start edge, not on every speech-positive frame
- Memory queries during struggle detection add meaningful context but must be non-blocking
