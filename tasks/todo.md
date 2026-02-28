# Rio Build Progress

## Current Layer: L3 — Tools (Day 8) ✅ COMPLETE

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

## Upcoming
- L1: Voice via Live API (Days 3-5 rework — parallel)
- L4: Struggle Detection (Days 9-10)
- L5: Memory (Day 11)
- L6: Dashboard (Day 12)
- L7: Polish (Days 13-14)
- L8: Demo (Days 15-16)

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
