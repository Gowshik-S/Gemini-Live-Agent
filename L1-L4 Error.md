# Rio-Agent — Complete Code Review: Issues & Hallucinations

## 🔴 CRITICAL BUGS (Will crash or silently fail at runtime)

### 1. **Audio binary prefix is empty — audio frames are indistinguishable from image frames**
- **Files:** `local/audio_io.py` line `AUDIO_PREFIX = b""`, `local/screen_capture.py` line `IMAGE_PREFIX = b""`
- **Impact:** Both prefixes are empty strings. The wire protocol specifies `0x01` for audio and `0x02` for images, but the actual code sends `b"" + payload` for both. On the cloud side (`cloud/main.py`), the prefix check `if prefix == b""` is identical for BOTH audio and image branches — the second `elif prefix == b""` clause is **dead code** and images are being handled as audio.
- **Fix:** Set `AUDIO_PREFIX = b"\x01"` and `IMAGE_PREFIX = b"\x02"`.

### 2. **Audio playback silence padding is broken — uses empty bytes instead of zero bytes**
- **File:** `local/audio_io.py`, `AudioPlayback._playback_callback()`
- **Line:** `audio_bytes = bytes(collected) + b"" * (bytes_needed - len(collected))`
- **Impact:** `b"" * N` is always `b""` (empty). The silence padding does nothing. When there's insufficient audio data, `np.frombuffer` will get a buffer smaller than expected, causing a **ValueError crash** when reshaping (`outdata[:] = ...reshape(-1, self._channels)`).
- **Fix:** Should be `b"\x00" * (bytes_needed - len(collected))`.

### 3. **Cloud relay sends audio with empty prefix — client can't distinguish response types**
- **File:** `cloud/main.py`, `_relay_live_to_client()`
- **Line:** `await websocket.send_bytes(b"" + item["data"])`
- **Impact:** Same issue — should be `b"\x01"` prefix. On the receive side in `local/main.py`, the check `if prefix == b""` matches the first byte of actual audio data, not a protocol prefix.

### 4. **`config.py` default value access on class attributes fails**
- **File:** `local/config.py`, `_from_dict()` method
- **Lines:** `cloud_url=d.get("cloud_url", cls.cloud_url)` and `session_mode=d.get("session_mode", cls.session_mode)`
- **Impact:** `cls.cloud_url` accesses a **dataclass field descriptor**, not the default value string `"ws://localhost:8080/ws/rio/live"`. In Python, accessing `RioConfig.cloud_url` on the class (not instance) returns the default value only because dataclasses set it as a class-level attribute. This works by accident but is fragile and non-idiomatic. Should use `RioConfig().cloud_url` or hardcode the defaults.

### 5. **`_build()` helper uses `.values()` instead of direct iteration on `__dataclass_fields__`**
- **File:** `local/config.py`
- **Line:** `valid_keys = {f.name for f in klass.__dataclass_fields__.values()}`
- **Impact:** This actually works correctly. No bug here (false alarm on closer inspection).

---

## 🟠 SIGNIFICANT LOGIC ISSUES

### 6. **Struggle detection feeds force-captured frames — defeats delta detection for screen_capture_loop**
- **File:** `local/main.py`, `struggle_detection_loop()`
- **Line:** `jpeg = await screen.capture_async(force=True)`
- **Impact:** Every 2 seconds, this force-captures a new frame, which **updates `_last_hash`** in `ScreenCapture`. Since `force=True` skips the delta check but doesn't skip updating internal state... wait, actually looking at `screen_capture.py`: when `force=True`, the `_last_hash` is **not updated** because the delta detection block is skipped entirely. So the periodic `screen_capture_loop` still works correctly. However, the forced captures are **not fed into the screen's delta state**, meaning: if the struggle loop captures a frame identical to the autonomous loop's last frame, the autonomous loop will still send it. Minor inefficiency, not a bug.

### 7. **Struggle Signal 1 (Repeated Error) uses screen JPEG hash, NOT error text hash**
- **File:** `local/struggle_detector.py`, `_signal_repeated_error()`
- **Impact:** The plan says "Hash match on screen text" and "OCR'd error text hash → same error 3+ times". The implementation hashes the **entire JPEG screenshot**. This means pixel-identical screenshots trigger the signal, but the same error with a slightly different cursor position, scrollbar, or clock will produce a different hash and NOT trigger. The signal is far less reliable than designed. The plan mentions "Gemini vision or local OCR" but neither is used — it's raw JPEG MD5.

### 8. **Struggle Signal 2 clears error state too aggressively**
- **File:** `local/struggle_detector.py`, `feed_gemini_response()`
- **Impact:** If Gemini sends ANY response without error keywords, `_error_detected` is set to `False`. This means: User gets error → Gemini mentions it (error_detected=True) → User asks "how do I fix this?" → Gemini responds with fix instructions (no error keywords) → error_detected is cleared → Signal 2 (long pause on error) becomes impossible to fire even though the user is still stuck on the error.

### 9. **Struggle Signal 3 and Signal 1 can be mutually exclusive by design**
- **Impact:** Signal 1 fires when the same hash appears 3+ times (screen unchanged = stuck). Signal 3 fires when 5+ distinct hashes appear (rapid changes = trial-and-error). These two signals can't fire simultaneously for the same timeframe, reducing the chance of reaching the 0.85 threshold. Max confidence with only one of these = 0.35 + 0.25 + 0.20 = 0.80 (still below 0.85 if only 3 signals fire). Actually: S1(0.35) + S2(0.25) = 0.60, S1 + S4 = 0.55, S3 + S4 = 0.40, S3 + S2 = 0.45. The ONLY combination reaching 0.85 is S1+S2+S4 (0.80) — still not enough! Or S1+S2+S3 impossible. S1+S2+S4 = 0.80, S1+S3+S4 = 0.75. **In normal mode (threshold 0.85), it's impossible to trigger with any combination of 3 signals!** You need all 4 (sum=1.0), which is contradictory since S1 and S3 conflict. This means the struggle detector can **never trigger in normal mode** unless S1+S2+S3+S4 all fire simultaneously.

### 10. **`record_decline()` is never called anywhere**
- **Files:** `local/struggle_detector.py` defines `record_decline()`, but it's never invoked
- **Impact:** The decline cooldown (10 minutes) never activates. Context.txt acknowledges this: "record_decline() integration (needs Gemini to detect 'no thanks')". But the system will re-trigger every 5 minutes indefinitely even if the user says "no."

### 11. **Text mode relay: context frames don't trigger a relay task**
- **File:** `cloud/main.py`
- **Impact:** When a `context` (struggle) frame arrives in text mode, `gemini.send_context()` appends to history, but no relay task is created to call `gemini.receive()`. The proactive response from Gemini will only be retrieved when the NEXT user text message triggers a relay. The struggle detector fires → context is sent → nothing happens until the user types something. In live mode this works because the persistent relay handles it.

### 12. **Rate limiter `can_call()` and `record_call()` are separate — race condition**
- **File:** `cloud/main.py`
- **Lines:** `if not rate_limiter.can_call(...)` then later `rate_limiter.record_call(...)`
- **Impact:** Between the check and the record, another concurrent request could slip through. Not a practical issue with single-client usage, but architecturally unsound.

---

## 🟡 INCONSISTENCIES BETWEEN PLAN/DOCS AND CODE

### 13. **Model mismatch: Plan says gemini-2.0-flash, code uses gemini-2.5-flash**
- **Plan:** `PRIMARY = gemini-2.0-flash (Live API)` throughout Rio-Plan.md
- **Config:** `models.primary: "gemini-2.0-flash"` in config.yaml
- **Code:** `TEXT_MODEL = "gemini-2.5-flash"` hardcoded in gemini_session.py
- **Impact:** Config is completely ignored. The context.txt documents this known issue but it's still a drift.

### 14. **Plan says 6 struggle signals, code implements 4**
- **Plan (Rio-Plan.md §9):** 6 signals with sklearn RandomForest scorer
- **Code:** 4 signals, pure weighted sum, no sklearn
- **Context.txt** correctly documents this as "deferred to L7" but the plan was never updated.

### 15. **Plan says sklearn RandomForest, code uses simple weighted sum**
- **Plan:** "Feature vector [6 floats] → sklearn RandomForest (50KB)"
- **Code:** Pure `confidence = sum(weights)` — no ML model at all
- **Impact:** The tech stack table in Rio-Plan.md lists "Rule engine + sklearn RF (~5 MB)" which is misleading.

### 16. **Plan says "OCR'd error text hash" for Signal 1, code uses raw JPEG hash**
- See issue #7 above. No OCR is performed. No text extraction from screenshots.

### 17. **Plan says `capture_screen` tool doesn't exist — but code adds it**
- **Plan (Rio-Plan.md §4):** Tool list is `read_file, write_file, patch_file, run_command`
- **Code:** 5 tools — includes `capture_screen` (added in v0.5.1)
- **Impact:** Doc/plan drift — not a bug, but the plan is stale.

### 18. **Plan says "confirmation gate: auto-approve reads, confirm writes/deletes"**
- **Code:** There is NO confirmation gate. All tool calls are auto-executed.
- **Impact:** `write_file`, `patch_file`, and `run_command` all execute without user confirmation, contrary to the architecture plan.

### 19. **Wire protocol doc says `0x01` and `0x02` prefixes, code uses `b""`**
- See issue #1. The wire protocol specification in both Rio-Plan.md and context.txt clearly documents `0x01 = audio, 0x02 = image`, but the actual implementation uses empty byte strings.

### 20. **Config.yaml `struggle` label says "Layer 3+" but struggle is Layer 4**
- **File:** `rio/config.yaml`, comment `# Struggle detection settings (Layer 3+)`
- **Should be:** `# Struggle detection settings (Layer 4+)`

### 21. **Config.yaml `memory` label says "Layer 4+" but memory is Layer 5**
- **File:** `rio/config.yaml`, comment `# Local memory / RAG settings (Layer 4+)`
- **Should be:** `# Local memory / RAG settings (Layer 5+)`

---

## 🔵 SECURITY ISSUES

### 22. **API key exposed in `.env` file committed to repository**
- **File:** `Rio-Agent/rio/cloud/.env`
- **Content:** `GEMINI_API_KEY=AIzaSyBgEptVaxEkhgG26Di83dGOy73BhHlMFVg`
- **Impact:** The actual API key is checked into the repo. The `.gitignore` should exclude it, but the file exists in the project tree. **This key should be rotated immediately.**

### 23. **Command blocklist has regex gaps**
- **File:** `local/tools.py`
- **Example:** `rm -rf /` is blocked but `rm -rf /*` is NOT blocked. Also `rm -rf $HOME` is not blocked. `sudo rm -rf /` bypasses the patterns. The pattern `r"\brm\s+(-\w+\s+)*-r\w*\s+/\s*$"` requires the line to end with `/` — any trailing content (like `*`) evades it.
- Other gaps: `curl | bash`, `wget | sh`, `pip install` (arbitrary packages), `python -c "import os; os.system('rm -rf /')"` all bypass the blocklist.

### 24. **Path traversal protection has edge case**
- **File:** `local/tools.py`, `_resolve()`
- **Line:** `if not str(resolved).startswith(str(cwd_resolved) + os.sep) and resolved != cwd_resolved`
- **Impact:** If `cwd_resolved` is `/home/user/project`, a path like `/home/user/project_evil/hack.py` would pass the `startswith` check because `/home/user/project_evil` starts with `/home/user/project`. Should use `resolved.is_relative_to(cwd_resolved)` (Python 3.9+).

### 25. **`shlex.split` doesn't work correctly on Windows**
- **File:** `local/tools.py`, `_run_sync()`
- **Impact:** `shlex.split()` uses POSIX quoting rules. On Windows, this will incorrectly parse commands. The plan claims "Supported OS: Windows 10+" but the command execution is Linux-centric.

---

## 🟣 MINOR ISSUES & CODE QUALITY

### 26. **`_handle_sigint()` function defined but never registered**
- **File:** `local/main.py`
- **Line:** `def _handle_sigint() -> None:` — exists but is never called or registered as a signal handler.

### 27. **Duplicate heartbeat mechanisms**
- `local/main.py` has a `heartbeat_loop()` that sends `{"type": "heartbeat"}` every 10s
- `local/ws_client.py` has an internal `_heartbeat_loop()` that sends WebSocket pings every 10s
- **Impact:** Redundant — both accomplish keep-alive but via different mechanisms.

### 28. **`screen_capture.py` log message has comma in wrong place**
- **File:** `local/screen_capture.py`
- **Line:** `log.warning("screen_capture.no_primary_monitor, using monitors[0]")`
- **Impact:** This is a single string, not a structured log. Should be: `log.warning("screen_capture.no_primary_monitor", note="using monitors[0]")`

### 29. **`audio_io.py` AUDIO_PREFIX is `b""` but `main.py` also imports it and uses it**
- The import guard in `main.py` sets fallback `AUDIO_PREFIX = b""` — same empty value as the actual module. This means the fallback is indistinguishable from the real value.

### 30. **Dashboard files exist but are not wired to struggle/memory data**
- Context.txt confirms: "Dashboard UI exists but needs wiring to struggle/memory"
- The dashboard JS files connect to `/ws/dashboard` but the cloud only sends health and transcript data reliably. Struggle gauge updates depend on the cloud correctly broadcasting struggle events.

### 31. **`session_manager.py` `touch()` may not exist or may be incomplete**
- **File:** `cloud/main.py` calls `await session_manager.touch(client_id)` on heartbeat
- **Impact:** Need to verify `SessionManager.touch()` is implemented. Based on earlier reads, `session_manager.py` does define lifecycle management but the `touch()` method existence couldn't be re-verified in this session.

### 32. **No `__init__.py` files in `cloud/` or `local/` directories**
- **Impact:** These directories rely on being run with the directory as the working directory (e.g., `cd local && python main.py`). Not an issue for the intended usage pattern but makes the code non-importable as packages.

---

## Summary

| Severity | Count | Categories |
|----------|-------|------------|
| 🔴 Critical (will crash/fail) | 3 | Empty wire protocol prefixes (#1, #3), silence padding crash (#2) |
| 🟠 Significant logic issues | 7 | Impossible threshold (#9), dead code paths (#11), signal unreliability (#7, #8) |
| 🟡 Plan/doc inconsistencies | 9 | Model mismatch, signal count, missing confirmation gate |
| 🔵 Security | 4 | Exposed API key, blocklist gaps, path traversal edge case |
| 🟣 Minor/quality | 7 | Unused code, redundant heartbeats, logging issues |
| **Total** | **30** | |

**The most impactful issues to fix immediately are #1/#3 (wire protocol prefixes), #2 (silence crash), #9 (impossible threshold), and #22 (exposed API key).**