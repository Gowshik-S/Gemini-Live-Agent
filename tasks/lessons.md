# Rio — Lessons Learned

## Audio Pipeline
- PortAudio (via sounddevice/pyaudio) has persistent issues with PipeWire virtual ALSA devices on Linux — prefer probing hardware devices first
- WASAPI exclusive mode causes -9984 errors on shared-mode devices — always implement fallback
- Linear interpolation (`np.interp`) for audio resampling produces metallic/buzzy artifacts — always use `scipy.signal.resample_poly` with proper anti-aliasing
- Callback-based audio + asyncio queues are a race condition minefield — the audio thread and event loop thread must be carefully synchronized
- Jitter buffer sizing is critical: too small = underruns/clicks, too large = perceptible latency
- Chunk size must align with callback block_size to avoid partial reads → silence padding → clicks

## Screen Navigation (Research — March 5, 2026)
- **Coordinate mapping is THE #1 bug source in AI computer-use.** The `resize_factor` in `screen_capture.py` means Gemini sees a scaled-down image. Every coordinate must be inverse-scaled before pyautogui execution: `real = screenshot_coord / resize_factor`
- **DPI awareness MUST be set before any screen capture or pyautogui calls.** On Windows, call `ctypes.windll.shcore.SetProcessDpiAwareness(2)` at process startup. Without it, both mss and pyautogui get virtualized coordinates on >100% DPI displays.
- **pyautogui is NOT thread-safe.** All calls must be serialized through a single thread (asyncio.to_thread or dedicated executor). Never call pyautogui from multiple async tasks.
- **Wayland breaks everything:** pyautogui, pynput, xdotool all fail on Wayland. Must detect at startup and degrade gracefully. ydotool and wtype are the best Wayland alternatives but require root.
- **Post-action screenshot is essential for reliability.** AI computer-use systems that don't verify their actions have <50% reliability. Always capture + send a new screenshot after click/type actions so Gemini can verify the result.
- **Set-of-Mark (numbered element labels) dramatically improves accuracy** over raw coordinate guessing. OCR bounding boxes are sufficient as element detectors for text-heavy developer UIs.
- **Always store capture metadata (monitor offset, original resolution, resize factor) alongside JPEG bytes.** Without this, coordinate mapping is impossible to get right in multi-monitor or non-standard DPI setups.
- **pyautogui.typewrite() does NOT support Unicode.** Use pynput.keyboard.Controller.type() or clipboard paste (pyperclip + ctrl+v) for non-ASCII text.
- **Rate limit screen actions.** An infinite click loop can crash/corrupt the system. Max 20 actions/min + pyautogui.FAILSAFE on screen corner.
