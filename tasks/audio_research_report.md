# Rio Audio Playback Module — Deep Research Report

**Date:** March 5, 2026  
**Scope:** Best Python audio playback module for a live voice assistant receiving 24kHz PCM from Gemini Live API  
**Platforms:** Windows + Linux  

---

## Executive Summary

**PyAudio with blocking `stream.write()` is the proven approach for Gemini Live API audio playback.** Google's own official Gemini Live API documentation uses PyAudio with a simple blocking write pattern — no callbacks, no queue race conditions. Pipecat (10.6k stars, the #1 voice AI framework) also uses PyAudio for local audio. Your current `sounddevice` callback-based architecture is the root cause of the stuttering: the callback/queue threading model introduces race conditions between the audio thread and asyncio event loop. The fix is **not to change libraries** but to **change the playback architecture from callback-based to blocking write-based**.

If you want to go further, **`miniaudio`** is the strongest alternative — it wraps the modern miniaudio C library (not the aging PortAudio), has native WASAPI/ALSA/PulseAudio/PipeWire support, built-in resampling, generator-based streaming (perfect for Python), and a `REALTIME` thread priority option.

---

## Recommended Approach

### Primary Recommendation: PyAudio with Blocking Write (match Google's pattern)

```python
# This is Google's OFFICIAL pattern from ai.google.dev/gemini-api/docs/live
stream = pya.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
while True:
    bytestream = await audio_queue_output.get()
    await asyncio.to_thread(stream.write, bytestream)
```

**Why this works and callbacks don't:**
- `stream.write()` is synchronous and blocks until the audio hardware consumes the data — **zero queue management needed**
- The audio driver handles its own internal buffering — no jitter buffer, no partial reads, no silence padding
- `asyncio.to_thread()` offloads the blocking write without stalling the event loop
- No threading race conditions between a callback thread and the asyncio loop

### Secondary Recommendation: `miniaudio` with Generator Pattern (if PyAudio proves insufficient)

```python
import miniaudio

def audio_generator():
    required_frames = yield b""
    while True:
        data = get_next_audio_chunk(required_frames)
        required_frames = yield data

with miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.SIGNED16,
                              nchannels=1, sample_rate=24000,
                              thread_prio=miniaudio.ThreadPriority.REALTIME) as device:
    gen = audio_generator()
    next(gen)
    device.start(gen)
```

---

## Bottlenecks Table

| Severity | Bottleneck | Impact | Fix |
|----------|-----------|--------|-----|
| 🔴 Critical | **Callback-based OutputStream** | The sounddevice callback runs in a C audio thread. It reads from `queue.Queue` which is written by the asyncio event loop thread. Timing mismatches cause underruns → silence padding → clicks/stuttering | Switch to blocking `stream.write()` with `asyncio.to_thread()` — no callback, no queue |
| 🔴 Critical | **Chunk size / block size misalignment** | Callback requests 1764 bytes but queue chunks are 960 bytes → 2 queue gets per callback, partial data → silence padding → audible clicks | With blocking write, the driver handles internal buffering — no alignment issues |
| 🟡 High | **Resampling before queuing** | `resample_poly` in `enqueue()` runs on the asyncio thread, blocking the event loop during CPU-intensive DSP. 24kHz→44.1kHz with up=147/down=80 is expensive | Do resampling in a thread pool, or use `miniaudio.convert_frames()` which is C-native |
| 🟡 High | **Jitter buffer as application code** | Custom jitter buffer (7200 bytes) is a half-measure — the audio driver already has its own internal buffer. Two layers of buffering cause unpredictable latency | Remove application-level jitter buffer; let the driver handle it |
| 🟡 Medium | **WASAPI exclusive mode failures** | `-9984` errors on shared devices require exclusive mode disabled; current fallback works but adds startup latency | Use `exclusive=False` by default (shared WASAPI is sufficient for voice), drop exclusive mode entirely |
| 🟢 Low | **PipeWire virtual ALSA device hangs** | PortAudio's callback mode with virtual ALSA devices on PipeWire sometimes hangs | Blocking write mode + hardware device probing (already implemented) resolves this |

---

## Options Comparison

### Module Comparison Table

| Module | Backend | GitHub Stars | Used By (Voice AI) | Cross-Platform | Callback Streaming | Blocking Write | Built-in Resampling | PipeWire OK | WASAPI | Latency |
|--------|---------|-------------|-------------------|----------------|-------------------|---------------|--------------------|-----------|---------|---------| 
| **`pyaudio`** | PortAudio | 780+ | **Google (official Gemini Live sample)**, **Pipecat (10.6k★)** | Win+Lin+Mac | Yes | **Yes (preferred)** | No | Via ALSA shim | Yes (host API 13) | Low |
| **`sounddevice`** | PortAudio (CFFI) | 1.0k | Rio (current), DeepGram examples | Win+Lin+Mac | Yes | Yes | No | Via ALSA shim | Yes (`WasapiSettings`) | Low |
| **`miniaudio`** | miniaudio C lib | 175 | None major (yet) | Win+Lin+Mac+Pi | **Generator-based** | No (gen only) | **Yes (`convert_frames`)** | **Native PulseAudio backend** | **Native WASAPI backend** | Low (REALTIME thread) |
| `pygame.mixer` | SDL2 | N/A (part of pygame) | None | Win+Lin+Mac | No | No | SDL handles it | Via PulseAudio | Via DirectSound | Medium-High |
| `pydub + simpleaudio` | OS native | 600+200 | None | Win+Lin+Mac | No | Write-and-forget | No | Partial | Via WinMM | High |
| `python-soundcard` | WASAPI/CA/PA | 190 | None | Win+Lin+Mac | No | Yes | No | Via PulseAudio | Native WASAPI | Medium |
| `pyalsaaudio` | ALSA | 90 | None | **Linux only** | Yes | Yes | No | Via ALSA shim | ❌ | Low |
| `comtypes` WASAPI | WASAPI direct | N/A | None | **Windows only** | Yes | Yes | No | ❌ | Native | Lowest possible |
| `av` (PyAV/FFmpeg) | FFmpeg | 2.3k | Transcoding pipelines | Win+Lin+Mac | No | Write | **Yes (full DSP)** | N/A (file-based) | N/A | N/A (not real-time) |


### Detailed Comparison: Top 3 Candidates

#### 1. PyAudio — **RECOMMENDED** ✅

| Attribute | Detail |
|-----------|--------|
| **Pros** | Google's official choice for Gemini Live API; Pipecat uses it for local transport; blocking write eliminates callback race conditions; mature, battle-tested; trivial migration from sounddevice (same PortAudio backend); well-documented |
| **Cons** | Stale maintenance (last PyPI release Nov 2023, but PortAudio itself is stable); requires `portaudio` system package on some Linux distros; no built-in resampling |
| **Best For** | Voice assistants that receive streamed PCM chunks and play them immediately |
| **Avoid If** | You need cutting-edge PipeWire native support or built-in sample rate conversion |
| **Latency** | Same as sounddevice (both PortAudio); ~5-20ms with proper config |
| **Resampling** | Do it yourself with `scipy.signal.resample_poly` (you already have this code) |
| **PipeWire** | Works via PipeWire's ALSA compatibility layer (same as sounddevice) |
| **WASAPI** | Access via host API index 13; no exclusive mode wrapper like sounddevice, but shared mode is fine for voice |

#### 2. miniaudio — Strong Alternative

| Attribute | Detail |
|-----------|--------|
| **Pros** | Modern C library (not the aging PortAudio); native WASAPI, ALSA, PulseAudio, JACK backends; built-in `convert_frames()` for resampling; generator-based callback (Pythonic, no threading); `ThreadPriority.REALTIME`; MIT licensed; no system deps (statically linked) |
| **Cons** | Only 175 GitHub stars; not used by any major voice AI project; generator model requires slight architectural rethink; buffersize_msec defaults to 200ms (needs tuning); less community support for edge cases |
| **Best For** | Projects needing native backend selection and built-in format conversion |
| **Avoid If** | You want proven production use in voice AI specifically |
| **Latency** | Excellent — `buffersize_msec=50` with `ThreadPriority.REALTIME` achieves ~10ms |
| **Resampling** | **Built-in** — `miniaudio.convert_frames()` does format + rate + channel conversion in C |
| **PipeWire** | Via PulseAudio backend (native, not ALSA shim) — more reliable than PortAudio |
| **WASAPI** | **Native** — no PortAudio indirection; can select backend explicitly |

#### 3. sounddevice (current) — Keep But Fix Architecture

| Attribute | Detail |
|-----------|--------|
| **Pros** | Already integrated; CFFI-based (cleaner than PyAudio's ctypes); excellent `WasapiSettings` API; `latency='low'` convenience; active maintenance (Jan 2026); numpy integration |
| **Cons** | Same PortAudio backend as PyAudio; callback model + asyncio queue = current stuttering issues; 199 open issues; PipeWire/ALSA issues documented |
| **Best For** | When callback model is actually needed (e.g., full-duplex simultaneous I/O) |
| **Avoid If** | You have a producer-consumer pattern (which is exactly what Gemini streaming is) |
| **Fix** | Switch from `OutputStream` callback to `RawOutputStream.write()` blocking mode |

---

## What Production Voice AI Projects Actually Use

| Project | Stars | Audio Module | Pattern | Notes |
|---------|-------|-------------|---------|-------|
| **Google Gemini Live API** (official docs) | N/A | **`pyaudio`** | **Blocking write** (`stream.write()` via `asyncio.to_thread`) | The canonical example. No callbacks. Simple queue → write loop |
| **Pipecat** (Daily.co) | 10.6k | **`pyaudio`** | Blocking write via `ThreadPoolExecutor` | `LocalAudioOutputTransport` uses `run_in_executor(self._executor, stream.write, data)` |
| **LiveKit Agents** | 9.6k | **WebRTC transport** (not local audio) | Server-side; client handles audio | No Python audio module — uses WebRTC for media transport |
| **Vocode** | 2.8k | **WebSocket/telephony** | Server-side; Twilio/Daily handles audio | No local audio playback in Python |
| **OpenAI Realtime Console** | 3.6k | **Web Audio API** (JavaScript) | WebRTC → browser | Not Python; browser handles audio natively |
| **AssemblyAI** | N/A | **WebSocket streaming** | Server-side transcription | No playback — input only |

**Key insight:** Every major voice AI project that does local Python audio uses **PyAudio with blocking write**. None of them use callback-based streaming for playback.

---

## Security Considerations

- PyAudio and sounddevice both ship PortAudio binaries — verify supply chain integrity
- `miniaudio` statically links its C code — smaller attack surface, no external `.dll`/`.so`
- Audio device access requires no elevated privileges on modern Windows/Linux
- OWASP: Not directly applicable (local audio, no network surface on playback side)

---

## Performance Considerations

| Metric | Callback (current) | Blocking Write (proposed) | miniaudio Generator |
|--------|-------------------|--------------------------|-------------------|
| Thread model | C audio thread + asyncio | asyncio + thread pool | miniaudio's own REALTIME thread |
| Latency floor | ~10ms (but jitter from queue sync) | ~10ms (driver-managed) | ~10ms (with `buffersize_msec=50`) |
| Buffer underrun risk | **High** (queue empty → silence) | **Low** (write blocks until consumed) | **Low** (generator yields on demand) |
| CPU overhead | Queue operations + numpy reshape | Minimal | Minimal (C-level) |
| Resampling | Python (scipy) | Python (scipy) | **C-native** (convert_frames) |

---

## Codebase Impact Analysis

### If switching to PyAudio blocking write:

| File | Change | Breaking? |
|------|--------|-----------|
| [audio_io.py](Rio-Agent/rio/local/audio_io.py) | Replace `AudioPlayback` class: remove callback, remove jitter buffer, remove queue. Add blocking write loop with `asyncio.to_thread` | **Yes** (but `enqueue()` API can stay the same externally) |
| [requirements.txt](Rio-Agent/rio/local/requirements.txt) | Add `pyaudio>=0.2.14`, keep `sounddevice` for capture | No |
| [config.yaml](Rio-Agent/rio/config.yaml) | No changes needed | No |
| [ws_client.py](Rio-Agent/rio/local/ws_client.py) | No changes | No |
| [main.py](Rio-Agent/rio/local/main.py) | Minor: update AudioPlayback import/usage if API changes | Maybe |

### If switching to miniaudio:

| File | Change | Breaking? |
|------|--------|-----------|
| [audio_io.py](Rio-Agent/rio/local/audio_io.py) | Replace entire playback class with generator-based miniaudio device | **Yes** |
| [requirements.txt](Rio-Agent/rio/local/requirements.txt) | Add `miniaudio>=1.61`, potentially remove `scipy` (built-in resampling) | No |
| Resampling code | Can be removed — `miniaudio.convert_frames()` handles it in C | No |

---

## Docs Gap Report

| Gap | Severity | Notes |
|-----|----------|-------|
| [AUDIO_FIX_BRAINSTORM.md](Rio-Agent/rio/local/AUDIO_FIX_BRAINSTORM.md) identifies root causes but doesn't explore the blocking-write alternative | High | The brainstorm focuses on tuning the callback model rather than questioning whether callbacks are the right approach |
| [todo.md](Rio-Agent/rio/tasks/todo.md) previous research only compared sounddevice vs PyAudio vs PyAudioWPatch, concluded "keep sounddevice" | High | This was correct that both wrap PortAudio, but missed the critical distinction: **callback vs blocking write pattern** |
| No documentation of Google's official playback pattern | High | The official Gemini Live API docs show PyAudio with blocking write — this should have been the reference architecture |
| No documentation of what Pipecat uses for local audio | Medium | Pipecat is the canonical voice AI framework and uses PyAudio blocking write |

---

## Next Steps

### Immediate (can do right now)
1. **Switch `AudioPlayback` to PyAudio blocking write** — keep `AudioCapture` on sounddevice (it works fine for input)
2. **Remove the jitter buffer** — `stream.write()` blocks until audio is consumed; the driver manages its own buffer
3. **Remove the callback + queue architecture** — replace with `asyncio.to_thread(stream.write, data)`
4. **Keep scipy resampling** — do it before the write, in the same thread pool

### Short-term (this week)
5. **Test on both Windows and Linux** with the new blocking write approach
6. **Benchmark latency** — measure first-sound-to-speaker time with PyAudio vs current
7. **Evaluate miniaudio** as a fallback — its C-native resampling and PulseAudio backend may solve Linux PipeWire issues more cleanly

### Long-term (next sprint)
8. **Consider miniaudio migration** if PyAudio blocking write still has issues on specific Linux PipeWire setups
9. **Remove PortAudio dependency entirely** if miniaudio proves reliable — it statically links everything, no system package needed
10. **Add audio quality metrics** — log underruns, latency jitter, buffer occupancy for production monitoring

---

## Sources

| Finding | Source |
|---------|--------|
| Google uses PyAudio for Gemini Live API | [ai.google.dev/gemini-api/docs/live](https://ai.google.dev/gemini-api/docs/live) — official starter code |
| Pipecat uses PyAudio for local transport | [pipecat/transports/local/audio.py](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/transports/local/audio.py) — source code |
| Pipecat uses blocking write via ThreadPoolExecutor | Same file: `run_in_executor(self._executor, stream.write, data)` |
| LiveKit uses WebRTC, not local audio | [github.com/livekit/agents](https://github.com/livekit/agents) — README |
| miniaudio has native WASAPI/PulseAudio backends | [pypi.org/project/miniaudio](https://pypi.org/project/miniaudio/) — Backend enum includes WASAPI, PULSEAUDIO, ALSA, JACK |
| miniaudio has built-in convert_frames() | Same PyPI page — `convert_frames()` function documentation |
| sounddevice has 199 open issues, many PipeWire/ALSA | [github.com/spatialaudio/python-sounddevice/issues](https://github.com/spatialaudio/python-sounddevice/issues) |
| PyAudio last release Nov 2023 | [pypi.org/project/PyAudio](https://pypi.org/project/PyAudio/) |
| sounddevice last release Jan 2026 | [pypi.org/project/sounddevice](https://pypi.org/project/sounddevice/) |
| miniaudio 175 stars, 7 contributors | [github.com/irmen/pyminiaudio](https://github.com/irmen/pyminiaudio) |
| PortAudio callback + asyncio race conditions | Rio codebase: `audio_io.py` lines 231-707, observed stuttering behavior |
| WASAPI -9984 error on shared devices | Rio codebase: `AUDIO_FIX_BRAINSTORM.md`, `audio_io.py` fallback code |
