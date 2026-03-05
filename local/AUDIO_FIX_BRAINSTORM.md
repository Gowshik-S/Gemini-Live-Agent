# Rio Audio Pipeline — Brainstorm & Root Cause Analysis

## Current Architecture

```
[Mic] → AudioCapture(16kHz PCM16) → WebSocket → Cloud → Gemini Live API
                                                          ↓
[Speaker] ← AudioPlayback ← WebSocket ← Cloud ← Gemini (24kHz PCM16)
                 ↑
         resample 24kHz→44.1kHz
         (np.interp — LINEAR!)
```

## Root Causes Found

### 🔴 Critical: WASAPI Capture Has No Fallback (causes total audio input failure)

```
PortAudioError: Incompatible host API specific stream info [PaErrorCode -9984]
```

`AudioCapture.start()` tries WASAPI → fails → exception → capture=None → **text-only mode**.

Meanwhile `AudioPlayback.start()` has a try/except that retries without WASAPI. The capture is **missing this identical fallback**. This is why the user gets no mic input.

### 🔴 Critical: Resampling Uses Linear Interpolation (terrible audio quality)

```python
# Current code in _resample():
resampled = np.interp(new_indices, old_indices, samples)  # LINEAR!
```

`np.interp` does **linear interpolation** between samples. For audio resampling 24kHz→44.1kHz (ratio 1.8375), this means:
- **No anti-aliasing filter** — introduces high-frequency artifacts
- **Spectral images** — creates phantom frequencies above Nyquist
- Sounds "metallic", "buzzy", "muffled", or "underwater"
- Every single sample is interpolated (non-integer ratio)

Proper audio resampling requires a **sinc/polyphase filter** (band-limited interpolation). Linear interp is literally the worst method for audio.

### 🟡 Medium: Jitter Buffer Config Mismatch

Jitter buffer = 1920 bytes (40ms). But first audio frame from Gemini = ~46KB (~960ms). The jitter buffer is bypassed instantly. Meanwhile, smaller follow-up chunks may arrive with network jitter and underrun.

The jitter buffer should be tuned for the **tail** of the stream, not the initial burst.

### 🟡 Medium: Chunk Size In Playback Queue

After resampling, data goes into queue in 960-byte chunks. The callback requests `frames * 2` bytes per call. When block_size=882 (for 44.1kHz), that's 1764 bytes needed per callback. So each callback pull needs ~2 queue gets, and may get partial data → silence padding → **clicks**.

### 🟢 Low: Device Probing Selects Sound Mapper Instead of Direct Device

"Microsoft Sound Mapper" (device 3) is the Windows audio indirection layer, not the actual hardware device. It adds latency and potential format conversion issues. The probe should prefer the actual Realtek device directly.

## Fix Plan

### Fix 1: Add WASAPI Fallback to AudioCapture (mirrors playback)
Copy the same try/except retry pattern from `AudioPlayback.start()` into `AudioCapture.start()`.

### Fix 2: Replace np.interp with scipy.signal.resample_poly
- `scipy` is already available (it's a dependency of scikit-learn which is installed)
- `resample_poly(samples, up=147, down=80)` for 24000→44100
- GCD(24000, 44100) = 300, so up=44100/300=147, down=24000/300=80
- Built-in anti-aliasing FIR filter, orders of magnitude better quality
- Falls back to np.interp if scipy unavailable

### Fix 3: Increase jitter buffer to ~150ms
Change from 1920 bytes (40ms) to ~7200 bytes (150ms at 24kHz×1.8375 resampled). Better trade-off between latency and smoothness.

### Fix 4: Align playback chunk sizes with callback block_size
Make queue chunk size = block_size * 2 * channels to reduce partial reads.
