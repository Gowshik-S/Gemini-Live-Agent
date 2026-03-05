# Rio Screen Navigation — Complete Implementation Plan

## Executive Summary

Rio needs full screen navigation so Gemini can **see the screen and interact with it** — clicking buttons, typing text, scrolling, managing windows, and navigating arbitrary applications. This turns Rio from a "look-only" assistant into a true **computer-use agent**.

**Primary module:** `pyautogui` (cross-platform, 10k+ GitHub stars, proven in hundreds of automation projects)  
**Accuracy boost:** Set-of-Mark (SoM) numbered overlays on screenshots  
**Architecture:** Gemini decides → cloud relays → local client executes via pyautogui

---

## Architecture

```
┌──────────────────────────────────────────┐
│              Gemini Live API             │
│ (sees screenshots, decides actions)      │
│                                          │
│ "Click the Submit button at (450, 320)"  │
│ → tool_call: screen_click(450, 320)      │
└─────────────────┬────────────────────────┘
                  │ WebSocket
                  ▼
┌──────────────────────────────────────────┐
│           Cloud Relay (main.py)          │
│ Forwards tool_call to local client       │
└─────────────────┬────────────────────────┘
                  │ WebSocket
                  ▼
┌──────────────────────────────────────────┐
│           Local Client (tools.py)        │
│                                          │
│ screen_navigator.py:                     │
│  ├─ CoordinateMapper  (resize → real)    │
│  ├─ screen_click(x, y, button)           │
│  ├─ screen_type(text)                    │
│  ├─ screen_scroll(x, y, clicks)          │
│  ├─ screen_hotkey(keys)                  │
│  ├─ screen_drag(x1,y1, x2,y2)            │
│  ├─ find_window(title) → list            │
│  ├─ focus_window(title)                  │
│  └─ screen_annotate() → SoM overlay      │ 
│                                          │
│ Dependencies: pyautogui, pygetwindow     │
└──────────────────────────────────────────┘
```

---

## Critical Problem: Coordinate Mapping

Rio's `screen_capture.py` resizes screenshots by `resize_factor=0.5` (1920×1080 → 960×540). Gemini sees the **resized** image. When Gemini says "click at (450, 320)", those are **screenshot coordinates**, not real screen coordinates.

### Solution: CoordinateMapper

```python
class CoordinateMapper:
    """Maps screenshot coords → real screen coords."""
    
    def __init__(self, resize_factor: float, monitor_offset: tuple[int, int]):
        self.resize_factor = resize_factor
        self.monitor_left = monitor_offset[0]
        self.monitor_top = monitor_offset[1]
    
    def to_real(self, sx: int, sy: int) -> tuple[int, int]:
        """Convert screenshot (sx, sy) → real screen (rx, ry)."""
        rx = int(sx / self.resize_factor) + self.monitor_left
        ry = int(sy / self.resize_factor) + self.monitor_top
        return rx, ry
```

### DPI Awareness (Windows)

Windows DPI scaling (125%, 150%, etc.) causes pyautogui coordinates to NOT match real pixels. **Must call at process startup:**

```python
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
```

### screen_capture.py Changes Needed

`capture()` must return metadata alongside the JPEG:
```python
@dataclass
class CaptureResult:
    jpeg: bytes
    original_width: int    # before resize
    original_height: int   # before resize
    resized_width: int     # after resize
    resized_height: int    # after resize
    monitor_left: int      # monitor X offset
    monitor_top: int       # monitor Y offset
    resize_factor: float
```

---

## Tool Declarations (9 New Tools)

### Phase 1: Core Navigation (MVP)

| Tool | Description | Parameters |
|------|-------------|------------|
| `screen_click` | Click at coordinates on screen | `x`, `y`, `button` (left/right/middle), `clicks` (1/2/3) |
| `screen_type` | Type text at current cursor position | `text`, `interval` (delay between keys) |
| `screen_scroll` | Scroll at position | `x`, `y`, `clicks` (+up/-down) |
| `screen_hotkey` | Press keyboard shortcut | `keys` (e.g., "ctrl+s", "alt+tab") |
| `screen_move` | Move mouse without clicking | `x`, `y` |

### Phase 2: Accuracy Enhancement

| Tool | Description | Parameters |
|------|-------------|------------|
| `screen_annotate` | Take screenshot with numbered element labels (SoM) | None |
| `click_element` | Click a numbered element from annotated screenshot | `element_number` |

### Phase 3: Window Management

| Tool | Description | Parameters |
|------|-------------|------------|
| `find_window` | Search for windows by title | `title_contains` |
| `focus_window` | Bring window to foreground | `title_contains` |
| `screen_drag` | Click-and-drag from A to B | `start_x`, `start_y`, `end_x`, `end_y`, `duration` |

---

## Implementation Plan

### Phase 1: Core Navigation (2-3 days)

**Files to create:**
- `rio/local/screen_navigator.py` — All screen navigation logic
- Tool declarations added to `rio/cloud/gemini_session.py`
- Tool dispatch added to `rio/local/tools.py`

**Step 1: Create `screen_navigator.py`**

```python
"""
Rio Local — Screen Navigator

Provides screen interaction capabilities for the AI:
click, type, scroll, hotkey, drag, window management.

Uses pyautogui for cross-platform mouse/keyboard control.
Coordinate mapping handles resize factor and DPI scaling.

Dependencies: pyautogui, pygetwindow (Windows), Pillow
"""

import pyautogui
import platform

# Safety: 0.1s pause between actions, failsafe enabled
pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True  # Move mouse to corner to abort

class ScreenNavigator:
    def __init__(self, resize_factor: float = 0.5):
        self._resize_factor = resize_factor
        self._monitor_left = 0
        self._monitor_top = 0
        self._dpi_aware = False
        self._setup_dpi()
    
    def _setup_dpi(self):
        """Enable DPI awareness on Windows."""
        if platform.system() == "Windows":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
                self._dpi_aware = True
            except Exception:
                pass
    
    def update_monitor_offset(self, left: int, top: int):
        """Update monitor offset from latest capture metadata."""
        self._monitor_left = left
        self._monitor_top = top
    
    def _map_coords(self, sx: int, sy: int) -> tuple[int, int]:
        """Map screenshot coords → real screen coords."""
        rx = int(sx / self._resize_factor) + self._monitor_left
        ry = int(sy / self._resize_factor) + self._monitor_top
        return rx, ry
    
    async def click(self, x, y, button="left", clicks=1):
        rx, ry = self._map_coords(x, y)
        pyautogui.click(rx, ry, button=button, clicks=clicks)
        return {"success": True, "real_coords": [rx, ry]}
    
    async def type_text(self, text, interval=0.02):
        pyautogui.typewrite(text, interval=interval)
        return {"success": True, "typed": len(text)}
    
    async def scroll(self, x, y, clicks):
        rx, ry = self._map_coords(x, y)
        pyautogui.scroll(clicks, rx, ry)
        return {"success": True}
    
    async def hotkey(self, keys):
        key_list = [k.strip() for k in keys.split("+")]
        pyautogui.hotkey(*key_list)
        return {"success": True, "keys": key_list}
    
    async def move(self, x, y):
        rx, ry = self._map_coords(x, y)
        pyautogui.moveTo(rx, ry)
        return {"success": True}
    
    async def drag(self, sx, sy, ex, ey, duration=0.5):
        rsx, rsy = self._map_coords(sx, sy)
        rex, rey = self._map_coords(ex, ey)
        pyautogui.moveTo(rsx, rsy)
        pyautogui.drag(rex - rsx, rey - rsy, duration=duration)
        return {"success": True}
```

**Step 2: Update `screen_capture.py`**

- Add `CaptureResult` dataclass with monitor metadata
- Return monitor offset (left, top) so navigator can map correctly
- Expose the current resize_factor as a property

**Step 3: Add Tool Declarations**

Add 5 new `FunctionDeclaration` entries to `RIO_TOOL_DECLARATIONS` in `gemini_session.py`:
- `screen_click`, `screen_type`, `screen_scroll`, `screen_hotkey`, `screen_move`

**Step 4: Add Tool Dispatch**

In `tools.py`, add execution handlers for each screen nav tool.
Connect the `ScreenNavigator` instance in `main.py`.

**Step 5: Add System Instruction**

Update the Gemini system prompt to explain:
- Screenshots are at 50% resolution
- Coordinates should be in screenshot space (the AI sees the resized image)
- Available screen actions and when to use them

### Phase 2: SoM Accuracy (1-2 days)

**What:** When uncertain about coordinates, Gemini calls `screen_annotate` which:
1. Takes a screenshot
2. Runs OCR + edge detection to find UI elements
3. Draws numbered labels (①②③...) on each element
4. Returns the annotated image to Gemini
5. Gemini says `click_element(3)` → we click the center of element 3

**Why:** Pure coordinate guessing has ~70% accuracy. SoM boosts to ~90-95%.

**How:**
```python
async def annotate_screen(self) -> tuple[bytes, list[dict]]:
    """Capture screen, detect elements, return labeled image + element map."""
    jpeg = await screen_capture.capture_async(force=True)
    img = Image.open(io.BytesIO(jpeg))
    
    # Detect elements via OCR (text buttons, labels) 
    # + contour detection (icons, buttons without text)
    elements = self._detect_elements(img)
    
    # Draw numbered labels
    annotated = self._draw_labels(img, elements)
    
    # Return annotated image + element coordinate map
    return annotated_jpeg, elements
```

### Phase 3: Window Management (1 day)

```python
# Windows: pygetwindow
import pygetwindow as gw

async def find_window(title_contains):
    windows = gw.getWindowsWithTitle(title_contains)
    return [{"title": w.title, "position": (w.left, w.top), 
             "size": (w.width, w.height)} for w in windows]

async def focus_window(title_contains):
    windows = gw.getWindowsWithTitle(title_contains)
    if windows:
        windows[0].activate()
        return {"success": True, "title": windows[0].title}
    return {"success": False, "error": "Window not found"}
```

Linux: Use `xdotool` or `wmctrl` via subprocess as fallback.

---

## Safety Considerations

### Confirmation Gate

All screen navigation tools execute **without confirmation** by default (like `read_file`), EXCEPT:
- `screen_type` with length > 100 characters → prompt user
- Any action when the active window title contains: "Terminal", "Console", "PowerShell", "bash" → warn about command execution risk
- Sequential rapid actions (>10 in 5 seconds) → pause and confirm

### Failsafe

`pyautogui.FAILSAFE = True` — moving mouse to top-left corner (0,0) aborts any action. This is pyautogui's built-in emergency stop.

### Action Log

Every screen action is logged with:
- Timestamp
- Action type
- Screenshot coordinates → real coordinates mapping
- Active window title at time of action

---

## Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Coordinate misalignment** | High | High | CoordinateMapper + DPI awareness + SoM Phase 2 |
| **Wrong click target** | Medium | High | Confirmation for destructive windows + SoM |
| **DPI scaling varies per monitor** | Medium | Medium | Per-monitor DPI via `SetProcessDpiAwareness(2)` |
| **pyautogui blocked by UAC/admin windows** | Low | Medium | Document limitation; suggest running as admin |
| **Wayland blocks pyautogui (Linux)** | Medium | High | Use `ydotool` or `wtype` as Wayland alternatives |
| **Performance: SoM overlay too slow** | Low | Low | Cache element detection; timeout at 2s |
| **Runaway automation** | Low | Critical | pyautogui.FAILSAFE + rate limiting + max actions per turn |

---

## Dependencies to Install

```bash
pip install pyautogui pygetwindow  # Windows
pip install pyautogui python-xlib  # Linux (X11)
# For Wayland: pip install pyautogui; apt install ydotool
```

Note: `pyautogui` depends on:
- Windows: no additional system deps
- Linux X11: `python3-xlib`, `xdotool`, `xclip`
- Linux Wayland: Limited support; `ydotool` for mouse/keyboard, `wtype` for typing

---

## Files Changed / Created

| File | Action | Description |
|------|--------|-------------|
| `rio/local/screen_navigator.py` | **CREATE** | All screen navigation logic, CoordinateMapper |
| `rio/local/screen_capture.py` | **MODIFY** | Add `CaptureResult` with monitor metadata |
| `rio/local/tools.py` | **MODIFY** | Add screen_* tool dispatch handlers |
| `rio/cloud/gemini_session.py` | **MODIFY** | Add 9 new FunctionDeclaration entries |
| `rio/local/main.py` | **MODIFY** | Initialize ScreenNavigator, wire up to tools |
| `rio/local/som.py` | **CREATE** (Phase 2) | Set-of-Mark overlay + element detection |

---

## System Prompt Addition

```
SCREEN NAVIGATION:
You can interact with the user's screen. Screenshots are captured at 50% resolution.
When you want to interact, use coordinate values as they appear in the screenshot image.
The system automatically maps them to real screen positions.

Available tools:
- screen_click(x, y) — click at a position
- screen_type(text) — type text at current cursor
- screen_scroll(x, y, clicks) — scroll at position (+up/-down)
- screen_hotkey(keys) — press keyboard shortcut (e.g. "ctrl+s")
- screen_move(x, y) — move mouse without clicking
- screen_annotate() — get numbered element overlay (use when unsure of exact coords)
- click_element(n) — click element N from annotated screenshot
- find_window(title) — search for windows
- focus_window(title) — bring window to front
- screen_drag(start_x, start_y, end_x, end_y) — drag from A to B

WORKFLOW for clicking UI elements:
1. Look at the most recent screenshot
2. If you can clearly identify the target, use screen_click(x, y)
3. If unsure, call screen_annotate() first for numbered labels, then click_element(N)
4. After any action, wait for the next screenshot to verify the result
```

---

## Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Click accuracy (known target) | >85% | Manual testing on 50 UI targets |
| Click accuracy with SoM | >93% | Same test with screen_annotate |
| Action latency | <200ms | Time from tool_call to pyautogui execution |
| Cross-platform support | Win + Linux | Test on both |
| No accidental destructive actions | 0 incidents | Safety gate + logging audit |
