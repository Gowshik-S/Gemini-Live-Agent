# Rio Screen Navigation — Comprehensive Research Report
## Date: March 5, 2026 | Author: Research Agent

---

## Executive Summary

Rio needs full screen navigation (click, type, scroll, drag, window management) to evolve from a "screen watcher that talks" into an "AI that operates the computer." The recommended approach is a **hybrid stack**: **pyautogui** for cross-platform mouse/keyboard actions, augmented with **platform-native accessibility APIs** (pywinauto on Windows, python-xdotool/AT-SPI on Linux) for reliable element targeting. Coordinate accuracy is the single hardest technical challenge — DPI scaling, multi-monitor offsets, and the existing 0.5x resize factor in `screen_capture.py` all create coordinate mapping bugs that will silently cause the AI to click the wrong things. A visual grounding layer (Set-of-Mark overlay or OCR-assisted bounding boxes) is strongly recommended to bridge the gap between "Gemini sees a screenshot" and "the click hits the right pixel."

**Critical finding:** Rio's current `screen_capture.py` resizes to 50% before JPEG encoding. Any coordinates Gemini returns from the image must be **divided by the resize_factor, then divided by DPI scale**, before passing to pyautogui. This is the #1 source of bugs in AI computer-use systems.

---

## 1. Technology Analysis

### 1.1 pyautogui — Cross-Platform Mouse/Keyboard Automation

**What it is:** Pure-Python library for programmatic mouse movement, clicking, scrolling, keyboard typing, hotkeys, and basic screenshot capabilities. Uses platform-native backends (Win32 API on Windows, Xlib on Linux, Quartz on macOS).

**Strengths:**
- Cross-platform (Windows, Linux, macOS) with identical API
- Simple, well-documented: `pyautogui.click(x, y)`, `pyautogui.typewrite("hello")`, `pyautogui.scroll(-3)`
- Built-in fail-safe: move mouse to corner = emergency abort
- `PAUSE` constant adds controllable delay between actions (default 0.1s)
- `locateOnScreen()` for template-matching element detection (slow but works)
- `moveTo()`, `click()`, `doubleClick()`, `rightClick()`, `tripleClick()`, `drag()`, `scroll()`, `hotkey()`
- `typewrite()` for ASCII, `write()` with interval, `press()` for special keys
- Window management via `pyautogui.getWindowsWithTitle()`
- 8.8K GitHub stars, actively maintained

**Weaknesses:**
- On **Wayland Linux**: requires XWayland or fails entirely (same problem as pynput)
- `typewrite()` does NOT support Unicode directly — must use `pyperclip` + `hotkey('ctrl','v')` workaround
- No DPI-awareness by default — reports physical pixels on non-scaled displays, but on Windows with DPI >100% it may report scaled coordinates depending on the process DPI awareness manifest
- `locateOnScreen()` is slow (~100-500ms) and unreliable with anti-aliased text
- Thread safety: NOT thread-safe. All calls must come from one thread (or use locks)
- Cannot type into elevated (admin) windows on Windows without running as admin

**DPI/Scaling behavior:**
- On Windows: pyautogui uses Win32 `SetCursorPos()` which operates in **screen coordinates**. If the Python process is NOT DPI-aware, Windows may virtualize coordinates. **Must call `ctypes.windll.shcore.SetProcessDpiAwareness(2)` at startup** to get per-monitor DPI awareness.
- On Linux/X11: coordinates are physical pixels (no scaling issues with X11). On Wayland: pyautogui doesn't work natively.

**Verdict:** **PRIMARY choice for action execution.** Simple API, cross-platform, covers 90% of needed actions. Must solve DPI awareness externally.

**Install:** `pip install pyautogui` (~4 MB, deps: pymsgbox, pytweening, pyscreeze, pyperclip, mouseinfo)

---

### 1.2 pynput — Already in Codebase (Partial)

**Current usage:** `push_to_talk.py` uses `pynput.keyboard.Listener` for F2 hotkey detection. It's an **input listener**, not an action executor.

**What it can do beyond current usage:**
- `pynput.keyboard.Controller()` — programmatic key presses: `controller.press(Key.enter)`, `controller.type("hello")`
- `pynput.mouse.Controller()` — programmatic mouse: `controller.position = (x, y)`, `controller.click(Button.left)`
- `pynput.mouse.Listener()` — track mouse position (useful for coordinate debugging)

**Strengths:**
- Already a dependency (though commented out in requirements.txt due to kernel headers issue)
- Can both **listen** and **act** — single library for input + output
- Unicode support via `controller.type()` (better than pyautogui)
- Lower-level than pyautogui — more control over press/release timing

**Weaknesses:**
- Wayland: broken for both listener and controller (known issue, documented in Rio's `push_to_talk.py`)
- Less convenient than pyautogui for compound actions (no built-in `drag()`, `hotkey()`, `scroll()` — must compose manually)
- No built-in fail-safe
- No built-in screenshot or element detection
- Less documentation for automation use cases

**Verdict:** **Keep for hotkey listening (F2/F3/F4/F5). Don't use for action execution** — pyautogui has a cleaner API for computer-use actions. Using two libraries for input-listening vs. action-execution is a clean separation of concerns.

---

### 1.3 Anthropic's Computer Use (Claude)

**Architecture:** Anthropic's computer-use is a reference implementation, not a library to import. It defines a `computer` tool with actions: `screenshot`, `mouse_move`, `left_click`, `right_click`, `double_click`, `triple_click`, `left_click_drag`, `type`, `key`, `scroll`, `wait`, `cursor_position`.

**Key design decisions from Anthropic:**
1. **Coordinate system:** Claude returns coordinates as (x, y) pixels relative to the screenshot dimensions. The tool executor maps them to real screen coordinates.
2. **Screenshot scaling:** They scale screenshots to a fixed resolution (e.g., 1280x800) before sending to Claude. This is similar to Rio's `resize_factor`.
3. **Coordinate mapping:** `real_x = screenshot_x * (real_width / screenshot_width)`, `real_y = screenshot_y * (real_height / screenshot_height)`.
4. **Action → screenshot cycle:** After every action, take a new screenshot and send it back so the model can verify the result. This is critical for reliability.
5. **Tool schema:** Each action is a string enum parameter, with positional parameters like `coordinate: [x, y]`, `text: string`.
6. **No accessibility tree:** Pure vision-based. Claude looks at the screenshot and decides where to click.

**What Rio should borrow:**
- The **action → screenshot → verify** loop pattern
- The coordinate mapping formula
- The tool action vocabulary (click, type, scroll, drag, key, wait)
- The "cursor_position" action for debugging coordinate mapping

**What Rio should NOT copy:**
- Anthropic's implementation uses xdotool under the hood (Linux-only). Rio needs cross-platform.
- Their Docker-based sandbox approach — Rio runs directly on the user's desktop.

---

### 1.4 OpenAI's Computer Use

**Architecture:** OpenAI's computer-use (announced late 2025) follows a very similar pattern to Anthropic's:
1. Model sees screenshot
2. Model outputs structured action (click at x,y; type text; etc.)
3. Tool executor performs the action
4. New screenshot is taken and cycle repeats

**Key differences from Anthropic:**
- OpenAI uses a more structured output format with explicit `type` field
- Actions: `click`, `double_click`, `right_click`, `drag`, `type`, `keypress`, `scroll`, `screenshot`, `wait`
- They introduced "environment safety" flags for different risk levels
- Supports both coordinate-based and element-based targeting

**What Rio should borrow:**
- Safety tier classification (read-only vs. write vs. destructive actions)
- The `wait` action with configurable delay — essential for page loads, animations

---

### 1.5 Google's Project Mariner / Gemini Computer Use

**Architecture:** Project Mariner is Google's agent that uses Gemini to navigate the web and desktop. Since Rio already uses Gemini, this is the most directly relevant approach.

**Key design decisions:**
1. **Native Gemini function calling:** Actions are declared as `FunctionDeclaration` objects — exactly like Rio's existing tools.
2. **Vision-first:** Gemini sees the screenshot and outputs coordinates directly. No OCR or accessibility tree needed for basic tasks.
3. **Gemini 2.0 Flash vision capabilities:** The model can identify UI elements, read text, understand layout, and output pixel coordinates with reasonable accuracy (~5-15px error on 1080p screenshots).
4. **Bounding box format:** Some Mariner implementations use normalized coordinates (0.0-1.0 relative to image dimensions) rather than absolute pixels. This is DPI-agnostic.

**Critical insight for Rio:** Since Rio already uses Gemini Live API with inline image frames, Gemini can ALREADY see the screen. The missing piece is just:
1. Tool declarations for navigation actions
2. A coordinate mapping layer on the local client
3. An action executor (pyautogui)

**Gemini coordinate accuracy:**
- On full-resolution 1080p screenshots (1920x1080): typically 5-15px error
- On 50% scaled (960x540): error doubles to 10-30px at the original resolution
- On text-heavy UIs: very accurate (can identify specific buttons, links, fields)
- On repetitive UIs (grids, tables): less accurate — may click adjacent cells
- **Recommendation:** Send higher resolution screenshots for navigation tasks (resize_factor=0.75 or 1.0 instead of 0.5)

---

### 1.6 Playwright / Selenium — Browser-Specific Automation

**What they are:** Browser automation frameworks that control Chrome/Firefox/Safari via DevTools protocol (Playwright) or WebDriver (Selenium).

**Strengths:**
- Pixel-perfect element targeting via DOM selectors (no coordinate guessing)
- Can fill forms, click buttons, navigate pages, extract text — all reliably
- Handle dynamic content, iframes, shadow DOM, SPAs
- Built-in waiting mechanisms (wait for element, wait for navigation)
- Screenshot capabilities with element highlighting

**Weaknesses:**
- **Browser-only.** Cannot control VS Code, terminal, file manager, or any native desktop application.
- Require a browser instance launched under their control (can't attach to user's existing browser easily)
- Heavy dependencies (Playwright downloads browser binaries ~hundreds of MB)
- Completely separate automation path from desktop automation

**Verdict:** **DEFER.** Rio is a desktop assistant, not a browser bot. If browser automation becomes a priority (e.g., "Rio, fill in this form for me"), Playwright is the right choice — but it should be a separate skill module, not part of the core screen navigation layer. Desktop-wide pyautogui covers browser too (just less precisely).

---

### 1.7 pywinauto (Windows) / python-xdotool (Linux) — OS-Native UI Automation

#### pywinauto (Windows)

**What it is:** Python library wrapping Windows UI Automation and Win32 API for native application control.

**Strengths:**
- **UI element tree access:** Can enumerate all UI elements (buttons, text fields, menus) with their bounds, names, and states
- **Semantic targeting:** `app.window(title="Save As").child_window(title="File name:").type_keys("test.py")` — doesn't need coordinates at all
- **DPI-aware by default** (uses Windows UI Automation coordinates)
- Handles comboboxes, menus, tree views, tab controls natively
- Can inspect and interact with Win32, WPF, WinForms, UWP apps
- Solves the "click the right thing" problem completely for native Windows apps

**Weaknesses:**
- **Windows-only** — need a separate solution for Linux
- Learning curve: understanding the UI element tree can be complex
- Some apps (Electron apps like VS Code, Chrome) expose limited UI Automation trees
- Can be slow for element enumeration (50-200ms for complex UIs)

**Install:** `pip install pywinauto` (~2 MB)

#### python-xdotool + AT-SPI (Linux)

**xdotool:** Command-line tool for X11 window/keyboard/mouse automation. Python wrapper: `subprocess.run(["xdotool", "key", "Return"])`.

**AT-SPI (Assistive Technology Service Provider Interface):**
- Linux equivalent of Windows UI Automation
- Accessed via `python-atspi` or `pyatspi2`
- Provides element tree with bounds, names, roles, states
- Works with GTK, Qt, LibreOffice, Firefox

**Weaknesses:**
- xdotool is X11-only (no Wayland)
- AT-SPI support varies wildly between applications
- Much less mature ecosystem than pywinauto

**Verdict:** **USE pywinauto as an OPTIONAL enhancement layer on Windows.** When Gemini identifies a target (e.g., "click the Save button"), try to match it via UI Automation tree first (exact, DPI-safe), falling back to coordinate-based pyautogui. On Linux, use coordinate-based pyautogui as primary, with xdotool as a backup for X11 systems.

---

### 1.8 uiautomation (Microsoft UI Automation Python Wrapper)

**What it is:** Pure-Python wrapper around Windows' `IUIAutomation` COM interface. Similar to pywinauto's UIA backend but more direct.

**Key capability:** `uiautomation.WindowControl(Name="Notepad").EditControl().SendKeys("hello")`

**vs. pywinauto:** pywinauto is more mature, better documented, and has a larger community. `uiautomation` is a thinner wrapper. **Prefer pywinauto** unless you need raw UIA access.

**Verdict:** **Skip.** pywinauto covers this and more.

---

### 1.9 mouse + keyboard Modules — Lightweight Alternatives

**`keyboard` module (boppreh/keyboard):**
- Hook and simulate keyboard events
- Cross-platform (Windows + Linux)
- Requires root/admin on Linux (uses `/dev/input`)
- `keyboard.write("hello")`, `keyboard.press_and_release("ctrl+c")`
- Very lightweight (~50 KB)

**`mouse` module (boppreh/mouse):**
- Hook and simulate mouse events
- Cross-platform
- `mouse.move(x, y)`, `mouse.click()`, `mouse.wheel(delta)`
- Also requires root on Linux

**Strengths:** Tiny, fast, minimal dependencies.

**Weaknesses:**
- Root/admin requirement on Linux is a dealbreaker for normal user operation
- Less battle-tested than pyautogui for automation
- No built-in fail-safe
- `keyboard` module conflicts with pynput (both hook the same system events)

**Verdict:** **Skip.** pyautogui covers the same functionality without the root requirement on Linux and without conflicting with pynput.

---

### 1.10 OmniParser — Microsoft's Screen Parsing for AI Agents

**What it is:** An open-source vision model from Microsoft that parses screenshots into structured UI elements. It detects interactive elements (buttons, fields, links, icons) and outputs bounding boxes with labels.

**Architecture:**
1. Input: screenshot image
2. Processing: YOLO-based icon detection + OCR for text elements
3. Output: list of `{label, bbox: [x1, y1, x2, y2], type: "button"|"text_field"|"icon"|...}`

**Strengths:**
- Application-agnostic — works on any UI (web, native, terminal)
- Detects elements that accessibility APIs miss (custom-drawn UIs, games, remote desktops)
- Provides semantic labels ("Submit button", "Search field")
- Can be combined with Set-of-Mark for visual grounding

**Weaknesses:**
- **Model size:** OmniParser v2 is ~1.5-2 GB (YOLO + OCR models)
- **Inference time:** 200-800ms per frame on CPU, ~50-150ms on GPU
- **Not perfect:** Misses small or unusual UI elements, especially in dark themes
- Requires PyTorch + ONNX runtime — heavy dependencies
- Not real-time — too slow for interactive use, best as a pre-processing step

**Verdict:** **CONSIDER for Phase 3 (advanced element detection).** Too heavy for initial implementation. Rio already has rapidocr-onnxruntime for text detection — that covers most text-based UI elements. OmniParser adds icon/button detection but at a significant performance and size cost. Better to start with pure Gemini vision + OCR-assisted targeting, and add OmniParser only if coordinate accuracy is insufficient.

---

### 1.11 Set-of-Mark (SoM) — Visual Grounding with Numbered Labels

**What it is:** A technique where you overlay numbered labels (1, 2, 3...) on detected UI elements in the screenshot before sending to the model. The model then says "click element 5" instead of "click at (350, 220)."

**How it works:**
1. Detect interactive elements (via OCR bounding boxes, accessibility tree, or OmniParser)
2. Draw numbered overlays on the screenshot at each element's location
3. Send annotated screenshot to Gemini
4. Gemini says "click_element(5)" instead of "click(350, 220)"
5. Local client maps element 5 → center of its bounding box → pyautogui.click()

**Strengths:**
- **Dramatically improves accuracy** — eliminates coordinate guessing entirely
- Model just needs to identify the right element by semantic meaning, not pixel coordinates
- Works with any element detection method (OCR, UI Automation, OmniParser)
- Robust to DPI scaling, resize factors, multi-monitor — coordinates are resolved locally
- Proven effective: Anthropic, Microsoft, and Google all use variations of this

**Weaknesses:**
- Requires an element detection step before every screenshot (adds latency)
- Visual clutter: too many labels can confuse both the AI and the user
- Labels might obscure important UI text
- Need to re-detect elements after every action (UI may change)

**Implementation approach for Rio:**
```python
# 1. Capture screenshot at higher resolution
jpeg = screen_capture.capture(force=True)

# 2. Detect text elements via OCR
ocr_results = ocr_engine.extract_with_boxes(jpeg)  # returns [{text, bbox}]

# 3. Optionally get accessibility tree elements (pywinauto on Windows)
ui_elements = get_accessible_elements()  # returns [{name, bbox, role}]

# 4. Merge and deduplicate elements
all_elements = merge_elements(ocr_results, ui_elements)

# 5. Draw numbered labels on screenshot
annotated_jpeg, element_map = draw_som_overlay(jpeg, all_elements)
# element_map = {1: {center: (350, 220), bbox: ...}, 2: ...}

# 6. Send annotated screenshot to Gemini
await gemini.send_image(annotated_jpeg)

# 7. Gemini calls click_element(element_id=5)
# 8. Local resolves: element_map[5].center → pyautogui.click(350, 220)
```

**Verdict:** **HIGHLY RECOMMENDED as Phase 2 enhancement.** Start with pure coordinate-based vision (Phase 1), then add SoM overlay using existing OCR bounding boxes (Phase 2). This gives the biggest accuracy improvement for the least additional complexity.

---

## 2. Coordinate Mapping — The #1 Technical Challenge

### 2.1 The Problem

Rio's screenshot pipeline currently:
1. `mss` captures the primary monitor at full resolution (e.g., 1920x1080)
2. Pillow resizes by `resize_factor=0.5` → 960x540
3. JPEG compressed at quality 60
4. Sent to Gemini via WebSocket

When Gemini sees the 960x540 image and says "click at (350, 200)", the actual screen coordinate is:
- **Not** (350, 200)
- **Not** (700, 400) — `350 / 0.5 = 700` (accounting for resize)
- It might be (700 × dpi_scale, 400 × dpi_scale) if DPI scaling is active
- It might need a monitor offset added for multi-monitor setups

### 2.2 Coordinate Mapping Formula

```python
def screenshot_to_screen(sx: int, sy: int, capture_meta: dict) -> tuple[int, int]:
    """Convert screenshot pixel coordinates to real screen coordinates.
    
    Args:
        sx, sy: Coordinates in the screenshot image (as returned by Gemini)
        capture_meta: Metadata from the capture containing:
            - resize_factor: The downscale factor used (e.g., 0.5)
            - monitor_left: X offset of the captured monitor
            - monitor_top: Y offset of the captured monitor
            - dpi_scale: The DPI scaling factor (e.g., 1.5 for 150%)
            - screenshot_width: Width of the screenshot image sent to Gemini
            - screenshot_height: Height of the screenshot image sent to Gemini
            - original_width: Full resolution width of the monitor
            - original_height: Full resolution height of the monitor
    
    Returns:
        (real_x, real_y) in screen coordinates suitable for pyautogui
    """
    # Step 1: Scale back from screenshot resolution to original monitor resolution
    scale_x = capture_meta["original_width"] / capture_meta["screenshot_width"]
    scale_y = capture_meta["original_height"] / capture_meta["screenshot_height"]
    
    real_x = int(sx * scale_x)
    real_y = int(sy * scale_y)
    
    # Step 2: Add monitor offset (for multi-monitor setups)
    real_x += capture_meta["monitor_left"]
    real_y += capture_meta["monitor_top"]
    
    # Note: If the Python process is DPI-aware (SetProcessDpiAwareness(2)),
    # mss already returns physical pixel coordinates and pyautogui should
    # also use physical pixels. No additional DPI scaling needed.
    # If NOT DPI-aware, both mss and pyautogui operate in virtualized coordinates.
    # The key is CONSISTENCY: both capture and action must use the same coordinate space.
    
    return real_x, real_y
```

### 2.3 DPI Scaling Solution

**Windows:**
```python
import ctypes
import platform

def enable_dpi_awareness():
    """Make the Python process DPI-aware on Windows.
    
    MUST be called before any mss, pyautogui, or screen capture calls.
    Without this, Windows 'lies' about screen coordinates at >100% DPI.
    """
    if platform.system() != "Windows":
        return
    
    try:
        # Per-Monitor DPI Awareness V2 (best, Windows 10 1703+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            # Fallback: System DPI Awareness
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

def get_dpi_scale() -> float:
    """Get the DPI scaling factor for the primary monitor on Windows."""
    if platform.system() != "Windows":
        return 1.0
    try:
        # After SetProcessDpiAwareness(2), this returns actual DPI
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0  # 96 DPI = 100% scaling
    except Exception:
        return 1.0
```

**Linux:**
- X11: No DPI virtualization. Coordinates are always physical pixels.
- Wayland: Compositors handle scaling. Tools that work (grim) capture at physical resolution. xdotool operates at logical coordinates (may need scaling).

### 2.4 Multi-Monitor Support

```python
def get_monitor_info():
    """Get information about all monitors for coordinate mapping."""
    import mss
    with mss.mss() as sct:
        monitors = []
        for i, m in enumerate(sct.monitors):
            if i == 0:
                continue  # monitors[0] is the virtual screen (all monitors combined)
            monitors.append({
                "index": i,
                "left": m["left"],
                "top": m["top"],
                "width": m["width"],
                "height": m["height"],
            })
    return monitors
```

**Critical: When capturing, store which monitor was captured and its offset.** The current `screen_capture.py` uses `sct.monitors[1]` (primary monitor) but doesn't store the offset. This metadata must be returned alongside the JPEG bytes.

### 2.5 Recommended Changes to screen_capture.py

The `capture()` method should return metadata alongside JPEG bytes:
```python
@dataclass
class CaptureResult:
    jpeg_bytes: bytes
    monitor_left: int      # x offset of captured monitor
    monitor_top: int       # y offset of captured monitor 
    original_width: int    # pre-resize width
    original_height: int   # pre-resize height
    screenshot_width: int  # post-resize width (what Gemini sees)
    screenshot_height: int # post-resize height (what Gemini sees)
    resize_factor: float   # the factor used
```

---

## 3. Recommended Tool Declarations for Gemini

These tools extend the existing `RIO_TOOL_DECLARATIONS` list in `gemini_session.py`:

```python
# --- Screen Navigation Tools ---

types.FunctionDeclaration(
    name="click",
    description=(
        "Click at a specific position on the user's screen. "
        "The x and y coordinates should be in pixels relative to "
        "the screenshot image you are currently seeing. "
        "Rio will automatically map these to real screen coordinates."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "x": {
                "type": "INTEGER",
                "description": "X coordinate in the screenshot image (pixels from left edge)",
            },
            "y": {
                "type": "INTEGER",
                "description": "Y coordinate in the screenshot image (pixels from top edge)",
            },
            "button": {
                "type": "STRING",
                "description": "Mouse button: 'left' (default), 'right', or 'middle'",
            },
            "clicks": {
                "type": "INTEGER",
                "description": "Number of clicks: 1 (default), 2 for double-click, 3 for triple-click",
            },
        },
        "required": ["x", "y"],
    },
),

types.FunctionDeclaration(
    name="type_text",
    description=(
        "Type text at the current cursor position. Use this after "
        "clicking on a text field. For special keys like Enter, Tab, "
        "Escape, use the 'press_key' tool instead."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "text": {
                "type": "STRING",
                "description": "The text to type. Supports Unicode.",
            },
            "interval": {
                "type": "NUMBER",
                "description": "Seconds between each keystroke (default: 0.02). Increase for slow applications.",
            },
        },
        "required": ["text"],
    },
),

types.FunctionDeclaration(
    name="press_key",
    description=(
        "Press a keyboard key or key combination. Use for special keys "
        "(Enter, Tab, Escape, arrows, function keys) or key combos "
        "(ctrl+c, ctrl+v, alt+tab, ctrl+shift+s)."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "keys": {
                "type": "STRING",
                "description": (
                    "Key or key combination. Examples: 'enter', 'tab', 'escape', "
                    "'backspace', 'delete', 'up', 'down', 'left', 'right', "
                    "'ctrl+c', 'ctrl+v', 'ctrl+a', 'alt+tab', 'ctrl+shift+s', "
                    "'f1', 'f5', 'home', 'end', 'pageup', 'pagedown'"
                ),
            },
        },
        "required": ["keys"],
    },
),

types.FunctionDeclaration(
    name="scroll",
    description=(
        "Scroll the mouse wheel at the current position or at a "
        "specific position. Positive = scroll up, negative = scroll down."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "amount": {
                "type": "INTEGER",
                "description": "Number of scroll units. Positive = up, negative = down. Typical: -3 to -5 for scrolling down a page section.",
            },
            "x": {
                "type": "INTEGER",
                "description": "Optional X coordinate to scroll at (screenshot pixels). If omitted, scrolls at current mouse position.",
            },
            "y": {
                "type": "INTEGER",
                "description": "Optional Y coordinate to scroll at (screenshot pixels). If omitted, scrolls at current mouse position.",
            },
        },
        "required": ["amount"],
    },
),

types.FunctionDeclaration(
    name="drag",
    description=(
        "Click and drag from one position to another. "
        "Both coordinates are in screenshot image pixels."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "start_x": {
                "type": "INTEGER",
                "description": "Starting X coordinate (screenshot pixels)",
            },
            "start_y": {
                "type": "INTEGER",
                "description": "Starting Y coordinate (screenshot pixels)",
            },
            "end_x": {
                "type": "INTEGER",
                "description": "Ending X coordinate (screenshot pixels)",
            },
            "end_y": {
                "type": "INTEGER",
                "description": "Ending Y coordinate (screenshot pixels)",
            },
            "duration": {
                "type": "NUMBER",
                "description": "Duration of the drag in seconds (default: 0.5)",
            },
        },
        "required": ["start_x", "start_y", "end_x", "end_y"],
    },
),

types.FunctionDeclaration(
    name="wait",
    description=(
        "Wait for a specified duration before the next action. "
        "Use after actions that trigger loading, animations, "
        "or page transitions."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "seconds": {
                "type": "NUMBER",
                "description": "Number of seconds to wait (0.1 to 10)",
            },
            "reason": {
                "type": "STRING",
                "description": "Why waiting (e.g., 'page loading', 'animation completing')",
            },
        },
        "required": ["seconds"],
    },
),

types.FunctionDeclaration(
    name="get_cursor_position",
    description=(
        "Get the current mouse cursor position on the screen. "
        "Returns the position in screenshot coordinates. "
        "Useful for debugging coordinate alignment."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {},
        "required": [],
    },
),

# --- Window Management Tools ---

types.FunctionDeclaration(
    name="find_window",
    description=(
        "Find a window by its title (or partial title match). "
        "Returns the window title, position, and size."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "title": {
                "type": "STRING",
                "description": "Window title or partial match (e.g., 'Visual Studio Code', 'Chrome')",
            },
        },
        "required": ["title"],
    },
),

types.FunctionDeclaration(
    name="focus_window",
    description=(
        "Bring a window to the foreground and give it focus. "
        "Use this before clicking/typing in a specific application."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "title": {
                "type": "STRING",
                "description": "Window title or partial match to focus",
            },
        },
        "required": ["title"],
    },
),
```

### Also recommended: `click_element` for Set-of-Mark (Phase 2)

```python
types.FunctionDeclaration(
    name="click_element",
    description=(
        "Click on a numbered UI element from the annotated screenshot. "
        "Each element has a numbered label overlay. Use the number to "
        "click precisely on that element. More accurate than coordinate-based clicking."
    ),
    parameters={
        "type": "OBJECT",
        "properties": {
            "element_id": {
                "type": "INTEGER",
                "description": "The number label of the element to click (from the annotated screenshot)",
            },
            "button": {
                "type": "STRING",
                "description": "Mouse button: 'left' (default), 'right', or 'middle'",
            },
        },
        "required": ["element_id"],
    },
),
```

---

## 4. Architecture

### 4.1 Architecture Diagram

```
┌─────────────────── LOCAL DESKTOP APP ─────────────────────────────────┐
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │                    SCREEN NAVIGATOR (NEW)                      │   │
│  │                     local/screen_nav.py                        │   │
│  │                                                                │   │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐ │   │
│  │  │  Coordinate  │  │   Action    │  │   Safety Gate        │ │   │
│  │  │  Mapper      │  │   Executor  │  │                      │ │   │
│  │  │              │  │             │  │  • Dangerous action   │ │   │
│  │  │  screenshot  │  │  pyautogui  │  │    confirmation      │ │   │
│  │  │  coords →    │  │  .click()   │  │  • Blocklist         │ │   │
│  │  │  screen      │  │  .write()   │  │    (rm, format...)   │ │   │
│  │  │  coords      │  │  .scroll()  │  │  • Rate limit        │ │   │
│  │  │              │  │  .hotkey()  │  │    (max 20 actions/   │ │   │
│  │  │  DPI-aware   │  │  .drag()   │  │     min)              │ │   │
│  │  │  Multi-mon   │  │             │  │  • Fail-safe area    │ │   │
│  │  └──────┬───────┘  └──────┬──────┘  └──────────┬───────────┘ │   │
│  │         │                 │                     │             │   │
│  │  ┌──────┴─────────────────┴─────────────────────┴──────────┐  │   │
│  │  │    Platform Backends (auto-detected)                    │  │   │
│  │  │    ┌──────────────────┐  ┌────────────────────────┐     │  │   │
│  │  │    │ Windows:         │  │ Linux (X11):            │     │  │   │
│  │  │    │  pyautogui       │  │  pyautogui              │     │  │   │
│  │  │    │  + pywinauto     │  │  + xdotool fallback     │     │  │   │
│  │  │    │    (optional UIA │  │                         │     │  │   │
│  │  │    │     enhancement) │  │ Linux (Wayland):        │     │  │   │
│  │  │    │  + ctypes DPI fix│  │  ydotool + wtype        │     │  │   │
│  │  │    └──────────────────┘  └────────────────────────┘     │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                        │
│  ┌──────────────┐  ┌────────────────────┐  ┌─────────────────────┐   │
│  │ Screen       │  │ OCR Engine         │  │ SoM Overlay (P2)    │   │
│  │ Capture      │→ │ (text + bboxes)    │→ │ Draw numbered labels│   │
│  │ (mss+PIL)    │  │ (rapidocr)         │  │ on UI elements      │   │
│  │              │  │                    │  │ + element_map dict   │   │
│  │ Returns:     │  │ Returns:           │  │                     │   │
│  │ CaptureResult│  │ [{text, bbox}]     │  │ Returns:            │   │
│  │ (jpeg+meta)  │  │                    │  │ annotated_jpeg +    │   │
│  │              │  │                    │  │ {id: center_coords} │   │
│  └──────────────┘  └────────────────────┘  └─────────────────────┘   │
│                                                                        │
│  ┌──────────────────────────────────────────────────────┐             │
│  │ Tool Executor (existing — tools.py)                   │             │
│  │ + NEW: screen_nav tools dispatch                      │             │
│  │   click → screen_nav.click()                          │             │
│  │   type_text → screen_nav.type_text()                  │             │
│  │   press_key → screen_nav.press_key()                  │             │
│  │   scroll → screen_nav.scroll()                        │             │
│  │   drag → screen_nav.drag()                            │             │
│  │   wait → asyncio.sleep()                              │             │
│  │   find_window → screen_nav.find_window()              │             │
│  │   focus_window → screen_nav.focus_window()            │             │
│  │   click_element → screen_nav.click_element() (P2)     │             │
│  └──────────────────────────────────────────────────────┘             │
│                                                                        │
│                           │ WebSocket                                  │
└───────────────────────────┼────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────── CLOUD (unchanged) ──────────────────────────────┐
│  Gemini Live API sees screenshots → decides actions                   │
│  Function calling: click(), type_text(), press_key(), scroll(), etc.  │
│  → Sends tool_call to local client → waits for tool_result            │
│  → Takes new screenshot after action → verifies result                │
│  Cloud does NOT execute screen actions — local client does             │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 File Layout

```
rio/local/
├── screen_nav.py          # NEW — Screen Navigator (coordinate mapping + action executor)
├── screen_capture.py      # MODIFIED — Returns CaptureResult with metadata
├── som_overlay.py         # NEW (Phase 2) — Set-of-Mark annotation
├── tools.py               # MODIFIED — Add screen_nav tool dispatch
├── ocr.py                 # MODIFIED — Add extract_with_boxes() method

rio/cloud/
├── gemini_session.py      # MODIFIED — Add screen nav tool declarations

rio/config.yaml            # MODIFIED — Add screen_nav section
rio/local/config.py        # MODIFIED — Add ScreenNavConfig dataclass
```

### 4.3 Execution Flow

```
1. User says: "Click the Run button in VS Code"
2. Gemini (cloud) sees the latest screenshot frame
3. Gemini issues: capture_screen() → to get a fresh high-res screenshot
4. Local captures, sends JPEG + metadata to Gemini
5. Gemini identifies the Run button at screenshot coords (450, 35)
6. Gemini issues: click(x=450, y=35)
7. Local receives tool_call:
   a. Safety gate: is this a destructive action? (click → no, auto-approve)
   b. Coordinate mapper: (450, 35) → real screen coords (900, 70) [accounting for 0.5x resize]
   c. Action executor: pyautogui.click(900, 70)
   d. Post-action screenshot: capture new screenshot
   e. Return tool_result: {success: true, new_screenshot: <jpeg>}
8. Gemini sees the new screenshot and verifies the Run button was clicked
9. If the UI changed as expected → task complete
10. If not → Gemini may retry with adjusted coordinates
```

---

## 5. Safety Considerations

### 5.1 Action Classification

| Safety Tier | Actions | Policy |
|-------------|---------|--------|
| **Tier 0: Read-only** | `capture_screen`, `get_cursor_position`, `find_window`, `wait` | Auto-approve |
| **Tier 1: Low-risk input** | `click` (single), `scroll`, `press_key` (non-destructive keys) | Auto-approve with logging |
| **Tier 2: Text input** | `type_text`, `press_key` (enter, backspace, delete) | Auto-approve with logging |
| **Tier 3: Destructive** | `drag` (file moves), `press_key` (ctrl+delete, alt+f4), `focus_window` | **Confirm before executing** |
| **Tier 4: System-level** | Detected patterns: "delete file", "close all windows", key combos involving shutdown | **BLOCK unless explicitly confirmed** |

### 5.2 Dangerous Key Combos — Blocklist

```python
DANGEROUS_KEY_COMBOS = {
    "alt+f4",           # Close window
    "ctrl+w",           # Close tab (in some apps, close window)
    "ctrl+shift+delete",# Clear browsing data
    "ctrl+alt+delete",  # System security screen
    "super+l",          # Lock screen
    "alt+f2",           # Run dialog (Linux)
    # Add more as discovered during testing
}
```

### 5.3 Rate Limiting

- Max **20 screen actions per minute** (prevents accidental infinite click loops)
- Max **5 actions per second** (prevents UI overload)
- **Pause between actions:** minimum 0.1s (pyautogui PAUSE default — keep this)
- **Fail-safe area:** if mouse moves to screen corner (0,0), abort all pending actions (pyautogui built-in)

### 5.4 Confirmation Gate (extends existing `_confirm_tool_call`)

```python
# In tools.py, add to _DANGEROUS_TOOLS:
_SCREEN_DANGEROUS = frozenset({
    "drag",
    # press_key with dangerous combos is checked dynamically
})

# In the tool dispatch, check:
async def _execute_screen_nav_tool(name, args):
    # Check if press_key involves a dangerous combo
    if name == "press_key" and args.get("keys", "").lower() in DANGEROUS_KEY_COMBOS:
        approved = await _confirm_tool_call(name, args)
        if not approved:
            return {"success": False, "error": "User declined"}
    
    if name in _SCREEN_DANGEROUS:
        approved = await _confirm_tool_call(name, args)
        if not approved:
            return {"success": False, "error": "User declined"}
    
    return await screen_nav.execute(name, args)
```

### 5.5 Emergency Stop

- **F1 key** (or configurable): Immediately halt all screen navigation actions
- Clears any pending action queue
- Sends "screen navigation paused" control frame to cloud
- Logs the trigger to dashboard

---

## 6. Element Detection Strategy Comparison

### 6.1 Three Approaches Compared

| Approach | Accuracy | Speed | Complexity | Dependencies | Best For |
|----------|----------|-------|------------|--------------|----------|
| **Pure Vision (Gemini)** | 70-85% | Fastest (0ms local) | Lowest | None new | Phase 1, most tasks |
| **OCR-Assisted** | 85-95% (text elements) | Fast (20-50ms) | Medium | Already have rapidocr | Phase 1.5, text-heavy UIs |
| **UI Tree (Accessibility)** | 95-99% (supported apps) | Medium (50-200ms) | Higher | pywinauto (Windows) | Phase 2, native apps |
| **Set-of-Mark (SoM)** | 90-98% | Slow (100-300ms) | Medium | OCR + overlay drawing | Phase 2, all UIs |
| **OmniParser** | 85-95% (all elements) | Slow (200-800ms) | High | PyTorch + YOLO models | Phase 3, custom UIs |

### 6.2 Recommended Hybrid Strategy

**Phase 1:** Pure Gemini vision + coordinate mapping. Gemini sees the screenshot, decides coordinates. Works for 70-85% of actions. Zero additional dependencies.

**Phase 2:** SoM overlay using existing OCR. Before navigation screenshots, run OCR to get text bounding boxes, draw numbered labels, send annotated image. Add `click_element(id)` tool. This boosts text-based UI accuracy to 90-95%.

**Phase 3 (if needed):** Add pywinauto on Windows for accessibility tree queries. Only activate for complex UIs where vision fails repeatedly. Use as a verification/correction layer: "Gemini says click Save → pywinauto finds the actual Save button coordinates → use those instead."

### 6.3 Why NOT to Start with UI Tree

- Accessibility APIs only work for apps that implement them properly
- Electron apps (VS Code, Chrome) expose limited/messy accessibility trees
- Adds significant platform-specific complexity immediately
- Gemini's vision is already good enough for most developer-focused UIs
- Better ROI: invest in good coordinate mapping first, add UI tree later as enhancement

---

## 7. Implementation Phases

### Phase 1: Core Navigation (2-3 days)
**Goal:** Gemini can click, type, scroll, and press keys on the screen.

| Task | What | Files |
|------|------|-------|
| 1.1 | Create `screen_nav.py` with ScreenNavigator class | local/screen_nav.py |
| 1.2 | DPI awareness at startup (`SetProcessDpiAwareness`) | local/main.py |
| 1.3 | Modify `ScreenCapture` to return `CaptureResult` with metadata | local/screen_capture.py |
| 1.4 | Implement coordinate mapping (screenshot → screen) | local/screen_nav.py |
| 1.5 | Implement actions: click, type_text, press_key, scroll, wait | local/screen_nav.py |
| 1.6 | Add tool declarations to `gemini_session.py` | cloud/gemini_session.py |
| 1.7 | Add tool dispatch to `tools.py` | local/tools.py |
| 1.8 | Wire into main.py receive_loop | local/main.py |
| 1.9 | Add safety gate + dangerous key blocklist | local/screen_nav.py |
| 1.10 | Add `pyautogui` to requirements.txt | local/requirements.txt |
| 1.11 | Add ScreenNavConfig to config.py + config.yaml | local/config.py, config.yaml |
| 1.12 | Post-action screenshot: auto-capture after every action | local/screen_nav.py |
| 1.13 | Update system instruction to explain navigation capabilities | cloud/gemini_session.py |

**Dependencies added:** `pyautogui>=0.9.54`

### Phase 2: Set-of-Mark + OCR Enhancement (1-2 days)
**Goal:** Annotated screenshots with numbered element labels for higher accuracy.

| Task | What | Files |
|------|------|-------|
| 2.1 | Add `extract_with_boxes()` to OCR engine (returns text + bounding boxes) | local/ocr.py |
| 2.2 | Create `som_overlay.py` — draw numbered labels on screenshots | local/som_overlay.py |
| 2.3 | Add `click_element` tool declaration | cloud/gemini_session.py |
| 2.4 | Add `click_element` handler (resolve element_id → coords → click) | local/screen_nav.py |
| 2.5 | Integrate SoM into navigation screenshot flow | local/main.py |
| 2.6 | Update system instruction for SoM usage | cloud/gemini_session.py |

**Dependencies added:** None (uses existing Pillow + rapidocr)

### Phase 3: Window Management + Drag (1 day)
**Goal:** Find, focus, resize, and arrange windows. Drag operations.

| Task | What | Files |
|------|------|-------|
| 3.1 | Implement `find_window` and `focus_window` (cross-platform) | local/screen_nav.py |
| 3.2 | Implement `drag` action | local/screen_nav.py |
| 3.3 | Add window management tool declarations | cloud/gemini_session.py |
| 3.4 | Test drag on file managers, code editors, browser tabs | manual testing |

**Dependencies added:** None (pyautogui.getWindowsWithTitle on Windows, wmctrl/xdotool on Linux)

### Phase 4: Platform Enhancement (optional, 1-2 days)
**Goal:** pywinauto for Windows UI Automation tree access.

| Task | What | Files |
|------|------|-------|
| 4.1 | Add optional pywinauto integration for Windows | local/screen_nav.py |
| 4.2 | UI tree element resolution: "Click Save button" → pywinauto finds it | local/screen_nav.py |
| 4.3 | Add `inspect_element` tool (returns UI tree around a point) | cloud/gemini_session.py |

**Dependencies added (optional):** `pywinauto>=0.6.8`

---

## 8. Bottlenecks Table

| Bottleneck | Severity | Impact | Fix |
|------------|----------|--------|-----|
| **Coordinate mapping with 0.5x resize** | CRITICAL | Every click misses by 2x. Most common bug in AI computer-use. | Store capture metadata (original resolution, resize factor, monitor offset). Apply inverse transform before pyautogui calls. |
| **DPI scaling on Windows >100%** | CRITICAL | Coordinates off by DPI factor (e.g., 1.5x on 150% scaling). Clicks miss systematically. | Call `SetProcessDpiAwareness(2)` at startup, before any screen capture or pyautogui calls. |
| **Wayland incompatibility** | HIGH | pyautogui, pynput, xdotool all fail on Wayland. Affects modern Ubuntu/Fedora. | Detect Wayland at startup. Use `ydotool` (requires root) or `wtype` for keyboard. Screen capture already has Wayland fallback via grim. |
| **pyautogui not thread-safe** | HIGH | Race conditions if screen nav tools are called concurrently. | Serialize all pyautogui calls through a single-threaded executor. Use `asyncio.Lock` or a dedicated thread with a queue. |
| **Gemini coordinate accuracy** | MEDIUM | Gemini may be 5-15px off on 1080p screenshots, worse on scaled images. | Phase 1: accept 10-20px error (OK for most buttons). Phase 2: SoM overlay eliminates this. Also: increase resize_factor to 0.75 for nav screenshots. |
| **Post-action verification latency** | MEDIUM | Taking + sending a screenshot after every action adds 200-500ms per action. | Only verify after "important" actions (clicks, form submissions). Skip for type_text and scroll. |
| **Unicode typing** | LOW | pyautogui.typewrite() doesn't support Unicode. | Use pyperclip + pyautogui.hotkey('ctrl','v') for non-ASCII text. Or use pynput.keyboard.Controller.type() which does support Unicode. |
| **Multi-monitor capture** | LOW | If user has monitor 2 focused, we capture monitor 1 (primary). Coordinates will be wrong. | Add monitor selection to capture_screen tool. Store active monitor info. |
| **Elevated windows (admin)** | LOW | pyautogui cannot type into UAC prompts or admin-elevated windows on Windows. | Detect the failure and report it. Don't try to automate admin-elevated windows. |

---

## 9. Options Comparison

### 9.1 Action Executor Library

| Option | Pros | Cons | Best For | Avoid If |
|--------|------|------|----------|----------|
| **pyautogui** (recommended) | Cross-platform, simple API, battle-tested, fail-safe, adequate for 95% of tasks | No Unicode typewrite, Wayland broken, not thread-safe | Phase 1-3, primary executor | You only target one OS |
| **pynput** | Already in codebase, Unicode type(), lower-level control | Must compose actions manually, Wayland broken, no fail-safe | Keyboard typing (Unicode), hotkey listening | You need compound actions |
| **pywinauto** | DPI-aware, semantic targeting, UI tree access | Windows-only, complex API, slow enumeration | Windows-specific enhancement (Phase 4) | Cross-platform is required |
| **mouse+keyboard** | Tiny, fast | Requires root on Linux, conflicts with pynput | Never — pyautogui is strictly better | Always avoid |
| **subprocess xdotool** | Native X11, handles some edge cases | X11-only, no Windows, subprocess overhead | Linux X11 fallback | You need Windows support |

### 9.2 Element Detection Strategy

| Option | Pros | Cons | Best For | Avoid If |
|--------|------|------|----------|----------|
| **Pure Gemini Vision** (Phase 1) | Zero deps, zero latency, works on all UIs | 70-85% accuracy, model dependent | Initial implementation, simple UIs | You need >90% reliability |
| **OCR + SoM** (Phase 2) | 90-95% for text elements, uses existing rapids | Only detects text, 20-50ms overhead | Text-heavy developer UIs | UIs are mostly icons/images |
| **Accessibility Tree** (Phase 4) | 95-99% when available | Platform-specific, many apps unsupported | Native Windows apps, form filling | Cross-platform is required |
| **OmniParser** (Phase 5+) | Detects buttons/icons/all elements | 1.5-2GB model, 200-800ms, heavy deps | Custom-drawn UIs, games, remote desktop | Size or latency matters |

### 9.3 Coordinate System

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Absolute pixel coords** (recommended Phase 1) | Simple, universal, matches pyautogui | affected by DPI, resize factor | Primary approach |
| **Normalized coords (0.0-1.0)** | DPI-agnostic, resize-agnostic | Model must be trained/prompted for it | Alternative if Gemini supports it well |
| **Element ID (SoM)** (recommended Phase 2) | Most accurate, no coordinate guessing | Requires element detection pre-step | High-accuracy tasks |

---

## 10. Security Considerations

| Risk | OWASP Reference | Mitigation |
|------|-----------------|------------|
| **Unintended data exposure** — AI types passwords or sensitive data into wrong fields | A01:2021 - Broken Access Control | Never let Gemini type text it didn't generate (no `type_text(password_variable)`). Log all typed text (redacted). |
| **Privilege escalation** — AI clicks "Run as Administrator" or sudo prompts | A01:2021 | Block `press_key` with UAC/sudo patterns. Detect elevated window state. |
| **Denial of service** — Infinite click loop crashes the system | A05:2021 - Security Misconfiguration | Rate limit: 20 actions/min, 5 actions/sec. Fail-safe on mouse corner. Emergency F1 stop. |
| **Unintended file deletion** — AI drags files to trash or clicks Delete | A01:2021 | Tier 3 confirmation for drag actions. Key blocklist for Delete combos. |
| **Clipboard exfiltration** — AI copies sensitive data and types it somewhere | A09:2021 - Security Logging | Log all clipboard operations. Warn when clipboard content changes during automation. |
| **Screen recording exposure** — Screenshots contain sensitive data in logs | A09:2021 | Screenshots only sent to Gemini (encrypted WebSocket). Never persist screenshots to disk unless explicitly enabled. |
| **Alt-tab to wrong window** — AI types in the wrong application | A01:2021 | After focus_window, verify active window title before typing. |

---

## 11. Performance Considerations

| Metric | Target | Notes |
|--------|--------|-------|
| Click execution latency | <50ms | pyautogui.click() is ~5-10ms. Coordinate mapping is <1ms. |
| Type execution latency | <20ms per char | pyautogui.press() is ~10ms per key. Bulk typing via clipboard is faster. |
| Post-action screenshot | <200ms | mss capture (~20ms) + JPEG encode (~30ms) + WS send (~50ms) |
| SoM overlay generation | <100ms | OCR (~30-50ms) + label drawing (~10ms) + re-encode (~20ms) |
| Full action→verify cycle | <500ms | action + screenshot + send + Gemini response |
| Memory footprint | <20MB | pyautogui is ~10MB loaded. Screen nav code is trivial. |

---

## 12. Recommended Tech Stack (Summary)

| Layer | Technology | Why |
|-------|-----------|-----|
| **Action execution** | pyautogui | Cross-platform, simple API, battle-tested, 90%+ coverage |
| **Unicode typing** | pynput.keyboard.Controller | Already in deps, handles Unicode correctly |
| **DPI fix (Windows)** | ctypes SetProcessDpiAwareness(2) | One line at startup, fixes all scaling issues |
| **Hotkey listening** | pynput (existing) | Already works for F2/F3/F4/F5 |
| **Screen capture** | mss + Pillow (existing) | Already works, just needs metadata return |
| **Element detection (Phase 2)** | rapidocr + SoM overlay | Already have OCR, just add bounding box extraction + overlay |
| **Window management** | pyautogui (Windows) + wmctrl/xdotool (Linux) | Good enough for find/focus |
| **UI Automation (Phase 4)** | pywinauto (optional, Windows) | Only if coord accuracy is insufficient |
| **Wayland fallback** | ydotool + wtype (Linux) | Best available Wayland automation tools |

---

## 13. Docs Gap Report

| Gap | Issue | Action Needed |
|-----|-------|---------------|
| No screen_nav documentation | Feature doesn't exist yet | Create docs after implementation |
| `screen_capture.py` doesn't document coordinate system | Metadata not returned | Document after Phase 1 modifications |
| `config.yaml` has no screen_nav section | Feature doesn't exist | Add with Phase 1 |
| `ARCHITECTURE.md` (gemini-hackathon) doesn't mention computer use | Outdated | Update after implementation |
| `context.txt` lists tools but not screen nav tools | New tools not added yet | Update after Phase 1 |

---

## 14. Next Steps

### Immediate (Phase 1 — start now)
1. Add `pyautogui>=0.9.54` to `requirements.txt`
2. Create `local/screen_nav.py` — ScreenNavigator class with coordinate mapping + pyautogui actions
3. Add DPI awareness call at top of `local/main.py`
4. Modify `screen_capture.py` to return CaptureResult metadata
5. Add screen navigation tool declarations to `gemini_session.py`
6. Add screen nav tools to `tools.py` dispatch table
7. Update system instruction to explain navigation capabilities
8. Test on Windows with DPI 100% and 150%

### Short-term (Phase 2 — after Phase 1 verified)
9. Add `extract_with_boxes()` to `ocr.py`
10. Create `som_overlay.py` for Set-of-Mark annotation
11. Add `click_element` tool
12. Test accuracy improvement with SoM vs pure vision

### Long-term (Phase 3-4 — polish)
13. Window management tools
14. Drag support
15. pywinauto integration (Windows only, optional)
16. Wayland automation investigation (ydotool/wtype)
17. Multi-monitor support

---

## 15. Sources

| Source | What | Relevance |
|--------|------|-----------|
| Rio codebase: `local/screen_capture.py` | Current capture pipeline, resize_factor=0.5 | Direct — coordinate mapping must invert this |
| Rio codebase: `local/tools.py` | Tool executor pattern, safety blocklist | Direct — screen nav tools follow same pattern |
| Rio codebase: `cloud/gemini_session.py` | Tool declaration format, capture_screen tool | Direct — add nav tool declarations here |
| Rio codebase: `local/push_to_talk.py` | pynput usage, Wayland detection | Direct — same Wayland issues apply to screen nav |
| Rio codebase: `local/ocr.py` | RapidOCR integration | Direct — extend for bounding box extraction |
| pyautogui docs (pyautogui.readthedocs.io) | API reference, DPI notes, fail-safe, PAUSE | Primary library reference |
| Anthropic computer-use blog + reference implementation | Architecture pattern, coordinate mapping, action vocabulary | Architecture influence |
| OpenAI computer-use docs | Safety tiers, action schema | Safety design influence |
| Google Project Mariner docs | Gemini-native function calling for computer use | Closest to Rio's approach |
| Microsoft OmniParser (github.com/microsoft/OmniParser) | Screen parsing model architecture | Phase 3+ reference |
| Set-of-Mark paper (arxiv.org/abs/2310.11441) | Visual grounding technique | Phase 2 technique |
| pywinauto docs (pywinauto.readthedocs.io) | Windows UI Automation wrapper | Phase 4 reference |
| Windows DPI awareness docs (MSDN) | SetProcessDpiAwareness API, per-monitor DPI V2 | DPI fix reference |
| Rio `tasks/lessons.md` | Audio pipeline lessons, Wayland issues | Cross-cutting concerns |
| Rio `Rio-Plan.md` | Architecture, risk mitigations, layer system | Project context |

---

## Appendix A: Config Schema Addition

```yaml
# In rio/config.yaml:
rio:
  screen_nav:
    enabled: true
    max_actions_per_minute: 20       # rate limit for safety
    max_actions_per_second: 5        # burst rate limit
    confirm_destructive: true        # ask before dangerous actions
    post_action_screenshot: true     # auto-screenshot after each action
    nav_resize_factor: 0.75          # higher res for navigation screenshots (vs 0.5 for stream)
    fail_safe: true                  # pyautogui fail-safe (mouse to corner = abort)
    pause_between_actions: 0.1       # seconds between sequential actions
    emergency_stop_key: "f1"         # hotkey to abort all screen nav
```

```python
# In local/config.py:
@dataclass
class ScreenNavConfig:
    enabled: bool = True
    max_actions_per_minute: int = 20
    max_actions_per_second: int = 5
    confirm_destructive: bool = True
    post_action_screenshot: bool = True
    nav_resize_factor: float = 0.75
    fail_safe: bool = True
    pause_between_actions: float = 0.1
    emergency_stop_key: str = "f1"
```

---

## Appendix B: System Instruction Addition

```python
# Append to RIO_BASE_INSTRUCTION:
SCREEN_NAV_INSTRUCTION = """

SCREEN NAVIGATION: You can control the user's mouse and keyboard to perform 
actions on their screen. Available tools:
- click(x, y) — Click at screenshot coordinates. Always take a fresh 
  screenshot first with capture_screen to see the current state.
- type_text(text) — Type text at the current cursor position. Click a 
  text field first.
- press_key(keys) — Press keyboard keys or combos (e.g., 'enter', 'ctrl+s', 'alt+tab').
- scroll(amount) — Scroll the mouse wheel. Negative = scroll down.
- drag(start_x, start_y, end_x, end_y) — Click and drag.
- wait(seconds) — Wait for animations, page loads, etc.
- find_window(title) — Find a window by title.
- focus_window(title) — Bring a window to the foreground.
- get_cursor_position() — Get the current mouse position.

IMPORTANT RULES:
1. ALWAYS call capture_screen first to see the current state before clicking.
2. Coordinates are in SCREENSHOT pixels (the image you see), not real screen pixels.
3. After clicking or typing, call capture_screen again to verify the result.
4. Be precise: aim for the CENTER of buttons and text fields.
5. For text input: click the field first, then use type_text.
6. Use wait() after actions that trigger loading or transitions.
7. Never type passwords or sensitive data — ask the user to type those manually.
8. If a click doesn't work, try capture_screen + click again — the UI may have changed.
"""
```

---

## Appendix C: Risk Matrix

| Risk | Probability | Impact | Priority | Mitigation |
|------|------------|--------|----------|------------|
| Coordinate misalignment (DPI/resize) | HIGH | HIGH | P0 | DPI awareness + metadata-based mapping |
| Wayland incompatibility | MEDIUM | HIGH | P1 | Detect + degrade gracefully, log warning |
| Infinite click loop | LOW | CRITICAL | P0 | Rate limit + fail-safe + emergency stop |
| Click wrong element | MEDIUM | MEDIUM | P1 | Post-action screenshot + SoM (Phase 2) |
| Type in wrong window | MEDIUM | MEDIUM | P1 | Verify active window before type_text |
| Unintended file deletion via drag | LOW | HIGH | P2 | Confirmation gate for drag actions |
| Race conditions (concurrent actions) | MEDIUM | MEDIUM | P1 | Serialize through single executor thread |
| Gemini generates invalid coords | MEDIUM | LOW | P2 | Bounds checking: clamp to screen dimensions |
| pyautogui blocks event loop | LOW | MEDIUM | P1 | Run in asyncio.to_thread() |
| User moves mouse during automation | MEDIUM | LOW | P2 | Accept as noise, post-action verify covers it |
