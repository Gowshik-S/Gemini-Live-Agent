# Rio Agent

Rio is a proactive AI pair programmer built for the Gemini Live Agent Challenge. It combines a local desktop client, a FastAPI cloud relay, Gemini Live and text sessions, screen capture, local tool execution, autonomous screen navigation, struggle detection, dashboard telemetry, and optional memory and ML layers so the agent can notice when a developer is stuck and step in before being asked.

One-line pitch: "The AI pair programmer that does not wait to be asked."

## What Rio Can Do

- Talk with the user through live audio or plain text.
- See the screen through on-demand screenshots or continuous capture.
- Execute local tools: `read_file`, `write_file`, `patch_file`, `run_command`.
- Operate the desktop with screen actions such as click, type, scroll, drag, and window focus.
- Detect likely struggle states from screen behavior and trigger proactive help.
- Persist chat history, user behavior patterns, tickets, tutor progress, and optional vector memory.
- Run domain-specific modes for customer care and tutoring from deterministic profile files.
- Expose a dashboard for transcript, tool log, connection health, and struggle telemetry.

## Feature Status

| Area | Status | Notes |
| --- | --- | --- |
| Live audio + text fallback | Implemented | `session_mode: live` falls back to text mode if Live API connection fails |
| Screen capture | Implemented | F3 on-demand capture plus periodic autonomous capture |
| Autonomous computer-use loop | Implemented | Screen actions auto-capture a follow-up screenshot for the next step |
| Local code/file tools | Implemented | Safe path resolution, backups, command blocklist |
| Struggle detection | Implemented | Four screen-based signals plus cooldowns and demo mode |
| Dashboard UI | Implemented | Transcript, struggle gauge, tool log, health, setup page |
| Customer care + tutor skills | Implemented | Fixed-schema profiles and local persistence |
| Chat history + pattern tracking | Implemented | SQLite-backed stores |
| ML ensemble pipeline | Implemented/optional | Active when `scikit-learn` is available |
| Vector memory | Optional | Requires `chromadb` and `sentence-transformers` |
| Gemini Pro routing | Partial | Heuristic exists, but `ModelRouter.call_pro()` is still a stub |

## Architecture

```text
Local desktop client                Cloud relay                     Gemini
--------------------                -----------                     ------
rio/local/main.py   <--- WS --->    rio/cloud/main.py   <------->   Live/text models
  - mic + speaker                    - FastAPI
  - VAD + wake word                  - session manager
  - screen capture                   - live/text relay
  - tool executor                    - dashboard websocket
  - screen navigation                - rate limiter
  - struggle detector                - model router
  - local stores
```

High-level flow:

1. The local client captures audio, text input, screenshots, and optional OCR or ML context.
2. The cloud service maintains a Gemini session and relays audio, text, images, and tool calls.
3. Tool calls execute on the local machine, not in the cloud.
4. Screen actions can trigger an automatic screenshot so Gemini can verify the result and continue the task.
5. The dashboard subscribes to transcript, tool, and health events from the cloud service.

## Repository Layout

```text
Rio-Agent/
  README.md
  Rio-Plan.md
  context.txt
  rio/
    config.yaml
    cloud/          FastAPI relay, Gemini session wrapper, Dockerfile, Cloud Run config
    local/          Desktop client, tools, screen nav, OCR, wake word, memory, chat store
    ml/             Feature extraction, ensemble model, training pipeline, bootstrap model
    setup/          Cross-platform setup and run scripts
    ui/dashboard/   Dashboard and setup UI
    tests/          Smoke/integration-style test script
```

## Quick Start

### Prerequisites

- Python 3.11+ (3.10+ may work but 3.11+ is recommended)
- A Gemini API key from Google AI Studio

### Installation

#### Option 1: Using setup scripts (Recommended)

**Windows:**

```cmd
cd Rio-Agent\rio\setup
setup.bat
```

Then run:

```cmd
run-cloud.bat
run-local.bat
```

**Linux / macOS:**

```bash
cd Rio-Agent/rio/setup
chmod +x *.sh
./setup.sh
```

Then run:

```bash
./run-cloud.sh
./run-local.sh
```

#### Option 2: Manual installation

```bash
cd Rio-Agent/rio

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements-dev.txt
```

After startup:

- Cloud service: `http://localhost:8080/health`
- Dashboard: `http://localhost:8080/dashboard`
- Skill setup page: `http://localhost:8080/dashboard/setup.html`

## Runtime Controls

Hotkeys only work when `pynput` is installed and your OS allows global keyboard hooks.

| Key | Behavior |
| --- | --- |
| `F2` | Push to talk |
| `F3` | Force a screenshot |
| `F4` | Force a proactive struggle trigger in demo mode |
| `F5` | Toggle screen mode between `on_demand` and `autonomous` |
| `F6` | Toggle Live Mode (continuous screen + autonomous agentic behavior) |

Default screen modes:

- `on_demand`: screenshots are sent only when requested by the user, by F3, or by the `capture_screen` tool.
- `autonomous`: screenshots are sent periodically based on `vision.fps`.
- `Live Mode`: forces autonomous screen capture and wake-word-style listening so Rio can behave like a continuously active desktop agent.

## Configuration

Main runtime config lives in [`rio/config.yaml`](rio/config.yaml).

Important settings:

| Setting | Purpose |
| --- | --- |
| `cloud_url` | WebSocket URL for the cloud relay |
| `session_mode` | `live` or `text` |
| `vision.default_mode` | `on_demand` or `autonomous` |
| `struggle.enabled` | Enables proactive struggle detection |
| `struggle.demo_mode` | Lowers thresholds for demos |
| `hotkeys.*` | Push-to-talk, screenshot, mode toggles |
| `memory.*` | Local vector memory path and recall count |
| `models.*` | Primary, secondary, and Pro RPM budget settings |
| `skills.*` | Ticket and tutor progress directories plus defaults |

Cloud-side environment variables:

- `GEMINI_API_KEY` required
- `SESSION_MODE` optional override
- `TEXT_MODEL` optional override
- `LIVE_MODEL` optional override
- `RIO_WS_TOKEN` optional shared-secret WebSocket auth
- `PRO_RPM_BUDGET` optional override for Pro routing budget

## Optional Dependencies

The setup scripts install the base cloud stack, the base local stack, and CPU PyTorch. Some capabilities are intentionally optional and need extra packages or assets.

### Hotkeys

`pynput` is commented out in the checked-in local requirements, so install it manually if you want F2-F6 controls:

```bash
cd Rio-Agent/rio/local
pip install pynput
```

### Screen navigation / computer use

For click, type, drag, scroll, and window management:

```bash
cd Rio-Agent/rio/local
pip install pyautogui pygetwindow pyperclip
```

Notes:

- Windows is the best-supported target because Rio enables DPI awareness before coordinate-based actions.
- Linux Wayland sessions degrade gracefully but global input and screen automation are still less reliable than Windows or X11.
- `pyautogui` fail-safe remains enabled: moving the mouse to the top-left corner aborts pending automation.

### Vector memory

For persistent semantic recall:

```bash
cd Rio-Agent/rio/local
pip install chromadb sentence-transformers
```

### Offline wake word

The code already includes `vosk`, but true local wake-word spotting needs a Vosk model in one of these locations:

- `~/.rio/vosk-model-small-en-us`
- `rio/local/vosk-model/`

Without a model, Rio falls back to an energy-based wake-word heuristic.

## Safety Model

Rio is designed to degrade gracefully and avoid destructive behavior by default.

- Dangerous shell patterns such as `rm -rf`, `mkfs`, `shutdown`, `curl | bash`, and similar commands are blocked.
- `write_file` and `patch_file` create `.rio.bak` backups before modifying files.
- File paths are resolved relative to the tool working directory and blocked from escaping it.
- Dangerous tool calls are confirmed locally before execution.
- Screen actions are rate-limited and logged, and `pyautogui.FAILSAFE` stays enabled.

## Working Directory Scope

Rio's local file tools are scoped to the process working directory. The provided `run-local` scripts start the client from `rio/local`, so by default Rio can only read or modify files in that directory tree.

If you want Rio to operate on a different project, launch `rio/local/main.py` from that project's root so the tool executor resolves paths there.

## Skills and Persistence

### Customer Care

- Deterministic profile schema in `rio/local/profiles.py`
- Setup UI saves `customer_care_profile.json`
- Ticket tools persist JSON tickets under `rio_tickets/`

### Tutor

- Deterministic student profile schema in `rio/local/profiles.py`
- Setup UI saves `tutor_profile.json`
- Tutor tools persist progress under `rio_progress/`

### Local state generated by Rio

- `rio/rio_chats.db` - chat history
- `rio/rio_patterns.db` - user behavior and struggle patterns
- `rio/rio_memory/` - optional vector memory store
- `rio/rio_profiles/` - saved customer care and tutor profiles
- `rio/local/*.rio.bak` or project-local `.rio.bak` files - backups from edits

## Testing

Rio includes a smoke/integration-style test script at [`rio/tests/test_all.py`](rio/tests/test_all.py).

Run it with:

```bash
cd Rio-Agent
python rio/tests/test_all.py
```

The suite validates imports, config loading, rate limiting, model routing, tool execution, dashboard files, and wire protocol constants. Optional features may be skipped if their dependencies are not installed.

## Cloud Run Deployment

The repo includes a deploy script and container spec:

- [`rio/deploy.sh`](rio/deploy.sh)
- [`rio/cloud/service.yaml`](rio/cloud/service.yaml)
- [`rio/cloud/Dockerfile`](rio/cloud/Dockerfile)

Deployment assumptions:

1. `gcloud` is installed and authenticated.
2. Cloud Run, Cloud Build, and Secret Manager are enabled.
3. A secret named `gemini-api-key` exists in Secret Manager.

Then run:

```bash
cd Rio-Agent/rio
chmod +x deploy.sh
./deploy.sh
```

The script prints the HTTP URL and the WebSocket URL to put back into `rio/config.yaml`.

## Known Limitations

- Gemini Pro routing is not complete yet. The router can decide that Pro should be used, but the actual Pro call path still returns `None`.
- The checked-in model settings and the cloud runtime are not perfectly aligned. `cloud/gemini_session.py` defaults to `gemini-2.5-flash` and `gemini-2.5-flash-native-audio-latest`, regardless of the older model notes elsewhere in the repo.
- Memory, some hotkeys, and some screen-navigation features are optional and may appear disabled until their dependencies are installed.
- Desktop automation on Wayland and elevated/admin windows is inherently less reliable than standard Windows desktop automation.
- The codebase contains version strings from multiple milestones (`v0.6.x`, `v0.7.x`, `v0.9.0`) because the project evolved layer by layer.

## Tech Stack

| Layer | Tech |
| --- | --- |
| Cloud relay | FastAPI, uvicorn, websockets, structlog |
| Gemini integration | `google-genai`, Gemini 2.5 Flash text mode, Gemini Live audio mode |
| Audio | `sounddevice`, CPU PyTorch, Silero VAD |
| Vision | `mss`, Pillow, RapidOCR |
| Desktop automation | `pyautogui`, `pygetwindow`, `pyperclip` |
| Memory and local state | SQLite, optional ChromaDB, sentence-transformers |
| ML | NumPy, scikit-learn ensemble pipeline |
| UI | Static HTML/CSS/JS dashboard served by FastAPI |

## Related Docs

- [`Rio-Plan.md`](Rio-Plan.md) - architecture and hackathon plan
- [`context.txt`](context.txt) - project-wide context snapshot
- [`rio/setup/README.md`](rio/setup/README.md) - setup-script-focused quick reference
