# Rio — Setup Scripts

One-command setup for the Rio AI Pair Programmer. Works on **Windows**, **Linux**, and **macOS**.

## Prerequisites

- **Python 3.11+** installed and on PATH
  - Windows: [python.org](https://www.python.org/downloads/) — check "Add Python to PATH"
  - Linux: `sudo apt install python3 python3-venv` (Debian/Ubuntu)
  - macOS: `brew install python@3.12` or [python.org](https://www.python.org/downloads/)
- **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Quick Start

### Windows

```cmd
cd Rio-Agent\rio\setup
setup.bat
```

Then in two separate terminals:

```cmd
run-cloud.bat     REM Terminal 1 — Cloud server
run-local.bat     REM Terminal 2 — Local client
```

### Linux / macOS

```bash
cd Rio-Agent/rio/setup
chmod +x *.sh
./setup.sh
```

Then in two separate terminals:

```bash
./run-cloud.sh    # Terminal 1 — Cloud server
./run-local.sh    # Terminal 2 — Local client
```

## What Setup Does

1. **Checks Python 3.11+** is available
2. **Creates `cloud/venv`** — installs FastAPI, uvicorn, google-genai, etc.
3. **Creates `local/venv`** — installs PyTorch CPU, sounddevice, mss, Pillow, scikit-learn, etc.
4. **Prompts for `GEMINI_API_KEY`** — saves to `cloud/.env`
5. **Prompts for cloud/dashboard port** — writes `PORT` to `cloud/.env` and updates `cloud_url` in `rio/config.yaml`
6. **Asks start-on-boot consent**
7. **(Linux/macOS) creates optional `systemd --user` service files** for cloud and local
8. **Runs mandatory onboarding** via `rio configure`
9. **Prints live URL acknowledgement**: `Rio is live at: http://localhost:<port>/dashboard`

## Mandatory Onboarding

- Installer scripts (`scripts/install.sh`, `scripts/install.ps1`) now launch `rio configure` immediately after installation.
- Local setup scripts (`setup/setup.sh`, `setup/setup.bat`) also launch `rio configure` before finishing.
- `rio run` enforces configure flow by default if portal key setup is incomplete.

## File Overview

| File | Platform | Purpose |
|------|----------|---------|
| `setup.bat` | Windows | Full environment setup |
| `setup.sh` | Linux/Mac | Full environment setup |
| `run-cloud.bat` | Windows | Start cloud server |
| `run-cloud.sh` | Linux/Mac | Start cloud server |
| `run-local.bat` | Windows | Start local client |
| `run-local.sh` | Linux/Mac | Start local client |

## Re-running Setup

Running setup again is safe — it skips venv creation if the folder already exists and lets you optionally update `.env` values and port settings. To force a fresh install, delete the `venv/` folders first:

```cmd
REM Windows
rmdir /s /q ..\cloud\venv
rmdir /s /q ..\local\venv
setup.bat
```

```bash
# Linux/Mac
rm -rf ../cloud/venv ../local/venv
./setup.sh
```

## Configuration

- **API Key**: `rio/cloud/.env`
- **Rio Settings**: `rio/config.yaml` (cloud URL, audio, vision, struggle detection, etc.)
