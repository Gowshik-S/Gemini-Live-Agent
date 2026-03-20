#!/usr/bin/env bash
# =============================================================================
#  Rio — One-Command Setup (Linux / macOS)
#
#  Usage:
#    chmod +x setup.sh && ./setup.sh
#
#  What it does:
#    1. Checks Python 3.11+ is installed
#    2. Creates cloud/venv and installs cloud dependencies
#    3. Creates local/venv and installs local dependencies (incl. PyTorch CPU)
#    4. Prompts for GEMINI_API_KEY and writes cloud/.env
#    5. Asks dashboard/live-feed port and updates cloud/.env + config.yaml
#    6. Asks permission to configure start-on-boot via systemd user services
#    7. Prints summary
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
#  Resolve project root (rio/ folder — parent of setup/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CLOUD_DIR="$RIO_ROOT/cloud"
LOCAL_DIR="$RIO_ROOT/local"
CONFIG_PATH="$RIO_ROOT/config.yaml"

echo ""
echo "============================================================"
echo "  Rio — Automated Setup (Linux / macOS)"
echo "============================================================"
echo ""
echo "  Project root : $RIO_ROOT"
echo "  Cloud dir    : $CLOUD_DIR"
echo "  Local dir    : $LOCAL_DIR"
echo ""

upsert_env_var() {
    local file="$1"
    local key="$2"
    local value="$3"

    touch "$file"
    if grep -qE "^${key}=" "$file"; then
        sed -i.bak -E "s|^${key}=.*|${key}=${value}|" "$file"
        rm -f "${file}.bak"
    else
        echo "${key}=${value}" >> "$file"
    fi
}

set_cloud_url_port() {
    local port="$1"
    "$PYTHON_CMD" - "$CONFIG_PATH" "$port" <<'PY'
import pathlib
import re
import sys

config_path = pathlib.Path(sys.argv[1])
port = sys.argv[2]
text = config_path.read_text(encoding="utf-8")
updated = re.sub(r"(cloud_url:\s*ws://localhost:)\d+(/ws/rio/live)", rf"\\g<1>{port}\\2", text)
if updated == text:
    updated = re.sub(r"cloud_url:\s*.*", f"cloud_url: ws://localhost:{port}/ws/rio/live", text, count=1)
config_path.write_text(updated, encoding="utf-8")
PY
}

create_systemd_service_files() {
    local port="$1"
    local user_systemd_dir="$HOME/.config/systemd/user"
    mkdir -p "$user_systemd_dir"

    cat > "$user_systemd_dir/rio-cloud.service" <<EOF
[Unit]
Description=Rio Cloud Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$CLOUD_DIR
Environment=PORT=$port
ExecStart=$CLOUD_DIR/venv/bin/python $CLOUD_DIR/main.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF

    cat > "$user_systemd_dir/rio-local.service" <<EOF
[Unit]
Description=Rio Local Client
After=network-online.target rio-cloud.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$LOCAL_DIR
ExecStart=$LOCAL_DIR/venv/bin/python $LOCAL_DIR/main.py
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
EOF
}

# ---------------------------------------------------------------------------
#  Step 0 — Find Python 3.11+
# ---------------------------------------------------------------------------
echo "[1/7] Checking Python installation..."

# Try python3 first, then python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo ""
    echo "ERROR: Python not found."
    echo "  Install Python 3.11+ from https://www.python.org/downloads/"
    echo "  or via your package manager (e.g. sudo apt install python3 python3-venv)"
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

echo "  Found $PYTHON_CMD $PY_VERSION"

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 11 ]]; }; then
    echo "ERROR: Python 3.11+ required. Found $PY_VERSION."
    exit 1
fi

echo "  Python version OK."
echo ""

# ---------------------------------------------------------------------------
#  Step 1 — Ensure venv module is available
# ---------------------------------------------------------------------------
echo "[2/7] Checking venv module..."

$PYTHON_CMD -m venv --help &>/dev/null || {
    echo ""
    echo "ERROR: Python venv module not available."
    echo "  Install it: sudo apt install python3-venv  (Debian/Ubuntu)"
    echo "              or: sudo dnf install python3-venv  (Fedora)"
    exit 1
}
echo "  venv module OK."
echo ""

# ---------------------------------------------------------------------------
#  Step 2 — Cloud virtual environment
# ---------------------------------------------------------------------------
echo "[3/7] Setting up Cloud environment..."

if [[ ! -d "$CLOUD_DIR" ]]; then
    echo "ERROR: Cloud directory not found at $CLOUD_DIR"
    exit 1
fi

if [[ -d "$CLOUD_DIR/venv" ]]; then
    echo "  Cloud venv already exists, skipping creation."
else
    echo "  Creating cloud/venv..."
    $PYTHON_CMD -m venv "$CLOUD_DIR/venv"
fi

echo "  Installing cloud dependencies..."
source "$CLOUD_DIR/venv/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$CLOUD_DIR/requirements.txt" --quiet
deactivate
echo "  Cloud environment ready."
echo ""

# ---------------------------------------------------------------------------
#  Step 3 — Local virtual environment
# ---------------------------------------------------------------------------
echo "[4/7] Setting up Local environment..."

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "ERROR: Local directory not found at $LOCAL_DIR"
    exit 1
fi

if [[ -d "$LOCAL_DIR/venv" ]]; then
    echo "  Local venv already exists, skipping creation."
else
    echo "  Creating local/venv..."
    $PYTHON_CMD -m venv "$LOCAL_DIR/venv"
fi

echo "  Installing local dependencies..."
source "$LOCAL_DIR/venv/bin/activate"
pip install --upgrade pip --quiet

# Install PyTorch CPU separately (requires --index-url)
echo "  Installing PyTorch (CPU-only)..."
pip install "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu --quiet || {
    echo "  WARNING: PyTorch CPU install failed. VAD will be disabled."
    echo "  You can retry manually: pip install torch --index-url https://download.pytorch.org/whl/cpu"
}

# Install remaining local deps (filter out torch line and --index-url)
echo "  Installing remaining local dependencies..."
grep -v -i "torch" "$LOCAL_DIR/requirements.txt" | grep -v "index-url" | pip install -r /dev/stdin --quiet || {
    echo "ERROR: Failed to install local dependencies."
    deactivate
    exit 1
}

deactivate
echo "  Local environment ready."
echo ""

# ---------------------------------------------------------------------------
#  Step 4 — GEMINI_API_KEY
# ---------------------------------------------------------------------------
echo "[5/7] Configuring API key..."

if [[ -f "$CLOUD_DIR/.env" ]]; then
    echo "  .env file already exists in cloud/."
    read -rp "  Update GEMINI_API_KEY now? [y/N]: " UPDATE_KEY
    if [[ "$UPDATE_KEY" =~ ^[Yy]$ ]]; then
        read -rp "  Enter your GEMINI_API_KEY (or press Enter to keep current): " API_KEY
        if [[ -n "$API_KEY" ]]; then
            upsert_env_var "$CLOUD_DIR/.env" "GEMINI_API_KEY" "$API_KEY"
            echo "  API key updated in cloud/.env"
        fi
    fi
else
    echo ""
    echo "  Rio needs a Google Gemini API key to function."
    echo "  Get one at: https://aistudio.google.com/app/apikey"
    echo ""
    read -rp "  Enter your GEMINI_API_KEY (or press Enter to skip): " API_KEY
    if [[ -z "$API_KEY" ]]; then
        echo "GEMINI_API_KEY=your_key_here" > "$CLOUD_DIR/.env"
    else
        echo "GEMINI_API_KEY=$API_KEY" > "$CLOUD_DIR/.env"
        echo "  API key saved to cloud/.env"
    fi
fi

echo ""

# ---------------------------------------------------------------------------
#  Step 5 — Dashboard/live-feed port
# ---------------------------------------------------------------------------
echo "[6/7] Configuring dashboard/live-feed port..."
DEFAULT_PORT="8080"
read -rp "  Enter cloud/dashboard port [${DEFAULT_PORT}]: " PORT_INPUT
RIO_PORT="${PORT_INPUT:-$DEFAULT_PORT}"

if ! [[ "$RIO_PORT" =~ ^[0-9]+$ ]] || [[ "$RIO_PORT" -lt 1 ]] || [[ "$RIO_PORT" -gt 65535 ]]; then
    echo "  Invalid port '$RIO_PORT'. Falling back to ${DEFAULT_PORT}."
    RIO_PORT="$DEFAULT_PORT"
fi

upsert_env_var "$CLOUD_DIR/.env" "PORT" "$RIO_PORT"
set_cloud_url_port "$RIO_PORT"
echo "  Port set to $RIO_PORT"
echo "  Updated cloud_url in config.yaml"
echo ""

# ---------------------------------------------------------------------------
#  Step 6 — Ask permission for start-on-boot (systemd)
# ---------------------------------------------------------------------------
echo "[7/7] Start-on-boot configuration"
read -rp "  Enable Rio Cloud/Local on every boot (systemd user services)? [y/N]: " ENABLE_BOOT
if [[ "$ENABLE_BOOT" =~ ^[Yy]$ ]]; then
    create_systemd_service_files "$RIO_PORT"
    echo "  Created systemd service files in ~/.config/systemd/user"

    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user daemon-reload
        systemctl --user enable rio-cloud.service rio-local.service
        read -rp "  Start services now? [y/N]: " START_NOW
        if [[ "$START_NOW" =~ ^[Yy]$ ]]; then
            systemctl --user start rio-cloud.service rio-local.service
            echo "  Services started."
        fi
    else
        echo "  systemctl not found. Services created but not enabled automatically."
    fi
else
    echo "  Skipped start-on-boot setup."
fi

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  To start Rio manually:"
echo ""
echo "  1. Start Cloud server (in one terminal):"
echo "       setup/run-cloud.sh"
echo "     Dashboard: http://localhost:${RIO_PORT}/dashboard"
echo "     Live feed WS: ws://localhost:${RIO_PORT}/ws/dashboard"
echo ""
echo "  2. Start Local client (in another terminal):"
echo "       setup/run-local.sh"
echo ""
echo "  Config: rio/config.yaml"
echo "  API Key + PORT: rio/cloud/.env"
echo ""
echo "  Launching mandatory onboarding now: rio configure"
"$PYTHON_CMD" "$RIO_ROOT/cli.py" configure
echo ""
echo "  Rio is live at: http://localhost:${RIO_PORT}/dashboard"
echo "============================================================"
echo ""
