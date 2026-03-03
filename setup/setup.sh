#!/usr/bin/env bash
# =============================================================================
#  Rio — One-Command Setup (Linux / macOS)
#
#  Usage:
#    chmod +x setup.sh && ./setup.sh
#
#  What it does:
#    1. Checks Python 3.10+ is installed
#    2. Creates cloud/venv and installs cloud dependencies
#    3. Creates local/venv and installs local dependencies (incl. PyTorch CPU)
#    4. Prompts for GEMINI_API_KEY and writes cloud/.env
#    5. Prints summary
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
#  Resolve project root (rio/ folder — parent of setup/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CLOUD_DIR="$RIO_ROOT/cloud"
LOCAL_DIR="$RIO_ROOT/local"

echo ""
echo "============================================================"
echo "  Rio — Automated Setup (Linux / macOS)"
echo "============================================================"
echo ""
echo "  Project root : $RIO_ROOT"
echo "  Cloud dir    : $CLOUD_DIR"
echo "  Local dir    : $LOCAL_DIR"
echo ""

# ---------------------------------------------------------------------------
#  Step 0 — Find Python 3.10+
# ---------------------------------------------------------------------------
echo "[1/5] Checking Python installation..."

# Try python3 first, then python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo ""
    echo "ERROR: Python not found."
    echo "  Install Python 3.10+ from https://www.python.org/downloads/"
    echo "  or via your package manager (e.g. sudo apt install python3 python3-venv)"
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

echo "  Found $PYTHON_CMD $PY_VERSION"

if [[ "$PY_MAJOR" -lt 3 ]] || { [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 10 ]]; }; then
    echo "ERROR: Python 3.10+ required. Found $PY_VERSION."
    exit 1
fi

echo "  Python version OK."
echo ""

# ---------------------------------------------------------------------------
#  Step 1 — Ensure venv module is available
# ---------------------------------------------------------------------------
echo "[2/5] Checking venv module..."

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
echo "[3/5] Setting up Cloud environment..."

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
echo "[4/5] Setting up Local environment..."

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
echo "[5/5] Configuring API key..."

if [[ -f "$CLOUD_DIR/.env" ]]; then
    echo "  .env file already exists in cloud/. Skipping."
    echo "  Edit $CLOUD_DIR/.env to change your API key."
else
    echo ""
    echo "  Rio needs a Google Gemini API key to function."
    echo "  Get one at: https://aistudio.google.com/app/apikey"
    echo ""
    read -rp "  Enter your GEMINI_API_KEY (or press Enter to skip): " API_KEY
    if [[ -z "$API_KEY" ]]; then
        echo "  Skipped. Create cloud/.env manually later:"
        echo "    GEMINI_API_KEY=your_key_here"
        echo "GEMINI_API_KEY=your_key_here" > "$CLOUD_DIR/.env"
    else
        echo "GEMINI_API_KEY=$API_KEY" > "$CLOUD_DIR/.env"
        echo "  API key saved to cloud/.env"
    fi
fi

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  To start Rio:"
echo ""
echo "  1. Start Cloud server (in one terminal):"
echo "       setup/run-cloud.sh"
echo ""
echo "  2. Start Local client (in another terminal):"
echo "       setup/run-local.sh"
echo ""
echo "  Config: rio/config.yaml"
echo "  API Key: rio/cloud/.env"
echo "============================================================"
echo ""
