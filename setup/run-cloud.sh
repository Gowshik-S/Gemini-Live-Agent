#!/usr/bin/env bash
# =============================================================================
#  Rio — Start Cloud Server (Linux / macOS)
#  Activates cloud venv and runs cloud/main.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLOUD_DIR="$RIO_ROOT/cloud"

if [[ ! -f "$CLOUD_DIR/venv/bin/activate" ]]; then
    echo "ERROR: Cloud venv not found. Run setup.sh first."
    exit 1
fi

if [[ ! -f "$CLOUD_DIR/.env" ]]; then
    echo "WARNING: No .env file found. GEMINI_API_KEY may not be set."
    echo "  Create $CLOUD_DIR/.env with: GEMINI_API_KEY=your_key"
    echo ""
fi

echo "Starting Rio Cloud Server..."
echo "  Dir: $CLOUD_DIR"
echo "  Press Ctrl+C to stop."
echo ""

cd "$CLOUD_DIR"
source venv/bin/activate
python main.py
