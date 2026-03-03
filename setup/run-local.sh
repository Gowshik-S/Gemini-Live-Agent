#!/usr/bin/env bash
# =============================================================================
#  Rio — Start Local Client (Linux / macOS)
#  Activates local venv and runs local/main.py
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_DIR="$RIO_ROOT/local"

if [[ ! -f "$LOCAL_DIR/venv/bin/activate" ]]; then
    echo "ERROR: Local venv not found. Run setup.sh first."
    exit 1
fi

echo "Starting Rio Local Client..."
echo "  Dir: $LOCAL_DIR"
echo "  Press Ctrl+C to stop."
echo ""

cd "$LOCAL_DIR"
source venv/bin/activate
python main.py
