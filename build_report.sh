#!/bin/bash
set -e

# Change to the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run the Python build script using .venv if present, else default python3
if [ -f ".venv/bin/python" ]; then
    .venv/bin/python build_report.py "$@"
elif command -v python3 &>/dev/null; then
    python3 build_report.py "$@"
else
    echo "[ERROR] Python 3 not found."
    exit 1
fi
