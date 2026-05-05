#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m uvicorn progress_note_module.test_ui_app:app --host 0.0.0.0 --port 8080 --reload
