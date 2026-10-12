#!/usr/bin/env bash
# Driver of the final-phase pipeline: bash final/run_all.sh --dry-run | --run [--only ID] [--from ID] (see final/run_all.py)
# Runs from the repository root, found from this script's own location. Interpreter: $PYTHON if set, else the repository's
# virtual environment (.venv/Scripts/python.exe on Windows, .venv/bin/python elsewhere), else python3 / python on the path.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
if [ -n "$PYTHON" ]; then PY="$PYTHON"
elif [ -x "$REPO/.venv/Scripts/python.exe" ]; then PY="$REPO/.venv/Scripts/python.exe"
elif [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi
exec "$PY" "$REPO/final/run_all.py" "$@"
