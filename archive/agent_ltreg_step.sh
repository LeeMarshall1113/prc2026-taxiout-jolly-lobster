#!/usr/bin/env bash
# experiment ltreg: run ONE python step under capped_run inside the slot.
#   bash archive/agent_ltreg_step.sh <script.py> <log> [args]
cd "${PRC_ROOT:-$PWD}"
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH=. OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
S="$1"; L="$2"; shift 2
echo "[$(date -u +%H:%M:%SZ)] step $S -> $L"
$PY archive/capped_run.py --cap-gb ${LR_CAP:-4} -- $PY -u "$S" "$@" > "$L" 2>&1
RC=$?
tail -120 "$L"
echo "[$(date -u +%H:%M:%SZ)] step rc $RC"
exit $RC
