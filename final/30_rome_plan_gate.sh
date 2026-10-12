#!/usr/bin/env bash
# Rome pre-rules (day-early bands and the bagged plan gate) on a prediction file, February and June 2026. Heavy: builds the feature matrix of the served months
# and fits 20 x (2 classifiers + 2 regressors) of LightGBM on the 2025 Rome plan rows. ESTIMATE, NOT MEASURED: peak 6-8 GB commit, about 150 minutes with 4 threads.
# Smoke test first (1 seed, about 10 minutes, same memory):  bash final/30_rome_plan_gate.sh IN.parquet OUT.parquet 1
# Do not start before the base refit (9 GB slot, until about 00:30 EDT) has ended unless a 3 GB fit is enough (it is not).
#   bash final/30_rome_plan_gate.sh <prediction parquet> <output parquet> [seeds=20]
set -uo pipefail
REPO="${PRC_ROOT:-$PWD}"
V="./.venv/Scripts/python.exe"
if [ "${1:-}" != "--inner" ]; then
  export PY=/c/Python314/python.exe; cd "$REPO" || exit 1; mkdir -p logs data/final
  exec archive/run_job.sh --detach --project prc-taxiout-final --min-gb 6 --want-gb 8 --minutes 240 --eta-minutes 150 --preemptible \
    --why "PRC final: Rome plan gate (20-seed bag) on February and June 2026" \
    --argument "experiment run" \
    --match "prc-taxiout-2026.*30_general_rules" -- bash "$REPO/final/30_rome_plan_gate.sh" --inner "$@"
fi
shift; IN="$1"; OUT="$2"; SEEDS="${3:-20}"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$REPO" OMP_NUM_THREADS=4
"$V" "$REPO/archive/capped_run.py" --cap-gb 9 -- "$V" "$REPO/final/30_general_rules.py" --pred "$IN" --out "$OUT" --stage pre --months 2,6 --gate-seeds "$SEEDS" >> "$REPO/logs/final_30_rome_gate.log" 2>&1
