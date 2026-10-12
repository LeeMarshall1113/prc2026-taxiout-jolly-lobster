#!/usr/bin/env bash
# experiment blend runner: scheduler slot + OS commit cap (STANDING RULE). Copy of agent_backx_run.sh (docs/LAB-NOTES.md).
#   bash archive/agent_blend_run.sh <eta_min> script.py [args]            (python script, capped at 4 GB)
#   BL_BASH=1 bash archive/agent_blend_run.sh <eta_min> script.sh [args]  (bash driver; every python inside is capped itself)
cd "${PRC_ROOT:-$PWD}"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)" OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
M="$1"; shift 1
PY=./.venv/Scripts/python.exe
if [ -n "$BL_BASH" ]; then CMD=(bash "$@"); else CMD=("$PY" archive/capped_run.py --cap-gb 4 -- "$PY" -u "$@"); fi
archive/run_job.sh ${BL_DETACH:+--detach} --project prc-taxiout --min-gb 3 --want-gb 4 --minutes $((M * 2 + 10)) --eta-minutes "$M" --preemptible \
  --why "PRC no-new-data per-airport blend (super-learner over saved ADS-B-stage arms): $(basename "$1")" \
  --match "agent_blend|splice_apt_parquet|splice_npz|apply_seq_residual|agent_smallbundle_06_check|agent_lens_09" \
  --argument "experiment run" -- \
  "${CMD[@]}"
