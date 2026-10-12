#!/usr/bin/env bash
# experiment frontier: 60-day stage-1 check of anchors in the floor stack (docs/LAB-NOTES.md). Run under run_job.sh.
# Days: 30 Oct-Nov 2025 days (the parity NOT used by the replica dump) + 30 Jan-Sep days (every 6th of the 203 calibration days
# that have fold predictions). LODO over all 60. Arms, same evaluator (archive/agent_frontier_stage1_eval.py), differing ONLY in
# --floor-stack-feats: Q = quality (v97/v98 serving), F = quality_anch. Judge: archive/agent_frontier_04_judge.py (rule fixed there).
# Each arm runs under capped_run at 5 GB. Resumable. No backslashes.
set -uo pipefail
cd "${PRC_ROOT:-$PWD}"
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 RAYON_NUM_THREADS=4
LOG=logs/agent_frontier_s1.log
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ON=$(ls data/adsb/points/2025-1[01]-*.parquet | sed 's/.*points\///;s/\.parquet//' | sed -n '2~2p' | paste -sd, -)
JS=$(cat data/interim/calib_days_v41_203.txt | tr , '\n' | grep -vE "^2025-04" | sed -n '1~6p' | head -30 | paste -sd, -)
DAYS="$JS,$ON"
N=$(echo "$DAYS" | tr ',' '\n' | grep -c .)
PREDS="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,data/interim/calib2026/preds_oct_f4.parquet,data/interim/calib2026/preds_nov_c.parquet"
SHIP="--stack gbm --floor on --floor-stack on --stack-skip-airports LIRF --floor-stack-snapshot on --floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG --blends both_guard --floor-q zero --osm-fill on --spill-tracks on --stack-feats quality_anch"
say "frontier stage-1 check on $N days"
for CFG in "Q:quality" "F:quality_anch"; do
  TAG=${CFG%%:*}; FF=${CFG#*:}
  if [ -f "data/interim/frontier_s1_$TAG.npz" ] && grep -q "saved blends" "logs/agent_frontier_s1_$TAG.txt" 2>/dev/null; then say "$TAG reused"; continue; fi
  say "$TAG (--floor-stack-feats $FF): start"
  $PY archive/capped_run.py --cap-gb 5 -- $PY -u -W ignore archive/agent_frontier_stage1_eval.py $SHIP --floor-stack-feats "$FF" \
      --preds "$PREDS" --days "$DAYS" --report-days "$DAYS" --save-blends "data/interim/frontier_s1_$TAG.npz" > "logs/agent_frontier_s1_$TAG.txt" 2>&1
  grep -q "saved blends" "logs/agent_frontier_s1_$TAG.txt" || { say "STOP: arm $TAG failed: $(tail -3 logs/agent_frontier_s1_$TAG.txt | tr '\n' ' ')"; exit 1; }
  say "$TAG: $(grep 'all departures' logs/agent_frontier_s1_$TAG.txt | tail -1 | sed 's/^ *//')"
done
$PY archive/capped_run.py --cap-gb 1 -- $PY archive/agent_frontier_04_judge.py 2>&1 | tee -a "$LOG"
say "frontier stage-1 check finished"
