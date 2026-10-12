#!/usr/bin/env bash
# experiment frontier: FULL-DAY CONFIRMATION of build B1 (floor-stack anchors, LIRF excluded from the start). docs/LAB-NOTES.md.
#
# DESIGN (fixed before launch, 2026-09-23 ~09:45Z):
#   evaluator  archive/agent_frontier_stage1_eval.py (adsbq stage-1 copy + --floor-stack-feats quality_anch + --floor-stack-anch-skip).
#   days       v98's 264-day calibration set (data/interim/calib_days_264_octnov.txt): 177 Jan-Mar/Jul-Sep + 61 Oct-Nov days carry
#              fold predictions and are scored; the 26 April days have none (as in serving, where v98's --calib-preds holds no April
#              file), so they contribute arrival stand samples only. Leave-one-day-out: each scored day is calibrated on the other 263.
#   preds      v96/v98 calib-preds list: n_feat_1_7, n_feat_2_8, n3_bulk_3_9, calib2026/preds_oct_f4, calib2026/preds_nov_c.
#   shared     --stack gbm --stack-feats quality_anch --stack-skip-airports LIRF --floor on --floor-stack on --floor-stack-snapshot on
#              --floor-q zero --floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG --osm-fill on --spill-tracks on
#              --blends both_guard   (= what v98 serves)
#   arm Q      --floor-stack-feats quality                                      (v98)
#   arm F      --floor-stack-feats quality_anch --floor-stack-anch-skip LIRF    (B1 exactly as it would ship)
#
# PASS RULE, PRE-REGISTERED (the team's four clauses; judged by archive/agent_frontier_07_judge.py), d = RMSE(F) - RMSE(Q):
#   1. all scored rows: d <= -0.10 s
#   2. no airport worse than +0.10 s
#   3. d < 0 after deleting the 50 rows F helps most
#   4. d < 0 on the Jan-Mar days AND on the Jul-Sep days
#   Reported, not gating: Oct-Nov d, core, per airport, per month, days better.
#
# MEMORY: the 60-day arms peaked at 1.86 GB commit; linear in days that is ~5.5 GB at 264 days, so each arm runs under a hard
# 7 GB capped_run cap inside a 6-7 GB slot. Arms are sequential (~25-35 min each). Resumable. No backslashes.
set -uo pipefail
cd "${PRC_ROOT:-$PWD}"
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 RAYON_NUM_THREADS=4
LOG=logs/agent_frontier_confirm.log
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
DAYS=$(cat data/interim/calib_days_264_octnov.txt)
N=$(echo "$DAYS" | tr ',' '\n' | grep -c .)
[ "$N" -eq 264 ] || { say "STOP: expected 264 days, found $N"; exit 1; }
REP=$(echo "$DAYS" | tr ',' '\n' | grep -v "^2025-04" | paste -sd, -)
PREDS="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,data/interim/calib2026/preds_oct_f4.parquet,data/interim/calib2026/preds_nov_c.parquet"
SHIP="--stack gbm --stack-feats quality_anch --stack-skip-airports LIRF --floor on --floor-stack on --floor-stack-snapshot on --floor-q zero --floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG --osm-fill on --spill-tracks on --blends both_guard"
say "full-day confirmation of B1 on $N days ($(echo "$REP" | tr ',' '\n' | grep -c .) scored)"
for CFG in "Q:--floor-stack-feats quality" "F:--floor-stack-feats quality_anch --floor-stack-anch-skip LIRF"; do
  TAG=${CFG%%:*}; X=${CFG#*:}
  if [ -f "data/interim/frontier_c238_$TAG.npz" ] && grep -q "saved blends" "logs/agent_frontier_c238_$TAG.txt" 2>/dev/null; then say "$TAG reused"; continue; fi
  say "$TAG ($X): start"
  $PY archive/capped_run.py --cap-gb 7 -- $PY -u -W ignore archive/agent_frontier_stage1_eval.py $SHIP $X \
      --preds "$PREDS" --days "$DAYS" --report-days "$REP" --save-blends "data/interim/frontier_c238_$TAG.npz" > "logs/agent_frontier_c238_$TAG.txt" 2>&1
  grep -q "saved blends" "logs/agent_frontier_c238_$TAG.txt" || { say "STOP: arm $TAG failed: $(tail -3 logs/agent_frontier_c238_$TAG.txt | tr '\n' ' ')"; exit 1; }
  say "$TAG: $(grep 'all departures' logs/agent_frontier_c238_$TAG.txt | tail -1 | sed 's/^ *//'); $(tail -1 logs/agent_frontier_c238_$TAG.txt)"
done
$PY archive/capped_run.py --cap-gb 1.5 -- $PY archive/agent_frontier_07_judge.py 2>&1 | tee -a "$LOG"
say "full-day confirmation finished"
