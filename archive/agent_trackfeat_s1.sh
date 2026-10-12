#!/usr/bin/env bash
# experiment trackfeat: 60-day screen (30 Oct-Nov + 30 Jan-Sep, same recipe as archive/agent_frontier_s1.sh) of NEW
# track-derived features (archive/agent_trackfeat_features.py) in the FLOOR stack, on top of v99/v100 production
# (quality_anch main+floor stack, LIRF anchor-skipped). docs/LAB-NOTES.md.
# Arms, same evaluator (archive/agent_trackfeat_stage1_eval.py), all identical except --floor-stack-track /
# --track-cols:
#   Q       production, no track cols
#   T_all   + all 8 TRACKFEAT_COLS
#   T_path  + tf_path_m, tf_avgspd_kt                          (path length / congestion)
#   T_stop  + tf_nstops, tf_stop_s, tf_maxstop_s                (stop count / total / max)
#   T_gap   + tf_movegap_s                                      (gap: last parked fix -> first moving fix)
#   T_hold  + tf_holdrun_s, tf_fmhold_s                          (late stop = runway-holding-point proxy)
# Judge: archive/agent_trackfeat_04_judge.py (rule fixed there, same 5 clauses as frontier's). Each arm capped at 3 GB
# (frontier's own 60-day arms peaked at 1.86 GB; scheduler capacity was tight when this ran, so this stays lower than
# frontier's 5 GB precedent). Resumable. No backslashes.
set -uo pipefail
cd "${PRC_ROOT:-$PWD}"
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 RAYON_NUM_THREADS=4
LOG=logs/agent_trackfeat_s1.log
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ON=$(ls data/adsb/points/2025-1[01]-*.parquet | sed 's/.*points\///;s/\.parquet//' | sed -n '2~2p' | paste -sd, -)
JS=$(cat data/interim/calib_days_v41_203.txt | tr , '\n' | grep -vE "^2025-04" | sed -n '1~6p' | head -30 | paste -sd, -)
DAYS="$JS,$ON"
N=$(echo "$DAYS" | tr ',' '\n' | grep -c .)
PREDS="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,data/interim/calib2026/preds_oct_f4.parquet,data/interim/calib2026/preds_nov_c.parquet"
SHIP="--stack gbm --stack-feats quality_anch --stack-skip-airports LIRF --floor on --floor-stack on --floor-stack-snapshot on --floor-q zero --floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG --osm-fill on --spill-tracks on --blends both_guard --floor-stack-feats quality_anch --floor-stack-anch-skip LIRF"
say "trackfeat 60-day screen on $N days"
for CFG in "Q:off:" "T_all:on:" "T_path:on:tf_path_m,tf_avgspd_kt" "T_stop:on:tf_nstops,tf_stop_s,tf_maxstop_s" "T_gap:on:tf_movegap_s" "T_hold:on:tf_holdrun_s,tf_fmhold_s"; do
  TAG=${CFG%%:*}; REST=${CFG#*:}; TRK=${REST%%:*}; COLS=${REST#*:}
  X="--floor-stack-track $TRK"
  [ -n "$COLS" ] && X="$X --track-cols $COLS"
  if [ -f "data/interim/trackfeat_s1_$TAG.npz" ] && grep -q "saved blends" "logs/agent_trackfeat_s1_$TAG.txt" 2>/dev/null; then say "$TAG reused"; continue; fi
  say "$TAG ($X): start"
  $PY archive/capped_run.py --cap-gb 3 -- $PY -u -W ignore archive/agent_trackfeat_stage1_eval.py $SHIP $X \
      --preds "$PREDS" --days "$DAYS" --report-days "$DAYS" --save-blends "data/interim/trackfeat_s1_$TAG.npz" > "logs/agent_trackfeat_s1_$TAG.txt" 2>&1
  grep -q "saved blends" "logs/agent_trackfeat_s1_$TAG.txt" || { say "STOP: arm $TAG failed: $(tail -3 logs/agent_trackfeat_s1_$TAG.txt | tr '\n' ' ')"; exit 1; }
  say "$TAG: $(grep 'all departures' logs/agent_trackfeat_s1_$TAG.txt | tail -1 | sed 's/^ *//')"
done
say "trackfeat 60-day screen finished"
