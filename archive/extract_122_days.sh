#!/usr/bin/env bash
# Extract April, May, October and November 2025: 122 days, taking the ADS-B calibration base from 177
# to 299 days. NOT June or December -- no fold file holds out those months, so their days would carry
# no out-of-fold model prediction and the evaluation would drop them silently. These four months DO
# have fold files (preds_s4_ctrl_4_10, preds_c_ctrl_5_11), but from different suite arms than the three
# the calibration uses today, so the day-curve test that follows must treat the mixed-arm baseline as
# part of what it is testing.
# RUN 2026-09-20: same source as before (github.com/adsblol/globe_history_2025, the
# organiser-cleared open archive, ODbL, declared): ~2 GB read per day and ~10 MB kept, so roughly
# 250 GB transferred over about three hours. The day list is data/interim/extract122_days.txt.
# RESUMABLE and PARALLEL: one extractor call with every remaining day, --workers 3 being a pool over
# DAYS; each day's parquet is written only on completion and days already on disk are filtered out, so
# a relaunch after a teardown costs only the days in flight. No backslashes.
set -uo pipefail
REPO="${PRC_ROOT:-$PWD}"
cd "$REPO"
PY=./.venv/Scripts/python.exe
LOG="$REPO/logs/extract122.log"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ALL=$(cat data/interim/extract122_days.txt)
BEFORE=$(ls data/adsb/points/2025-*.parquet | wc -l)
REMAINING=""
for D in $(echo "$ALL" | tr ',' ' '); do
  [ -f "data/adsb/points/$D.parquet" ] || REMAINING="${REMAINING:+$REMAINING,}$D"
done
NREM=$(echo "$REMAINING" | tr ',' '\n' | grep -c .)
say "extracting $NREM days; 2025 days on disk before: $BEFORE"
if [ "$NREM" -gt 0 ]; then
  $PY -u -m prc.adsb extract --dates "$REMAINING" --workers 3 --out data/adsb/points >> "$LOG" 2>&1
fi
FAILED=""
for D in $(echo "$ALL" | tr ',' ' '); do
  [ -f "data/adsb/points/$D.parquet" ] || FAILED="$FAILED $D"
done
AFTER=$(ls data/adsb/points/2025-*.parquet | wc -l)
say "extraction finished: $((AFTER - BEFORE)) new this run, missing:${FAILED:- none}"
say "2025 days now on disk: $AFTER"
