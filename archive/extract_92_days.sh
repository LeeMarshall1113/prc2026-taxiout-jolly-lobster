#!/usr/bin/env bash
# Extract the remaining 92 days of Jan-Mar and Jul-Sep 2025, taking the ADS-B calibration base from
# 89 to 181 days and completing the six months the fold predictions cover.
# RUN 2026-09-19: same source as before (github.com/adsblol/globe_history_2025, the
# organiser-cleared open archive), about 200 GB transferred (winter days ~1.7 GB read, summer
# ~2.7 GB), about two hours. The day list is data/interim/extract92_days.txt; 2025-01-18 is left out
# because it is not in the archive at all.
# RESUMABLE and PARALLEL, as the 60-day run ended up: one extractor call with every remaining day,
# --workers 3 being a pool over DAYS; each day's parquet is written only on completion and days
# already on disk are filtered out, so a relaunch after a teardown costs only the days in flight.
# No backslashes.
set -uo pipefail
REPO="${PRC_ROOT:-$PWD}"
cd "$REPO"
PY=./.venv/Scripts/python.exe
LOG="$REPO/logs/extract92.log"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
ALL=$(cat data/interim/extract92_days.txt)
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
