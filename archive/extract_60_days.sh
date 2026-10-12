#!/usr/bin/env bash
# Extract 60 more 2025 days of adsb.lol ground tracks, taking the calibration base from 30 to 90.
#
# RUN 2026-09-19: source github.com/adsblol/globe_history_2025 (the same open archive
# the existing 30 days came from, organiser-cleared), about 119 GB transferred, ~1.5 hours.
# Per day: a 2.65 GB release of which the 0.67 GB heatmap block is skipped with range requests, so
# ~1.98 GB is read at ~9.5 MB/s; nothing is stored but the 6.5 MB output parquet.
#
# WHY THESE DAYS. The learning curve (run_daycurve.sh, same ten test days throughout, the
# calibration pool growing) reads -9.28 / -9.92 / -10.22 / -10.37 / -10.68s at 10 / 15 / 20 / 25 /
# 30 days: decaying but still climbing at 30. The gain is roughly logarithmic, so 30 -> 90 should be
# worth about +0.6s in evaluation terms and -0.4 to -0.5s on the board at the ADS-B exchange rate.
# That projection is to be CHECKED by re-running the curve at 90 days, not assumed -- three
# projections today have been too high (notes sections 28 and 29).
#
# The days fill in the six months the folds and the scored set actually cover (Jan-Mar, Jul-Sep),
# ten per month, skipping the 30 already on disk. Seasonal match matters: the scored set is Jan and
# Jul 2026.
#
# RESUMABLE. Each day is extracted on its own and skipped if its parquet already exists, so a
# preempt or a crash costs at most one day's work. Run under a slot; heartbeats come from
# run_job.sh. No backslashes.
set -uo pipefail
REPO="${PRC_ROOT:-$PWD}"
cd "$REPO"
PY=./.venv/Scripts/python.exe
LOG="$REPO/logs/extract60.log"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

JAN="2025-01-02,2025-01-04,2025-01-08,2025-01-12,2025-01-18,2025-01-20,2025-01-22,2025-01-27,2025-01-28,2025-01-30"
FEB="2025-02-02,2025-02-06,2025-02-07,2025-02-11,2025-02-16,2025-02-21,2025-02-23,2025-02-24,2025-02-27,2025-02-28"
MAR="2025-03-01,2025-03-05,2025-03-10,2025-03-12,2025-03-17,2025-03-19,2025-03-24,2025-03-26,2025-03-27,2025-03-31"
JUL="2025-07-01,2025-07-03,2025-07-09,2025-07-11,2025-07-15,2025-07-19,2025-07-23,2025-07-25,2025-07-30,2025-07-31"
AUG="2025-08-01,2025-08-05,2025-08-10,2025-08-12,2025-08-17,2025-08-22,2025-08-24,2025-08-29,2025-08-30,2025-08-31"
SEP="2025-09-01,2025-09-03,2025-09-08,2025-09-13,2025-09-15,2025-09-19,2025-09-21,2025-09-26,2025-09-28,2025-09-29"
ALL="$JAN,$FEB,$MAR,$JUL,$AUG,$SEP"

TOTAL=$(echo "$ALL" | tr ',' '\n' | wc -l)
BEFORE=$(ls data/adsb/points/2025-*.parquet 2>/dev/null | wc -l)
# ONE CALL, ALL REMAINING DAYS. prc.adsb's --workers is a process pool over DAYS, so the first run,
# which passed one date per call, kept two of its three workers idle and ran strictly in sequence.
# The extractor still writes each day's parquet only when that day completes, so this stays
# resumable: a day already on disk is filtered out here, and a day killed mid-read leaves nothing.
# Relaunched 2026-09-19 after the first run died at 3 of 60 when its Windows session was torn down
# (fork failing with 0xC000026B, STATUS_DLL_INIT_FAILED) -- --detach survives a session
# exiting but not a logoff or reboot.
REMAINING=""
SKIPPED=0
for D in $(echo "$ALL" | tr ',' ' '); do
  if [ -f "data/adsb/points/$D.parquet" ]; then
    SKIPPED=$((SKIPPED + 1))
  else
    REMAINING="${REMAINING:+$REMAINING,}$D"
  fi
done
NREM=$(echo "$REMAINING" | tr ',' '\n' | grep -c .)
say "extracting $NREM of $TOTAL days ($SKIPPED already on disk); 2025 days on disk before: $BEFORE"
if [ "$NREM" -gt 0 ]; then
  $PY -u -m prc.adsb extract --dates "$REMAINING" --workers 3 --out data/adsb/points >> "$LOG" 2>&1
fi
FAILED=""
for D in $(echo "$ALL" | tr ',' ' '); do
  [ -f "data/adsb/points/$D.parquet" ] || FAILED="$FAILED $D"
done
AFTER=$(ls data/adsb/points/2025-*.parquet 2>/dev/null | wc -l)
say "extraction finished: $((AFTER - BEFORE)) new this run, missing:${FAILED:- none}"
say "2025 days now on disk: $AFTER"
