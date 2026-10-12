#!/usr/bin/env bash
# ADS-B Stage 2: stream-extract the 62 scored days (January and July 2026) from adsb.lol.
#
# NOT RUN YET. Stage 2 follows the Stage 1 leave-one-day-out numbers (~205 GB of bandwidth,
# <5 GB disk). Written ahead of them; launch it only with those numbers in hand.
#
# Same extractor and filters as Stage 1 (run_adsb_stage1.sh): the raw tar is never
# stored, only points inside the ten airport boxes on the ground or below 3,000 ft. Days
# already on disk are skipped, so a preempted or interrupted run resumes where it stopped.
# Order: January first, then July, each in date order, so a partial run still yields whole
# months for calibration checks.
#
# Preemptible: touching data/adsb/STOP stops new days from starting. No backslashes.
set -uo pipefail

REPO="${PRC_ROOT:-$PWD}"
SLOTS="archive/job_slots.py"
SLOT="${1:?usage: run_adsb_stage2.sh <slot-id>}"
LOG="$REPO/logs/adsb_stage2.log"

cd "$REPO"; mkdir -p logs data/adsb/points; PY=./.venv/Scripts/python.exe
DAYS=$($PY -c 'import datetime as d; print(",".join(str(d.date(2026, m, 1) + d.timedelta(days=i)) for m in (1, 7) for i in range(31)))')
N=$(echo "$DAYS" | tr "," " " | wc -w)
[ "$N" = "62" ] || { echo "expected 62 days, got $N -- refusing to run" | tee -a "$LOG"; exit 1; }

$PY "$SLOTS" wait "$SLOT" --timeout 86400 || { echo "[wait] not granted -- stopping" | tee -a "$LOG"; exit 1; }
rm -f data/adsb/STOP
$PY "$SLOTS" heartbeat "$SLOT" >/dev/null 2>&1
( while true; do sleep 240; $PY "$SLOTS" heartbeat "$SLOT" >/dev/null 2>&1 || { echo "[heartbeat] yield requested" >&2; touch data/adsb/STOP; }; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; $PY "$SLOTS" release "$SLOT" >/dev/null 2>&1' EXIT

export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
echo "=== adsb stage 2 start $(date -u +%H:%M:%SZ): $N days ===" | tee -a "$LOG"
$PY -u -m prc.adsb extract --dates "$DAYS" --workers 3 --out data/adsb/points --stop-file data/adsb/STOP 2>&1 | tee -a "$LOG"
echo "=== adsb stage 2 exit at $(date -u +%H:%M:%SZ) ===" | tee -a "$LOG"
echo "2026 days on disk: $(ls data/adsb/points/2026-*.parquet 2>/dev/null | wc -l) of 62" | tee -a "$LOG"

kill $HB 2>/dev/null
$PY "$SLOTS" release "$SLOT" >/dev/null 2>&1
echo "lease released at $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
