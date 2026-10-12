#!/usr/bin/env bash
# Waits for all 58 February/June day files (and for the download slot to end), then runs the real stage-20 dumps for months 2,6
# under a 3 GB slot and files the four-month coverage table in logs/final_progress.log. Safe to start twice. Log: logs/final_2x_dumps.log.
REPO=.
cd "$REPO" || exit 1
LOG=logs/final_2x_dumps.log
say () { echo "[$(date '+%a %H:%M:%S')] $*" >> "$LOG"; }
mkdir data/final/.dumps2x 2>/dev/null || { echo "already running" >> "$LOG"; exit 0; }
trap 'rm -r data/final/.dumps2x' EXIT
V=./.venv/Scripts/python.exe
say "waiting for 58 day files"
while true; do
  n=$(ls data/adsb/points | grep -c "^2026-0[26]")
  if [ "$n" -ge 58 ] && ! python archive/job_slots.py status 2>/dev/null | grep -q "prc-taxiout-adsb-final"; then break; fi
  sleep 120
done
say "all day files present"
bash final/2x_run.sh "real stage-20 dumps for Feb/Jun" final/20_adsb_dumps.py --months 2,6 >> "$LOG" 2>&1 || { say "dumps failed"; exit 1; }
say "dumps done"
PYTHONIOENCODING=utf-8 $V final/29_adsb_coverage.py > logs/final_2x_coverage.txt 2>&1
{ echo "progress 2x $(date +%H:%M): real J6 dumps for Feb/Jun written. Coverage (ground aircraft per movement, all four months):"; cat logs/final_2x_coverage.txt; } >> logs/final_progress.log
say "coverage filed"
