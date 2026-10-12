#!/usr/bin/env bash
# Final phase, J1: stream-extract the ADS-B days of February and June 2026 (58 days) from adsb.lol.
#
# Derived from archive/run_adsb_stage2.sh (January and July, 62 days). Same extractor (python -m prc.adsb extract),
# same filters, same output folder (data/adsb/points, one parquet per UTC day). Differences:
#   * the days are 2026-02-01..28 and 2026-06-01..30 (set MONTHS below to change them);
#   * the slot is requested here through run_job.sh (the original took a slot id made by the caller);
#   * after the first pass, days that failed or are still missing are tried once more, then the script reports them.
#
# NOT STARTED by anyone yet. Launch (from Git Bash, in any directory):
#     bash final/05_adsb_download.sh
# That requests a 3 GB slot (network + 3 worker processes, about 1 GB each; stage 2 did not log its RAM) for up to
# 180 minutes (estimate 110: 153 GB streamed at 7-19 MB/s per worker, 3 workers), then detaches and returns at once.
# Log: logs/adsb_final_download.log (and the run_job log under logs/detached/).
#
# Resumable: a finished day is a parquet file, written under a temporary name and renamed, so an interrupted run leaves
# no half day; the extractor skips days already on disk. Run the script again to continue.
# Preemptible: when the scheduler asks the slot to yield, the watcher below touches data/adsb/STOP, no new day starts, the
# days in flight finish, and the script exits; start it again later to continue. No backslashes in this file.
set -uo pipefail

REPO="${PRC_ROOT:-$PWD}"
V="$REPO/.venv/Scripts/python.exe"
# (year, month) pairs to fetch; day count comes from the calendar.
MONTHS="2026-2 2026-6"
LOG="$REPO/logs/adsb_final_download.log"

if [ "${1:-}" != "--inner" ]; then
  export PY=/c/Python314/python.exe      # run_job itself needs an absolute interpreter that sees AppData
  cd "$REPO" || exit 1
  exec archive/run_job.sh --detach --project prc-taxiout-adsb-final \
    --min-gb 3 --want-gb 3 --minutes 180 --eta-minutes 110 --preemptible \
    --why "PRC final phase: stream 58 ADS-B day files (Feb and Jun 2026) from adsb.lol, network bound" \
    --argument "experiment run" \
    --match "prc-taxiout-2026.[.]venv.*(prc[.]adsb|multiprocessing)" -- bash "$REPO/final/05_adsb_download.sh" --inner
fi

cd "$REPO" || exit 1
mkdir -p logs data/adsb/points
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$REPO"

DAYS=$($V -c "
import calendar, datetime as d
out = []
for ym in '$MONTHS'.split():
    y, m = map(int, ym.split('-'))
    out += [str(d.date(y, m, k)) for k in range(1, calendar.monthrange(y, m)[1] + 1)]
print(','.join(out))")
N=$(echo "$DAYS" | tr "," " " | wc -w)
[ "$N" = "58" ] || { echo "expected 58 days, got $N -- refusing to run" | tee -a "$LOG"; exit 1; }

# Turn a scheduler yield request (a file named by run_job) into the extractor's stop file.
rm -f data/adsb/STOP
( while true; do sleep 30; [ -n "${JOB_PREEMPT_FILE:-}" ] && [ -f "$JOB_PREEMPT_FILE" ] && touch data/adsb/STOP; done ) &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

missing () { for d in $(echo "$DAYS" | tr "," " "); do [ -f "data/adsb/points/$d.parquet" ] || echo -n "$d,"; done; }

echo "=== adsb final download start $(date -u +%H:%M:%SZ): $N days ===" | tee -a "$LOG"
TODO="$DAYS"
for pass in 1 2; do
  $V -u -m prc.adsb extract --dates "$TODO" --workers 3 --out data/adsb/points --stop-file data/adsb/STOP 2>&1 | tee -a "$LOG"
  [ -f data/adsb/STOP ] && { echo "stop requested, exiting; run again to continue" | tee -a "$LOG"; break; }
  TODO=$(missing | sed 's/,$//')
  [ -z "$TODO" ] && break
  echo "pass $pass left missing: $TODO" | tee -a "$LOG"
done
LEFT=$(missing | sed 's/,$//')
echo "=== adsb final download exit $(date -u +%H:%M:%SZ): missing days: ${LEFT:-none} ===" | tee -a "$LOG"
echo "February/June 2026 day files on disk: $(ls data/adsb/points/2026-02-*.parquet data/adsb/points/2026-06-*.parquet 2>/dev/null | wc -l) of $N" | tee -a "$LOG"
[ -z "$LEFT" ]
