#!/usr/bin/env bash
# Ship v20 = v19 + the Rome extreme-gap band rule (archive/apply_rome_dayearly.py).
# Upload v20 was planned on 2026-09-16 (to be sent when ready).
#
# Waits for the v19 build to exit cleanly, then gates every step and stops at the first
# failure without uploading anything:
#   1. v19 passes the template checks, and is a plausible successor of v18
#   2. the Rome rule is applied, v20 passes the template checks, and only 1-10 rows moved
#   3. v20 -- and ONLY v20 -- is uploaded. v19 is never uploaded on its own: uploading both
#      would let the board isolate the handful of Rome rows the rule touches, which is
#      the row-level probing the organisers despise. The rule stands on 2025 evidence.
#   4. the board is polled until v20 is scored.
# No backslashes anywhere, on purpose (see run_suite.sh).
set -uo pipefail

REPO="${PRC_ROOT:-$PWD}"
cd "$REPO"; PY=./.venv/Scripts/python.exe
LOG="$REPO/logs/ship_v20.log"
V18=submissions/jolly-lobster_v18.parquet
V19=submissions/jolly-lobster_v19.parquet
V20=submissions/jolly-lobster_v20.parquet
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage

say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
stop () { say "STOPPED, NOTHING UPLOADED: $*"; exit 1; }

say "ship_v20 armed; waiting for the v19 build to finish"
for i in $(seq 1 300); do
  grep -q "=== v19 exit" logs/v19.log 2>/dev/null && break
  sleep 120
done
EXITLINE=$(grep "=== v19 exit" logs/v19.log 2>/dev/null | tail -1)
[ -n "$EXITLINE" ] || stop "v19 never reported an exit within 10 hours"
say "v19 finished: $EXITLINE"
case "$EXITLINE" in
  *"exit 0 at"*) ;;
  *) stop "v19 did not exit cleanly" ;;
esac
[ -f "$V19" ] || stop "$V19 does not exist"
[ -f "$V20" ] && stop "$V20 already exists; refusing to overwrite it"

say "gate 1: v19 template checks"
$PY archive/check_submission.py "$V18" "$V19" > logs/ship_v20_check_v19.txt 2>&1
cat logs/ship_v20_check_v19.txt >> "$LOG"
grep -q "template checks pass" logs/ship_v20_check_v19.txt || stop "v19 failed the template checks"
say "gate 1: v19 is a plausible successor of v18"
$PY archive/ship_helpers.py sanity "$V18" "$V19" 2>&1 | tee -a "$LOG" | grep -q "sanity PASS" || stop "v19 failed the sanity gate"

say "applying the Rome extreme-gap rule"
$PY archive/apply_rome_dayearly.py --in "$V19" --out "$V20" 2>&1 | grep -v -i warning | tee -a "$LOG"
[ -f "$V20" ] || stop "the Rome rule did not write $V20"

say "gate 2: v20 template checks and scope"
$PY archive/check_submission.py "$V19" "$V20" > logs/ship_v20_check_v20.txt 2>&1
cat logs/ship_v20_check_v20.txt >> "$LOG"
grep -q "template checks pass" logs/ship_v20_check_v20.txt || stop "v20 failed the template checks"
$PY archive/ship_helpers.py changed "$V19" "$V20" 1 10 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "v20 changed an unexpected number of rows"

say "gate 3: uploading v20"
$PY archive/ship_helpers.py upload "$V20" 2>&1 | grep -v -i warning | tee -a "$LOG"
grep -q "uploaded jolly-lobster_v20.parquet" "$LOG" || { say "UPLOAD DID NOT CONFIRM -- state unknown; check the team bucket before retrying"; exit 1; }

say "polling the board for the v20 score (up to 3 hours)"
$PY archive/ship_helpers.py poll jolly-lobster_v20.parquet 180 2>&1 | grep -v -i warning | tee -a "$LOG"
say "ship_v20 done"
