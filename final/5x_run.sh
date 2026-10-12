#!/usr/bin/env bash
export SLOT_WAIT=${SLOT_WAIT:-28800}   # run_job gives up after 2 h in the queue by default; wait up to 8 h
# Run one final-phase job of the corrector and Istanbul stages (experiment 5x) under a 3 GB scheduler slot, the cross-experiment job lock and a 3 GB
# commit cap. usage: bash final/5x_run.sh "<one line why>" <python script relative to repo> [args...]
# The job lock is the directory data/final/.joblock (one job over 1 GB at a time across the final-phase scripts); it is
# taken only after the slot is granted, so waiting for the slot does not block anyone.
REPO=.
LOCK=$REPO/data/final/.joblock
WHY="$1"; shift
cd "$REPO" || exit 1
mkdir -p data/final logs
if [ "${INNER:-0}" != "1" ]; then
  export INNER=1 PY=/c/Python314/python.exe
  exec archive/run_job.sh --project prc-taxiout-final --min-gb ${SLOT_GB:-3} --want-gb ${SLOT_GB:-3} --minutes ${SLOT_MIN:-40} --eta-minutes ${SLOT_ETA:-10} \
    --why "PRC final: $WHY" --match "prc-taxiout-2026" -- bash "$REPO/final/5x_run.sh" "$WHY" "$@"
fi
# take the lock (retry each minute; a lock older than 40 minutes with no repo python running may be removed)
while ! mkdir "$LOCK" 2>/dev/null; do
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +40 2>/dev/null)" ] && ! tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -q python; then
    rm -r "$LOCK"; continue
  fi
  echo "[$(date -u +%H:%M:%S)] job lock held by $(cat "$LOCK"/* 2>/dev/null); waiting"; sleep 60
done
echo "5x-assemble $$" > "$LOCK/owner"
trap 'rm -r "$LOCK"' EXIT
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$REPO" OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
V=./.venv/Scripts/python.exe
$V $REPO/archive/capped_run.py --cap-gb ${SLOT_GB:-3} -- $V "$@"
