#!/usr/bin/env bash
# Chain of the ADS-B stage jobs for February and June, run unattended (start with final/2x_chain_start.ps1 or `bash final/2x_chain.sh`).
#  1. wait until data/final/J2_v94model.parquet exists and holds every template departure;
#  2. Rome pre-rules (stage 21, step 1) under a 3 GB slot;
#  3. wait until CHAIN_BIG_AFTER (default Fri 2026-10-09 07:30 local), then stage 21 (ADS-B realisations, 3 workers) under an 11 GB slot
#     (a serving process needs a little over 3 GB: a 3 GB cap stops it), then stage 22 under a 3 GB slot.
# Every step skips outputs that exist, so a restart resumes. Safe to start twice (directory lock data/final/.chain2x). Each slot is
# released by final/2x_run.sh / run_job.sh on exit. Log: logs/final_2x_chain.log.
REPO=.
cd "$REPO" || exit 1
mkdir -p logs data/final
LOG=logs/final_2x_chain.log
say () { echo "[$(date '+%a %H:%M:%S')] $*" >> "$LOG"; }
mkdir data/final/.chain2x 2>/dev/null || { if [ -n "$(find data/final/.chain2x -maxdepth 0 -mmin +1440 2>/dev/null)" ]; then rm -r data/final/.chain2x; mkdir data/final/.chain2x; else echo "already running" >> "$LOG"; exit 0; fi; }
trap 'rm -r data/final/.chain2x' EXIT
V=./.venv/Scripts/python.exe
BIG_AFTER=$(date -d "${CHAIN_BIG_AFTER:-2026-10-09 07:30}" +%s)
say "chain started (pid $$)"
until $V -c "
import polars as pl, sys
t=pl.read_parquet('data/raw/final_submitting.parquet',columns=['MVT_ID_mvt'])
b=pl.read_parquet('data/final/J2_v94model.parquet',columns=['MVT_ID_mvt'])
sys.exit(0 if b.height==t.height and b['MVT_ID_mvt'].n_unique()==t.height else 1)" 2>/dev/null; do sleep 300; done
say "J2_v94model.parquet complete"
bash final/2x_run.sh "ADS-B stage pre-rules for Feb/Jun" final/21_adsb_stage.py --months 2,6 --prerules-only >> "$LOG" 2>&1 || { say "pre-rules failed"; exit 1; }
say "pre-rules done; waiting for the big-slot window (or for the cloud outputs if data/final/.cloud21 exists)"
stage_outputs_exist () {
  for s in 103 105 105sc; do
    [ -s data/final/J5_v${s}_gatebag_s1.parquet ] || return 1
    for k in $(seq 1 10); do [ -s data/final/J5_v${s}_adsb_$k.parquet ] || return 1; done
  done
  return 0
}
# The operator starts the Modal run (modal/final_adsb_stage_app.py) and creates data/final/.cloud21 after fetching or while waiting:
# then this chain does not serve the stage itself, it waits for the 33 outputs and only averages them (stage 21 skips what exists).
while [ "$(date +%s)" -lt "$BIG_AFTER" ] && [ ! -f data/final/.cloud21 ] && ! stage_outputs_exist; do sleep 300; done
if stage_outputs_exist; then
  say "all stage outputs present: averaging only"
  bash final/2x_run.sh "ADS-B stage averaging for Feb/Jun" final/21_adsb_stage.py --months 2,6 >> "$LOG" 2>&1 || { say "stage 21 (average) failed"; exit 1; }
elif [ -f data/final/.cloud21 ]; then
  say "cloud run flagged (.cloud21): waiting for the outputs"
  until stage_outputs_exist; do sleep 300; done
  bash final/2x_run.sh "ADS-B stage averaging for Feb/Jun" final/21_adsb_stage.py --months 2,6 >> "$LOG" 2>&1 || { say "stage 21 (average) failed"; exit 1; }
else
  n=$(ls data/adsb/points | grep -c "^2026-0[26]")
  say "points day files for Feb/Jun: $n"
  SLOT_GB=11 SLOT_MIN=420 SLOT_ETA=240 bash final/2x_run.sh "ADS-B stage realisations for Feb/Jun, 3 workers" final/21_adsb_stage.py --months 2,6 --workers 3 >> "$LOG" 2>&1 || { say "stage 21 failed (re-run resumes)"; exit 1; }
fi
say "stage 21 done"
SLOT_MIN=60 SLOT_ETA=40 bash final/2x_run.sh "re-stack for Feb/Jun" final/22_restack.py >> "$LOG" 2>&1 || { say "stage 22 failed"; exit 1; }
say "stage 22 done"
