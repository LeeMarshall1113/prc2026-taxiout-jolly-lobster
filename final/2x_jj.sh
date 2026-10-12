#!/usr/bin/env bash
# January/July through stages 21 and 22 from the final file's take-off times (tags J5jj, J6jj, J7jj). Stage 20 is run before.
REPO=.
cd "$REPO" || exit 1
LOG=logs/final_2x_jj.log
say () { echo "[$(date '+%a %H:%M:%S')] $*" >> "$LOG"; }
cleanup () { rm -rf "$LOCALAPPDATA"/Temp/adsb_tracks_* "$LOCALAPPDATA"/Temp/adsb_apply_tracks_* 2>/dev/null; T="$(cygpath -u "${TEMP:-/tmp}" 2>/dev/null || echo /tmp)"; rm -rf "$T"/adsb_apply_tracks_* "$T"/adsb_tracks_* 2>/dev/null; }
say "stage 21 start"
SLOT_GB=9 SLOT_MIN=900 SLOT_ETA=720 bash final/2x_run.sh "ADS-B stage for January/July from the final file, 1 worker (2 workers exceeded 11 GB on the second stream)" final/21_adsb_stage.py --months 1,7 --tag J5jj --base data/final/J2_v94model.parquet --workers 1 >> "$LOG" 2>&1 || { say "stage 21 failed"; cleanup; exit 1; }
cleanup
say "stage 21 done"
SLOT_GB=6 SLOT_MIN=90 SLOT_ETA=40 bash final/2x_run.sh "re-stack for January/July from the final file" final/22_restack.py --tag J7jj --frame data/final/J5jj_frame.parquet --stage-tag J5jj_ --rows data/final/J6jj_rows.parquet --evt data/final/J6jj_evt.parquet --seq data/final/J6jj_seq.parquet --bx data/final/J6jj_bx.parquet >> "$LOG" 2>&1 || { say "stage 22 failed"; exit 1; }
say "stage 22 done"
