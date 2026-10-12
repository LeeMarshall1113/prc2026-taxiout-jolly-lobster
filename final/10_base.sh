#!/usr/bin/env bash
# Final phase, J2: refit the base model (the v92 recipe plus the v94 member change) and serve all four months.
#
# Derived from archive/build_v92model.sh (flags, seeds, KEEP list) and archive/fit_members_full.sh + auto_v94.sh
# (members and member_delta). The Python is final/10_base.py, a parameterised copy of prc/final.py main().
#
# NOT STARTED by anyone yet. Launch from Git Bash, in any directory:
#     bash final/10_base.sh
# That requests the slot below, then detaches (no window) and returns at once. Log: logs/final_10_base.log.
# Check progress with: tail -f logs/final_10_base.log      Cancel/inspect: python archive/job_slots.py status
#
# Steps (each is skipped when its output already exists, so running the script again resumes):
#   1  smoke      30 iterations, 1 seed             -> data/final/smoke_base.parquet            (about 3 minutes)
#   2  refit      5 bulk seeds x 2500, depth-10 member, LightGBM member, 20 no-plan seeds, regime correction
#                 -> data/final/J2_base.parquet (all months in template order)                  (about 11-12 h)
#   3  members    LightGBM members lgb, lgb_routed, lgb_extratrees -> data/final/J2_members.npz (about 1 h)
#   4  delta      archive/member_delta.py, mode set:lgb_routed,lgb_extratrees -> data/final/J2_v94model.parquet (seconds)
# Resume inside step 2: every fitted blend member is saved to data/final/J2_members/ as it finishes (one CatBoost seed is
# about 1.5-2 h with the per-airport models), and a restart reuses them. Step 3 keeps members already in the npz.
# Leftover from a crash that is NOT resumable: the no-plan model (20 seeds, small) and the post-processing are redone.
set -uo pipefail

REPO="${PRC_ROOT:-$PWD}"
V="$REPO/.venv/Scripts/python.exe"
LOG="$REPO/logs/final_10_base.log"

if [ "${1:-}" != "--inner" ]; then
  export PY=/c/Python314/python.exe      # run_job needs an absolute interpreter that can see AppData
  cd "$REPO" || exit 1
  mkdir -p logs data/final
  exec archive/run_job.sh --detach --project prc-taxiout-final-base \
    --min-gb 9 --want-gb 10 --minutes 1000 --eta-minutes 810 --preemptible \
    --why "PRC final phase J2: refit the base model (5 CatBoost seeds, depth-10, LightGBM, 20 no-plan seeds) and serve Jan/Feb/Jun/Jul 2026" \
    --argument "experiment run" \
    --match "prc-taxiout-2026.*final.10_base" -- bash "$REPO/final/10_base.sh" --inner
fi

cd "$REPO" || exit 1
mkdir -p logs data/final
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$REPO"
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# Same feature switches as the v92 build.
WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
WX="wx_temp_c,wx_dewpoint_c,wx_wind_kt,wx_gust_kt,wx_visibility_km,wx_ceiling_ft,wx_precip,wx_heavy_precip,wx_deice_risk,wx_thunder,wx_snow,wx_freezing,wx_lowvis,wx_age_min"
NM=$($V -c "from prc.features import NM_NEIGHBOUR_FEATURES as N; print(','.join(N))")
LATE=$($V -c "from prc.features import LATE_NEIGHBOUR_FEATURES as L; print(','.join(L))")
KEEP="$WAVE2,$WX,callsign_op,$NM,$LATE"
COMMON="--noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF --regime-correction 1.0 --drop-extra doy,month --blend-cat-depth 10 --blend-lightgbm 0.70 --bulk-plan-only --residual --baseline blend_apt --bulk-routes ALL"
SERVE="--ranking data/raw/final_ranking.parquet --template data/raw/final_submitting.parquet --months 1,2,6,7 --out-dir data/final"
# 3 GB cap in the repo's usual wrapper is for scripts; the real fit gets a cap above the slot so a leak fails here, not on the machine.
CAP="$V archive/capped_run.py --cap-gb 18 --"

if [ ! -f data/final/smoke_base.parquet ]; then
  say "=== smoke: 30 iterations ==="
  $CAP $V -u final/10_base.py --seeds 1 --iterations 30 --threads 6 --noplan-seeds 1 --noplan-iterations 30 $COMMON --keep "$KEEP" $SERVE --tag smoke > logs/final_10_base_smoke.log 2>&1 \
    || { say "SMOKE FAILED"; tail -20 logs/final_10_base_smoke.log | tee -a "$LOG"; exit 1; }
  grep -qE "\(80 features" logs/final_10_base_smoke.log || { say "SMOKE: not 80 features"; exit 1; }
  say "smoke passed"
fi

if [ ! -f data/final/J2_base.parquet ]; then
  say "=== refit: seeds 5, noplan-seeds 20, iterations 2500 ==="
  $CAP $V -u final/10_base.py --seeds 5 --iterations 2500 --threads 6 --noplan-seeds 20 $COMMON --keep "$KEEP" $SERVE --tag J2 >> "$LOG" 2>&1
  say "=== refit exit $? ==="
fi
[ -f data/final/J2_base.parquet ] || { say "J2 BASE MISSING -- rerun this script to resume from the cached members"; exit 1; }

say "=== members: lgb, lgb_routed, lgb_extratrees ==="
$CAP $V -u final/10_base.py --seeds 1 --iterations 2500 --threads 6 $COMMON --keep "$KEEP" $SERVE --tag J2 \
  --fit-members lgb,lgb_routed,lgb_extratrees --members-out data/final/J2_members.npz --cache-dir data/final/J2_fitmembers_cache >> "$LOG" 2>&1
grep -q "FIT-MEMBERS DONE" "$LOG" || { say "MEMBERS FAILED -- rerun to resume"; exit 1; }

say "=== member delta: set:lgb_routed,lgb_extratrees ==="
$V archive/member_delta.py --base data/final/J2_base.parquet --members data/final/J2_members.npz \
  --mode "set:lgb_routed,lgb_extratrees" --out data/final/J2_v94model.parquet 2>&1 | tee -a "$LOG"
[ -f data/final/J2_v94model.parquet ] && say "J2 DONE: data/final/J2_v94model.parquet" || { say "member_delta failed"; exit 1; }
