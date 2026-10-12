#!/usr/bin/env bash
# Fit the six candidate blend members (prc.members) on the FULL v92 training matrix and save their
# ranking predictions + regime masks to data/interim/members_full.npz, for archive/member_delta.py.
# v92's exact COMMON/KEEP. No CatBoost. Each member is saved as it finishes. No backslashes.
set -uo pipefail
REPO="${PRC_ROOT:-$PWD}"; cd "$REPO"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$REPO"
LOG="$REPO/logs/fit_members_full.log"
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
WX="wx_temp_c,wx_dewpoint_c,wx_wind_kt,wx_gust_kt,wx_visibility_km,wx_ceiling_ft,wx_precip,wx_heavy_precip,wx_deice_risk,wx_thunder,wx_snow,wx_freezing,wx_lowvis,wx_age_min"
NM=$($PY -c "from prc.features import NM_NEIGHBOUR_FEATURES as N; print(','.join(N))")
LATE=$($PY -c "from prc.features import LATE_NEIGHBOUR_FEATURES as L; print(','.join(L))")
KEEP="$WAVE2,$WX,callsign_op,$NM,$LATE"
COMMON="--noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF --regime-correction 1.0 --drop-extra doy,month --blend-cat-depth 10 --blend-lightgbm 0.70 --bulk-plan-only --residual --baseline blend_apt --bulk-routes ALL"
say "=== smoke (30 iterations, 3 kinds) ==="
$PY -u -m prc.final --seeds 1 --iterations 30 --threads 6 $COMMON --keep "$KEEP" --fit-members lgb,lgb_routed,xgb_hist --members-out data/interim/members_smoke.npz > logs/fit_members_smoke.log 2>&1
grep -q "FIT-MEMBERS DONE" logs/fit_members_smoke.log && grep -qE "\(80 features" logs/fit_members_smoke.log || { say "SMOKE FAILED"; tail -20 logs/fit_members_smoke.log | tee -a "$LOG"; exit 1; }
grep -E "regime:|member " logs/fit_members_smoke.log | tee -a "$LOG"; say "smoke passed"
say "=== full: 2500 iterations, 6 kinds ==="
$PY -u -m prc.final --seeds 1 --iterations 2500 --threads 6 $COMMON --keep "$KEEP" --fit-members lgb,xgb_hist,lgb_routed,lgb_huber600,lgb_extratrees,lgb_leaves63 --members-out data/interim/members_full.npz >> "$LOG" 2>&1
say "=== exit $? ==="
grep -q "FIT-MEMBERS DONE" "$LOG" && say "MEMBERS FULL BUILT" || say "MEMBERS FULL FAILED"
