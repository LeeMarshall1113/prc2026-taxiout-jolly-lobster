#!/usr/bin/env bash
# v92 MODEL = v91 (the lateness model) with the seed counts scaled up, because bagging just paid 3x
# its estimate: bulk seeds 3 -> 5, no-plan seeds 5 -> 20. Same flags, same KEEP (v90's keep list plus
# the seven late_* columns). MODEL ONLY -- the bagged chain runs on it afterwards. No backslashes.
set -uo pipefail
REPO="${PRC_ROOT:-$PWD}"; cd "$REPO"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$REPO"
LOG="$REPO/logs/build_v92model.log"
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
WAVE2="dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min"
WX="wx_temp_c,wx_dewpoint_c,wx_wind_kt,wx_gust_kt,wx_visibility_km,wx_ceiling_ft,wx_precip,wx_heavy_precip,wx_deice_risk,wx_thunder,wx_snow,wx_freezing,wx_lowvis,wx_age_min"
NM=$($PY -c "from prc.features import NM_NEIGHBOUR_FEATURES as N; print(','.join(N))")
LATE=$($PY -c "from prc.features import LATE_NEIGHBOUR_FEATURES as L; print(','.join(L))")
KEEP="$WAVE2,$WX,callsign_op,$NM,$LATE"
COMMON="--noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF --regime-correction 1.0 --drop-extra doy,month --blend-cat-depth 10 --blend-lightgbm 0.70 --bulk-plan-only --residual --baseline blend_apt --bulk-routes ALL"
[ -f submissions/jolly-lobster_v92.parquet ] && { say "v92 exists; refusing to overwrite"; exit 1; }
say "=== smoke ==="
rm -f submissions/jolly-lobster_v999.parquet
$PY -u -m prc.final --seeds 1 --iterations 30 --threads 6 --noplan-seeds 1 --noplan-iterations 30 $COMMON --keep "$KEEP" --team jolly-lobster --version 999 > logs/v92_smoke.log 2>&1 || { say "SMOKE FAILED"; exit 1; }
grep -qE "\(80 features" logs/v92_smoke.log || { say "SMOKE: not 80 features"; exit 1; }
rm -f submissions/jolly-lobster_v999.parquet; say "smoke passed"
say "=== full model: seeds 5, noplan-seeds 20, iterations 2500 -> v92 ==="
$PY -u -m prc.final --seeds 5 --iterations 2500 --threads 6 --noplan-seeds 20 $COMMON --keep "$KEEP" --team jolly-lobster --version 92 >> "$LOG" 2>&1
say "=== model exit $? ==="
[ -f submissions/jolly-lobster_v92.parquet ] && say "V92 MODEL BUILT" || say "V92 MODEL FAILED"
