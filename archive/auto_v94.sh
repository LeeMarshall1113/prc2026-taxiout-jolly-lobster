#!/usr/bin/env bash
# Detached, slot-free waiter: lean screen fold 4 -> decide_v94 (rule fixed before fold 4) -> wait for the
# full-data member fits -> member_delta on v92 -> bag_chain v94 (own slot, guards, upload) -> report score.
# Gives up after 20h. No backslashes.
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
arms4 () { $PY -c "import polars as pl; print(len([c for c in pl.read_parquet('results/lean_members_f4.parquet').columns if c.startswith(('lgb_','xgb_'))]))" 2>/dev/null || echo 0; }
for i in $(seq 1 1200); do [ "$(arms4)" = "6" ] && break; sleep 60; done
[ "$(arms4)" = "6" ] || { say "lean screen fold 4 never completed -- stopping"; exit 1; }
say "lean screen complete -- deciding"
$PY archive/decide_v94.py > logs/decide_v94.log 2>&1; cat logs/decide_v94.log
MODE=$(tail -1 logs/decide_v94.log)
[ "$MODE" = "NONE" ] && { say "NO SET QUALIFIES ON BOTH FOLDS -- v94 not built"; exit 0; }
case "$MODE" in set:*) ;; *) say "decide failed: '$MODE'"; exit 1;; esac
say "chosen: $MODE -- waiting for full-data member fits"
for i in $(seq 1 1200); do
  grep -q "MEMBERS FULL BUILT" logs/fit_members_full.log 2>/dev/null && break
  grep -qE "MEMBERS FULL FAILED|SMOKE FAILED" logs/fit_members_full.log 2>/dev/null && { say "member fits FAILED -- stopping"; exit 1; }
  sleep 60
done
grep -q "MEMBERS FULL BUILT" logs/fit_members_full.log || { say "member fits never finished"; exit 1; }
$PY archive/member_delta.py --base submissions/jolly-lobster_v92.parquet --members data/interim/members_full.npz --mode "$MODE" --out data/interim/v94_model.parquet || { say "member_delta failed"; exit 1; }
say "v94 model built -- running the bagged chain under a slot"
archive/run_job.sh --project prc-taxiout --min-gb 9 --want-gb 10 --minutes 180 --eta-minutes 90 --why "PRC v94: new LightGBM members through the bagged chain (10 ADS-B runs), guards, upload" --match "bag_chain|adsb_apply|apply_rome|apply_gap_clamp|apply_lobt_box|apply_min_floor|apply_v35_rules|apply_upper_bound|apply_hidden_reading|apply_eddm_january" --argument "experiment run" -- bash archive/bag_chain.sh data/interim/v94_model.parquet 94 51 upload
for i in $(seq 1 40); do
  R=$($PY - <<'PYEOF' 2>/dev/null
from prc.leaderboard import fetch_all, team_bests
subs = fetch_all()
v = [s for s in subs if s.team_name == "jolly-lobster" and s.filename == "jolly-lobster_v94.parquet"]
if v:
    tb = team_bests(subs); rank = next(i for i, (t, _, _) in enumerate(tb, 1) if t == "jolly-lobster")
    print(f"SCORED v94 {v[0].score:.4f} (v51 239.3220, delta {v[0].score-239.3220:+.4f}s) rank {rank}")
    for i, (t, sc, c) in enumerate(tb[:4], 1): print(f"  {i}. {t:20} {sc:.4f}")
PYEOF
)
  if echo "$R" | grep -q SCORED; then echo "$R"; break; fi; sleep 45
done
say "AUTO V94 DONE"
