#!/usr/bin/env bash
# Generic bagged chain (the v51 recipe) for any model output:
#   bash archive/bag_chain.sh <model.parquet> <version> <prior_version> [upload]
#   non-LIRF: shipped gate -> ADS-B bagged over 10 realisations (3 in parallel) -> downstream
#   LIRF:     gate bagged over 20 seeds at the v47 retune -> ADS-B -> downstream
# Strict guards vs the prior best; uploads ONLY if the 4th argument is "upload" AND every guard passes.
# No backslashes.
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
MODEL="$1"; V="$2"; PRIOR="$3"; UP="${4:-}"
export OUTF="submissions/jolly-lobster_v${V}.parquet" PRIORF="submissions/jolly-lobster_v${PRIOR}.parquet" P="data/interim/v${V}" V PRIOR
[ -f "$MODEL" ] || { say "no model $MODEL -- stopping"; exit 1; }
[ -f "$PRIORF" ] || { say "no prior $PRIORF -- stopping"; exit 1; }
[ -f "$OUTF" ] && { say "$OUTF exists -- stopping"; exit 1; }
CD="$(cat data/interim/calib_days_v41_203.txt)"
FL="--osm-fill on --floor-stack-feats quality --stack-feats quality --floor-q zero --spill-tracks on --calib-days $CD --floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG"
avg () { $PY - "$@" <<'PYEOF'
import sys, polars as pl, numpy as np
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; out, ins = sys.argv[1], sys.argv[2:]
fr = [pl.read_parquet(p).sort(ID) for p in ins]
for f in fr[1:]: assert (f[ID] == fr[0][ID]).all()
M = np.vstack([f[T].to_numpy().astype(float) for f in fr])
fr[0].with_columns(pl.Series(T, np.rint(M.mean(0))).cast(fr[0][T].dtype)).write_parquet(out)
print(f"averaged {len(ins)} -> {out}")
PYEOF
}
chain () {
  $PY archive/apply_gap_clamp.py --in "$1" --out "$2_s2.parquet" >/dev/null 2>&1
  $PY archive/apply_lobt_box.py --hard-binding-blend 0.5 --in "$2_s2.parquet" --out "$2_s3.parquet" >/dev/null 2>&1
  $PY -u archive/apply_min_floor.py --in "$2_s3.parquet" --out "$2_s35.parquet" >/dev/null 2>&1
  $PY -u archive/apply_v35_rules.py --in "$2_s35.parquet" --out "$2_s4.parquet" >/dev/null 2>&1
  $PY -u archive/apply_upper_bound.py --in "$2_s4.parquet" --out "$2_s5.parquet" >/dev/null 2>&1
  $PY -u archive/apply_hidden_reading.py --in "$2_s5.parquet" --out "$2_s6.parquet" >/dev/null 2>&1
  $PY -u archive/apply_eddm_january.py --in "$2_s6.parquet" --out "$2_final.parquet" >/dev/null 2>&1
}
say "v$V: base = day-early rule on $MODEL"
$PY archive/apply_rome_dayearly.py --in "$MODEL" --out ${P}_base.parquet >/dev/null 2>&1 || { say "dayearly failed"; exit 1; }
say "non-LIRF path: shipped gate, then 10 ADS-B realisations, 3 at a time"
$PY -u archive/apply_rome_plan_gate.py --in ${P}_base.parquet --out ${P}_s0ship.parquet >/dev/null 2>&1
for batch in "1 2 3" "4 5 6" "7 8 9" "10"; do
  for k in $batch; do $PY -u archive/adsb_apply.py --in ${P}_s0ship.parquet --out ${P}_adsb_$k.parquet $FL >/dev/null 2>&1 & done
  wait; say "ADS-B batch [$batch] done"
done
avg ${P}_s1.parquet $(for k in $(seq 1 10); do echo ${P}_adsb_$k.parquet; done)
chain ${P}_s1.parquet ${P}_nl
say "LIRF path: gate bagged over 20 seeds at the retune"
for sd in $(seq 1 20); do $PY -u archive/apply_rome_plan_gate.py --gap-cap 10800 --blend-weight 0.60 --seed $sd --in ${P}_base.parquet --out ${P}_gate_$sd.parquet >/dev/null 2>&1; done
avg ${P}_gatebag_s0.parquet $(for sd in $(seq 1 20); do echo ${P}_gate_$sd.parquet; done)
$PY -u archive/adsb_apply.py --in ${P}_gatebag_s0.parquet --out ${P}_gatebag_s1.parquet $FL >/dev/null 2>&1
chain ${P}_gatebag_s1.parquet ${P}_li
say "splice + guards"
$PY - <<'PYEOF'
import os, sys, datetime as dt, polars as pl, numpy as np
from prc.config import RAW_DIR
from prc.leaderboard import fetch_all
C = pl.col; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; P, OUTF, PRIORF = os.environ["P"], os.environ["OUTF"], os.environ["PRIORF"]
nl = pl.read_parquet(f"{P}_nl_final.parquet")
li = pl.read_parquet(f"{P}_li_final.parquet").select(ID, C(T).alias("_l"))
apt = pl.scan_parquet(RAW_DIR / "ranking.parquet").select(C(ID), C("ADEP_mvt")).collect()
j = nl.join(li, on=ID, how="left", maintain_order="left").join(apt, on=ID, how="left", maintain_order="left")
out = j.with_columns(pl.when(C("ADEP_mvt") == "LIRF").then(C("_l")).otherwise(C(T)).cast(nl[T].dtype).alias(T)).select(nl.columns)
out.write_parquet(OUTF)
pr = pl.read_parquet(PRIORF).sort(ID); o = out.sort(ID)
a, b = pr[T].to_numpy().astype(float), o[T].to_numpy().astype(float)
corr, shift, neg = float(np.corrcoef(a, b)[0, 1]), float((b - a).mean()), int((b < 0).sum())
today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
used = len([s for s in fetch_all() if s.team_name == "jolly-lobster" and str(s.processed_at)[:10] == today])
g = {"rows == 344841": out.height == 344841, "corr vs prior > 0.99": corr > 0.99,
     "|mean shift| < 5s": abs(shift) < 5, "no negatives": neg == 0, "slots used today < 5": used < 5}
print(f"{OUTF} vs {PRIORF}: corr {corr:.4f}, mean shift {shift:+.2f}s, RMS {np.sqrt(((b-a)**2).mean()):.1f}s, negatives {neg}; slots used today {used}")
for k, v in g.items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if all(g.values()) else 3)
PYEOF
G=$?
$PY archive/check_submission.py ${P}_base.parquet "$OUTF" > logs/v${V}_check.txt 2>&1
grep -q "template checks pass" logs/v${V}_check.txt || G=4
[ "$G" = "0" ] || { say "GUARDS FAILED (code $G) -- v$V built but NOT uploaded"; exit 0; }
[ "$UP" = "upload" ] || { say "all guards pass -- v$V BUILT, not uploaded (no 'upload' argument)"; exit 0; }
say "all guards pass -- uploading v$V"
$PY archive/ship_helpers.py upload "$OUTF" 2>&1 | grep -viE "warning|deprecat" | tail -1
say "V$V PIPELINE DONE"
