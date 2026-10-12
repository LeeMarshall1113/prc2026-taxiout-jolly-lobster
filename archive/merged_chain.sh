#!/usr/bin/env bash
# MERGED chain (2026-09-23, docs/LAB-NOTES.md): a copy of agent_frontier_chain.sh whose ADS-B step
# runs archive/agent_merged_apply.py (frontier knobs + the postchain airborne-callsign matching) and appends $AIRCS to the
# ADS-B flags. AIRCS defaults to "--air-cs-all on --air-cs-skip LIRF" (the 177-day-gated setting); AIRCS="--air-cs-all off"
# reproduces agent_frontier_chain.sh. Frontier knobs still come in through FL_EXTRA
# (e.g. FL_EXTRA="--floor-stack-anchors on --floor-stack-anch-skip LIRF"). Everything else is identical.
# experiment frontier COPY of bag_chain4.sh (2026-09-23, docs/LAB-NOTES.md). Identical recipe; three things are
# env-overridable and default to exactly what bag_chain4 does:
#   ADSB        the ADS-B apply script           (default archive/agent_frontier_apply.py = agent_adsbq_apply.py + two OFF knobs)
#   LOWER_SKIP  --floor-stack-lower-skip list    (default LIRF,LFPG)
#   FL_EXTRA    extra ADS-B flags, e.g. "--kind-off LIRF:appear" or "--calib-weight-recent 3"
#   CALIB_FILE / CALIB_PREDS as in bag_chain3/4.
# One difference: each ADS-B run's stdout (calibration + serving counts) is kept in data/interim/v<V>_adsb_<k>.log.
#   bash agent_frontier_chain.sh <model.parquet> <version> <prior_version> [upload]
# No backslashes.
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-5}
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-5} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-5} MKL_NUM_THREADS=${MKL_NUM_THREADS:-5} NUMEXPR_MAX_THREADS=5 RAYON_NUM_THREADS=5
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
MODEL="$1"; V="$2"; PRIOR="$3"; UP="${4:-}"
export OUTF="submissions/jolly-lobster_v${V}.parquet" PRIORF="submissions/jolly-lobster_v${PRIOR}.parquet" P="data/interim/v${V}" V PRIOR
[ -f "$MODEL" ] || { say "no model $MODEL -- stopping"; exit 1; }
[ -f "$PRIORF" ] || { say "no prior $PRIORF -- stopping"; exit 1; }
[ -f "$OUTF" ] && { say "$OUTF exists -- stopping"; exit 1; }
CALIB_FILE="${CALIB_FILE:-data/interim/calib_days_v41_203.txt}"
CD="$(cat $CALIB_FILE)"
CPX=""; [ -n "${CALIB_PREDS:-}" ] && CPX="--calib-preds $CALIB_PREDS"
ADSB="${ADSB:-scripts/agent_merged_apply.py}"
AIRCS="${AIRCS---air-cs-all on --air-cs-skip LIRF}"
LOWER_SKIP="${LOWER_SKIP:-LIRF,LFPG}"
FL="--osm-fill on --floor-stack-feats quality --stack-feats quality_anch --floor-q zero --spill-tracks on --calib-days $CD --floor-stack-lower 1800 --floor-stack-lower-skip $LOWER_SKIP $CPX ${FL_EXTRA:-} $AIRCS"
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
say "merged chain: ADSB=$ADSB AIRCS=${AIRCS:-none} LOWER_SKIP=$LOWER_SKIP FL_EXTRA=${FL_EXTRA:-none}; OMP_NUM_THREADS=$OMP_NUM_THREADS; calib $CALIB_FILE ($(echo "$CD" | tr , " " | wc -w) days); calib-preds ${CALIB_PREDS:-default}"
say "v$V: base = day-early rule on $MODEL"
$PY archive/apply_rome_dayearly.py --in "$MODEL" --out ${P}_base.parquet >/dev/null 2>&1 || { say "dayearly failed"; exit 1; }
say "non-LIRF path: shipped gate, then 10 ADS-B realisations, 3 at a time"
$PY -u archive/apply_rome_plan_gate.py --in ${P}_base.parquet --out ${P}_s0ship.parquet >/dev/null 2>&1
for batch in "1 2 3" "4 5 6" "7 8 9" "10"; do
  for k in $batch; do [ -s ${P}_adsb_$k.parquet ] && continue; $PY -u $ADSB --in ${P}_s0ship.parquet --out ${P}_adsb_$k.parquet $FL > ${P}_adsb_$k.log 2>&1 & done
  wait; say "ADS-B batch [$batch] done"
done
avg ${P}_s1.parquet $(for k in $(seq 1 10); do echo ${P}_adsb_$k.parquet; done)
chain ${P}_s1.parquet ${P}_nl
say "LIRF path: gate bagged over 20 seeds at the retune"
for sd in $(seq 1 20); do $PY -u archive/apply_rome_plan_gate.py --gap-cap 10800 --blend-weight 0.60 --seed $sd --in ${P}_base.parquet --out ${P}_gate_$sd.parquet >/dev/null 2>&1; done
avg ${P}_gatebag_s0.parquet $(for sd in $(seq 1 20); do echo ${P}_gate_$sd.parquet; done)
$PY -u $ADSB --in ${P}_gatebag_s0.parquet --out ${P}_gatebag_s1.parquet $FL > ${P}_gatebag_adsb.log 2>&1
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
