#!/usr/bin/env bash
# experiment stagecap stage serving (docs/LAB-NOTES.md): the v105 ADS-B path of archive/merged_chain.sh
# (as run by archive/build_v105_taxinm.sh) with archive/agent_stagecap_apply_taxinm.py and the variant's stack flags.
# Inputs are v105's own stage inputs (data/interim/v105_s0ship.parquet, v105_gatebag_s0.parquet; they do not depend on the
# stacks). Writes data/interim/v105sc_adsb_{1..10}.parquet -> v105sc_s1.parquet (mean, rint, as merged_chain's avg) and
# v105sc_gatebag_s1.parquet. RECON=1 also runs the copy at DEFAULT stack settings once (v105sc_recon_1.parquet) for the
# reconstruction check vs the shipped realisations v105_adsb_1..3. Whole tree under capped_run --cap-gb $SC_CAP.
#   bash archive/agent_stagecap_stageserve.sh "<stack flags>"
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)" OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
SF="$1"
P=data/interim/v105sc
ADSB=archive/agent_stagecap_apply_taxinm.py
CD="$(cat data/interim/calib_days_264_octnov.txt)"
CALIB_PREDS="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,data/interim/calib2026/preds_oct_f4.parquet,data/interim/calib2026/preds_nov_c.parquet"
AIRCS="--air-cs-all on --air-cs-skip LIRF --noplan-callsign on --floor-stack-taxinm-apts EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH"
FL="--osm-fill on --floor-stack-feats quality --stack-feats quality_anch --floor-q zero --spill-tracks on --calib-days $CD --floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG --calib-preds $CALIB_PREDS --floor-stack-anchors on --floor-stack-anch-skip LIRF $AIRCS"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
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
say "stage serve: stack flags '$SF'; RECON=${RECON:-0}"
JOBS=()
[ "${RECON:-0}" = "1" ] && [ ! -s ${P}_recon_1.parquet ] && JOBS+=("recon")
for k in $(seq 1 10); do [ -s ${P}_adsb_$k.parquet ] || JOBS+=("$k"); done
[ -s ${P}_gatebag_s1.parquet ] || JOBS+=("gb")
say "jobs: ${JOBS[*]}"
i=0
for J in "${JOBS[@]}"; do
  case "$J" in
    recon) $PY -u $ADSB --in data/interim/v105_s0ship.parquet --out ${P}_recon_1.parquet $FL > ${P}_recon_1.log 2>&1 & ;;
    gb)    $PY -u $ADSB --in data/interim/v105_gatebag_s0.parquet --out ${P}_gatebag_s1.parquet $FL $SF > ${P}_gatebag_adsb.log 2>&1 & ;;
    *)     $PY -u $ADSB --in data/interim/v105_s0ship.parquet --out ${P}_adsb_$J.parquet $FL $SF > ${P}_adsb_$J.log 2>&1 & ;;
  esac
  i=$((i + 1))
  if [ $((i % 3)) = 0 ]; then wait; say "batch done (${i}/${#JOBS[@]})"
    # yield checkpoint (scheduler heartbeat exit 10 -> run_job touches JOB_PREEMPT_FILE); finished outputs are skipped on resume
    if [ -n "${JOB_PREEMPT_FILE:-}" ] && [ -f "$JOB_PREEMPT_FILE" ] && [ $i -lt ${#JOBS[@]} ]; then say "scheduler asked to yield -- stopping after this batch (exit 10); re-run resumes"; exit 10; fi
  fi
done
wait
for k in $(seq 1 10); do [ -s ${P}_adsb_$k.parquet ] || { say "realisation $k missing"; tail -5 ${P}_adsb_$k.log; exit 1; }; done
[ -s ${P}_gatebag_s1.parquet ] || { say "gatebag missing"; tail -5 ${P}_gatebag_adsb.log; exit 1; }
avg ${P}_s1.parquet $(for k in $(seq 1 10); do echo ${P}_adsb_$k.parquet; done)
if [ "${RECON:-0}" = "1" ]; then
$PY - <<'PYEOF'
import itertools, numpy as np, polars as pl
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
L = lambda p: pl.read_parquet(p).sort(ID)[T].to_numpy().astype(float)  # noqa: E731
F = {f"v105_adsb_{k}": L(f"data/interim/v105_adsb_{k}.parquet") for k in (1, 2, 3)}
R = L("data/interim/v105sc_recon_1.parquet")
band = []
for a, b in itertools.combinations(F, 2):
    d = F[a] - F[b]; band.append((int((d != 0).sum()), float(np.sqrt((d ** 2).mean()))))
    print(f"shipped {a} vs {b}: rows moved {band[-1][0]:,} rms {band[-1][1]:.2f}")
ok = True
for a in F:
    d = R - F[a]; n, r = int((d != 0).sum()), float(np.sqrt((d ** 2).mean()))
    print(f"copy@defaults vs {a}: rows moved {n:,} rms {r:.2f}")
    ok &= n <= max(x[0] for x in band) and r <= max(x[1] for x in band)
print(f"STAGE-APPLY RECONSTRUCTION (copy at defaults inside the shipped realisation band): {'PASS' if ok else 'FAIL'}")
PYEOF
fi
say "stage serve done -> ${P}_s1.parquet, ${P}_gatebag_s1.parquet"
