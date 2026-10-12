#!/usr/bin/env bash
# experiment stagecap E9S build (docs/LAB-NOTES.md): S1-stack stage serving (v105 path, 10 realisations + gatebag,
# yield checkpoint between batches, resumable), then the E9f bx_big re-stack serve build, the M splice as a delta on v116
# added to v119_big_blend_ltfmrefit, checks, lens. Exit 10 = yielded; re-run resumes (finished outputs are skipped).
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH=. OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
I=data/interim; M=EDDF,EDDM,EHAM,LEBL,LSZH
V116=$I/v116_rs5_v109_big_final.parquet; V119=$I/v119_big_blend_ltfmrefit_final.parquet; OUTF=$I/v122_stage9s_final.parquet
SF="--stack-trees 600 --stack-leaves 31 --stack-minchild 50 --stack-lr 0.05 --floor-stack-trees 500 --floor-stack-leaves 31 --floor-stack-minchild 100 --floor-stack-lr 0.05"
CAP="$PY archive/capped_run.py --cap-gb 6 --"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
grep -q "^E9S_VERDICT PASS" logs/agent_stagecap_e9s.log || { say "E9S did not pass -- nothing to build"; exit 1; }
say "=== 1. stage serving (S1 stacks, v105 path)"
if [ ! -s $I/v105sc_s1.parquet ] || [ ! -s $I/v105sc_gatebag_s1.parquet ]; then
  $PY archive/capped_run.py --cap-gb 24 -- bash archive/agent_stagecap_stageserve.sh "$SF" >> logs/agent_stagecap_stageserve.log 2>&1
  RC=$?
  grep -v capped_run logs/agent_stagecap_stageserve.log | tail -6
  [ $RC -eq 10 ] && { say "yielded during stage serving (exit 10)"; exit 10; }
  [ $RC -eq 0 ] || { say "stage serving failed rc $RC"; exit 1; }
fi
[ -s $I/v105sc_s1.parquet ] && [ -s $I/v105sc_gatebag_s1.parquet ] || { say "stage outputs missing"; exit 1; }
if [ -n "${JOB_PREEMPT_FILE:-}" ] && [ -f "$JOB_PREEMPT_FILE" ]; then say "yield requested -- stop before the re-stack build (exit 10)"; exit 10; fi
say "=== 2. E9f bx_big re-stack serve build (train CAND_E9f, v105sc stage at S6 + EDDM,EHAM,LEBL)"
B=agent_stagecap_9s_bx_big
if [ ! -s $I/${B}_final.parquet ]; then
  SC_PFX=sc9s_ SC_TRAIN=$I/agent_stagecap_CAND_E9f.npz SC_P105=$I/v105sc SB_EXTRA_SPL=EDDM,EHAM,LEBL \
    bash archive/agent_stagecap_bxbuild.sh 1 est_v2 cap,evt,trk,bx,big $B >> logs/agent_stagecap_build9s_detail.log 2>&1 || { say "bx build failed"; tail -5 logs/agent_stagecap_build9s_detail.log; exit 1; }
fi
say "=== 3. S_M = v116 with the build at M; v122 = max(v119 + (S_M - v116), 0)"
$CAP $PY archive/splice_apt_parquet.py $V116 $I/${B}_final.parquet $M $I/agent_stagecap_9s_SM.parquet 2>&1 | grep -v capped_run
$CAP $PY - <<'PYEOF' 2>&1 | grep -v capped_run
import numpy as np, polars as pl
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; I = "data/interim/"
v116 = pl.read_parquet(I + "v116_rs5_v109_big_final.parquet").sort(ID)
sm = pl.read_parquet(I + "agent_stagecap_9s_SM.parquet").sort(ID)
v119 = pl.read_parquet(I + "v119_big_blend_ltfmrefit_final.parquet").sort(ID)
assert v116[ID].equals(sm[ID]) and v116[ID].equals(v119[ID]) and v116.columns == v119.columns == sm.columns
apt = pl.scan_parquet("data/raw/ranking.parquet").select(pl.col(ID), pl.col("ADEP_mvt")).collect()
a = v119.select(ID).join(apt, on=ID, how="left", maintain_order="left")["ADEP_mvt"].to_numpy().astype(str)
M = np.isin(a, ["EDDF", "EDDM", "EHAM", "LEBL", "LSZH"])
x116, xsm, x119 = (f[T].to_numpy().astype(float) for f in (v116, sm, v119))
d = xsm - x116
print(f"E9S delta: rows moved {int((d != 0).sum()):,} (all at M: {bool((d[~M] == 0).all())}); rms at M {np.sqrt((d[M] ** 2).mean()):.2f} s; mean at M {d[M].mean():+.3f} s")
bd = x119 - x116
print(f"v119 - v116 at M (blend delta + LTFM refit, kept): rows {int((bd[M] != 0).sum()):,}, rms {np.sqrt((bd[M] ** 2).mean()):.2f} s; "
      f"a plain row splice into v119 would have deleted it there")
out = np.maximum(x119 + d, 0)
print(f"LTFM rows untouched: {bool((out[a == 'LTFM'] == x119[a == 'LTFM']).all())}; rows changed vs v119 {int((out != x119).sum()):,}; floor binds on {int((x119 + d < 0).sum())} rows")
v119.with_columns(pl.Series(T, out).cast(v119[T].dtype)).write_parquet(I + "v122_stage9s_final.parquet")
print("wrote data/interim/v122_stage9s_final.parquet")
PYEOF
say "=== 4. checks vs v119 and lens"
$CAP $PY archive/agent_smallbundle_06_check.py $OUTF $V119 2>&1 | grep -v capped_run | tee logs/agent_stagecap_check9s_v119.log
grep -q "ALL CHECKS PASS" logs/agent_stagecap_check9s_v119.log || { say "CHECKS FAILED vs v119"; exit 1; }
$CAP $PY archive/agent_lens_09_predict.py --fold-ctrl $I/agent_restack5_V9TOP1.npz --fold-treat $I/agent_stagecap_RS_E9S_s0.npz \
  --sub-ctrl $V119 --sub-treat $OUTF --board-ctrl 225.84 2>&1 | grep -v capped_run
say "BUILT $OUTF"
