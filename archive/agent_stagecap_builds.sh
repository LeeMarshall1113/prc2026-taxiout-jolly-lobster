#!/usr/bin/env bash
# experiment stagecap builds (docs/LAB-NOTES.md): the exact v116 (= queued v111) path of
# agent_restack5_v109builds.sh step 1, through the copied re-stack builds (scripts/agent_stagecap_{bx,}build.sh).
#   bash archive/agent_stagecap_builds.sh repro        copies at defaults (TX_S6, v105 stage) must rebuild restack5's three
#                                                     recipe builds and, spliced into v109, v116 -- 0 rows differ
#   bash archive/agent_stagecap_builds.sh new TAG      SC_TRAIN = CAND_<TAG>, v105 stage = data/interim/v105sc_* ->
#                                                     data/interim/v119_stagecap_final.parquet; LTFM-order check; checks; lens
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)" OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
CAP="$PY archive/capped_run.py --cap-gb 6 --"
I=data/interim
V109=submissions/jolly-lobster_v109.parquet
V109PRE=$I/agent_restack5_chk_v112.parquet     # v109 before its LTFM post-step (restack5 s6.3 reproduction)
V116=$I/v116_rs5_v109_big_final.parquet
BOARD=225.84
R6=EDDF,EDDM,EGLL,EHAM,LEBL,LSZH
MODE="$1"; VT="${2:-}"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
cmp_ () { $CAP $PY archive/agent_restack5_cmp.py "$1" "$2" 2>&1 | grep -v capped_run; return ${PIPESTATUS[0]}; }
if [ "$MODE" = "repro" ]; then
  export SC_PFX=scr_; unset SC_TRAIN SC_P105; OT=agent_stagecap_repro
else
  [ -n "$VT" ] || { echo "need variant tag"; exit 2; }
  export SC_PFX=scx_ SC_TRAIN=$I/agent_stagecap_CAND_${VT}.npz SC_P105=$I/v105sc; OT=agent_stagecap_new
  [ -s $SC_TRAIN ] && [ -s $I/v105sc_s1.parquet ] && [ -s $I/v105sc_gatebag_s1.parquet ] || { say "missing new inputs"; exit 1; }
fi
say "=== $MODE ${VT}: SC_TRAIN=${SC_TRAIN:-default} SC_P105=${SC_P105:-default} SC_PFX=$SC_PFX"
b () {  # script recipe out
  [ -s $I/${3}_final.parquet ] || bash scripts/$1 1 est_v2 "$2" "$3" >> logs/${OT}_detail.log 2>&1 || { say "build $3 failed"; tail -5 logs/${OT}_detail.log; exit 1; }
  say "built $I/${3}_final.parquet"
}
b agent_stagecap_bxbuild.sh cap,evt,trk,bx,big ${OT}_bx_big
b agent_stagecap_bxbuild.sh cap,evt,trk,seq,bx,big ${OT}_seq_bx_big
b agent_stagecap_build.sh cap,evt,trk,seq,big ${OT}_seq_big
if [ "$MODE" = "repro" ]; then
  cmp_ $I/${OT}_bx_big_final.parquet $I/agent_restack5_x_cap_evt_trk_bx_big_final.parquet || { say "REPRODUCTION FAILED (bx,big)"; exit 1; }
  cmp_ $I/${OT}_seq_bx_big_final.parquet $I/agent_restack5_x_cap_evt_trk_seq_bx_big_final.parquet || { say "REPRODUCTION FAILED (seq,bx,big)"; exit 1; }
  cmp_ $I/${OT}_seq_big_final.parquet $I/agent_restack5_b_cap_evt_trk_seq_big_final.parquet || { say "REPRODUCTION FAILED (seq,big)"; exit 1; }
fi
say "--- splice into v109 (bx,big at $R6; seq,bx,big at LEMD; seq,big at LFPG,LIRF), and into v109-pre-LTFM"
for W in post pre; do
  if [ $W = post ]; then CUR=$V109; else CUR=$V109PRE; fi
  $CAP $PY archive/splice_apt_parquet.py $CUR $I/${OT}_bx_big_final.parquet $R6 $I/${OT}_${W}_g1.parquet 2>&1 | grep -v capped_run
  $CAP $PY archive/splice_apt_parquet.py $I/${OT}_${W}_g1.parquet $I/${OT}_seq_bx_big_final.parquet LEMD $I/${OT}_${W}_g2.parquet 2>&1 | grep -v capped_run
  $CAP $PY archive/splice_apt_parquet.py $I/${OT}_${W}_g2.parquet $I/${OT}_seq_big_final.parquet LFPG,LIRF $I/${OT}_${W}_g3.parquet 2>&1 | grep -v capped_run
done
if [ "$MODE" = "repro" ]; then
  cmp_ $I/${OT}_post_g3.parquet $V116 || { say "REPRODUCTION FAILED (v116)"; exit 1; }
  say "REPRODUCTION OK: the copied path at defaults rebuilds v116 exactly"
  exit 0
fi
OUTF=$I/v119_stagecap_final.parquet
cp $I/${OT}_post_g3.parquet $OUTF
$CAP $PY archive/apply_seq_residual.py --airports LTFM --in $I/${OT}_pre_g3.parquet --out $I/${OT}_ltfm.parquet > logs/${OT}_ltfm.log 2>&1
cmp_ $I/${OT}_ltfm.parquet $OUTF || { say "LTFM-order check FAILED"; exit 1; }
say "--- checks vs v116 (the v111 build) and vs v109"
$CAP $PY archive/agent_smallbundle_06_check.py $OUTF $V116 2>&1 | grep -v capped_run | tee logs/agent_stagecap_check_v116.log
grep -q "ALL CHECKS PASS" logs/agent_stagecap_check_v116.log || { say "CHECKS FAILED vs v116"; exit 1; }
$CAP $PY archive/agent_smallbundle_06_check.py $OUTF $V109 2>&1 | grep -v capped_run | tee logs/agent_stagecap_check_v109.log
say "lens: fold V9TOP1 -> RS_${VT}_s0, sub v116 -> v119"
$CAP $PY archive/agent_lens_09_predict.py --fold-ctrl $I/agent_restack5_V9TOP1.npz --fold-treat $I/agent_stagecap_RS_${VT}_s0.npz \
  --sub-ctrl $V116 --sub-treat $OUTF --board-ctrl $BOARD 2>&1 | grep -v capped_run
say "BUILT $OUTF"
