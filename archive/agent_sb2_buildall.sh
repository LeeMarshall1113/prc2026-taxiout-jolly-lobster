#!/usr/bin/env bash
# experiment sb2 build driver (docs/LAB-NOTES.md): only after FINAL=PASS* in logs/agent_sb2_chain.log.
# Serving builds (treatment + no-token control through the same sb2 scripts; finished builds are skipped), compose + guard,
# lens, and only if both guard ratios <= 2.0: v129 + checks. Exit 10 = yielded to a scheduler preempt between builds.
cd "${PRC_ROOT:-$PWD}"
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH=. OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
CAP="$PY archive/capped_run.py --cap-gb ${SB2_CAP:-6} --"
I=data/interim
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
yield_ () { if [ -n "${JOB_PREEMPT_FILE:-}" ] && [ -f "$JOB_PREEMPT_FILE" ]; then say "PREEMPT -- yielding between builds"; exit 10; fi; }
L=logs/agent_sb2_chain2.log
FINAL=$(grep "^FINAL=" $L | tail -1 | cut -d= -f2); TOK=$(grep "^TOKENS=" $L | tail -1 | cut -d= -f2)
C2=$(grep "^C2=" $L | tail -1 | cut -d= -f2); NAME=$(grep "^NAME=" $L | tail -1 | cut -d= -f2)
say "chain verdict: FINAL=$FINAL TOKENS=$TOK C2=$C2 NAME=$NAME"
case "$FINAL" in PASS*) ;; *) say "no pass -- nothing built"; exit 0;; esac
SQT=""; BXT=""
for t in $(echo "$TOK" | tr ',' ' '); do SQT="$SQT,$t"; [ "$t" != "lw" ] && BXT="$BXT,$t"; done
build () {  # set(c|t) piece(a|b|c|d) tokens-bx tokens-sq
  local S=$1 PC=$2 B=$3 Q=$4 OUT=agent_sb2_$1_$2
  [ -s $I/${OUT}_final.parquet ] && { say "$OUT exists"; return 0; }
  yield_
  say "build $OUT"
  case $PC in
    a) SC_PFX=sb2${S}a_ bash archive/agent_sb2_bxbuild.sh 1 est_v2 cap,evt,trk,bx,big$B $OUT ;;
    b) SC_PFX=sb2${S}b_ bash archive/agent_sb2_bxbuild.sh 1 est_v2 cap,evt,trk,seq,bx,big$B $OUT ;;
    c) SC_PFX=sb2${S}c_ bash archive/agent_sb2_build.sh 1 est_v2 cap,evt,trk,seq,big$Q $OUT ;;
    d) SC_PFX=sb2${S}d_ SC_TRAIN=$I/agent_stagecap_CAND_E9f.npz SC_P105=$I/v105sc SB_EXTRA_SPL=EDDM,EHAM,LEBL \
         bash archive/agent_sb2_bxbuild.sh 1 est_v2 cap,evt,trk,bx,big$B $OUT ;;
  esac >> logs/agent_sb2_build_detail.log 2>&1 || { say "build $OUT failed"; tail -8 logs/agent_sb2_build_detail.log; exit 1; }
}
if [ -n "$TOK" ]; then
  for PC in a b c d; do build t $PC "$BXT" "$SQT"; done
  for PC in a b c d; do build c $PC "" ""; done
fi
yield_
$CAP $PY -u archive/agent_sb2_compose.py --name "$NAME" --tokens "$TOK" --c2 "$C2" 2>&1 | grep -v capped_run | tee logs/agent_sb2_compose.log
grep -q "^GUARD=PASS" logs/agent_sb2_compose.log || { say "serving guard (like-for-like) FAILED -- no v129"; exit 0; }
$CAP $PY archive/agent_lens_09_predict.py --fold-ctrl $I/agent_sb2_ctrl_r0.npz --fold-treat $I/agent_sb2_${NAME}_s0.npz \
  --sub-ctrl $I/v122_stage9s_final.parquet --sub-treat $I/agent_sb2_cand.parquet --board-ctrl 224.88 2>&1 | grep -v capped_run | tee logs/agent_sb2_lens.log
LR=$($PY -c "
import re,sys
t=open('logs/agent_sb2_lens.log',encoding='utf-8').read()
m=re.search(r'fold: dMSE .*?sum dp\^2/row\s*([0-9.]+)\s+2026: sum dp\^2/row\s*([0-9.]+)',t)
print(f'{float(m.group(2))/float(m.group(1)):.4f}' if m else 'NA')")
say "lens footprint ratio: $LR"
OK=$($PY -c "print(1 if '$LR'!='NA' and float('$LR')<=2.0 else 0)")
[ "$OK" = "1" ] || { say "serving guard (lens) FAILED or unreadable ($LR) -- no v129"; exit 0; }
cp $I/agent_sb2_cand.parquet $I/v129_smallbundle2_final.parquet
say "v129 written: $I/v129_smallbundle2_final.parquet"
$CAP $PY archive/agent_smallbundle_06_check.py $I/v129_smallbundle2_final.parquet $I/v122_stage9s_final.parquet 2>&1 | grep -v capped_run | tee logs/agent_sb2_check.log
say "ALL DONE"
