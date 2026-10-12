#!/usr/bin/env bash
# experiment md: label-free feature dumps for the 116 new 2025 days (docs/LAB-NOTES.md), resumable; each step
# is skipped when its output exists; yields (exit 10) between steps when the scheduler asks. Existing scripts UNEDITED.
cd "${PRC_ROOT:-$PWD}"
PY=./.venv/Scripts/python.exe
I=data/interim
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
yieldchk () { if [ -n "$JOB_PREEMPT_FILE" ] && [ -e "$JOB_PREEMPT_FILE" ]; then say "preempt requested -> exit 10"; exit 10; fi; }
step () {  # step <out> <cmd...>
  local out=$1; shift
  [ -s "$out" ] && { say "have $out"; return 0; }
  yieldchk; say "RUN $*"; "$@" || { say "FAILED rc $? : $*"; exit 1; }
}
for h in a b; do
  D=$(cat $I/agent_md_newdays_$h.txt)
  step $I/agent_md_rows_new_$h.parquet $PY -u -W ignore archive/agent_offblock_01_dump.py --mode 2025 --days $D --out $I/agent_md_rows_new_$h.parquet
  step $I/agent_md_evt_new_$h.parquet $PY -u -W ignore archive/agent_monster2_01_events.py --mode 2025 --days $D --out $I/agent_md_evt_new_$h.parquet
  step $I/agent_md_trk_new_$h.parquet $PY -u -W ignore archive/agent_restack3_01_dump.py --mode 2025 --days $D --out $I/agent_md_trk_new_$h.parquet
  step $I/agent_md_ff_new_$h.parquet $PY -u -W ignore archive/agent_backx_01_dump.py --mode 2025 --days $D --out $I/agent_md_ff_new_$h.parquet
done
for k in rows evt trk ff; do
  [ -s $I/agent_md_${k}_new.parquet ] || $PY -c "
import polars as pl
a = pl.concat([pl.read_parquet('$I/agent_md_${k}_new_a.parquet'), pl.read_parquet('$I/agent_md_${k}_new_b.parquet')], how='diagonal_relaxed')
print('$k', a.height, 'rows', a['MVT_ID_mvt'].n_unique(), 'unique ids')
a.write_parquet('$I/agent_md_${k}_new.parquet')" || exit 1
done
step $I/agent_md_seq_new.parquet $PY -u -W ignore archive/agent_md_seqnew.py
say "dumps done"
