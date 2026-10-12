#!/usr/bin/env bash
# After v138 and v139 have scored: v140 (Amsterdam hedge) and v141 (corrector refit), each on the SAME best base, both uploaded regardless of each other's score.
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
until [ "$(date -u +%Y%m%d)" -ge 20261004 ]; do sleep 120; done
sc () { $PY archive/score_of.py "jolly-lobster_v$1.parquet" 219.0687 2>/dev/null | head -1 | awk '{print $3}'; }
for i in $(seq 1 150); do S9=$(sc 139); [ -n "$S9" ] && break; sleep 60; done
S8=$(sc 138); echo "[$(date -u +%H:%MZ)] v138=$S8 v139=$S9"
BASE=137; PS=219.0687
if [ -n "$S8" ] && awk "BEGIN{exit !($S8 < $PS)}"; then BASE=138; PS=$S8; fi
if [ -n "$S9" ] && awk "BEGIN{exit !($S9 < $PS)}"; then BASE=139; PS=$S9; fi
$PY archive/agent_v14z_build.py 140 $BASE agent_snowsev_hedge_final agent_basefix_final || { echo "BUILD 140 FAILED"; exit 1; }
$PY archive/agent_v14z_build.py 141 $BASE agent_mega_final agent_mega_no4_final || { echo "BUILD 141 FAILED"; exit 1; }
for V in 140 141; do $PY archive/agent_smallbundle_06_check.py data/interim/agent_bundle_v${V}_final.parquet submissions/jolly-lobster_v${BASE}.parquet | tail -1; done
bash archive/upload_when_slot.sh 140 $BASE $PS 20261004
bash archive/upload_when_slot.sh 141 $BASE $PS 20261004
