#!/usr/bin/env bash
# One planned upload for UTC 2026-10-06, written down in docs/LAB-NOTES.md before it runs.
# Preferred file: round three WITHOUT the view-geometry group (arm KYQSR-pw-s0-r0), if its 12 serve fits and its temporal-frame fits arrive, the temporal
# difference against the control is at least 0.10 s better, and the build passes its checks. Otherwise the already built and checked file with the group
# (data/interim/agent_arith3_final.parquet). Hard deadline for the choice: 22:15 UTC. No decision in this script uses a leaderboard score.
cd "${PRC_ROOT:-$PWD}"; V=./.venv/Scripts/python.exe; SYS=/c/Python314/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
ARM=KYQSR-pw-s0-r0; PICK=""
while [ "$(date -u +%H%M)" -lt 2215 ]; do
  $SYS modal/agent_arith_app.py fetch >/dev/null 2>&1
  ns=$(ls data/interim/agent_arith3_serve 2>/dev/null | grep -c "^${ARM}_serve_s"); nt=$(ls data/interim/agent_arith3_out 2>/dev/null | grep -c "^${ARM}_s[0-9]_t[ab]")
  say "serve fits $ns/12, temporal fits $nt/6"
  if [ "$ns" -ge 12 ] && [ "$nt" -ge 6 ]; then
    $V archive/capped_run.py --cap-gb 3 -- "$(pwd -W)/.venv/Scripts/python.exe" "$(pwd -W)/archive/agent_arith3_03_judge.py" temp $ARM 0-pw-s0-r0 0,1,2 > logs/auto_v152_temp.log 2>&1
    D=$($V -c "
import re
t=open('logs/auto_v152_temp.log').read()
a=re.search(r'cand-ctrl: pooled ([-+0-9.]+)',t); b=re.search(r'pcd-ctrl: pooled ([-+0-9.]+)',t)
print(round(float(a.group(1))-float(b.group(1)),3) if a and b else 'nan')")
    say "temporal difference candidate minus control: $D"
    OK=$($V -c "print(1 if '$D' not in ('nan','') and float('$D') <= -0.10 else 0)")
    if [ "$OK" = "1" ]; then
      $V archive/capped_run.py --cap-gb 3 -- "$(pwd -W)/.venv/Scripts/python.exe" "$(pwd -W)/archive/agent_arith3_07_build.py" build --arm $ARM --suffix _noG > logs/auto_v152_build.log 2>&1
      if grep -q "ALL CHECKS PASS" logs/auto_v152_build.log && [ -s data/interim/agent_arith3_final_noG.parquet ]; then PICK=data/interim/agent_arith3_final_noG.parquet; say "built and checked the version without the group"; else say "build of the version without the group failed its checks"; fi
    else say "temporal clause not met for the version without the group"; fi
    break
  fi
  sleep 300
done
[ -z "$PICK" ] && { PICK=data/interim/agent_arith3_final.parquet; say "using the version with the group (already built and checked)"; }
cp "$PICK" data/interim/agent_bundle_v152_final.parquet && cp "$PICK" submissions/jolly-lobster_v152.parquet && say "v152 = $PICK"
bash archive/upload_when_slot.sh 152 150 216.6698 20261006
say "AUTO V152 DONE"
