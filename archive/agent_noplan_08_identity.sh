#!/usr/bin/env bash
# experiment noplan 08: archive/agent_noplan_stage1_eval.py (LEVER A / IDEA-NP1, docs/LAB-NOTES.md) must
# reproduce archive/agent_trackfeat_stage1_eval.py bit-identically with --air-cs-all off and --noplan-callsign
# off (the Q baseline neither flag touches). 3 days, shipped flags, mirrors agent_trackfeat_08_identity.sh.
set -uo pipefail
cd "${PRC_ROOT:-$PWD}"
export OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4 PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
PY=./.venv/Scripts/python.exe
D=2025-01-15,2025-07-17,2025-08-20
SHIP="--stack gbm --stack-feats quality_anch --stack-skip-airports LIRF --floor on --floor-stack on --floor-stack-snapshot on --floor-q zero --floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG --osm-fill on --spill-tracks on --blends both_guard --floor-stack-feats quality_anch --floor-stack-anch-skip LIRF"
CAP="$PY archive/capped_run.py --cap-gb 3 --"
$CAP $PY -u -W ignore archive/agent_trackfeat_stage1_eval.py $SHIP --days $D --report-days $D --save-blends data/interim/agent_noplan_id_orig.npz > logs/agent_noplan_08_orig.txt 2>&1
$CAP $PY -u -W ignore archive/agent_noplan_stage1_eval.py $SHIP --days $D --report-days $D --save-blends data/interim/agent_noplan_id_copy.npz > logs/agent_noplan_08_copy.txt 2>&1
$PY - <<'PYEOF' > logs/agent_noplan_08_compare.txt 2>&1
import numpy as np
a = np.load("data/interim/agent_noplan_id_orig.npz", allow_pickle=True)
b = np.load("data/interim/agent_noplan_id_copy.npz", allow_pickle=True)
oa = np.argsort(a["ids"], kind="stable"); ob = np.argsort(b["ids"], kind="stable")
assert (a["ids"][oa] == b["ids"][ob]).all(), "id sets differ"
for k in ("y", "p0", "both_guard"):
    da = a[k][oa].astype(float); db = b[k][ob].astype(float)
    print(f"{k}: max|diff| = {np.max(np.abs(da - db)):.6g} over {len(da)} rows")
print("IDENTITY OK" if all(np.max(np.abs(a[k][oa].astype(float) - b[k][ob].astype(float))) == 0.0 for k in ("y","p0","both_guard")) else "IDENTITY MISMATCH")
PYEOF
echo DONE > logs/agent_noplan_08_done.txt
