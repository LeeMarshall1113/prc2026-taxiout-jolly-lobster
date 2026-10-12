#!/usr/bin/env bash
# Upload an ALREADY-BUILT submissions/jolly-lobster_v<V>.parquet once the UTC date reaches <DATE> and a slot is free.
# Re-checks the same guards as bag_chain against the prior best, then uploads and polls the score. Slot-free (no compute).
#   bash archive/upload_when_slot.sh <V> <prior V> <prior score> <YYYYMMDD> [wait-for-file-log]
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
V="$1"; PR="$2"; PS="$3"; D="$4"; say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
export OUTF="submissions/jolly-lobster_v${V}.parquet" PRIORF="submissions/jolly-lobster_v${PR}.parquet"
for i in $(seq 1 1440); do [ -s "$OUTF" ] && break; sleep 60; done
[ -s "$OUTF" ] || { say "$OUTF never appeared"; exit 1; }
until [ "$(date -u +%Y%m%d)" -ge "$D" ]; do sleep 120; done
say "v$V: date ok -- re-checking guards vs v$PR"
$PY - <<'PYEOF' || { say "GUARDS FAILED -- v$V NOT uploaded"; exit 0; }
import os, sys, datetime as dt, numpy as np, polars as pl
from prc.leaderboard import fetch_all
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
o = pl.read_parquet(os.environ["OUTF"]).sort(ID); p = pl.read_parquet(os.environ["PRIORF"]).sort(ID)
a, b = p[T].to_numpy().astype(float), o[T].to_numpy().astype(float)
today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
used = len([s for s in fetch_all() if s.team_name == "jolly-lobster" and str(s.processed_at)[:10] == today])
done = [s for s in fetch_all() if s.team_name == "jolly-lobster" and s.filename == os.path.basename(os.environ["OUTF"])]
g = {"rows": o.height == 344841, "ids match": bool((o[ID] == p[ID]).all()), "corr > 0.99": float(np.corrcoef(a, b)[0, 1]) > 0.99,
     "|shift| < 5": abs(float((b - a).mean())) < 5, "no negatives": int((b < 0).sum()) == 0, "slots < 5": used < 5, "not already uploaded": not done}
print(f"corr {np.corrcoef(a, b)[0, 1]:.4f} shift {(b - a).mean():+.2f}s RMS {np.sqrt(((b - a) ** 2).mean()):.1f}s used today {used}")
for k, v in g.items(): print(f"  {'PASS' if v else 'FAIL'}  {k}")
sys.exit(0 if all(g.values()) else 3)
PYEOF
say "uploading v$V"
$PY archive/ship_helpers.py upload "$OUTF" 2>&1 | grep -viE "warning|deprecat" | tail -1
for i in $(seq 1 40); do R=$($PY archive/score_of.py "jolly-lobster_v${V}.parquet" "$PS" 2>/dev/null); if echo "$R" | grep -q SCORED; then echo "$R"; break; fi; sleep 45; done
say "UPLOAD V$V DONE"
