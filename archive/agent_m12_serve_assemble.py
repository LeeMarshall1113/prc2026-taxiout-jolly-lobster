"""experiment m12: assemble the 12-month serving term into a board file on top of agent_slot2_v130_final.parquet (which already holds the 8-month C4ff7V3 x v9-kapt term).
python archive/agent_m12_serve_assemble.py [BASE=agent_slot2_v130_final.parquet] [OUT=agent_m12_slot_final.parquet]
out = rint(max(BASE + (dp12 - dp9), 0)) on the nine airports; dp12 = kapt12 x mean over seeds 0-2 of the Modal serve corr (data/interim/bigc12_serve/C4ff7V3_s*.npz, python modal/agent_bigc12_app.py fetch_serve);
dp9 = agent_bigc_v9_serve_delta['dp'] (the term inside BASE). Every other row (LTFM included) is untouched. Guard: 2026 sum dp^2/row of (dp12 - dp9) <= 2.0x the fold's (12- vs 8-month f on the 8 old months).
NOTE: slot1/slot2 differ from agent_bigc_v9_board_final by rms 3.8 s on the nine airports (not exactly v9): the replacement uses v9's dp as the term to remove; that residual is carried unchanged.
"""
import glob
import json
import subprocess
import sys
import numpy as np
import polars as pl
I = "data/interim/"
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
base_f = sys.argv[1] if len(sys.argv) > 1 else "agent_final_v130.parquet"
out_f = sys.argv[2] if len(sys.argv) > 2 else "agent_m12_slot_final.parquet"
TREAT = sys.argv[4] if len(sys.argv) > 4 else "C4ff7V3"
FOLD = sys.argv[5] if len(sys.argv) > 5 else "agent_m12_fold_C4ff7V4.npz"   # fold npz written by agent_m12_judge.py (f = treat 3-seed term, f8 = the term inside the base)
fs = sorted(glob.glob(I + f"bigc12_serve/{TREAT}_s*.npz"))
assert fs, "no serve fits: run modal/agent_bigc12_app.py fetch_serve"
D = [np.load(f, allow_pickle=True) for f in fs]
print("seed files", [f.split("/")[-1] for f in fs], "rounds", [int(d["rounds"]) for d in D])
corr = np.mean([d["pred"] for d in D], 0)
si = np.load(I + "agent_bigc12_serve_in.npz", allow_pickle=True)
ids, apt = si["ids"], si["apt"].astype(str)
is9 = np.isin(apt, AIR)
kap = json.load(open(I + f"agent_m12_kapt_{TREAT}.json"))
dp12 = np.array([kap.get(a, 0.0) for a in apt]) * corr
dp12[~is9] = 0
old = np.load(I + (sys.argv[3] if len(sys.argv) > 3 else "agent_slot_old_delta.npz"), allow_pickle=True)
assert (old["ids"] == ids).all()
dpn = dp12 - old["dp"]
bf = pl.read_parquet(I + base_f).sort(ID)
assert (bf[ID].to_numpy() == ids).all()
cur = bf[T].cast(pl.Float64).to_numpy()
o = np.rint(np.maximum(cur + dpn, 0))
mv = o != cur
bf.with_columns(pl.Series(T, o).cast(bf.schema[T])).write_parquet(I + out_f)
print(f"rows moved {int(mv.sum()):,}; mean shift {float((o - cur).mean()):+.3f}; rms dp(moved) {float(np.sqrt(((o - cur)[mv] ** 2).mean())):.2f}; max {np.abs(o - cur).max():.0f}; sum dp^2/row {float(((o - cur) ** 2).mean()):.2f}")
print("seed-to-seed corr of the serve term:", np.round(np.corrcoef([d["pred"][is9] for d in D])[0, 1:], 4) if len(D) > 1 else "n/a")
fz = np.load(I + FOLD, allow_pickle=True)   # f8 = the 8-month V4 fold term (the term inside the base)
f12, f8 = fz["f"], fz["f8"]
f = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
a_f = f["apt"].astype(str)
n8 = len(f8)
k12 = dict(zip(fz["air"], fz["kap"]))
k9 = json.load(open(I + "agent_bigc_v9_kapt.json"))   # V4 kept the v9 per-airport k
dpf = np.array([k12.get(x, 0.0) for x in a_f[:n8]]) * f12[:n8] - np.array([k9.get(x, 0.0) for x in a_f[:n8]]) * f8
ratio = float((dpn[is9] ** 2).mean() / (dpf ** 2).mean())
print(f"fold (8 old months) sum dp^2/row {float((dpf ** 2).mean()):.2f}; 2026 9-apt {float((dpn[is9] ** 2).mean()):.2f}; footprint ratio {ratio:.2f} ({'PASS' if ratio <= 2.0 else 'FAIL'} at <= 2.0)")
print("LTFM rows changed vs base:", int((o[apt == "LTFM"] != cur[apt == "LTFM"]).sum()))
r = subprocess.run([sys.executable, "archive/agent_smallbundle_06_check.py", I + out_f, I + base_f], capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(__import__("os").environ, PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage"))
print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in ("FAIL", "moved", "ALL", "shift", "corr"))))
