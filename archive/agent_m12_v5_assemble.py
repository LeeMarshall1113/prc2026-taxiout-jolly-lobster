"""experiment m12: 6-seed 12m-V5 term replaces the 9-seed 12m-V4 term (kapt4 x mean s0-8) on agent_m12v4_9seed_final.parquet; new = kapt5 x mean(V5 serve s0-5). Nine airports; LTFM untouched.
Also writes the lens fold pair: ctrl = agent_m12_v4_fold_treat.npz, treat = ctrl + k5 f5 - k4 f4 on the 8 old months (agent_m12_v5_fold_treat.npz)."""
import json
import subprocess
import sys
import numpy as np
import polars as pl
I = "data/interim/"
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
base_f, out_f = "agent_m12v4_9seed_final.parquet", "agent_m12v5_6seed_final.parquet"
L = lambda c, s: np.load(I + f"bigc12_serve/{c}_s{s}.npz", allow_pickle=True)  # noqa: E731
P4 = np.mean([L("C4ff7V4", s)["pred"] for s in range(9)], 0)
S5 = [L("C4ff7V5", s) for s in range(6)]
P5 = np.mean([d["pred"] for d in S5], 0)
print("V5 rounds", [int(d["rounds"]) for d in S5], "features", len(S5[0]["names"]), "pred sd", [round(float(d["pred"].std()), 1) for d in S5])
si = np.load(I + "agent_bigc12_serve_in.npz", allow_pickle=True)
ids, apt = si["ids"], si["apt"].astype(str)
is9 = np.isin(apt, AIR)
k4, k5 = json.load(open(I + "agent_m12_kapt_C4ff7V4.json")), json.load(open(I + "agent_m12_kapt_C4ff7V5.json"))
print("dose V4", k4, "\ndose V5", k5)
d4 = np.array([k4.get(a, 0.0) for a in apt]) * P4
d5 = np.array([k5.get(a, 0.0) for a in apt]) * P5
d4[~is9] = 0; d5[~is9] = 0
dn = d5 - d4
print(f"rms(V5 6-seed term - V4 9-seed term) nine airports {np.sqrt((dn[is9] ** 2).mean()):.2f} s; corr {np.corrcoef(d5[is9], d4[is9])[0, 1]:.4f}; rms terms V5 {np.sqrt((d5[is9] ** 2).mean()):.1f} V4 {np.sqrt((d4[is9] ** 2).mean()):.1f}")
bf = pl.read_parquet(I + base_f).sort(ID)
assert (bf[ID].to_numpy() == ids).all()
cur = bf[T].cast(pl.Float64).to_numpy()
o = np.rint(np.maximum(cur + dn, 0)); mv = o != cur
bf.with_columns(pl.Series(T, o).cast(bf.schema[T])).write_parquet(I + out_f)
print(f"rows moved {int(mv.sum()):,}; LTFM changed {int((o[apt == 'LTFM'] != cur[apt == 'LTFM']).sum())}; mean shift {float((o - cur).mean()):+.3f}; sum dp^2/row {float(((o - cur) ** 2).mean()):.2f}")
fz = np.load(I + "agent_m12_fold_C4ff7V5_vs_12C4ff7V4.npz", allow_pickle=True)
n8 = len(fz["f8"])
f = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
a8 = f["apt"].astype(str)[:n8]
kk5 = np.array([dict(zip(fz["air"], fz["kap"])).get(x, 0.0) for x in a8]); kk4 = np.array([k4.get(x, 0.0) for x in a8])
dpf = kk5 * fz["f"][:n8] - kk4 * fz["f8"]
ratio = float((dn[is9] ** 2).mean() / (dpf ** 2).mean())
print(f"fold (8 old months, 3-seed V5 vs 3-seed V4) sum dp^2/row {float((dpf ** 2).mean()):.2f}; 2026 9-apt {float((dn[is9] ** 2).mean()):.2f}; footprint ratio {ratio:.2f} ({'PASS' if ratio <= 2.0 else 'FAIL'} at <= 2.0)")
c = dict(np.load(I + "agent_m12_v4_fold_treat.npz", allow_pickle=True))
pos = {v: i for i, v in enumerate(c["ids"].astype(float))}
ix = np.array([pos[v] for v in f["ids"][:n8].astype(float)])
g = c["both_guard"].astype(float).copy(); g[ix] += dpf; c["both_guard"] = g
np.savez(I + "agent_m12_v5_fold_treat.npz", **c)
r = subprocess.run([sys.executable, "archive/agent_smallbundle_06_check.py", I + out_f, I + base_f], capture_output=True, text=True, encoding="utf-8", errors="replace",
                   env=dict(__import__("os").environ, PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage"))
print("\n".join(l for l in r.stdout.splitlines() if any(q in l for q in ("FAIL", "moved", "ALL", "corr", "shift"))))
