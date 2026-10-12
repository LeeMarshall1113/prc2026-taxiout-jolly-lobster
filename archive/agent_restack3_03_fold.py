"""experiment restack3 03: COPY of archive/agent_expensive_03_restack_fold.py with the pre-registered recipes of
docs/LAB-NOTES.md (archive/agent_restack3_restack.py). Recipe "" = RS_v2's est_v2 re-stack, bit for bit.

    python archive/agent_restack3_03_fold.py --recipe evt --seed 0 --out data/interim/agent_restack3_A.npz
Judge: python archive/agent_expensive_06_judge.py data/interim/modal/RS_v2.npz <out>
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_restack3_restack import features, fit, load_rows, predict

ap = argparse.ArgumentParser()
ap.add_argument("--fin-npz", default="data/interim/modal/TX_S6.npz")
ap.add_argument("--blend", default="both_guard")
ap.add_argument("--rows", default="data/interim/agent_offblock_rows_c238.parquet")
ap.add_argument("--evt", default="data/interim/agent_restack3_evt_2025.parquet")
ap.add_argument("--trk", default="data/interim/agent_restack3_trk_2025.parquet")
ap.add_argument("--recipe", default="")
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
T0 = time.time()

z = np.load(a.fin_npz, allow_pickle=True)
Z = {k: z[k] for k in z.keys()}
ids = Z["ids"].astype(float)
pos_of = {k: i for i, k in enumerate(ids)}
assert len(pos_of) == len(ids), "npz ids not unique"
r = load_rows(a.rows, a.recipe, a.evt, a.trk)
pos = np.array([pos_of.get(m, -1) for m in r["MVT_ID_mvt"].to_list()])
print(f"rows {r.height:,}; in npz {int((pos >= 0).sum()):,}; recipe '{a.recipe}'")
r = r.filter(pl.Series(pos >= 0))
pos = pos[pos >= 0]
fin = Z[a.blend][pos].astype(float)
y = Z["y"][pos].astype(float)
d_npz = Z["days"][pos].astype(str)
assert (d_npz == np.asarray(r["day"].to_list())).all(), "day mismatch between rows and npz"
X = features(r, fin, a.recipe)
print(f"X {X.shape}; non-null share of the added columns: "
      + " ".join(f"{np.isfinite(X[:, j]).mean():.2f}" for j in range(36, X.shape[1])))
t = y - fin
ud = sorted(set(d_npz))
fold = {d: i % a.folds for i, d in enumerate(ud)}
fo = np.array([fold[d] for d in d_npz])
add = np.zeros(len(t))
for k in range(a.folds):
    tr, te = fo != k, fo == k
    if te.any():
        add[te] = predict(fit(X[tr], t[tr], a.recipe, a.seed), X[te])
new = Z[a.blend].astype(float).copy()
new[pos] = np.maximum(fin + add, 0.0)
Z["both_guard_prers"] = Z[a.blend]
Z[a.blend] = new
np.savez(a.out, **Z)
N = len(Z["y"])
R0 = float(np.sqrt(((Z["both_guard_prers"] - Z["y"]) ** 2).mean()))
R1 = float(np.sqrt(((new - Z["y"]) ** 2).mean()))
print(f"re-stack recipe '{a.recipe}' seed {a.seed}: moved {int((add != 0).sum()):,} rows, sum dp^2 "
      f"{float((add ** 2).sum()):.3e}; all {N:,} rows {R0:.3f} -> {R1:.3f} ({R1 - R0:+.3f} s vs fin); "
      f"{time.time() - T0:.0f}s -> {a.out}")
