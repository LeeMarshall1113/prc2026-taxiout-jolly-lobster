"""experiment expensive 03: RE-STACK on a saved stage-1 fold npz (docs/LAB-NOTES.md).

    python archive/agent_expensive_03_restack_fold.py --fin-npz data/interim/modal/TX_S.npz \
        --rows data/interim/agent_offblock_rows_c238.parquet --feats est --folds 10 \
        --out data/interim/agent_expensive_RS.npz [--skip LIRF]

Writes a copy of --fin-npz whose `both_guard` is replaced by the re-stacked prediction on every matched row found in
--rows (K folds grouped by day, round-robin over sorted days; the model for a day never saw that day's labels). The
original is kept as `both_guard_prers`. Judge with the standing 4-clause judge:
    python archive/agent_trackfeat_07_judge.py <fin-npz> <out>
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_expensive_restack import features, fit, load_rows, predict

ap = argparse.ArgumentParser()
ap.add_argument("--fin-npz", default="data/interim/modal/TX_S.npz")
ap.add_argument("--blend", default="both_guard")
ap.add_argument("--rows", required=True)
ap.add_argument("--feats", choices=["est", "est_v2"], default="est")
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip", default="", help="airports left untouched (training still uses their rows)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
T0 = time.time()

z = np.load(a.fin_npz, allow_pickle=True)
Z = {k: z[k] for k in z.keys()}
ids = Z["ids"].astype(float)
pos_of = {k: i for i, k in enumerate(ids)}
assert len(pos_of) == len(ids), "npz ids not unique"
r = load_rows(a.rows)
pos = np.array([pos_of.get(m, -1) for m in r["MVT_ID_mvt"].to_list()])
print(f"rows {r.height:,}; in npz {int((pos >= 0).sum()):,}")
r = r.filter(pl.Series(pos >= 0))
pos = pos[pos >= 0]
fin = Z[a.blend][pos].astype(float)
y = Z["y"][pos].astype(float)
d_npz = Z["days"][pos].astype(str)
assert (d_npz == np.asarray(r["day"].to_list())).all(), "day mismatch between rows and npz"
X = features(r, fin, a.feats)
t = y - fin
ud = sorted(set(d_npz))
fold = {d: i % a.folds for i, d in enumerate(ud)}
fo = np.array([fold[d] for d in d_npz])
add = np.zeros(len(t))
for k in range(a.folds):
    tr, te = fo != k, fo == k
    if te.any():
        add[te] = predict(fit(X[tr], t[tr], a.seed), X[te])
skip = np.isin(np.asarray(r["apt"].to_list()), [s for s in a.skip.split(",") if s])
add[skip] = 0.0
new = Z[a.blend].astype(float).copy()
new[pos] = np.maximum(fin + add, 0.0)
Z["both_guard_prers"] = Z[a.blend]
Z[a.blend] = new
np.savez(a.out, **Z)
e0, e1 = (fin - y) ** 2, (new[pos] - y) ** 2
N = len(Z["y"])
R0 = float(np.sqrt(((Z["both_guard_prers"] - Z["y"]) ** 2).mean()))
R1 = float(np.sqrt(((new - Z["y"]) ** 2).mean()))
print(f"re-stack ({a.feats}, {a.folds} day-folds, skip '{a.skip}'): moved {int((add != 0).sum()):,} rows, sum dp^2 "
      f"{float((add ** 2).sum()):.3e}; matched-row dSSE {float(e1.sum() - e0.sum()):.3e}; all {N:,} rows "
      f"{R0:.3f} -> {R1:.3f} ({R1 - R0:+.3f} s); days in rows {len(ud)}; {time.time() - T0:.0f}s -> {a.out}")
