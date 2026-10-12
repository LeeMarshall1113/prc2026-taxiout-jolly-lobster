"""experiment smallbundle 03: COPY of archive/agent_restack3_03_fold.py (docs/LAB-NOTES.md) using
archive/agent_smallbundle_restack.py (adds the `seq` token) and, per fold, ALSO predicting the airports in --alt-apts with
the test-row fin taken from --alt-npz (the serve-side splice of another stage-1 path; training fins unchanged).

    python archive/agent_smallbundle_03_fold.py --recipe cap,evt,trk,seq --out X.npz --alt-npz data/interim/modal/TX.npz \
        --alt-apts LEBL --alt-out X_lebl.npz

--out     : exactly agent_restack3_03_fold.py's output for the recipe (with seq support).
--alt-out : same models; rows at --alt-apts take the alt npz's both_guard, and their dump rows get alt fin + re-stack(alt fin).
Both npz keep both_guard_prers = the TX_S6 fin (for the LIRF revert).
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_smallbundle_restack import features, fit, load_rows, predict

ap = argparse.ArgumentParser()
ap.add_argument("--fin-npz", default="data/interim/modal/TX_S6.npz")
ap.add_argument("--blend", default="both_guard")
ap.add_argument("--rows", default="data/interim/agent_offblock_rows_c238.parquet")
ap.add_argument("--evt", default="data/interim/agent_restack3_evt_2025.parquet")
ap.add_argument("--trk", default="data/interim/agent_restack3_trk_2025.parquet")
ap.add_argument("--seq", default="data/interim/agent_smallbundle_seq_2025.parquet")
ap.add_argument("--recipe", default="")
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--alt-npz", default="")
ap.add_argument("--alt-apts", default="")
ap.add_argument("--alt-out", default="")
a = ap.parse_args()
T0 = time.time()

z = np.load(a.fin_npz, allow_pickle=True)
Z = {k: z[k] for k in z.keys()}
ids = Z["ids"].astype(float)
pos_of = {k: i for i, k in enumerate(ids)}
assert len(pos_of) == len(ids), "npz ids not unique"
r = load_rows(a.rows, a.recipe, a.evt, a.trk, a.seq)
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
alt = bool(a.alt_npz and a.alt_apts and a.alt_out)
if alt:
    za = np.load(a.alt_npz, allow_pickle=True)
    zid = za["ids"].astype(float)
    oz = np.argsort(zid, kind="stable")
    ix = oz[np.searchsorted(zid[oz], ids)]
    assert (zid[ix] == ids).all(), "alt npz does not cover the fin npz ids"
    alt_full = za[a.blend].astype(float)[ix]   # aligned to the fin npz's row order
    ra = np.isin(np.asarray(r["apt"].to_list()), a.alt_apts.split(","))
    fin_alt = np.where(ra, alt_full[pos], fin)
    ria = np.flatnonzero(ra)
    Xa = features(r.filter(pl.Series(ra)), fin_alt[ra], a.recipe)   # row-wise features: only the alt rows are needed
    print(f"alt: {int(ra.sum()):,} dump rows at {a.alt_apts} use {a.alt_npz}; fin differs on {int((fin_alt != fin).sum()):,}")
t = y - fin
ud = sorted(set(d_npz))
fold = {d: i % a.folds for i, d in enumerate(ud)}
fo = np.array([fold[d] for d in d_npz])
add = np.zeros(len(t))
add_alt = np.zeros(len(t))
for k in range(a.folds):
    tr, te = fo != k, fo == k
    if te.any():
        m = fit(X[tr], t[tr], a.recipe, a.seed)
        add[te] = predict(m, X[te])
        if alt:
            kk = te[ria]
            if kk.any():
                add_alt[ria[kk]] = predict(m, Xa[kk])
    print(f"  fold {k} done ({time.time() - T0:.0f}s)", flush=True)
prers = Z[a.blend].astype(float).copy()
new = prers.copy()
new[pos] = np.maximum(fin + add, 0.0)
Z["both_guard_prers"] = prers
Z[a.blend] = new
np.savez(a.out, **Z)
N = len(Z["y"])
R0 = float(np.sqrt(((prers - Z["y"]) ** 2).mean()))
R1 = float(np.sqrt(((new - Z["y"]) ** 2).mean()))
print(f"re-stack recipe '{a.recipe}' seed {a.seed}: moved {int((add != 0).sum()):,} rows, sum dp^2 "
      f"{float((add ** 2).sum()):.3e}; all {N:,} rows {R0:.3f} -> {R1:.3f} ({R1 - R0:+.3f} s vs fin); "
      f"{time.time() - T0:.0f}s -> {a.out}")
if alt:
    apts_all = Z["apts"].astype(str)
    na = new.copy()
    at = np.isin(apts_all, a.alt_apts.split(","))
    na[at] = alt_full[at]                     # non-dump rows at the alt airports: the alt path's stage-1 value
    na[pos[ra]] = np.maximum(fin_alt[ra] + add_alt[ra], 0.0)
    Z[a.blend] = na
    np.savez(a.alt_out, **Z)
    print(f"alt arm: rows changed vs --out {int((na != new).sum()):,} (non-dump rows at alt apts changed "
          f"{int(((na != new) & at).sum() - (na[pos[ra]] != new[pos[ra]]).sum()):,}) -> {a.alt_out}")
