"""experiment sb2 fold4: COPY of archive/agent_restack6_fold4.py (docs/LAB-NOTES.md) importing
archive/agent_sb2_restack4.py (tokens mw, lw, unc, osm, occ) and adding --osm/--occ/--unc-npz as agent_sb2_bxnest.py.
Without the new tokens this is agent_restack6_fold4.py bit for bit. Original docstring follows.

experiment restack6 fold4: COPY of archive/agent_restack4_fold.py (docs/LAB-NOTES.md) importing
archive/agent_restack6_restack4.py (tokens r1, r2, r3 alongside the existing big) in place of agent_restack4_restack. Used
for the non-bx LFPG/LIRF recipe (cap,evt,trk,seq,r{1,2,3}). With --recipe cap,evt,trk,seq,big --seed 0 this must equal
data/interim/agent_restack5_f_cap_evt_trk_seq_big_s0.npz (the identity check of restack6.md s0.1). Original docstring
follows.

experiment restack4 fold: COPY of archive/agent_neighbour_05_restack_fold.py using archive/agent_restack4_restack.py (tokens
seq, seqm, arr, big, cb; --seq/--arr parquets) and ALSO storing the out-of-fold correction `rs_add` (0 off the dump) and
the dump mask `rs_dump` in the npz, for the diversity gate and the blend of docs/LAB-NOTES.md.
With recipe cap,evt,trk,seq it must equal agent_smallbundle_B.npz, with cap,evt,trk,nbr agent_neighbour_B.npz (s0.2).
Token cb: CatBoost instead of LightGBM; if the first fold takes > --cb-guard-s seconds the run stops (exit 3) before any
prediction is written (s0.3 compute guard). Original docstring follows.

experiment restack3 03: COPY of archive/agent_expensive_03_restack_fold.py with the pre-registered recipes of
docs/LAB-NOTES.md (archive/agent_restack3_restack.py). Recipe "" = RS_v2's est_v2 re-stack, bit for bit.
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np

def _et_add(Xtr, ttr, Xte, seed):
    from sklearn.ensemble import ExtraTreesRegressor
    A = np.nan_to_num(Xtr.astype(np.float32), nan=-99999.0)   # sklearn casts to float32 anyway; same values, less commit
    m = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, n_jobs=4, random_state=seed).fit(A, ttr)
    del A
    B = np.nan_to_num(Xte.astype(np.float32), nan=-99999.0)
    return np.clip(m.predict(B), -1800.0, 1800.0)

import polars as pl

from archive.agent_sb2_restack4 import features, fit, load_rows, parse, predict
from archive.agent_sb2_common import weights

ap = argparse.ArgumentParser()
ap.add_argument("--fin-npz", default="data/interim/modal/TX_S6.npz")
ap.add_argument("--blend", default="both_guard")
ap.add_argument("--rows", default="data/interim/agent_offblock_rows_c238.parquet")
ap.add_argument("--evt", default="data/interim/agent_restack3_evt_2025.parquet")
ap.add_argument("--trk", default="data/interim/agent_restack3_trk_2025.parquet")
ap.add_argument("--nbr", default="data/interim/agent_neighbour_nbr_2025.parquet")
ap.add_argument("--seq", default="data/interim/agent_smallbundle_seq_2025.parquet")
ap.add_argument("--arr", default="data/interim/agent_restack4_arr_2025.parquet")
ap.add_argument("--recipe", default="")
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cb-guard-s", type=float, default=900.0)
ap.add_argument("--out", required=True)
ap.add_argument("--osm", default="data/interim/agent_sb2_osm_2025.parquet")
ap.add_argument("--occ", default="data/interim/agent_sb2_occ_2025.parquet")
ap.add_argument("--unc-npz", default="")
ap.add_argument("--nov-fold", default="data/interim/agent_nov_fold.parquet")
a = ap.parse_args()
T0 = time.time()

z = np.load(a.fin_npz, allow_pickle=True)
Z = {k: z[k] for k in z.keys()}
ids = Z["ids"].astype(float)
pos_of = {k: i for i, k in enumerate(ids)}
assert len(pos_of) == len(ids), "npz ids not unique"
r = load_rows(a.rows, a.recipe, a.evt, a.trk, a.nbr, a.seq, a.arr, extra={"osm": a.osm, "occ": a.occ})
RID0 = r["MVT_ID_mvt"].to_numpy()
pos = np.array([pos_of.get(m, -1) for m in r["MVT_ID_mvt"].to_list()])
print(f"rows {r.height:,}; in npz {int((pos >= 0).sum()):,}; recipe '{a.recipe}'")
r = r.filter(pl.Series(pos >= 0))
pos = pos[pos >= 0]
fin = Z[a.blend][pos].astype(float)
y = Z["y"][pos].astype(float)
d_npz = Z["days"][pos].astype(str)
assert (d_npz == np.asarray(r["day"].to_list())).all(), "day mismatch between rows and npz"
TOKS = parse(a.recipe)
WALL = weights(r, TOKS)
UNC = None
if "unc" in TOKS:
    uz = np.load(a.unc_npz)
    uid = uz["ids"]
    upos = {v: i for i, v in enumerate(uid)}
    ui = np.array([upos[v] for v in r["MVT_ID_mvt"].to_numpy()])
    UNC = {k: uz[f"w{k}"][ui] for k in range(10)}
    r = r.with_columns(pl.Series("unc_w", UNC[0]))
NOV = None
if "nov" in TOKS:
    from archive.agent_nov_lib import nov_folds
    NOV = nov_folds(a.nov_fold, r["MVT_ID_mvt"].to_numpy())
    r = r.with_columns(pl.Series("nov_w", NOV[0]))
X = features(r, fin, a.recipe)
print(f"X {X.shape}; non-null share of the added columns: "
      + " ".join(f"{np.isfinite(X[:, j]).mean():.2f}" for j in range(36, X.shape[1])), flush=True)
t = y - fin
ud = sorted(set(d_npz))
fold = {d: i % a.folds for i, d in enumerate(ud)}
fo = np.array([fold[d] for d in d_npz])
add = np.zeros(len(t))
add_e = np.zeros(len(t))
cb = "cb" in parse(a.recipe)
for k in range(a.folds):
    tr, te = fo != k, fo == k
    if te.any():
        t1 = time.time()
        if UNC is not None or NOV is not None:
            rk = r
            if UNC is not None:
                rk = rk.with_columns(pl.Series("unc_w", UNC[k]))
            if NOV is not None:
                rk = rk.with_columns(pl.Series("nov_w", NOV[k]))
            X = features(rk, fin, a.recipe)
            del rk
        add[te] = predict(fit(X[tr], t[tr], a.recipe, a.seed, w=None if WALL is None else WALL[tr]), X[te])
        add_e[te] = _et_add(X[tr], t[tr], X[te], a.seed)
        if cb and k == 0 and time.time() - t1 > a.cb_guard_s:
            print(f"CB GUARD: first fold took {time.time() - t1:.0f}s > {a.cb_guard_s:.0f}s -> V4 dropped before any number exists")
            sys.exit(3)
    print(f"  fold {k} done ({time.time() - T0:.0f}s)", flush=True)
new = Z[a.blend].astype(float).copy()
new[pos] = np.maximum(fin + add, 0.0)
Z["both_guard_prers"] = Z[a.blend]
Z[a.blend] = new
full_add = np.zeros(len(new))
full_add[pos] = add
dump = np.zeros(len(new), bool)
dump[pos] = True
Z["rs_add"], Z["rs_dump"] = full_add, dump
np.savez(a.out, **Z)
Za = dict(Z)
new_e = Z['both_guard_prers'].astype(float).copy(); new_e[pos] = np.maximum(fin + 0.5 * (add + add_e), 0.0)
Za['both_guard'] = new_e; Za['et_add'] = add_e
np.savez(a.out.replace('.npz', '_avg.npz'), **Za)
N = len(Z["y"])
R0 = float(np.sqrt(((Z["both_guard_prers"] - Z["y"]) ** 2).mean()))
R1 = float(np.sqrt(((new - Z["y"]) ** 2).mean()))
print(f"re-stack recipe '{a.recipe}' seed {a.seed}: moved {int((add != 0).sum()):,} rows, sum dp^2 "
      f"{float((add ** 2).sum()):.3e}; all {N:,} rows {R0:.3f} -> {R1:.3f} ({R1 - R0:+.3f} s vs fin); "
      f"{time.time() - T0:.0f}s -> {a.out}")
