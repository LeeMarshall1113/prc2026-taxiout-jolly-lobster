"""experiment md fold4b (docs/LAB-NOTES.md): COPY of archive/agent_md_fold4.py (frac/sub-seed dropped) that ADDS
training-only rows from the 116 new days (--new-fin-npz + the new-day rows/evt/trk/seq parquets): fold k trains on the
238-day rows with fold != k PLUS every new-day row with a stage output; new rows are never test rows. With --new-fin-npz ""
this is agent_md_fold4.py at --frac 1.0 bit for bit. Original docstring follows.

experiment md fold4 (docs/LAB-NOTES.md): COPY of archive/agent_restack6_fold4.py adding ONLY --frac /
--sub-seed: the re-stack of fold k is trained on the rows of a random round(frac*|D_k|) subset of fold k's training days
D_k (numpy default_rng([sub_seed, k]).permutation, nested across frac). --frac 1.0 = the original bit for bit. Original
docstring follows.

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
import polars as pl

from archive.agent_restack6_restack4 import features, fit, load_rows, parse, predict


def osm_cols(frame, path):
    """moredays arm D: osm_dc, osm_dff (agent_sb2_common.OSM) aligned with frame's ids, as agent_sb2_common.extra_cols."""
    x = pl.read_parquet(path, columns=["MVT_ID_mvt", "osm_dc", "osm_dff"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
    assert x["MVT_ID_mvt"].n_unique() == x.height
    j = frame.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(x, on="MVT_ID_mvt", how="left", maintain_order="left")
    return np.column_stack([j[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in ("osm_dc", "osm_dff")])


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
ap.add_argument("--osm", action="store_true", help="arm D: append osm_dc, osm_dff (sb2 token osm)")
ap.add_argument("--osm-train", default="data/interim/agent_sb2_osm_2025.parquet")
ap.add_argument("--osm-new", default="data/interim/agent_md_osm_new.parquet")
ap.add_argument("--new-fin-npz", default="")
ap.add_argument("--new-rows", default="data/interim/agent_md_rows_new.parquet")
ap.add_argument("--new-evt", default="data/interim/agent_md_evt_new.parquet")
ap.add_argument("--new-trk", default="data/interim/agent_md_trk_new.parquet")
ap.add_argument("--new-seq", default="data/interim/agent_md_seq_new.parquet")
a = ap.parse_args()
T0 = time.time()

z = np.load(a.fin_npz, allow_pickle=True)
Z = {k: z[k] for k in z.keys()}
ids = Z["ids"].astype(float)
pos_of = {k: i for i, k in enumerate(ids)}
assert len(pos_of) == len(ids), "npz ids not unique"
r = load_rows(a.rows, a.recipe, a.evt, a.trk, a.nbr, a.seq, a.arr)
pos = np.array([pos_of.get(m, -1) for m in r["MVT_ID_mvt"].to_list()])
print(f"rows {r.height:,}; in npz {int((pos >= 0).sum()):,}; recipe '{a.recipe}'")
r = r.filter(pl.Series(pos >= 0))
pos = pos[pos >= 0]
fin = Z[a.blend][pos].astype(float)
y = Z["y"][pos].astype(float)
d_npz = Z["days"][pos].astype(str)
assert (d_npz == np.asarray(r["day"].to_list())).all(), "day mismatch between rows and npz"
X = features(r, fin, a.recipe)
if a.osm:
    X = np.column_stack([X, osm_cols(r, a.osm_train)])
print(f"X {X.shape}; non-null share of the added columns: "
      + " ".join(f"{np.isfinite(X[:, j]).mean():.2f}" for j in range(36, X.shape[1])), flush=True)
t = y - fin
NN = 0
if a.new_fin_npz:
    zn = np.load(a.new_fin_npz, allow_pickle=True)
    npos_of = {k: i for i, k in enumerate(zn["ids"].astype(float))}
    rn = load_rows(a.new_rows, a.recipe, a.new_evt, a.new_trk, a.nbr, a.new_seq, a.arr)
    npos = np.array([npos_of.get(m, -1) for m in rn["MVT_ID_mvt"].to_list()])
    print(f"new rows {rn.height:,}; with a stage output {int((npos >= 0).sum()):,}", flush=True)
    rn = rn.filter(pl.Series(npos >= 0))
    npos = npos[npos >= 0]
    assert not set(rn["MVT_ID_mvt"].to_list()) & set(r["MVT_ID_mvt"].to_list()), "a new row is also a 238-day row"
    fin_n = zn[a.blend][npos].astype(float)
    Xn = features(rn, fin_n, a.recipe)
    if a.osm:
        Xn = np.column_stack([Xn, osm_cols(rn, a.osm_new)])
    tn = zn["y"][npos].astype(float) - fin_n
    NN = len(tn)
    print(f"new training rows {NN:,} on {rn['day'].n_unique()} days", flush=True)
ud = sorted(set(d_npz))
fold = {d: i % a.folds for i, d in enumerate(ud)}
fo = np.array([fold[d] for d in d_npz])
add = np.zeros(len(t))
cb = "cb" in parse(a.recipe)
for k in range(a.folds):
    tr, te = fo != k, fo == k
    Xtr, ttr = (np.vstack([X[tr], Xn]), np.concatenate([t[tr], tn])) if NN else (X[tr], t[tr])
    print(f"  fold {k}: training rows {len(ttr):,} (238-day {int(tr.sum()):,} + new {NN:,})", flush=True)
    if te.any():
        t1 = time.time()
        add[te] = predict(fit(Xtr, ttr, a.recipe, a.seed), X[te])
        if cb and k == 0 and time.time() - t1 > a.cb_guard_s:
            print(f"CB GUARD: first fold took {time.time() - t1:.0f}s > {a.cb_guard_s:.0f}s -> V4 dropped before any number exists")
            sys.exit(3)
    del Xtr, ttr
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
N = len(Z["y"])
R0 = float(np.sqrt(((Z["both_guard_prers"] - Z["y"]) ** 2).mean()))
R1 = float(np.sqrt(((new - Z["y"]) ** 2).mean()))
print(f"re-stack recipe '{a.recipe}' seed {a.seed}: moved {int((add != 0).sum()):,} rows, sum dp^2 "
      f"{float((add ** 2).sum()):.3e}; all {N:,} rows {R0:.3f} -> {R1:.3f} ({R1 - R0:+.3f} s vs fin); "
      f"{time.time() - T0:.0f}s -> {a.out}")
