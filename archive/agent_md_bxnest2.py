"""experiment md bxnest2 (docs/LAB-NOTES.md): COPY of archive/agent_md_bxnest.py (frac/sub-seed dropped: always
the full fold) that ADDS training-only rows from the 116 new days. For outer fold k the re-stack trains on the 238-day rows
with fold != k (exactly as before) PLUS every new-day row that has a stage output; the new rows are never test rows.
New rows' bx columns: per outer fold k, E1/E2 (the agent_backx_08 definitions below) fitted on the 238-day rows with
fold != k and a first fix, applied to the new rows (no test-fold label reaches them). With --new-fin-npz "" this is
agent_md_bxnest.py at --frac 1.0 bit for bit. Original docstring follows.

experiment md bxnest (docs/LAB-NOTES.md): COPY of archive/agent_restack6_bxnest.py adding ONLY
--fin-npz (as archive/agent_stagecap_bxnest.py; default TX_S6 = the original) and --frac / --sub-seed: the re-stack of
outer fold k is trained on the rows of a random round(frac*|D_k|) subset of fold k's training days D_k
(numpy default_rng([sub_seed, k]).permutation, nested across frac). --frac 1.0 = the original bit for bit. Evaluation rows,
the cached nested bx columns and the stage input are unchanged. Original docstring follows.

experiment restack6 bxnest: COPY of archive/agent_restack5_bxnest.py (docs/LAB-NOTES.md) importing
archive/agent_restack6_restack.py (tokens r1, r2, r3 alongside the existing big) in place of agent_restack5_bxrestack.
--cols defaults to the EXISTING cache data/interim/agent_restack5_bxnest_cols.npz (read-only reuse: the per-outer-fold
nested bx columns do not depend on the re-stack recipe or seed, so the expensive E1/E2 inner-fold loop is skipped
whenever that cache is found -- this run never recomputes it). With --recipe cap,evt,trk,bx,big --seed 0 this must equal
data/interim/agent_restack5_bxn_cap_evt_trk_bx_big_s0.npz (the identity check of restack6.md s0.1). Original docstring
follows.

experiment restack5 bxnest: COPY of archive/agent_backx_08_nested.py (docs/LAB-NOTES.md) with --recipe / --seed /
--out and archive/agent_restack5_bxrestack.py (token big). The per-outer-fold nested bx columns do not depend on the recipe
or the seed, so they are cached in --cols (computed on the first run, reused after). The post-hoc W/M diagnostic of the
original is dropped. With --recipe cap,evt,trk,bx --seed 0 this must equal agent_backx_BXnest_s0.npz. Original docstring
follows.

experiment backx 08: NESTED leakage audit of arm BX (docs/LAB-NOTES.md addendum, added after stage A, disclosed).

In the registered arm the bx columns of the re-stack's TRAINING rows (fold j != k) come from E1/E2 models fitted on all
folds != j -- which include the test fold k's labels (second-level stacking leak). Here, for each outer fold k, the bx
columns of every training row are recomputed from E1/E2 fitted WITHOUT fold k (inner 9-fold OOF over the other folds),
and the test rows keep the model fitted on all folds != k (as registered). The re-stack (recipe cap,evt,trk,bx, seed 0)
is then refitted per outer fold exactly as archive/agent_backx_03_fold.py does. Also prints where the arm's gain lives
(W vs M rows, by airport) for BX_s0 and for the nested arm.
Writes data/interim/agent_backx_BXnest_s0.npz.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")
import lightgbm as lgb
import numpy as np
import polars as pl

from archive.agent_restack6_restack import features, fit, load_rows, parse, predict

C = pl.col
ID = "MVT_ID_mvt"
I = "data/interim/"
T0 = time.time()
ap = argparse.ArgumentParser()
ap.add_argument("--recipe", default="cap,evt,trk,bx")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--cols", default="data/interim/agent_restack5_bxnest_cols.npz")
ap.add_argument("--fin-npz", default="data/interim/modal/TX_S6.npz")
ap.add_argument("--osm", action="store_true", help="arm D: append osm_dc, osm_dff (sb2 token osm)")
ap.add_argument("--osm-train", default="data/interim/agent_sb2_osm_2025.parquet")
ap.add_argument("--osm-new", default="data/interim/agent_md_osm_new.parquet")
ap.add_argument("--new-fin-npz", default="")
ap.add_argument("--new-rows", default=I + "agent_md_rows_new.parquet")
ap.add_argument("--new-evt", default=I + "agent_md_evt_new.parquet")
ap.add_argument("--new-trk", default=I + "agent_md_trk_new.parquet")
ap.add_argument("--new-seq", default=I + "agent_md_seq_new.parquet")
ap.add_argument("--new-ff", default=I + "agent_md_ff_new.parquet")
args = ap.parse_args()
RECIPE = args.recipe
assert "bx" in parse(RECIPE)
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
# ---- E1/E2 definitions: verbatim from archive/agent_backx_02_estimate.py
DB = [0, 30, 60, 100, 150, 250, 400, 600, 900, 1300, 1800, 2500, 3500, 1e12]
TTCLIP = 3600.0
GP = dict(objective="l1", n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
          subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=0)
GF = ["apt_i", "tx", "hour", "plan_f", "first_gs", "first_dstand", "ff_lat", "ff_lon", "ff_trk", "st_lat", "st_lon",
      "brg_st", "mv_brg", "mv_spd", "rel_mv", "n120", "seg_start_gap", "w_f"]
FFC = ["e_start", "first_gs", "first_dstand", "ff_lat", "ff_lon", "ff_trk", "st_lat", "st_lon", "brg_st", "mv_brg",
       "mv_dt", "mv_spd", "rel_mv", "n120"]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def mat(j):
    g = lambda c: j[c].cast(pl.Float64).fill_null(np.nan).to_numpy()  # noqa: E731
    X = {c: g(c) for c in ["hour", "first_gs", "first_dstand", "ff_lat", "ff_lon", "ff_trk", "st_lat", "st_lon", "brg_st",
                           "mv_brg", "mv_spd", "rel_mv", "n120", "seg_start_gap"]}
    X["apt_i"] = np.array([APTS.index(a) if a in APTS else -1 for a in j["apt"].to_list()], float)
    X["tx"] = (j["src"] == "taxinm").cast(pl.Float64).to_numpy()
    X["plan_f"] = j["plan"].cast(pl.Float64).to_numpy()
    X["w_f"] = j["prod_t"].cast(pl.Float64).fill_null(np.nan).is_nan().cast(pl.Float64).to_numpy()
    return np.column_stack([X[c] for c in GF])


def dbin(d):
    b = np.digitize(np.nan_to_num(d, nan=-1.0), DB) - 1
    return np.where(np.isnan(d), -1, b)


def cell_fit(apt, db, tt):
    df = pl.DataFrame({"apt": apt, "db": db, "tt": tt})
    c = df.group_by(["apt", "db"]).agg(C("tt").median().alias("m"), pl.len().alias("n")).filter(C("n") >= 30)
    a = df.group_by("apt").agg(C("tt").median().alias("ma"))
    return c, a


def cell_apply(model, apt, db):
    c, a = model
    q = pl.DataFrame({"apt": apt, "db": db}).with_row_index("_i")
    q = q.join(c, on=["apt", "db"], how="left").join(a, on="apt", how="left").sort("_i")
    return q.select(pl.coalesce(C("m"), C("ma"))).to_series().cast(pl.Float64).fill_null(np.nan).to_numpy()



def osm_cols(frame, path):
    """moredays arm D: osm_dc, osm_dff (agent_sb2_common.OSM) aligned with frame's ids, as agent_sb2_common.extra_cols."""
    x = pl.read_parquet(path, columns=["MVT_ID_mvt", "osm_dc", "osm_dff"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
    assert x["MVT_ID_mvt"].n_unique() == x.height
    j = frame.select(pl.col("MVT_ID_mvt").cast(pl.Float64)).join(x, on="MVT_ID_mvt", how="left", maintain_order="left")
    return np.column_stack([j[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in ("osm_dc", "osm_dff")])


# ---- re-stack rows, exactly as archive/agent_backx_03_fold.py
z = np.load(args.fin_npz, allow_pickle=True)
Z = {k: z[k] for k in z.keys()}
ids = Z["ids"].astype(float)
pos_of = {k: i for i, k in enumerate(ids)}
r = load_rows("data/interim/agent_offblock_rows_c238.parquet", RECIPE, "data/interim/agent_restack3_evt_2025.parquet",
              "data/interim/agent_restack3_trk_2025.parquet", "data/interim/agent_smallbundle_seq_2025.parquet",
              "data/interim/agent_neighbour_nbr_2025.parquet", I + "agent_backx_bx_2025.parquet")
pos = np.array([pos_of.get(m, -1) for m in r[ID].to_list()])
assert (pos >= 0).all()
fin = Z["both_guard"][pos].astype(float)
y = Z["y"][pos].astype(float)
d_npz = Z["days"][pos].astype(str)
ud = sorted(set(d_npz))
fold = {d: i % 10 for i, d in enumerate(ud)}
fo = np.array([fold[d] for d in d_npz])



# ---- E1/E2 inputs aligned with r
rr = pl.read_parquet("data/interim/agent_offblock_rows_c238.parquet",
                     columns=[ID, "apt", "src", "plan", "hour", "block_t", "prod_t", "e_start", "first_gs", "first_dstand",
                              "seg_start_gap"]).with_columns(C(ID).cast(pl.Float64))
ff = (pl.read_parquet(I + "agent_backx_ff_2025.parquet").with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
      .select([ID] + FFC).drop(["e_start", "first_gs", "first_dstand"]))
j = r.select(ID).join(rr, on=ID, how="left", maintain_order="left").join(ff, on=ID, how="left", maintain_order="left")
assert j.height == r.height and (j[ID].to_numpy() == r[ID].to_numpy()).all()
es = j["e_start"].cast(pl.Float64).fill_null(np.nan).to_numpy()
blk = j["block_t"].to_numpy()
has = np.isfinite(es)
tt = es - blk
X2 = mat(j)
apt = np.asarray(j["apt"].to_list())
db = dbin(j["first_dstand"].cast(pl.Float64).fill_null(np.nan).to_numpy())
W = X2[:, GF.index("w_f")] == 1.0
bxc0 = r["bx_c"].cast(pl.Float64).fill_null(np.nan).to_numpy()
bxg0 = r["bx_g"].cast(pl.Float64).fill_null(np.nan).to_numpy()
log(f"rows {r.height:,}; with a first fix {int(has.sum()):,}; W {int(W.sum()):,}")

# ---- NEW-day rows (training only)
NN = 0
if args.new_fin_npz:
    zn = np.load(args.new_fin_npz, allow_pickle=True)
    npos_of = {k: i for i, k in enumerate(zn["ids"].astype(float))}
    rn = load_rows(args.new_rows, ",".join(t_ for t_ in RECIPE.split(",") if t_ != "bx"), args.new_evt, args.new_trk,
                   args.new_seq, None, None)
    npos = np.array([npos_of.get(m, -1) for m in rn[ID].to_list()])
    log(f"new rows {rn.height:,}; with a stage output {int((npos >= 0).sum()):,}")
    rn = rn.filter(pl.Series(npos >= 0))
    npos = npos[npos >= 0]
    assert not set(rn[ID].to_list()) & set(r[ID].to_list()), "a new row is also a 238-day row"
    rn = rn.with_columns(pl.lit(None, pl.Float64).alias("bx_c"), pl.lit(None, pl.Float64).alias("bx_g")).select(r.columns)
    fin_n = zn["both_guard"][npos].astype(float)
    y_n = zn["y"][npos].astype(float)
    rrn = pl.read_parquet(args.new_rows, columns=rr.columns).with_columns(C(ID).cast(pl.Float64))
    ffn = (pl.read_parquet(args.new_ff).with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
           .select([ID] + FFC).drop(["e_start", "first_gs", "first_dstand"]))
    jn = rn.select(ID).join(rrn, on=ID, how="left", maintain_order="left").join(ffn, on=ID, how="left", maintain_order="left")
    assert jn.height == rn.height and (jn[ID].to_numpy() == rn[ID].to_numpy()).all()
    es_n = jn["e_start"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    has_n = np.isfinite(es_n)
    X2n = mat(jn)
    apt_n = np.asarray(jn["apt"].to_list())
    db_n = dbin(jn["first_dstand"].cast(pl.Float64).fill_null(np.nan).to_numpy())
    NN = rn.height
    log(f"new training rows {NN:,}; with a first fix {int(has_n.sum()):,}; days {rn['day'].n_unique()}")

t = y - fin
add = np.zeros(len(t))
rid = r[ID].to_numpy()
cache = None
if os.path.exists(args.cols):
    cz = np.load(args.cols)
    assert (cz["ids"] == rid).all(), "bx column cache: row order differs"
    cache = {k: (cz[f"bc{k}"], cz[f"bg{k}"]) for k in range(10)}
    log(f"nested bx columns loaded from {args.cols}")
newcache = {}
for k in range(10):
    te = fo == k
    bc, bg = bxc0.copy(), bxg0.copy()          # test rows (fold k): the registered OOF values (models fitted on folds != k)
    for jj in (range(10) if cache is None else []):
        if jj == k:
            continue
        itr = (fo != k) & (fo != jj) & has
        ite = (fo == jj) & has
        bc[ite] = es[ite] - cell_apply(cell_fit(apt[itr], db[itr], tt[itr]), apt[ite], db[ite])
        m = lgb.LGBMRegressor(**GP).fit(X2[itr], np.clip(tt[itr], -TTCLIP, TTCLIP), categorical_feature=[0])
        bg[ite] = es[ite] - m.predict(X2[ite])
    if cache is not None:
        bc, bg = cache[k]
    else:
        newcache[f"bc{k}"], newcache[f"bg{k}"] = bc, bg
    rk = r.with_columns(pl.Series("bx_c", bc), pl.Series("bx_g", bg))
    X = features(rk, fin, RECIPE)
    if args.osm:
        X = np.column_stack([X, osm_cols(r, args.osm_train)])
    trm = fo != k
    if NN:
        itr = (fo != k) & has
        bcn, bgn = np.full(NN, np.nan), np.full(NN, np.nan)
        bcn[has_n] = es_n[has_n] - cell_apply(cell_fit(apt[itr], db[itr], tt[itr]), apt_n[has_n], db_n[has_n])
        mE2 = lgb.LGBMRegressor(**GP).fit(X2[itr], np.clip(tt[itr], -TTCLIP, TTCLIP), categorical_feature=[0])
        bgn[has_n] = es_n[has_n] - mE2.predict(X2n[has_n])
        Xn = features(rn.with_columns(pl.Series("bx_c", bcn), pl.Series("bx_g", bgn)), fin_n, RECIPE)
        if args.osm:
            Xn = np.column_stack([Xn, osm_cols(rn, args.osm_new)])
        Xtr, ttr = np.vstack([X[trm], Xn]), np.concatenate([t[trm], y_n - fin_n])
        del Xn
    else:
        Xtr, ttr = X[trm], t[trm]
    log(f"outer fold {k}: training rows {len(ttr):,} (238-day {int(trm.sum()):,} + new {NN:,})")
    add[te] = predict(fit(Xtr, ttr, RECIPE, args.seed), X[te])
    del X, rk, Xtr, ttr
    log(f"outer fold {k} done; training-row bx_g changed rms "
        f"{np.sqrt(np.nanmean((bg[~te & has] - bxg0[~te & has]) ** 2)):.1f} s")
new = Z["both_guard"].astype(float).copy()
new[pos] = np.maximum(fin + add, 0.0)
Z["both_guard_prers"] = Z["both_guard"]
Z["both_guard"] = new
np.savez(args.out, **Z)
if cache is None:
    np.savez(args.cols, ids=rid, **newcache)
    log(f"cached nested bx columns -> {args.cols}")
N = len(Z["y"])
R0 = float(np.sqrt(((Z["both_guard_prers"] - Z["y"]) ** 2).mean()))
R1 = float(np.sqrt(((new - Z["y"]) ** 2).mean()))
log(f"nested re-stack recipe {RECIPE} seed {args.seed}: all {N:,} rows {R0:.3f} -> {R1:.3f} -> {args.out}")
