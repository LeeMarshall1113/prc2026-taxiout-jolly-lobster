"""experiment backx 08: NESTED leakage audit of arm BX (docs/LAB-NOTES.md addendum, added after stage A, disclosed).

In the registered arm the bx columns of the re-stack's TRAINING rows (fold j != k) come from E1/E2 models fitted on all
folds != j -- which include the test fold k's labels (second-level stacking leak). Here, for each outer fold k, the bx
columns of every training row are recomputed from E1/E2 fitted WITHOUT fold k (inner 9-fold OOF over the other folds),
and the test rows keep the model fitted on all folds != k (as registered). The re-stack (recipe cap,evt,trk,bx, seed 0)
is then refitted per outer fold exactly as archive/agent_backx_03_fold.py does. Also prints where the arm's gain lives
(W vs M rows, by airport) for BX_s0 and for the nested arm.
Writes data/interim/agent_backx_BXnest_s0.npz.
"""
import sys
import time

sys.path.insert(0, ".")
import lightgbm as lgb
import numpy as np
import polars as pl

from archive.agent_backx_restack import features, fit, load_rows, predict

C = pl.col
ID = "MVT_ID_mvt"
I = "data/interim/"
T0 = time.time()
RECIPE = "cap,evt,trk,bx"
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


# ---- re-stack rows, exactly as archive/agent_backx_03_fold.py
z = np.load("data/interim/modal/TX_S6.npz", allow_pickle=True)
Z = {k: z[k] for k in z.keys()}
ids = Z["ids"].astype(float)
pos_of = {k: i for i, k in enumerate(ids)}
r = load_rows("data/interim/agent_offblock_rows_c238.parquet", RECIPE, "data/interim/agent_restack3_evt_2025.parquet",
              "data/interim/agent_restack3_trk_2025.parquet", None, None, I + "agent_backx_bx_2025.parquet")
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

t = y - fin
add = np.zeros(len(t))
for k in range(10):
    te = fo == k
    bc, bg = bxc0.copy(), bxg0.copy()          # test rows (fold k): the registered OOF values (models fitted on folds != k)
    for jj in range(10):
        if jj == k:
            continue
        itr = (fo != k) & (fo != jj) & has
        ite = (fo == jj) & has
        bc[ite] = es[ite] - cell_apply(cell_fit(apt[itr], db[itr], tt[itr]), apt[ite], db[ite])
        m = lgb.LGBMRegressor(**GP).fit(X2[itr], np.clip(tt[itr], -TTCLIP, TTCLIP), categorical_feature=[0])
        bg[ite] = es[ite] - m.predict(X2[ite])
    rk = r.with_columns(pl.Series("bx_c", bc), pl.Series("bx_g", bg))
    X = features(rk, fin, RECIPE)
    add[te] = predict(fit(X[~te], t[~te], RECIPE, 0), X[te])
    log(f"outer fold {k} done; training-row bx_g changed rms "
        f"{np.sqrt(np.nanmean((bg[~te & has] - bxg0[~te & has]) ** 2)):.1f} s")
new = Z["both_guard"].astype(float).copy()
new[pos] = np.maximum(fin + add, 0.0)
Z["both_guard_prers"] = Z["both_guard"]
Z["both_guard"] = new
np.savez(I + "agent_backx_BXnest_s0.npz", **Z)
log("wrote data/interim/agent_backx_BXnest_s0.npz")

# ---- where the gain lives (vs C0 = CAB), BX_s0 and the nested arm
c0 = np.load(I + "agent_restack3_CAB.npz", allow_pickle=True)
assert (c0["ids"].astype(float) == ids).all()
cab = c0["both_guard"].astype(float)[pos]
bx0 = np.load(I + "agent_backx_BX_s0.npz", allow_pickle=True)["both_guard"].astype(float)[pos]
nest = new[pos]
N = len(Z["y"])
SSE = float(((c0["both_guard"].astype(float) - Z["y"]) ** 2).sum())
R = np.sqrt(SSE / N)
for nm, p in (("BX_s0", bx0), ("BXnest", nest)):
    g = (cab - y) ** 2 - (p - y) ** 2
    tot = np.sqrt((SSE - g.sum()) / N) - R
    part = lambda m: np.sqrt((SSE - g[m].sum()) / N) - R  # noqa: E731
    print(f"{nm}: all-rows d {tot:+.3f} s; W rows {part(W):+.3f} (W with fix {part(W & has):+.3f}, W no fix "
          f"{part(W & ~has):+.3f}); M rows {part(~W):+.3f}")
    print("   by airport (W / M): " + "  ".join(f"{a} {part(W & (apt == a)):+.2f}/{part(~W & (apt == a)):+.2f}" for a in APTS))
