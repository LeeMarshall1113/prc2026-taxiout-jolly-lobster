"""Back-extrapolated off-block estimates (stage 20, part e).

Derived from the 2026 half of archive/agent_backx_02_estimate.py. Two estimators of the time from the recorded off-block to
the first ground fix of the departure segment, both fitted on ALL 2025 dump rows (stored files data/interim/
agent_offblock_rows_c238.parquet and agent_backx_ff_2025.parquet, not rebuilt) and applied to the rows of the given dump
files:  bx_c = cell median per (airport, first-fix-to-stand distance bucket);  bx_g = LightGBM (L1) on first-fix geometry.
Changes against the shipped script: the 2025 out-of-fold part and its printed measurements are dropped; the dump files are
arguments. Run from the repo root.

    python final/_dump_bx.py --rows FILE_rows --ff FILE_ff --out FILE_bx
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import lightgbm as lgb
import numpy as np
import polars as pl

C = pl.col
ID = "MVT_ID_mvt"
I = "data/interim/"
T0 = time.time()
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
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


def load(rows_p, ff_p):
    r = pl.read_parquet(rows_p, columns=[ID, "day", "apt", "src", "plan", "hour", "mvt_t", "block_t", "prod_kind",
                                         "prod_t", "e_start", "first_gs", "first_dstand", "seg_start_gap"])
    r = r.with_columns(C(ID).cast(pl.Float64))
    assert r[ID].n_unique() == r.height
    f = pl.read_parquet(ff_p).with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
    f = f.select([ID] + FFC).rename({"e_start": "ff_e_start", "first_gs": "ff_first_gs", "first_dstand": "ff_first_dstand"})
    j = r.join(f, on=ID, how="left")
    assert j.height == r.height
    return j


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--ff", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    j = load(I + "agent_offblock_rows_c238.parquet", I + "agent_backx_ff_2025.parquet")
    es = j["e_start"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    blk = j["block_t"].to_numpy()          # 2025 training labels (fold side)
    has = np.isfinite(es)
    apt = np.asarray(j["apt"].to_list())
    tt = es - blk
    X = mat(j)
    db = dbin(j["first_dstand"].cast(pl.Float64).fill_null(np.nan).to_numpy())
    cm_all = cell_fit(apt[has], db[has], tt[has])
    fi = lgb.LGBMRegressor(**GP).fit(X[has], np.clip(tt[has], -TTCLIP, TTCLIP), categorical_feature=[0])
    log(f"fitted on {int(has.sum()):,} 2025 rows")
    j6 = load(args.rows, args.ff)
    es6 = j6["e_start"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    has6 = np.isfinite(es6)
    X6 = mat(j6)
    apt6 = np.asarray(j6["apt"].to_list())
    tc6 = np.full(j6.height, np.nan)
    tg6 = np.full(j6.height, np.nan)
    tc6[has6] = cell_apply(cm_all, apt6[has6], dbin(X6[has6, 5]))
    tg6[has6] = fi.predict(X6[has6])
    pl.DataFrame({ID: j6[ID], "bx_c": es6 - tc6, "bx_g": es6 - tg6, "tt_g": tg6}).write_parquet(args.out)
    log(f"wrote {j6.height:,} rows, with a first fix {int(has6.sum()):,} -> {args.out}")
