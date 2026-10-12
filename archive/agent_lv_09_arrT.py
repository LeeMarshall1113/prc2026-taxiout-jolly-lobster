"""lv 09: arrival-to-departure transfer columns (group T; second-opinion hypothesis, docs/LAB-NOTES.md).
Taxi-IN model (LightGBM; runway, stand, operator, type, hour, dow) on LTFM arrivals, CROSS-FITTED BY MONTH (arrivals of month m are predicted by a model not trained on m);
2026 arrivals (labelled in ranking.parquet) are predicted by the model trained on all of 2025.  Residual = taxi-in - prediction.
For each departure (take-off t, runway side, stand area): over arrivals landing in (t-W, t]:
  t0 mean resid 30 min, t1 mean 60 min, t2 median 60 min, t3 count 60 min, t4 mean 60 min on the SAME runway side (W/C/E), t5 mean 60 min at the SAME stand area, t6 count same area 60 min.  NaN if none.
Writes data/interim/agent_lv_arrT.npz (fold rows = agent_lv_keys12_full order, serve rows = agent_bndl_rank_ltfm order)."""
import glob
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "archive")
from agent_ltfm_seqfeat import RWY_GROUP  # noqa: E402

I = "data/interim/"; ID = "MVT_ID_mvt"; C = pl.col
cols = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "TAXITIME_SEC_mvt"]
area = lambda c: C(c).cast(pl.Utf8).str.extract(r"^([A-Za-z]+)", 1).fill_null("NA")


def load(files):
    return pl.concat([pl.read_parquet(f, columns=cols).filter((C("ADEP_mvt") == "LTFM") | (C("ADES_mvt") == "LTFM")) for f in files])


def arrivals(d):
    a = d.filter((C("PHASE_mvt") == "ARR") & (C("ADES_mvt") == "LTFM")).select(
        C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"), C("MVT_TIME_UTC_mvt").dt.month().alias("m"), C("MVT_TIME_UTC_mvt").dt.hour().alias("hr"), C("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
        C("RUNWAY_mvt").cast(pl.Utf8).fill_null("NA").alias("rw"), C("STAND_mvt").cast(pl.Utf8).fill_null("NA").alias("st"), C("AIRCRAFT_OPERATOR_flt").cast(pl.Utf8).fill_null("NA").alias("op"),
        C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).fill_null("NA").alias("ty"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"))
    return a.drop_nulls(["t", "y"]).filter((C("y") > 30) & (C("y") < 3600)).sort("t").with_columns(area("st").alias("ar"))


CATS = ["rw", "st", "op", "ty"]


def design(a, levels=None):
    X = a.select(CATS + ["hr", "dow"]).to_pandas()
    for c in CATS:
        if levels is None: X[c] = X[c].astype("category")
        else: X[c] = pd.Categorical(X[c], categories=levels[c])
    return X


def fit(a):
    X = design(a); lv = {c: X[c].cat.categories for c in CATS}
    g = lgb.LGBMRegressor(objective="regression", n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbose=-1)
    g.fit(X, a["y"].to_numpy()); return g, lv


def predict(g, lv, a): return g.predict(design(a, lv))


def deps_cols(d, ids):
    dep = d.filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")).select(ID, C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"), C("RUNWAY_mvt").cast(pl.Utf8).alias("rw"), C("STAND_mvt").cast(pl.Utf8).alias("st"))
    return pl.DataFrame({ID: ids}).join(dep, on=ID, how="left", maintain_order="left").with_columns(area("st").alias("ar"))


def feats(dep, a, resid):
    t = dep["t"].fill_null(0).to_numpy().astype(np.int64); n = len(t); out = np.full((n, 7), np.nan)
    ta = a["t"].to_numpy(); side_a = np.array([RWY_GROUP.get(r, "?") for r in a["rw"].to_numpy()]); ar_a = a["ar"].to_numpy()
    side_d = np.array([RWY_GROUP.get(r, "?") for r in dep["rw"].fill_null("NA").to_numpy()]); ar_d = dep["ar"].to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(resid)])
    def win(W): return np.searchsorted(ta, t - W, side="right"), np.searchsorted(ta, t, side="right")
    lo, hi = win(1800); c = hi - lo; out[:, 0] = np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)
    lo, hi = win(3600); c = hi - lo; out[:, 1] = np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan); out[:, 3] = c
    for i in np.flatnonzero(c > 0): out[i, 2] = np.median(resid[lo[i]:hi[i]])
    for col, key_a, key_d in ((4, side_a, side_d), (5, ar_a, ar_d)):
        for k in np.unique(key_d):
            i = np.flatnonzero(key_d == k); idx = np.flatnonzero(key_a == k)
            if len(idx) == 0: continue
            tk = ta[idx]; ck = np.concatenate([[0.0], np.cumsum(resid[idx])])
            l2 = np.searchsorted(tk, t[i] - 3600, side="right"); h2 = np.searchsorted(tk, t[i], side="right"); cc = h2 - l2
            out[i, col] = np.where(cc > 0, (ck[h2] - ck[l2]) / np.maximum(cc, 1), np.nan)
            if col == 5: out[i, 6] = cc
    return out


if __name__ == "__main__":
    d25 = load(sorted(glob.glob("data/raw/training_2025-*.parquet"))); a25 = arrivals(d25)
    print("2025 arrivals", a25.height)
    res25 = np.zeros(a25.height); m = a25["m"].to_numpy(); y = a25["y"].to_numpy()
    for mo in range(1, 13):
        tr, te = m != mo, m == mo
        if te.sum() == 0: continue
        g, lv = fit(a25.filter(pl.Series(tr))); p = predict(g, lv, a25.filter(pl.Series(te))); res25[te] = y[te] - p
        print(f"  month {mo}: taxi-in model trained without it: RMSE {np.sqrt(np.mean((y[te] - p) ** 2)):.1f} s (mean resid {np.mean(y[te] - p):+.1f})", flush=True)
    g, lv = fit(a25)
    r26 = pl.read_parquet("data/raw/ranking.parquet", columns=cols); r26 = r26.filter((C("ADEP_mvt") == "LTFM") | (C("ADES_mvt") == "LTFM")); a26 = arrivals(r26)
    res26 = a26["y"].to_numpy() - predict(g, lv, a26); print("2026 arrivals", a26.height, "mean resid", float(res26.mean()), "rmse", float(np.sqrt((res26 ** 2).mean())))
    z = np.load(I + "agent_lv_keys12_full.npz", allow_pickle=True)
    fold = feats(deps_cols(d25, z["fid"]), a25, res25)
    rl = pl.read_parquet(I + "agent_bndl_rank_ltfm.parquet").select(ID)
    serve = feats(deps_cols(r26, rl[ID].to_numpy()), a26, res26)
    names = ["t_mean30", "t_mean60", "t_med60", "t_cnt60", "t_side60", "t_area60", "t_areacnt60"]
    for j, nm in enumerate(names):
        print(f"  {nm:12s} fold mean {np.nanmean(fold[:, j]):7.2f} sd {np.nanstd(fold[:, j]):7.2f} cov {np.isfinite(fold[:, j]).mean():.3f} | serve mean {np.nanmean(serve[:, j]):7.2f} sd {np.nanstd(serve[:, j]):7.2f} cov {np.isfinite(serve[:, j]).mean():.3f}")
    yy, ref = z["yf"], z["ref"]; r = np.clip(yy - ref, -900, 900)
    for j, nm in enumerate(names):
        ok = np.isfinite(fold[:, j]); print(f"  corr(dep residual, {nm}) {np.corrcoef(fold[ok, j], r[ok])[0, 1]:+.4f}")
    np.savez(I + "agent_lv_arrT.npz", fold=fold.astype(np.float32), serve=serve.astype(np.float32), names=np.array(names), fold_ids=z["fid"], serve_ids=rl[ID].to_numpy())
