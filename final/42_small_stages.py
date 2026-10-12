"""Stage 42: the small stages for the departures of chosen months: M1, M2, M3, M4, the tail component; nudge and meta stay zero.

Derived from archive/agent_bndl_02_serve.py (M1 uplift GBM at half dose, M2 arrival-lateness ridge, M3 matched-row operator
offset, M4 Istanbul corrector at half dose) and archive/agent_tailc_03_serve.py (tail component). Every member is fitted on all of
2025 and applied to the served rows; fit recipes, seeds, doses and clips are those of the shipped scripts. The body of the
bundle script is copied with only these changes: the ranking file, month set and base file are arguments, the full-file row-count
assert is gone, the off-block dump of M3 and the Istanbul rank features are read from the final-phase files, and the outputs go
to data/final/.

  --base   parquet (MVT_ID_mvt, TAXITIME_SEC_mvt): the chain value of the served departures before the small correctors
           (the v137 of the shipped chain: re-stack, rules, Istanbul bulk and snow refits).
  Outputs  <tag>_deltas.npz  ids, M1, M2, M3, M4, nudge (0), meta (0), ehat (tail, 0 unless --v5-glob and --serve-in12)
           <tag>_pv.parquet  base + M1 + M2 + M3: the value final/40 serves the correctors on when no later rule has touched
                             a row (the operator's assembly replaces it by the chain value before the corrector where rules moved rows)
           <tag>_small.parquet  max(base + M1 + M2 + M3 + M4, 0), the shipped bundle file (before rounding)

Not ported: the silent-start nudge and the NNLS meta layer (they read the ADS-B coverage census and the off-block dump of
the served months); they stay zero (flag --nudge-meta stops with a message). M3 needs the off-block rows dump (J6_rows) and
stays zero with a warning without it. The tail needs three serve fits of the earlier 12-month corrector (final/41 v5).

    python final/42_small_stages.py --months 2,6 --base data/final/J7_base.parquet
    python final/42_small_stages.py --ranking data/raw/ranking.parquet --months 1,7 --base <stored v137 as parquet>
        --rows data/interim/agent_offblock_rows_2026.parquet --rank-cache data/interim/agent_trans_rank_features.parquet --sample 3000
"""
import argparse, sys, glob, time
import numpy as np, polars as pl, lightgbm as lgb
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "final"))
import os; os.chdir(REPO)
from fserve import parse_months
C = pl.col; I = "data/interim/"; ID = "MVT_ID_mvt"; T = "TAXITIME_SEC_mvt"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
ap = argparse.ArgumentParser()
ap.add_argument("--ranking", default="data/raw/final_ranking.parquet")
ap.add_argument("--months", default="2,6")
ap.add_argument("--base", required=True, help="parquet MVT_ID_mvt, TAXITIME_SEC_mvt: the chain value of the served departures before the small correctors")
ap.add_argument("--rows", default="data/final/J6_rows.parquet", help="off-block rows dump (M3)")
ap.add_argument("--rank-cache", default=None, help="parquet of rank features (Istanbul rows are used by M4); default: data/final/rankfeat_<mm>.parquet of final/40")
ap.add_argument("--v5-glob", default=None, help="serve preds of the earlier 12-month corrector (final/41 v5), for the tail component")
ap.add_argument("--serve-in12", default=None, help="serve file made by final/40 with --fold-prefix agent_bigc12")
ap.add_argument("--tag", default="J8")
ap.add_argument("--outdir", default="data/final")
ap.add_argument("--sample", type=int, default=0, help="serve only the first N departures per airport (training side unchanged)")
ap.add_argument("--nudge-meta", action="store_true", help="NOT PORTED: the silent-start nudge and the NNLS meta layer need the ADS-B census; they stay zero")
ARGS = ap.parse_args()
MONTHS = parse_months(ARGS.months)
if ARGS.nudge_meta:
    sys.exit("nudge/meta are not ported (they need the ADS-B coverage census of the served months); the deltas stay zero")
base = pl.read_parquet(ARGS.base)
if ID not in base.columns:
    base = base.rename({"id": ID})
base = base.select(ID, T).with_columns(C(ID).cast(pl.Float64)).sort(ID)
_need = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt", "AIRCRAFT_OPERATOR_flt"]
rk = pl.read_parquet(ARGS.ranking, columns=_need).with_columns(C(ID).cast(pl.Float64))
_dm = rk.filter(C("PHASE_mvt") == "DEP").filter(C("MVT_TIME_UTC_mvt").dt.month().is_in(MONTHS)).select(ID, C("ADEP_mvt").alias("_a"))
base = base.join(_dm, on=ID, how="semi")
if ARGS.sample:
    _keep = _dm.sort(ID).group_by("_a", maintain_order=True).head(ARGS.sample).select(ID)
    base = base.join(_keep, on=ID, how="semi")
print("served departures", base.height, flush=True)
dep = rk.filter(C("PHASE_mvt") == "DEP").select(
    ID, C("ADEP_mvt").alias("apt"),
    (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
    (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
    (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
    C("AOBT_3_flt").is_not_null().cast(pl.Float64).alias("plan"),
    C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("hr"), C("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Float64).alias("dow"),
    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t"))
S = base.join(dep, on=ID, how="left", maintain_order="left")
assert S.height == base.height and S["apt"].null_count() == 0
p = S[T].cast(pl.Float64).to_numpy(); a = S["apt"].to_numpy().astype(str); N = len(p)
delta = {k: np.zeros(N) for k in ("M1", "M2", "M3", "M4")}
z = np.load(I + "agent_bulkr_fold_SA_s0.npz", allow_pickle=True); k9 = np.isin(z["apts"].astype(str), AIR)
ids25 = z["ids"][k9]; y25 = z["y"][k9].astype(float); p25 = z["both_guard"][k9].astype(float); a25 = z["apts"][k9].astype(str); r25 = y25 - p25
# ---------------- M1
ctx = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP").select(
    C(ID).cast(pl.Float64).alias("id"),
    (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"), (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
    (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"), C("AOBT_3_flt").is_not_null().cast(pl.Float64).alias("plan"),
    C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("hr"), C("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Float64).alias("dow")).collect())
j = pl.DataFrame({"id": ids25}).with_columns(pl.int_range(pl.len()).alias("_r")).join(ctx, on="id", how="left").sort("_r")
X25 = np.column_stack([p25, [AIR.index(v) for v in a25], *[j[c].to_numpy() for c in ["gs", "gl", "ga", "plan", "hr", "dow"]]]).astype(float)
m9 = np.isin(a, AIR)
Xs = np.column_stack([p, [AIR.index(v) if v in AIR else -1 for v in a], *[S[c].to_numpy() for c in ["gs", "gl", "ga", "plan", "hr", "dow"]]]).astype(float)
d1 = np.zeros(N)
for seed in (0, 1, 2):
    g = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=2000, subsample=0.7, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=4, verbose=-1)
    g.fit(X25, r25, categorical_feature=[1]); d1[m9] += g.predict(Xs[m9]) / 3
delta["M1"] = 0.5 * d1; print("M1", flush=True); del ctx, j


# ---------------- M2 (late features: same definitions as agent_wx2_01_diag.py)
def late_feats(arrdf, depdf):
    out = {}
    for ap in AIR:
        A = arrdf.filter(C("ADES_mvt") == ap).sort("t"); at = A["t"].to_numpy()
        late = np.clip((A["t"] - A["ts"]).to_numpy().astype(float), -3600, 10800)
        cs_l = np.concatenate([[0], np.cumsum(late)]); cs_s = np.concatenate([[0], np.cumsum((late > 900).astype(float))])
        D = depdf.filter(C("apt") == ap); Tt = D["t"].to_numpy(); F = {}
        for wn, wl in (("60", 3600), ("180", 10800)):
            hi = np.searchsorted(at, Tt, side="right"); lo = np.searchsorted(at, Tt - wl, side="right"); n = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                F["s" + wn] = np.where(n >= 3, (cs_s[hi] - cs_s[lo]) / np.maximum(n, 1), np.nan)
                F["m" + wn] = np.where(n >= 3, (cs_l[hi] - cs_l[lo]) / np.maximum(n, 1), np.nan)
        out[ap] = (D[ID].to_numpy(), np.column_stack([F["s60"], F["s180"], F["m60"], F["m180"]]))
    return out


arr26 = rk.filter((C("PHASE_mvt") == "ARR") & C("ADES_mvt").is_in(AIR)).select(
    "ADES_mvt", C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t"), C("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("ts"))
print("ranking ARR rows", arr26.height, flush=True)
L26 = late_feats(arr26, S.select(ID, "apt", "t"))
dg = pl.read_parquet(I + "agent_wx2_diag.parquet").select(["id", "apt", "r", "arr_late_share60", "arr_late_share180", "arr_late_mean60", "arr_late_mean180"])
idpos = {v: i for i, v in enumerate(S[ID].to_numpy())}
for ap in AIR:
    g = dg.filter(C("apt") == ap)
    Ft = np.column_stack([g[c].to_numpy().astype(float) for c in ["arr_late_share60", "arr_late_share180", "arr_late_mean60", "arr_late_mean180"]])
    mu0 = np.nanmean(Ft, 0); Ft = np.where(np.isnan(Ft), mu0, Ft); rr = g["r"].to_numpy()
    mu, sd = Ft.mean(0), Ft.std(0) + 1e-9; A_ = np.column_stack([np.ones(len(rr)), (Ft - mu) / sd])
    co = np.linalg.solve(A_.T @ A_ + 1e3 * np.eye(5), A_.T @ rr)
    sid, Fs = L26[ap]; Fs = np.where(np.isnan(Fs), mu0, Fs)
    delta["M2"][[idpos[v] for v in sid]] = np.clip(np.column_stack([np.ones(len(Fs)), (Fs - mu) / sd]) @ co, -60, 60)
print("M2", flush=True)
# ---------------- M3
op = pl.concat([pl.read_parquet(f, columns=[ID, "AIRCRAFT_OPERATOR_flt"]) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))]).unique(ID)
o25 = (pl.scan_parquet(I + "agent_offblock_rows_c238.parquet").select([ID, "apt", "day", "block_t", "e_revany"]).filter(C("e_revany").is_not_nan()).collect()
       .join(op, on=ID, how="left").rename({"AIRCRAFT_OPERATOR_flt": "op"})
       .with_columns((C("e_revany") - C("block_t")).clip(-300, 300).alias("ec"), C("day").str.slice(5, 2).alias("mon")).unique(ID))


def keyarr(df, keys):
    return np.array(["|".join(map(str, t)) for t in df.select([C(k).cast(pl.Utf8).fill_null("NA") for k in keys]).rows()])


def tab(df, keys, minn=30):
    g = pl.DataFrame({"k": keyarr(df, keys), "e": df["ec"].to_numpy()}).group_by("k").agg(C("e").median().alias("md"), pl.len().alias("n")).filter(C("n") >= minn)
    return dict(zip(g["k"].to_list(), g["md"].to_list()))


f25 = pl.DataFrame({ID: ids25, "r": r25}); o25r = o25.join(f25, on=ID, how="inner")
mon = o25r["mon"].to_numpy(); r_ = o25r["r"].to_numpy(); k2 = keyarr(o25r, ["apt", "op"]); k1 = keyarr(o25r, ["apt"]); x = np.full(len(r_), np.nan)
for m in np.unique(mon):
    tr = o25r.filter(C("mon") != m); t2, t1 = tab(tr, ["apt", "op"]), tab(tr, ["apt"]); idx = np.where(mon == m)[0]
    x[idx] = [t2.get(k, np.nan) - t1.get(kk, np.nan) for k, kk in zip(k2[idx], k1[idx])]
okx = ~np.isnan(x); b = (x[okx] * r_[okx]).sum() / (x[okx] ** 2).sum(); print("M3 slope b", b, flush=True)
T2, T1 = tab(o25, ["apt", "op"]), tab(o25, ["apt"])
o26 = (pl.read_parquet(ARGS.rows).rename({"id": ID} if "id" in pl.read_parquet_schema(ARGS.rows) else {}).select([ID, "apt", "e_revany"]).with_columns(C(ID).cast(pl.Float64)).filter(C("e_revany").is_not_nan())) if Path(ARGS.rows).exists() else pl.DataFrame(schema={ID: pl.Float64, "apt": pl.Utf8, "e_revany": pl.Float64})
if o26.height == 0:
    print("WARNING: no off-block rows dump: M3 stays zero", flush=True)
opr = rk.filter(C("PHASE_mvt") == "DEP").select(ID, C("AIRCRAFT_OPERATOR_flt").alias("op")).unique(ID)
o26 = o26.join(opr, on=ID, how="left")
kk2, kk1 = keyarr(o26, ["apt", "op"]), keyarr(o26, ["apt"])
x26 = np.array([T2.get(k, np.nan) - T1.get(kk, np.nan) for k, kk in zip(kk2, kk1)]); ok26 = ~np.isnan(x26)
ii = np.array([idpos.get(v, -1) for v in o26[ID].to_numpy()]); sel = ok26 & (ii >= 0)
sel &= np.isin(a[np.clip(ii, 0, None)], AIR)
delta["M3"][ii[sel]] = b * x26[sel]; print("M3 rows", sel.sum(), flush=True)
del rk, arr26, o25, o25r, o26, opr, op, dg, L26, dep
import gc; gc.collect()
# ---------------- M4 (copy of agent_ltfmd_02_corr.py recipe)
cols = ['hour', 'dow', 'minute_of_day', 'gap_sched', 'gap_lobt', 'gap_eobt', 'gap_aobt', 'dep_delay', 'sched_vs_eobt', 'lobt_minus_aobt', 'turn_sched_slack', 'turn_in_delay', 'turn_age', 'turn_taxi_in', 'dep_30min', 'arr_30min', 'dep_60min', 'arr_60min', 'dep_rwy_headway', 'rwy_dep_active', 'rwy_dep_share', 'rwy_arr_active', 'rwy_mixed_mode', 'wake_heavy_60min', 'wake_prev_heavy', 'prev_rwy_gap', 'wx_temp_c', 'wx_wind_kt', 'wx_visibility_km', 'wx_precip', 'wx_lowvis', 'wx_snow', 'wx_freezing', 'has_flight_plan', 'sched_demand_30min', 'late_apt_past60_mean', 'late_apt_past15_mean']
Tt = pl.read_parquet(I + "train_features.parquet", columns=[ID, "month", "RUNWAY_mvt", "WK_TBL_CAT_flt"] + cols)
zt = np.load(I + "modal/TX_S6.npz", allow_pickle=True)
tid = zt["ids"].astype("float64"); ty = zt["y"]; p0 = zt["p0"]; tap = zt["apts"]
mon_t = pl.DataFrame({ID: tid, "i": np.arange(len(tid))}).join(Tt.select(ID, "month"), on=ID, how="left").sort("i")["month"].to_numpy()
dl = {}
for f in glob.glob("results/preds_ltfmseq_treat_f*.parquet"):
    aa = pl.read_parquet(f).select(ID, C("bagged").alias("t")); bb = pl.read_parquet(f.replace("treat", "ctrl")).select(ID, C("bagged").alias("c"))
    jj = aa.join(bb, on=ID).with_columns((C("t") - C("c")).alias("d")); dl.update(zip(jj[ID].to_numpy(), jj["d"].to_numpy()))
L = np.isin(mon_t, [1, 2, 3, 7, 8, 9, 10]) & (tap == "LTFM"); d_ = np.array([dl.get(k, np.nan) for k in tid[L]]); ok = np.isfinite(d_); idxL = np.where(L)[0][ok]
ref = p0[idxL] + d_[ok]
Xt = pl.DataFrame({ID: tid[idxL]}).join(Tt.select([ID, 'RUNWAY_mvt', 'WK_TBL_CAT_flt'] + cols), on=ID, how="left", maintain_order="left")
rw_cat = sorted(set(Xt["RUNWAY_mvt"].drop_nulls().to_list())); wk_cat = sorted(set(Xt["WK_TBL_CAT_flt"].drop_nulls().to_list()))


def mat(df, refv):
    rw = np.array([rw_cat.index(v) if v in rw_cat else -1 for v in df["RUNWAY_mvt"].to_list()], float)
    wk = np.array([wk_cat.index(v) if v in wk_cat else -1 for v in df["WK_TBL_CAT_flt"].to_list()], float)
    Fm = np.column_stack([df.select(cols).to_numpy().astype(float), rw, wk])
    return np.column_stack([Fm, refv, Fm[:, cols.index('gap_lobt')] - refv, Fm[:, cols.index('gap_sched')] - refv])


Ftr = mat(Xt, ref); rtr = ty[idxL] - ref
del Tt
if ARGS.rank_cache:
    rl = pl.read_parquet(ARGS.rank_cache)
else:
    rl = pl.concat([pl.read_parquet(f"data/final/rankfeat_{m:02d}.parquet") for m in MONTHS])
rl = rl.filter(C("ADEP_mvt") == "LTFM").with_columns(C(ID).cast(pl.Float64)).join(base.select(ID), on=ID, how="semi").unique(ID)
print("LTFM rank rows", rl.height, flush=True)
rl = rl.join(base.select(ID, C(T).cast(pl.Float64).alias("v137")), on=ID, how="left", maintain_order="left")
Fs = mat(rl, rl["v137"].to_numpy()); snow_s = (Fs[:, cols.index('wx_snow')] == 1) | (Fs[:, cols.index('wx_freezing')] == 1)
d4 = np.zeros(len(Fs))
for seed in (0, 1, 2):
    g = lgb.LGBMRegressor(objective='huber', alpha=300, num_leaves=8, n_estimators=200, learning_rate=0.03, min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=seed, n_jobs=4, verbose=-1)
    g.fit(Ftr, rtr); d4 += np.clip(g.predict(Fs), -600, 600) / 3
d4 = 0.5 * d4; d4[snow_s] = 0
delta["M4"][[idpos[v] for v in rl[ID].to_numpy()]] = d4
# ---------------- combine
tot = sum(delta.values()); out = np.maximum(p + tot, 0)
print("MEMBER rows_moved mean_shift rms_moved")
for k, v in delta.items():
    print(k, int((v != 0).sum()), round(float(v.mean()), 3), round(float(np.sqrt((v[v != 0] ** 2).mean())) if (v != 0).any() else 0, 2))
print("total moved", int((tot != 0).sum()), "mean shift", float((out - p).mean()))
# ---------------- tail component (archive/agent_tailc_03_serve.py): ehat for plan rows at the nine airports, 3 seeds averaged
ehat = np.zeros(N)
if ARGS.v5_glob and ARGS.serve_in12:
    import json
    z = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
    y12, pv12, apt12 = z["y"], z["pv"], z["apt"].astype(str)
    cols12 = [str(c) for c in z["cols"]]
    N12, nx = len(y12), len(cols12)
    F12 = np.mean([np.load(I + f"agent_bigc12_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"] for s in (0, 1, 2)], 0)
    kap = json.load(open(I + "agent_m12_kapt_C4ff7V5.json"))
    cur = pv12 + np.array([kap[x] for x in apt12]) * F12
    Xt = np.empty((N12, nx + 2), np.float32)
    Xt[:, :nx] = z["X"]
    Xt[:, nx] = F12
    Xt[:, nx + 1] = cur - pv12
    plan = Xt[:, cols12.index("has_flight_plan")] == 1
    del z
    rw = np.clip(y12 - cur, -7200, 7200)
    e = rw - np.clip(rw, -900, 900)
    ev = e != 0
    si = np.load(ARGS.serve_in12, allow_pickle=True)
    sid = si["ids"].astype(float)
    assert np.array_equal(sid, S[ID].to_numpy()), "serve file of the old pv is not in the row order of the served departures"
    sa = si["apt"].astype(str)
    cs = [str(c) for c in si["cols_pre"]]
    Xv = np.empty((len(sa), nx + 2), np.float32)
    Xv[:, :nx] = si["Xs_pre"][:, [cs.index(c) for c in cols12]]
    fs = sorted(glob.glob(os.path.join(ARGS.v5_glob, 'C4ff7V5_s*.npz') if os.path.isdir(ARGS.v5_glob) else ARGS.v5_glob))
    assert len(fs) >= 3, f"need three serve fits of the 12-month corrector, found {len(fs)}"
    Fs = np.mean([np.load(f, allow_pickle=True)["pred"] for f in fs[:3]], 0)
    Xv[:, nx] = Fs
    Xv[:, nx + 1] = np.array([kap.get(x, 0.0) for x in sa]) * Fs
    use = np.isin(sa, AIR) & (Xv[:, cols12.index("has_flight_plan")] == 1)
    print("tail: serve rows with plan at the nine airports", int(use.sum()), flush=True)
    outs = []
    for sd_ in (0, 1, 2):
        rng = np.random.default_rng(100 + sd_)
        tr = np.where(plan & (ev | (rng.random(N12) < 0.2)))[0]
        w = np.where(ev[tr], 1.0, 5.0)
        prm = dict(objective="regression", verbosity=-1, num_threads=4, seed=sd_, bagging_seed=sd_, feature_fraction_seed=sd_, feature_fraction=0.5, bagging_fraction=0.5,
                   bagging_freq=1, max_bin=63, learning_rate=0.03, num_leaves=15, min_child_samples=200, lambda_l2=50.0)
        m = lgb.train(prm, lgb.Dataset(Xt[tr], e[tr], weight=w), 300)
        h = np.zeros(len(sa))
        h[use] = m.predict(Xv[use])
        outs.append(h)
    ehat = np.mean(outs, 0)
    print("tail ehat: mean %.2f sd %.1f" % (ehat[use].mean(), ehat[use].std()), flush=True)
else:
    print("WARNING: tail not computed (needs --v5-glob and --serve-in12): ehat = 0", flush=True)
# ---------------- outputs: the deltas, and the value the correctors are served on (chain value + M1 + M2 + M3, nothing else)
delta["nudge"], delta["meta"] = np.zeros(N), np.zeros(N)
od = Path(ARGS.outdir)
np.savez(od / f"{ARGS.tag}_deltas.npz", ids=S[ID].to_numpy(), ehat=ehat, **delta)
pv = p + delta["M1"] + delta["M2"] + delta["M3"]
pl.DataFrame({ID: S[ID].to_numpy(), "pv": pv}).write_parquet(od / f"{ARGS.tag}_pv.parquet")
pl.DataFrame({ID: S[ID].to_numpy(), "value_small": np.maximum(p + tot, 0)}).write_parquet(od / f"{ARGS.tag}_small.parquet")
print("per airport-month rows and mean shift of M1+M2+M3+M4:")
import datetime
mo = np.array([datetime.datetime.fromtimestamp(int(v), datetime.timezone.utc).month for v in S["t"].to_numpy()])
for ap_ in sorted(set(a.tolist())):
    for m_ in sorted(set(mo.tolist())):
        s_ = (a == ap_) & (mo == m_)
        if s_.any():
            print("  %s %02d rows %d mean shift %+.2f" % (ap_, m_, s_.sum(), tot[s_].mean()))
print("wrote", od / f"{ARGS.tag}_deltas.npz", od / f"{ARGS.tag}_pv.parquet")
