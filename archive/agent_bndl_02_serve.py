"""experiment bndl 02: serving twins of M1+M2+M3+M4 on v137. BOARD TEST, not a pass (docs/LAB-NOTES.md).
Each twin fitted on ALL 2025 rows with the LOMO recipe, applied to 2026 ranking rows. Writes data/interim/agent_bndl_board_final.parquet.
M4 = copy of the agent_ltfmd_02_corr.py recipe (that script is untouched)."""
import sys, glob, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, ".")
C = pl.col; I = "data/interim/"; ID = "MVT_ID_mvt"; T = "TAXITIME_SEC_mvt"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
base = pl.read_parquet(I + "v137_ltfmsnow_final.parquet").sort(ID)
rk = pl.read_parquet("data/raw/ranking.parquet")
dep = rk.filter(C("PHASE_mvt") == "DEP").select(
    ID, C("ADEP_mvt").alias("apt"),
    (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
    (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
    (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
    C("AOBT_3_flt").is_not_null().cast(pl.Float64).alias("plan"),
    C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("hr"), C("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Float64).alias("dow"),
    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t"))
S = base.join(dep, on=ID, how="left", maintain_order="left")
assert S.height == base.height == 344841 and S["apt"].null_count() == 0
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
o26 = pl.scan_parquet(I + "agent_offblock_rows_2026.parquet").select([ID, "apt", "e_revany"]).filter(C("e_revany").is_not_nan()).collect()
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
rl = pl.read_parquet(I + "agent_bndl_rank_ltfm.parquet")  # built by agent_bndl_02a_rankfeat.py (prc.features.build on ranking.parquet)
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
res = base.with_columns(pl.Series(T, np.rint(out)).cast(base.schema[T]))
res.write_parquet(I + "agent_bndl_board_final.parquet"); np.savez(I + "agent_bndl_serve_deltas.npz", ids=S[ID].to_numpy(), **delta)
print("wrote", I + "agent_bndl_board_final.parquet")
