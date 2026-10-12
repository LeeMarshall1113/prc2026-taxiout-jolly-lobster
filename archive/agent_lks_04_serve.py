"""lks 04: serving twin of the LTFM key-stat corrector (screen arm 'ks': ctrl cols + out-of-month key stats of y AND residual, k=0.5, 3 seeds, snow rows 0).
Replaces M4 in v128: out = max(v128 - M4_delta + lks_delta, 0). Fit on all 7 LTFM fold months (train stats leave-own-month-out), serve stats pool = all 7 months of 2025.
Ref feature for 2026 rows = v137 (as the M4 twin). Writes data/interim/agent_lks_ltfm_corr_final.parquet. BOARD TEST."""
import sys, glob, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "archive"); import agent_lks_te as TE
C = pl.col; I = "data/interim/"; ID = "MVT_ID_mvt"; T = "TAXITIME_SEC_mvt"; MONTHS = [1, 2, 3, 7, 8, 9, 10]
K = np.load(I + "agent_lks_keys.npz", allow_pickle=True); fid = K["fid"]; yf = K["yf"]; ref = K["ref"]; monf = K["monf"]; Kall = K["Kall"]; sid = K["sid"]; Nf = len(fid)
cols = ['hour', 'dow', 'minute_of_day', 'gap_sched', 'gap_lobt', 'gap_eobt', 'gap_aobt', 'dep_delay', 'sched_vs_eobt', 'lobt_minus_aobt', 'turn_sched_slack', 'turn_in_delay', 'turn_age', 'turn_taxi_in', 'dep_30min', 'arr_30min', 'dep_60min', 'arr_60min', 'dep_rwy_headway', 'rwy_dep_active', 'rwy_dep_share', 'rwy_arr_active', 'rwy_mixed_mode', 'wake_heavy_60min', 'wake_prev_heavy', 'prev_rwy_gap', 'wx_temp_c', 'wx_wind_kt', 'wx_visibility_km', 'wx_precip', 'wx_lowvis', 'wx_snow', 'wx_freezing', 'has_flight_plan', 'sched_demand_30min', 'late_apt_past60_mean', 'late_apt_past15_mean']
Tt = pl.read_parquet(I + "train_features.parquet", columns=[ID, "RUNWAY_mvt", "WK_TBL_CAT_flt"] + cols)
Xt = pl.DataFrame({ID: fid}).join(Tt, on=ID, how="left", maintain_order="left"); del Tt
rw_cat = sorted(set(Xt["RUNWAY_mvt"].drop_nulls().to_list())); wk_cat = sorted(set(Xt["WK_TBL_CAT_flt"].drop_nulls().to_list()))
def mat(df, refv):
    rw = np.array([rw_cat.index(v) if v in rw_cat else -1 for v in df["RUNWAY_mvt"].to_list()], float); wk = np.array([wk_cat.index(v) if v in wk_cat else -1 for v in df["WK_TBL_CAT_flt"].to_list()], float)
    Fm = np.column_stack([df.select(cols).to_numpy().astype(float), rw, wk]); return np.column_stack([Fm, refv, Fm[:, cols.index('gap_lobt')] - refv, Fm[:, cols.index('gap_sched')] - refv])
Ftr = mat(Xt, ref); res = yf - ref
base = pl.read_parquet(I + "v137_ltfmsnow_final.parquet").sort(ID)
rl = pl.read_parquet(I + "agent_bndl_rank_ltfm.parquet").join(base.select(ID, C(T).cast(pl.Float64).alias("v137")), on=ID, how="left", maintain_order="left")
pos = {v: i for i, v in enumerate(sid)}; ii = np.array([pos[v] for v in rl[ID].to_numpy()])   # rank row -> serve key row
Fs = mat(rl, rl["v137"].to_numpy()); snow_s = (Fs[:, cols.index('wx_snow')] == 1) | (Fs[:, cols.index('wx_freezing')] == 1)
mi = np.searchsorted(np.array(MONTHS), monf); eng = TE.KS(Kall); rf = np.arange(Nf); sr = Nf + ii
tr_y, sv_y = eng.build(None, rf, "y", yf, mi, 7, None, serve_rows=sr); tr_r, sv_r = eng.build(None, rf, "r", res, mi, 7, None, serve_rows=sr)
Ftr2 = np.hstack([Ftr, tr_y, tr_r]).astype(np.float32); Fs2 = np.hstack([Fs, sv_y, sv_r]).astype(np.float32)
print("train", Ftr2.shape, "serve", Fs2.shape, "serve unseen op share", float((sv_y[:, 3] == 0).mean()), flush=True)
d = np.zeros(len(Fs2))
for seed in (0, 1, 2):
    g = lgb.LGBMRegressor(objective='huber', alpha=300, num_leaves=8, n_estimators=200, learning_rate=0.03, min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=seed, n_jobs=4, verbose=-1)
    g.fit(Ftr2, res); d += np.clip(g.predict(Fs2), -600, 600) / 3
d = 0.5 * d; d[snow_s] = 0
v = pl.read_parquet(I + "agent_bigc_v4_board_final.parquet").sort(ID); sd = np.load(I + "agent_bndl_serve_deltas.npz"); assert (sd["ids"] == v[ID].to_numpy()).all()
p = v[T].cast(pl.Float64).to_numpy(); idpos = {x: i for i, x in enumerate(v[ID].to_numpy())}
new = np.zeros(len(p)); new[[idpos[x] for x in rl[ID].to_numpy()]] = d; m4 = sd["M4"]
out = np.maximum(p - m4 + new, 0); net = out - p
print("M4 term: rows", int((m4 != 0).sum()), "mean", float(m4.mean()), "| new term rows", int((new != 0).sum()), "mean on LTFM", float(d.mean()), "snow-zeroed", int(snow_s.sum()))
print("rows moved (net vs v128)", int((net != 0).sum()), "mean shift all rows", float(net.mean()), "mean shift on moved", float(net[net != 0].mean()), "rms moved", float(np.sqrt((net[net != 0] ** 2).mean())))
dpn = new ** 2; foot_s = float((new ** 2).sum() / len(p))
z = np.load(I + "agent_lks_screen_dlt.npz"); dk = np.mean([z[f"ks_s{s}"] for s in (0, 1, 2)], 0) if "ks_s0" in z.files else None
if dk is not None:
    dk = 0.5 * dk; dk[z["snow"]] = 0; zz = np.load(I + "modal/TX_S6.npz", allow_pickle=True); Nall = int(np.isin(zz["apts"], ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]).sum()) if False else None
res_ = pl.DataFrame({ID: v[ID]}).with_columns(pl.Series(T, np.rint(out)).cast(v.schema[T])); v.with_columns(pl.Series(T, np.rint(out)).cast(v.schema[T])).write_parquet(I + "agent_lks_ltfm_corr_final.parquet")
np.savez(I + "agent_lks_serve_delta.npz", ids=v[ID].to_numpy(), new=new, m4=m4); print("serve sum dp^2/row (new term)", foot_s, "wrote candidate")
