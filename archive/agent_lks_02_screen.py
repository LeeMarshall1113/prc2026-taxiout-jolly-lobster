"""lks step 2: cheap LOMO screen. Same corrector as agent_ltfmd_02_corr.py (LightGBM on LTFM rows, huber, k*clip(pred,+-600), snow rows 0) on REF = TX_S6 p0 + v113 analogue delta.
Arms: ctrl (37 serve-time cols, the ltfmd set), ks (ctrl + out-of-month key stats of y and of the base residual), ksonly (REF-derived cols + key stats only).
Key stats are rebuilt per held month (train rows pool excludes {held, own}; held rows pool = other months)."""
import sys, numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, "archive"); import agent_lks_te as TE
I = "data/interim/"; ID = "MVT_ID_mvt"; MONTHS = [1, 2, 3, 7, 8, 9, 10]
K = np.load(I + "agent_lks_keys.npz", allow_pickle=True); fid = K["fid"]; yf = K["yf"]; ref = K["ref"]; monf = K["monf"]; Kall = K["Kall"]; kn = list(K["kn"]); Nf = len(fid)
z = np.load(I + "modal/TX_S6.npz", allow_pickle=True); ids = z["ids"].astype("float64"); y = z["y"]; p0 = z["p0"]; apts = z["apts"]
cols = ['hour', 'dow', 'minute_of_day', 'gap_sched', 'gap_lobt', 'gap_eobt', 'gap_aobt', 'dep_delay', 'sched_vs_eobt', 'lobt_minus_aobt', 'turn_sched_slack', 'turn_in_delay', 'turn_age', 'turn_taxi_in',
        'dep_30min', 'arr_30min', 'dep_60min', 'arr_60min', 'dep_rwy_headway', 'rwy_dep_active', 'rwy_dep_share', 'rwy_arr_active', 'rwy_mixed_mode', 'wake_heavy_60min', 'wake_prev_heavy', 'prev_rwy_gap',
        'wx_temp_c', 'wx_wind_kt', 'wx_visibility_km', 'wx_precip', 'wx_lowvis', 'wx_snow', 'wx_freezing', 'has_flight_plan', 'sched_demand_30min', 'late_apt_past60_mean', 'late_apt_past15_mean']
T = pl.read_parquet(I + 'train_features.parquet', columns=[ID, 'month', 'RUNWAY_mvt', 'WK_TBL_CAT_flt'] + cols)
mon = pl.DataFrame({ID: ids, 'i': np.arange(len(ids))}).join(T.select(ID, 'month'), on=ID, how='left').sort('i')['month'].to_numpy()
X = pl.DataFrame({ID: fid}).join(T.select([ID, 'RUNWAY_mvt', 'WK_TBL_CAT_flt'] + cols), on=ID, how='left', maintain_order='left')
X = X.with_columns(pl.col('RUNWAY_mvt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('rwy'), pl.col('WK_TBL_CAT_flt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('wk'))
F = X.select(cols + ['rwy', 'wk']).to_numpy().astype(float); del T, X; import gc; gc.collect()
F = np.column_stack([F, ref, F[:, cols.index('gap_lobt')] - ref, F[:, cols.index('gap_sched')] - ref]); nF = F.shape[1]
snow = (F[:, cols.index('wx_snow')] == 1) | (F[:, cols.index('wx_freezing')] == 1)
res = yf - ref; months = np.array(MONTHS); mi = np.searchsorted(months, monf); rf = np.arange(Nf); ks_ = TE.KS(Kall[:Nf])
# other airports (unchanged) for the ALL metric, same population as ltfmd
keep = np.isin(mon, MONTHS); oth = keep & (apts != 'LTFM'); sse_oth = ((p0[oth] - y[oth]) ** 2).sum(); N = oth.sum() + Nf
base = (ref - yf) ** 2; top50 = np.argsort(-np.abs(res))[:50]; mk = np.ones(Nf, bool); mk[top50] = False
seas = {}
for s, ms in [('JanMar', [1, 2, 3]), ('JulSep', [7, 8, 9])]:
    sm = np.isin(mon, ms) & keep & (apts != 'LTFM'); seas[s] = (np.isin(monf, ms), ((p0[sm] - y[sm]) ** 2).sum(), sm.sum() + np.isin(monf, ms).sum())
SEEDS = [0, 1, 2]; ARMS = ['ctrl', 'ks', 'ksonly']; dlt = {(a, s): np.zeros(Nf) for a in ARMS for s in SEEDS}
for hi, h in enumerate(MONTHS):
    S = np.hstack([ks_.build(None, rf, 'y', yf, mi, len(MONTHS), hi), ks_.build(None, rf, 'r', res, mi, len(MONTHS), hi)])
    S = S.astype(np.float32); Fk = np.hstack([F.astype(np.float32), S]); Fo = np.hstack([F[:, [nF - 3, nF - 2, nF - 1]].astype(np.float32), S]); del S
    te = np.where(mi == hi)[0]; tr = np.where(mi != hi)[0]
    for a, M in (('ctrl', F), ('ks', Fk), ('ksonly', Fo)):
        for s in SEEDS:
            g = lgb.LGBMRegressor(objective='huber', alpha=300, num_leaves=8, n_estimators=200, learning_rate=0.03, min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=s, n_jobs=4, verbose=-1)
            g.fit(M[tr], res[tr]); dlt[(a, s)][te] = np.clip(g.predict(M[te]), -600, 600)
    del Fk, Fo; gc.collect(); print('fold month', h, 'done', flush=True)
def rm(a, b): return np.sqrt(a) - np.sqrt(b)
for a in ARMS:
    for s in SEEDS:
        d = dlt[(a, s)].copy(); d[snow] = 0
        for k in (1, 0.5):
            ns = (ref + k * d - yf) ** 2
            o = [f'{a} s{s} k{k}: LTFM {np.sqrt(ns.mean()) - np.sqrt(base.mean()):+.2f}', f'ALL {rm((sse_oth + ns.sum()) / N, (sse_oth + base.sum()) / N):+.3f}',
                 f'-top50 {rm((sse_oth + ns[mk].sum()) / (N - 50), (sse_oth + base[mk].sum()) / (N - 50)):+.3f}']
            for sn, (mm, so, n_s) in seas.items(): o.append(f'{sn} {rm((so + ns[mm].sum()) / n_s, (so + base[mm].sum()) / n_s):+.3f}')
            o.append('months<0 %d/7' % sum(((ns[mi == i].sum() - base[mi == i].sum()) < 0) for i in range(7)))
            print(' | '.join(o), flush=True)
# increment ks vs ctrl, k=0.5 and 1, seed-averaged deltas
for k in (1, 0.5):
    out = {}
    for a in ARMS:
        d = np.mean([dlt[(a, s)] for s in SEEDS], 0); d[snow] = 0; ns = (ref + k * d - yf) ** 2; out[a] = ns
        print(f'seedavg {a} k{k}: ALL vs REF {rm((sse_oth + ns.sum()) / N, (sse_oth + base.sum()) / N):+.3f}')
    print(f'seedavg k{k}: ks-minus-ctrl ALL {rm((sse_oth + out["ks"].sum()) / N, (sse_oth + out["ctrl"].sum()) / N):+.3f}  ksonly-minus-ctrl {rm((sse_oth + out["ksonly"].sum()) / N, (sse_oth + out["ctrl"].sum()) / N):+.3f}')
np.savez(I + 'agent_lks_screen_dlt.npz', **{f'{a}_s{s}': v for (a, s), v in dlt.items()}, snow=snow)
