import numpy as np, polars as pl, lightgbm as lgb, glob
z = np.load('data/interim/modal/TX_S6.npz', allow_pickle=True)
ids = z['ids'].astype('float64'); y = z['y']; p0 = z['p0']; apts = z['apts']; days = z['days'].astype(str)
T = pl.read_parquet('data/interim/train_features.parquet')
mon = (pl.DataFrame({'MVT_ID_mvt': ids, 'i': np.arange(len(ids))})
       .join(T.select('MVT_ID_mvt', 'month'), on='MVT_ID_mvt', how='left').sort('i')['month'].to_numpy())
MONTHS = [1, 2, 3, 7, 8, 9, 10]
dl = {}
for f in glob.glob('results/preds_ltfmseq_treat_f*.parquet'):
    c = f.replace('treat', 'ctrl')
    a = pl.read_parquet(f).select('MVT_ID_mvt', pl.col('bagged').alias('t'))
    b = pl.read_parquet(c).select('MVT_ID_mvt', pl.col('bagged').alias('c'))
    j = a.join(b, on='MVT_ID_mvt').with_columns((pl.col('t') - pl.col('c')).alias('d'))
    for k, v in zip(j['MVT_ID_mvt'].to_numpy(), j['d'].to_numpy()):
        dl[k] = v
keep = np.isin(mon, MONTHS)
L = keep & (apts == 'LTFM')
d = np.array([dl.get(k, np.nan) for k in ids[L]])
ok = np.isfinite(d)
print('ltfm rows', L.sum(), 'with delta', ok.mean())
idxL = np.where(L)[0][ok]
ref = p0[idxL] + d[ok]
cols = ['hour', 'dow', 'minute_of_day', 'gap_sched', 'gap_lobt', 'gap_eobt', 'gap_aobt', 'dep_delay', 'sched_vs_eobt',
        'lobt_minus_aobt', 'turn_sched_slack', 'turn_in_delay', 'turn_age', 'turn_taxi_in', 'dep_30min', 'arr_30min',
        'dep_60min', 'arr_60min', 'dep_rwy_headway', 'rwy_dep_active', 'rwy_dep_share', 'rwy_arr_active',
        'rwy_mixed_mode', 'wake_heavy_60min', 'wake_prev_heavy', 'prev_rwy_gap', 'wx_temp_c', 'wx_wind_kt',
        'wx_visibility_km', 'wx_precip', 'wx_lowvis', 'wx_snow', 'wx_freezing', 'has_flight_plan',
        'sched_demand_30min', 'late_apt_past60_mean', 'late_apt_past15_mean']
X = (pl.DataFrame({'MVT_ID_mvt': ids[idxL]})
     .join(T.select(['MVT_ID_mvt', 'RUNWAY_mvt', 'WK_TBL_CAT_flt'] + cols), on='MVT_ID_mvt', how='left', maintain_order='left'))
X = X.with_columns(pl.col('RUNWAY_mvt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('rwy'),
                   pl.col('WK_TBL_CAT_flt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('wk'))
F = X.select(cols + ['rwy', 'wk']).to_numpy().astype(float)
F = np.column_stack([F, ref, F[:, cols.index('gap_lobt')] - ref, F[:, cols.index('gap_sched')] - ref])
snow = (F[:, cols.index('wx_snow')] == 1) | (F[:, cols.index('wx_freezing')] == 1)
yl = y[idxL]; m_l = mon[idxL]; dd = days[idxL]
res = yl - ref
dayidx = np.array([int(s[-2:]) % 10 for s in dd])


def fit_pred(tr, te, seed):
    g = lgb.LGBMRegressor(objective='huber', alpha=300, num_leaves=8, n_estimators=200, learning_rate=0.03,
                          min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                          random_state=seed, n_jobs=4, verbose=-1)
    g.fit(F[tr], res[tr])
    return np.clip(g.predict(F[te]), -600, 600)


oth = keep & (apts != 'LTFM')
sse_oth = ((p0[oth] - y[oth]) ** 2).sum(); N = oth.sum() + len(idxL)
base = (ref - yl) ** 2
top50 = np.argsort(-np.abs(res))[:50]
mk = np.ones(len(res), bool); mk[top50] = False
seas = {}
for s, ms in [('JanMar', [1, 2, 3]), ('JulSep', [7, 8, 9])]:
    sm = np.isin(mon, ms) & keep & (apts != 'LTFM')
    seas[s] = (np.isin(m_l, ms), ((p0[sm] - y[sm]) ** 2).sum(), sm.sum() + np.isin(m_l, ms).sum())
for mode in ['month', 'day']:
    for seed in [0, 1, 2]:
        dlt = np.zeros(len(res))
        grp = m_l if mode == 'month' else dayidx
        for g in np.unique(grp):
            te = np.where(grp == g)[0]; tr = np.where(grp != g)[0]
            dlt[te] = fit_pred(tr, te, seed)
        dlt[snow] = 0
        for k in [1, 0.5]:
            ns = (ref + k * dlt - yl) ** 2
            o = [f'{mode} s{seed} k{k}: LTFM {np.sqrt(ns.mean()) - np.sqrt(base.mean()):+.2f}',
                 f'ALL {np.sqrt((sse_oth + ns.sum()) / N) - np.sqrt((sse_oth + base.sum()) / N):+.3f}',
                 f'-top50 {np.sqrt((sse_oth + ns[mk].sum()) / (N - 50)) - np.sqrt((sse_oth + base[mk].sum()) / (N - 50)):+.3f}']
            for s, (mm, so, n_s) in seas.items():
                o.append(f'{s} {np.sqrt((so + ns[mm].sum()) / n_s) - np.sqrt((so + base[mm].sum()) / n_s):+.3f}')
            o.append('exsnowLTFM %+.2f' % (np.sqrt(ns[~snow].mean()) - np.sqrt(base[~snow].mean())))
            print(' | '.join(o), flush=True)
