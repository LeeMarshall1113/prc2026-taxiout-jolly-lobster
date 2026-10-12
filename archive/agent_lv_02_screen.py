"""lv step 2: LOMO screen of the agent_lks corrector WITH variant views (docs/LAB-NOTES.md).
Same protocol as archive/agent_lks_02_screen.py (LightGBM huber on REF, ctrl cols + out-of-month key stats of y and residual, k*clip(pred,+-600), snow rows 0,
ALL pooled with the nine other airports unchanged) + view columns (archive/agent_lv_views.py).

    python archive/agent_lv_02_screen.py --stage 0|1 [--arms ks,A,C,AC] [--cfgs S1,S2] [--seeds 0,1,2]
stage 0 = informational: evaluated + trained on months 1,2,3,7,8,9 (month 10 has no member fold yet); key-stat pools still use all 7 months (as lks).
stage 1 = binding: all 7 months (needs fold 4 checkpoints).  Arm letters: ks = no views; A members; Q quantiles; C seq-on diffs.  Writes data/interim/agent_lv_screen_s<stage>.npz."""
import argparse
import gc
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "archive")
import agent_lks_te as TE  # noqa: E402
import agent_lv_views as VW  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", type=int, default=0)
ap.add_argument("--arms", default="ks,A,C,AC")
ap.add_argument("--cfgs", default="S1")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--tagx", default="")
ap.add_argument("--judge-only", action="store_true", help="keys12: hold out (predict) only the original 7 months")
ap.add_argument("--eval12", action="store_true", help="also report LTFM-only RMSE on all months / excluding Feb and ALL7 excluding Feb")
ap.add_argument("--trainmonths", default="", help="comma list: only these months are TRAINED on / held out (key-stat pools still use every month of the key table)")
ap.add_argument("--catmap", default="phys", help="phys = polars physical codes (validated screens), sorted = SORTED category index as agent_lv_04_serve.mat()")
ap.add_argument("--lgbcat", action="store_true", help="declare rwy/wk as categorical_feature in LightGBM")
ap.add_argument("--maxfolds", type=int, default=99)
ap.add_argument("--merge", default="", help="comma list of npz files (per-seed runs) to load dlt from instead of fitting")
ap.add_argument("--keys12", default="", help="key table with more training months (agent_lv_05_keys12.py): train on every month with views, JUDGE on months 1,2,3,7,8,9,10 only")
ap.add_argument("--ref", default="stale", help="stale = TX_S6 p0 + v113 delta (lks); served = v94 fold bagged + (treat-ctrl) = agent_ltfmm_judge SERVED (mirrors serve)")
a = ap.parse_args()
I = "data/interim/"; ID = "MVT_ID_mvt"; MONTHS = [1, 2, 3, 7, 8, 9, 10]; ORIG7 = list(MONTHS)
EV = [1, 2, 3, 7, 8, 9] if a.stage == 0 else MONTHS
ARMS = a.arms.split(","); SEEDS = [int(s) for s in a.seeds.split(",")]
CFG = {"S1": dict(num_leaves=8, n_estimators=200), "S2": dict(num_leaves=16, n_estimators=300), "S3": dict(num_leaves=31, n_estimators=300), "S4": dict(num_leaves=63, n_estimators=300, min_child_samples=200)}
K = np.load(a.keys12 or (I + "agent_lks_keys.npz"), allow_pickle=True); fid = K["fid"]; yf = K["yf"]; ref = K["ref"]; monf = K["monf"]; Kall = K["Kall"]; Nf = len(fid)
if a.keys12:
    REFOK = np.ones(len(fid), bool); MONTHS = sorted(set(int(m) for m in monf)); print("training months", MONTHS, flush=True)
if a.ref == "served" and not a.keys12:
    ref_stale = ref.copy(); ref = np.full(Nf, np.nan)
    for k_, t_ in ((1, "1_7"), (2, "2_8"), (3, "3_9"), (4, "4_10")):
        try:
            b = pl.read_parquet(f"results/preds_bulkr_f{k_}_{t_}.parquet").select(ID, "bagged")
            c0 = pl.read_parquet(f"results/preds_ltfmseq_ctrl_f{k_}_{t_}.parquet").select(ID, pl.col("bagged").alias("c0"))
            t1 = pl.read_parquet(f"results/preds_ltfmseq_treat_f{k_}_{t_}.parquet").select(ID, pl.col("bagged").alias("t1"))
        except FileNotFoundError:
            continue
        j = b.join(c0, on=ID).join(t1, on=ID).with_columns((pl.col("bagged") + pl.col("t1") - pl.col("c0")).alias("r"))
        dd = dict(zip(j[ID].to_numpy().astype("float64"), j["r"].to_numpy())); sel = np.isin(monf, [int(x) for x in t_.split("_")])
        ref[sel] = [dd.get(x, np.nan) for x in fid[sel]]
    REFOK = np.isfinite(ref); ref = np.where(REFOK, ref, ref_stale); print("served REF coverage", float(REFOK.mean()), flush=True)
z = np.load(I + "modal/TX_S6.npz", allow_pickle=True); ids = z["ids"].astype("float64"); y = z["y"]; p0 = z["p0"]; apts = z["apts"]
cols = ['hour', 'dow', 'minute_of_day', 'gap_sched', 'gap_lobt', 'gap_eobt', 'gap_aobt', 'dep_delay', 'sched_vs_eobt', 'lobt_minus_aobt', 'turn_sched_slack', 'turn_in_delay', 'turn_age', 'turn_taxi_in',
        'dep_30min', 'arr_30min', 'dep_60min', 'arr_60min', 'dep_rwy_headway', 'rwy_dep_active', 'rwy_dep_share', 'rwy_arr_active', 'rwy_mixed_mode', 'wake_heavy_60min', 'wake_prev_heavy', 'prev_rwy_gap',
        'wx_temp_c', 'wx_wind_kt', 'wx_visibility_km', 'wx_precip', 'wx_lowvis', 'wx_snow', 'wx_freezing', 'has_flight_plan', 'sched_demand_30min', 'late_apt_past60_mean', 'late_apt_past15_mean']
T = pl.read_parquet(I + 'train_features.parquet', columns=[ID, 'month', 'RUNWAY_mvt', 'WK_TBL_CAT_flt'] + cols)
mon = pl.DataFrame({ID: ids, 'i': np.arange(len(ids))}).join(T.select(ID, 'month'), on=ID, how='left').sort('i')['month'].to_numpy()
X = pl.DataFrame({ID: fid}).join(T.select([ID, 'RUNWAY_mvt', 'WK_TBL_CAT_flt'] + cols), on=ID, how='left', maintain_order='left')
if a.catmap == "sorted":
    _rc = sorted(set(X["RUNWAY_mvt"].drop_nulls().to_list())); _wc = sorted(set(X["WK_TBL_CAT_flt"].drop_nulls().to_list()))
    X = X.with_columns(pl.Series("rwy", [float(_rc.index(v)) if v in _rc else -1.0 for v in X["RUNWAY_mvt"].to_list()]), pl.Series("wk", [float(_wc.index(v)) if v in _wc else -1.0 for v in X["WK_TBL_CAT_flt"].to_list()]))
else:
    X = X.with_columns(pl.col('RUNWAY_mvt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('rwy'), pl.col('WK_TBL_CAT_flt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('wk'))
F = X.select(cols + ['rwy', 'wk']).to_numpy().astype(float); del T, X; gc.collect()
F = np.column_stack([F, ref, F[:, cols.index('gap_lobt')] - ref, F[:, cols.index('gap_sched')] - ref]); nF = F.shape[1]
snow = (F[:, cols.index('wx_snow')] == 1) | (F[:, cols.index('wx_freezing')] == 1)
res = yf - ref; months = np.array(MONTHS); mi = np.searchsorted(months, monf); rf = np.arange(Nf); ks_ = TE.KS(Kall[:Nf])
valid = np.isin(monf, EV) & (REFOK if a.ref == 'served' else True)
TRN = np.ones(Nf, bool)   # rows usable for training/holdout: all view arrays finite (set after the views are loaded)
# views per arm letter group
mem = VW.load_fold(fid, monf, "ctrl", VW.MEM); qq = VW.load_fold(fid, monf, "ctrl", VW.QS); tt = VW.load_fold(fid, monf, "treat", VW.MEM)
if "ks" in ARMS or "m4" in ARMS: pass
print("view coverage on valid rows: members %.4f quantiles %.4f treat %.4f" % tuple(float(np.isfinite(x[valid]).all(1).mean()) for x in (mem, qq, tt)), flush=True)
VIEW = {}
EX = {}
for L_, kk in VW.EXTRA.items():
    if any(L_ in x for x in ARMS):
        for k_, col in zip(kk, VW.load_fold(fid, monf, "ctrl", kk).T):
            EX[k_] = col; print("extra view", k_, "coverage", float(np.isfinite(col[valid]).mean()), flush=True)
def _align(Z, key="fold"):   # rows of a column file (keys12_full order) -> the rows of this run's key table (by id)
    if len(Z["fold_ids"]) == len(fid) and (Z["fold_ids"] == fid).all(): return Z[key]
    pos = {v: i for i, v in enumerate(Z["fold_ids"].tolist())}; return Z[key][np.array([pos[v] for v in fid.tolist()])]
if any("R" in x.replace("ks", "") for x in ARMS):
    HO = np.load(I + "agent_lv_handover.npz", allow_pickle=True); EX["R"] = _align(HO); print("handover columns coverage", float(np.isfinite(EX["R"][:, 0]).mean()), flush=True)
if any("W" in x.replace("ks", "") for x in ARMS):
    RW = np.load(I + "agent_lv_rwycfg.npz", allow_pickle=True); EX["W"] = _align(RW); print("rwycfg columns", list(RW["names"]), flush=True)
if any(("P" in x.replace("ks", "") or "D" in x.replace("ks", "")) for x in ARMS):
    FLW = np.load(I + "agent_lv_flow.npz", allow_pickle=True); EX["F"] = _align(FLW); print("flow columns", len(FLW["names"]), flush=True)
if any("T" in x.replace("ks", "") for x in ARMS):
    TT = np.load(I + "agent_lv_arrT.npz", allow_pickle=True); EX["T"] = _align(TT); print("arrT columns", EX["T"].shape[1], flush=True)
BULK = None
if any("X" in x.replace("ks", "") for x in ARMS):   # FULL bulk feature set (80 base-model columns minus the BASEDROP set; no date/month identifiers remain), categoricals as stable codes
    sys.path.insert(0, "."); import prc.features as PF
    BD_ = "airport_plan,arr_slow_180min,arr_slow_60min,arr_sub_day,arr_sub_stand,arr_taxi_180min,arr_taxi_trend,cfg_new,cfg_shift,clk_aobt_minus_sched,clk_arvt3_minus_aobt,clk_arvt3_minus_arvt1,clk_arvt3_minus_mvt,clk_eobt_minus_lobt,clk_iobt_minus_lobt,clk_lobt_minus_sched,clk_sec_aobt,clk_sec_mvt,clk_sec_sched,doy,lobt_minus_aobt,month,mvt_sec_00,prev_rwy_gap,prev_stand_gap,prev_stand_headway,ref_level,ref_taxi_s,turn_in_delay,turn_rwy_match,turn_sched_slack,wake_heavy_60min,wake_prev_heavy,wake_sep_req".split(",")
    BF = [c_ for c_ in PF.FEATURES if c_ not in BD_ and c_ != "ADEP_mvt"]
    TB = pl.DataFrame({ID: fid}).join(pl.read_parquet(I + "train_features.parquet", columns=[ID] + BF), on=ID, how="left", maintain_order="left")
    CATB = [c_ for c_ in BF if c_ in PF.CATEGORICAL]
    BULK = np.column_stack([TB[c_].cast(pl.Categorical).to_physical().cast(pl.Float64).to_numpy() if c_ in CATB else TB[c_].cast(pl.Float64).to_numpy() for c_ in BF]).astype(np.float32); del TB
    print("bulk columns", BULK.shape[1], flush=True)
NOKS = {}
for arm in ARMS:
    g = arm.replace("ks", "").replace("N", "").replace("Z", "").replace("X", "")   # N = drop the key-stat columns (views + ctrl cols only)
    NOKS[arm] = "N" in arm.replace("ks", "")
    if g and arm != "m4":
        VIEW[arm], vn = VW.make_views(ref, mem, qq if any(c in g for c in "QE") else None, tt if "C" in g else None, g, EX); print(arm, "view columns", len(vn), flush=True)
for _x in (mem, qq if any(c in x for x in ARMS for c in "QE") else mem, tt if any("C" in x for x in ARMS) else mem, *[v_ for k_, v_ in EX.items() if k_ not in ("R", "W", "F", "T")]):
    TRN &= np.isfinite(_x).all(1) if _x.ndim == 2 else np.isfinite(_x)
if a.trainmonths: TRN &= np.isin(monf, [int(x) for x in a.trainmonths.split(',')])
valid &= TRN
if a.keys12:
    print("trainable rows by month", {int(m): int(TRN[monf == m].sum()) for m in MONTHS}, flush=True)
keep = np.isin(mon, EV); oth = keep & (apts != 'LTFM'); sse_oth = ((p0[oth] - y[oth]) ** 2).sum(); N = oth.sum() + valid.sum()
base = (ref - yf) ** 2; sub = np.where(valid)[0]
top50 = sub[np.argsort(-np.abs(res[sub]))[:50]]; mk = valid.copy(); mk[top50] = False
seas = {}
for s, ms in [('JanMar', [1, 2, 3]), ('JulSep', [7, 8, 9])]:
    sm = np.isin(mon, ms) & keep & (apts != 'LTFM'); mm_ = np.isin(monf, ms) & valid; seas[s] = (mm_, ((p0[sm] - y[sm]) ** 2).sum(), sm.sum() + mm_.sum())
DAYM = any(x[-1] in "YGUVM" and x.startswith("AQC") for x in ARMS)
if DAYM:
    sys.path.insert(0, "archive"); import agent_lv_13_dayarr as DA
    DY = np.load(I + "agent_lv_dayarr.npz", allow_pickle=True); assert (DY["fold_ids"] == fid).all(), "day-level arm needs the keys8 key table"
    HMd = list(DY["HM"]); dep_t_, dep_m_ = DY["dep_t"], DY["dep_m"]; arr_t_, arr_m_ = DY["arr_t"], DY["arr_m"]
dlt_done = []
dlt = {(c, ar, s): np.zeros(Nf) for c in a.cfgs.split(",") for ar in ARMS for s in SEEDS}
dlz = {k_: np.zeros(Nf) for k_ in dlt if k_[1].endswith("Z")}
for hi, h in enumerate(MONTHS):
    if a.merge or sum(1 for q_ in dlt_done) >= a.maxfolds or (h not in EV and not a.keys12) or (a.judge_only and h not in ORIG7) or not TRN[mi == hi].any():
        continue
    S = np.hstack([ks_.build(None, rf, 'y', yf, mi, len(MONTHS), hi), ks_.build(None, rf, 'r', res, mi, len(MONTHS), hi)]).astype(np.float32)
    Fk = np.hstack([F.astype(np.float32), S]); del S
    te = np.where((mi == hi) & (TRN if a.keys12 else valid))[0]; tr = np.where((mi != hi) & (TRN if a.keys12 else valid))[0]
    for ar in ARMS:
        dayarm = DAYM and ar[-1] in "YGUVM" and ar.startswith("AQC")
        if dayarm:
            Yh = DA.day_features(dep_t_, dep_m_, arr_t_, arr_m_, DY["res"][HMd.index(h)].astype(float))
            if ar[-1] == "U":   # shuffled day-state within month (null)
                rng_ = np.random.default_rng(1000 + h); Yh = Yh.copy()
                for m_ in np.unique(monf):
                    ii_ = np.flatnonzero(monf == m_); Yh[ii_] = Yh[ii_[rng_.permutation(len(ii_))]]
        M = F.astype(np.float32) if ar == "m4" else (Fk if ar == "ks" else np.hstack([F.astype(np.float32) if NOKS[ar] else Fk, VIEW[ar]] + ([BULK] if "X" in ar.replace("ks", "") else [])))
        for c in a.cfgs.split(","):
            for s in SEEDS:
                g = lgb.LGBMRegressor(**{**dict(objective='huber', alpha=300, min_child_samples=500, learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=s, n_jobs=4, verbose=-1), **CFG[c]})
                Mfit = np.hstack([M, Yh.astype(np.float32)]) if dayarm and ar[-1] in "YU" else M
                if dayarm and ar[-1] == "M":   # post-hoc arm M: tree + 4 day columns (y_dmean, y_w3, y_wc3, y_dslow) with monotone +1 constraints (sign-stable response)
                    Mfit = np.hstack([M, Yh[:, [0, 3, 5, 2]].astype(np.float32)]); g.set_params(monotone_constraints=[0] * M.shape[1] + [1, 1, 1, 1])
                g.fit(Mfit[tr], res[tr], **({'categorical_feature': [len(cols), len(cols) + 1]} if a.lgbcat else {})); dlt[(c, ar, s)][te] = np.clip(g.predict(Mfit[te]), -600, 600)
                if dayarm and ar[-1] == "V":   # ROBUST ridge: snow rows excluded from the fit, target clipped +-300, standardised features winsorised at +-2.5, alpha 20000, output clipped +-150, no intercept
                    Xr = np.nan_to_num(Yh[:, [0, 3, 5, 2]].astype(float)); trv = tr[~snow[tr]]; mu_, sd_ = Xr[trv].mean(0), Xr[trv].std(0) + 1e-9; Xr = np.clip((Xr - mu_) / sd_, -2.5, 2.5)
                    rt = np.clip(res[trv], -300, 300); rt = rt - rt.mean(); beta = np.linalg.solve(Xr[trv].T @ Xr[trv] + 20000.0 * np.eye(Xr.shape[1]), Xr[trv].T @ rt)
                    dlt[(c, ar, s)][te] = dlt[(c, ar, s)][te] + np.clip(Xr[te] @ beta, -150, 150)
                if dayarm and ar[-1] == "G":   # per-airport linear response: ridge on 4 day-state columns (centred, standardised on the training rows), no intercept
                    Xr = np.nan_to_num(Yh[:, [0, 3, 5, 2]].astype(float)); mu_, sd_ = Xr[tr].mean(0), Xr[tr].std(0) + 1e-9; Xr = (Xr - mu_) / sd_
                    rt = res[tr] - res[tr].mean(); beta = np.linalg.solve(Xr[tr].T @ Xr[tr] + 2000.0 * np.eye(Xr.shape[1]), Xr[tr].T @ rt)
                    dlt[(c, ar, s)][te] = dlt[(c, ar, s)][te] + np.clip(Xr[te] @ beta, -300, 300)
                if ar.endswith("Z"):   # snow rows act ONLY through ref + the handover columns (separate small model on all training rows)
                    Mz = np.column_stack([ref, EX["R"]]).astype(np.float32); gz = lgb.LGBMRegressor(objective='huber', alpha=300, min_child_samples=500, learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.9, random_state=s, n_jobs=4, verbose=-1, num_leaves=8, n_estimators=200)
                    gz.fit(Mz[tr], res[tr]); dlz[(c, ar, s)][te] = np.clip(gz.predict(Mz[te]), -600, 600)
        if ar not in ("ks", "m4"):
            del M
    del Fk; gc.collect(); dlt_done.append(h); print('fold month', h, 'done', flush=True)


if a.merge:
    for f_ in a.merge.split(","):
        zz = np.load(f_)
        for k_ in zz.files:
            if k_.startswith("z") and k_ not in ("snow", "valid"):
                c_, ar_, s_ = k_[1:].split("_")[0], "_".join(k_[1:].split("_")[1:-1]), int(k_.split("_")[-1][1:])
                if (c_, ar_, s_) in dlz and np.abs(zz[k_]).sum() > 0: dlz[(c_, ar_, s_)] = zz[k_]
            elif k_ not in ("snow", "valid"):
                c_, ar_, s_ = k_.split("_")[0], "_".join(k_.split("_")[1:-1]), int(k_.split("_")[-1][1:])
                if (c_, ar_, s_) in dlt and np.abs(zz[k_]).sum() > 0: dlt[(c_, ar_, s_)] = zz[k_]


def rm(x, b): return np.sqrt(x) - np.sqrt(b)


def report(d, k, tag, dz=None):
    d = d.copy(); d[snow] = 0 if dz is None else dz[snow]; ns = (ref + k * d - yf) ** 2
    o = [f'{tag} k{k}: LTFM {np.sqrt(ns[valid].mean()) - np.sqrt(base[valid].mean()):+.2f}', f'ALL {rm((sse_oth + ns[valid].sum()) / N, (sse_oth + base[valid].sum()) / N):+.3f}',
         f'-top50 {rm((sse_oth + ns[mk].sum()) / (N - 50), (sse_oth + base[mk].sum()) / (N - 50)):+.3f}']
    for sn, (mm_, so, n_s) in seas.items(): o.append(f'{sn} {rm((so + ns[mm_].sum()) / n_s, (so + base[mm_].sum()) / n_s):+.3f}')
    nm = sum(((ns[(mi == i) & valid].sum() - base[(mi == i) & valid].sum()) < 0) for i, m_ in enumerate(MONTHS) if m_ in EV)
    o.append('months<0 %d/%d' % (nm, len(EV)))
    print(' | '.join(o), flush=True)
    return rm((sse_oth + ns[valid].sum()) / N, (sse_oth + base[valid].sum()) / N)


for c in a.cfgs.split(","):
    for ar in ARMS:
        for s in SEEDS:
            for k in (0.75, 1.0): report(dlt[(c, ar, s)], k, f'{c} {ar} s{s}', dlz.get((c, ar, s)))
    for k in (0.5, 0.75, 1.0):
        out = {ar: report(np.mean([dlt[(c, ar, s)] for s in SEEDS], 0), k, f'SEEDAVG {c} {ar}', (np.mean([dlz[(c, ar, s)] for s in SEEDS], 0) if ar.endswith('Z') else None)) for ar in ARMS}
        print(f'== {c} k{k}: seed-avg ALL ' + '  '.join(f'{ar} {v:+.3f}' for ar, v in out.items()), flush=True)
if a.eval12:
    mon_all = monf
    for c in a.cfgs.split(","):
        for ar in ARMS:
            for s_ in SEEDS + ["avg"]:
                d = (np.mean([dlt[(c, ar, q)] for q in SEEDS], 0) if s_ == "avg" else dlt[(c, ar, s_)]).copy()
                dz_ = None if not ar.endswith("Z") else (np.mean([dlz[(c, ar, q)] for q in SEEDS], 0) if s_ == "avg" else dlz[(c, ar, s_)])
                d[snow] = 0 if dz_ is None else dz_[snow]; ns = (ref + 0.75 * d - yf) ** 2
                o = []
                for nm_, msk in (("LTFM12", TRN), ("LTFM12-noFeb", TRN & (monf != 2)), ("LTFM7", TRN & np.isin(monf, ORIG7)), ("LTFM7-noFeb", TRN & np.isin(monf, ORIG7) & (monf != 2))):
                    o.append(f"{nm_} {np.sqrt(ns[msk].mean()) - np.sqrt(base[msk].mean()):+.3f}")
                for nm_, msk in (("LTFM17", TRN & np.isin(monf, [1, 7])), ("LTFM1", TRN & (monf == 1)), ("LTFM7m", TRN & (monf == 7))):
                    o.append(f"{nm_} {np.sqrt(ns[msk].mean()) - np.sqrt(base[msk].mean()):+.3f}")
                for nm_, mset in (("ALL17", [1, 7]), ("ALL7", ORIG7), ("ALL7-noFeb", [m_ for m_ in ORIG7 if m_ != 2])):
                    vm = np.isin(monf, mset) & TRN; om = np.isin(mon, mset) & (apts != 'LTFM'); so = ((p0[om] - y[om]) ** 2).sum(); n_ = om.sum() + vm.sum()
                    o.append(f"{nm_} {rm((so + ns[vm].sum()) / n_, (so + base[vm].sum()) / n_):+.3f}")
                if s_ == "avg":   # structural-bias check: mean residual before -> after by month (LTFM rows)
                    o.append("BIAS " + " ".join(f"m{m_}:{(yf - ref)[(monf == m_) & TRN].mean():+.0f}>{(yf - ref - 0.75 * d)[(monf == m_) & TRN].mean():+.0f}" for m_ in (1, 2, 3, 7, 8, 12)))
                print(f"EVAL12 {c} {ar} s{s_} k0.75: " + " | ".join(o), flush=True)
if a.keys12:   # new months: LTFM-only RMSE delta (no other-airport population for those months)
    for c in a.cfgs.split(","):
        for ar in ARMS:
            d = np.mean([dlt[(c, ar, s)] for s in SEEDS], 0); d[snow] = 0
            for k in (0.75,):
                ns = (ref + k * d - yf) ** 2
                print(f"NEWMONTHS {c} {ar} k{k}: " + " ".join(f"m{m} {np.sqrt(ns[(monf == m) & TRN].mean()) - np.sqrt(base[(monf == m) & TRN].mean()):+.2f}" for m in MONTHS), flush=True)
np.savez(I + f'agent_lv_screen_s{a.stage}_{a.ref}{a.tagx}.npz', **{f'{c}_{ar}_s{s}': v for (c, ar, s), v in dlt.items()}, **{f'z{c}_{ar}_s{s}': v for (c, ar, s), v in dlz.items()}, snow=snow, valid=valid)
