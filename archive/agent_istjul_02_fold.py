"""istjul 02 (copy of agent_lv_02_screen.py head + anomalous-window arms A/B/C; held-out months only, window = 5-26 July 2025).
lv step 2: LOMO screen of the agent_lks corrector WITH variant views (docs/LAB-NOTES.md).
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
ap.add_argument("--anomarms", default="A,B,C"); ap.add_argument("--hold", default="1,6,8,9"); ap.add_argument("--shrink", type=float, default=30.0)
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

# ======================= istjul tail =======================
import glob
C = pl.col
HOLD = [int(x) for x in a.hold.split(",")]; AA = a.anomarms.split(","); SEEDS = [int(s) for s in a.seeds.split(",")]
raw = pl.concat([pl.read_parquet(f, columns=[ID, "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "PHASE_mvt", "ADEP_mvt"]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
raw = raw.with_columns(C("MVT_TIME_UTC_mvt").dt.day().alias("dom"), C("MVT_TIME_UTC_mvt").dt.hour().alias("hr"), C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("ts"))
J = pl.DataFrame({ID: fid, "i": np.arange(Nf)}).join(raw, on=ID, how="left", maintain_order="left")
assert J.height == Nf and J["dom"].null_count() == 0, J["dom"].null_count()
dom = J["dom"].to_numpy(); rwy_s = J["RUNWAY_mvt"].fill_null("?").to_numpy(); hr = J["hr"].to_numpy(); ts = J["ts"].to_numpy()
# 30-min departure density from the whole LTFM departure list
allt = np.sort(raw["ts"].to_numpy()); dens = np.searchsorted(allt, ts + 1800) - np.searchsorted(allt, ts - 1800); db = dens // 10 * 10
snowrow = snow
win = (monf == 7) & (dom >= 5) & (dom <= 26)
print("window rows", int(win.sum()), "of July", int((monf == 7).sum()), flush=True)
# normal expectation by runway x density bin x hour from months other than 2 and 7 (non-snow, y < 3600)
ok_ = (~snowrow) & (yf < 3600) & ~np.isin(monf, [2, 7])
key = np.array([f"{r}|{d}|{h}" for r, d, h in zip(rwy_s, db, hr)])
ku, kinv = np.unique(key, return_inverse=True)
cs = np.bincount(kinv[ok_], weights=yf[ok_], minlength=len(ku)); cn = np.bincount(kinv[ok_], minlength=len(ku)).astype(float)
ex = np.where(cn[kinv] >= 15, cs[kinv] / np.maximum(cn[kinv], 1), np.nan)
# per (day, runway group) shift in the window
rg = np.where(np.isin(rwy_s, ["35L"]), "35L", np.where(rwy_s == "36", "36", "oth"))
shift = np.zeros(Nf); okw = win & (~snowrow) & np.isfinite(ex) & (yf < 3600)
for dd_ in range(5, 27):
    for g_ in ("35L", "36", "oth"):
        m_ = win & (dom == dd_) & (rg == g_); mo = okw & (dom == dd_) & (rg == g_)
        if mo.sum() == 0: continue
        sh = (yf[mo] - ex[mo]).mean() * mo.sum() / (mo.sum() + a.shrink)
        shift[m_] = max(sh, 0.0)
print("window shift: mean over window rows %.1f s; by group" % shift[win].mean(), {g_: round(float(shift[win & (rg == g_)].mean()), 1) for g_ in ("35L", "36", "oth")}, flush=True)
# arms
yv = {"A": yf.copy(), "B": yf.copy(), "C": yf - shift}
excl = {"A": np.zeros(Nf, bool), "B": win.copy(), "C": np.zeros(Nf, bool)}
resv = {k: yv[k] - ref for k in yv}
elig = {k: TRN & ~excl[k] for k in yv}
OUTD = {}
for hi, h in enumerate(MONTHS):
    if h not in HOLD or not TRN[mi == hi].any(): continue
    for arm in AA:
        pool = np.where(~excl[arm])[0]
        S = np.full((Nf, 2 * 6 * ks_.K), np.nan, np.float32)
        S[pool] = np.hstack([ks_.build(None, pool, 'y', yv[arm][pool], mi[pool], len(MONTHS), hi), ks_.build(None, pool, 'r', resv[arm][pool], mi[pool], len(MONTHS), hi)]).astype(np.float32)
        M = np.hstack([F.astype(np.float32), S, VIEW["AQCEHSBR"]]); del S
        te = np.where((mi == hi) & TRN)[0]; tr = np.where((mi != hi) & elig[arm])[0]
        for s in SEEDS:
            g = lgb.LGBMRegressor(**{**dict(objective='huber', alpha=300, min_child_samples=500, learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=s, n_jobs=6, verbose=-1), **CFG["S2"]})
            g.fit(M[tr], resv[arm][tr]); dd = np.zeros(Nf); dd[te] = np.clip(g.predict(M[te]), -600, 600); OUTD[(arm, h, s)] = dd[te]
        OUTD[("idx", h)] = te; del M; gc.collect()
        print("held", h, "arm", arm, "done; train rows", len(tr), flush=True)
    np.savez(I + "agent_istjul_02_fold%s.npz" % a.tagx, **{"_".join(map(str, k)): v for k, v in OUTD.items()})
# report
def rmse(e): return float(np.sqrt(e.mean()))
rng = np.random.default_rng(0)
dayid = dom + 100 * monf
print("\nIstanbul RMSE by held month (non-snow, y<=inf), served-style ref + 0.75 x corrector mean of %d seeds" % len(SEEDS))
SS = {}
for h in HOLD:
    if ("idx", h) not in OUTD: continue
    te = OUTD[("idx", h)]; ns = ~snowrow[te]
    for arm in AA: SS[(arm, h)] = np.mean([OUTD[(arm, h, s)] for s in SEEDS], 0)
    base_e = (yf[te] - ref[te]) ** 2
    line = f"held {h:2d} n {ns.sum():6d} ref {rmse(base_e[ns]):7.2f}"
    for arm in AA: line += f" | {arm} {rmse(((yf[te] - ref[te] - 0.75 * SS[(arm, h)]) ** 2)[ns]):7.2f}"
    for s in SEEDS:
        line += " | seeds " + " ".join(f"{arm}{s}:{rmse(((yf[te] - ref[te] - 0.75 * OUTD[(arm, h, s)]) ** 2)[ns]):.2f}" for arm in AA)
    print(line, flush=True)
def cmp(h_list, a_, b_, label):
    d = []; days = []; ys = []; ea = []; eb = []
    for h in h_list:
        te = OUTD[("idx", h)]; ns = ~snowrow[te]; te = te[ns]
        pa = ref[te] + 0.75 * SS[(a_, h)][ns]; pb = ref[te] + 0.75 * SS[(b_, h)][ns]
        ea.append((yf[te] - pa) ** 2); eb.append((yf[te] - pb) ** 2); days.append(dayid[te]); ys.append(yf[te])
    ea = np.concatenate(ea); eb = np.concatenate(eb); dy = np.concatenate(days); yy = np.concatenate(ys)
    gain = rmse(ea) - rmse(eb)    # positive = b better
    u, inv = np.unique(dy, return_inverse=True); sa = np.bincount(inv, weights=ea); sb = np.bincount(inv, weights=eb); cn_ = np.bincount(inv).astype(float)
    bs = []
    for _ in range(1000):
        k = rng.integers(0, len(u), len(u)); bs.append(np.sqrt(sa[k].sum() / cn_[k].sum()) - np.sqrt(sb[k].sum() / cn_[k].sum()))
    bs = np.array(bs); P = float((bs > 0).mean())
    dd = ea - eb; tot = dd.sum()
    body = dd[yy < 1500].sum() / tot if tot != 0 else np.nan; tail = dd[yy >= 1800].sum() / tot if tot != 0 else np.nan
    top20 = np.sort(dd)[::-1][:20].sum() / tot if tot != 0 else np.nan
    print(f"{label}: {b_} vs {a_}: Istanbul RMSE gain {gain:+.3f} s (pos = {b_} better) | day-boot P {P:.2f} [{np.percentile(bs, 5):+.2f},{np.percentile(bs, 95):+.2f}] | share of SSE gain body(y<1500) {body:+.2f} mid {1-body-tail:+.2f} tail(y>=1800) {tail:+.2f} top20 {top20:+.2f} | rows {len(yy)}", flush=True)
    return gain
have = [h for h in HOLD if ("idx", h) in OUTD]
for h in have:
    for b_ in AA:
        if b_ != "A": cmp([h], "A", b_, f"held {h}")
summer = [h for h in have if h in (6, 8, 9)]
if summer:
    for b_ in AA:
        if b_ != "A": cmp(summer, "A", b_, "summer pooled (6,8,9)")
