"""istarith 02 (copy of the istpcd 02 front end, unchanged recipe; extra columns = lib groups A+E): Istanbul views corrector with / without the plan-clock distance columns, month-held-out and temporal frames (experiment "istpcd", 2026-10-05).
Front end copied from agent_istjul_02_fold.py (itself agent_lv_02_screen.py): served recipe = views arm AQCEHSBR, S2 (16 leaves, 300 trees, huber alpha 300, min_child 500, rate 0.03, subsample 0.7, colsample 0.7),
key statistics of label and residual out of month (8-month key table keys8, served-style reference as the base), target = label - ref, prediction clipped +-600, 3 seeds, n_jobs pinned to 6.
Arms: ctrl = as shipped; pcd = + the 9 columns; shuf = + the same 9 with the prediction shuffled within month (draw = seed); rand = + 9 random N(0,1000) columns (draw = seed).
Frames: lomo = leave one month out over the 8 months; temporal = fit on months 1-4 and judge 7-10, and the reverse (key statistics from the fit half only).
Output data/interim/agent_istpcd_02_<frame>.npz (per held block: indices + per-seed clipped corrector values). Resumable. No 2026 label read."""
import argparse, gc, glob, os, sys, time
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "archive")
import agent_lks_te as TE  # noqa: E402
import agent_lv_views as VW  # noqa: E402
import agent_istarith_lib as LB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frame", default="lomo"); ap.add_argument("--arms", default="ctrl,wins,unp,shuf,rand"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--hold", default="1,2,3,4,7,8,9,10"); ap.add_argument("--threads", type=int, default=6); ap.add_argument("--tag", default="")
a = ap.parse_args()
I = "data/interim/"; ID = "MVT_ID_mvt"; C = pl.col
ARMS = a.arms.split(","); SEEDS = [int(s) for s in a.seeds.split(",")]
CFG = dict(num_leaves=16, n_estimators=(5 if os.environ.get("ISTARITH_SMOKE") else 300))
t0 = time.time()
def P(*x): print("[%6.0fs]" % (time.time() - t0), *x, flush=True)
K = np.load(I + "agent_lv_keys8.npz", allow_pickle=True); fid = K["fid"]; yf = K["yf"]; ref = K["ref"]; monf = K["monf"]; Kall = K["Kall"]; Nf = len(fid)
MONTHS = sorted(set(int(m) for m in monf)); P("months", MONTHS)
cols = ['hour', 'dow', 'minute_of_day', 'gap_sched', 'gap_lobt', 'gap_eobt', 'gap_aobt', 'dep_delay', 'sched_vs_eobt', 'lobt_minus_aobt', 'turn_sched_slack', 'turn_in_delay', 'turn_age', 'turn_taxi_in',
        'dep_30min', 'arr_30min', 'dep_60min', 'arr_60min', 'dep_rwy_headway', 'rwy_dep_active', 'rwy_dep_share', 'rwy_arr_active', 'rwy_mixed_mode', 'wake_heavy_60min', 'wake_prev_heavy', 'prev_rwy_gap',
        'wx_temp_c', 'wx_wind_kt', 'wx_visibility_km', 'wx_precip', 'wx_lowvis', 'wx_snow', 'wx_freezing', 'has_flight_plan', 'sched_demand_30min', 'late_apt_past60_mean', 'late_apt_past15_mean']
T = pl.read_parquet(I + 'train_features.parquet', columns=[ID, 'RUNWAY_mvt', 'WK_TBL_CAT_flt'] + cols)
X = pl.DataFrame({ID: fid}).join(T.select([ID, 'RUNWAY_mvt', 'WK_TBL_CAT_flt'] + cols), on=ID, how='left', maintain_order='left')
X = X.with_columns(pl.col('RUNWAY_mvt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('rwy'), pl.col('WK_TBL_CAT_flt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('wk'))
F = X.select(cols + ['rwy', 'wk']).to_numpy().astype(float); del T, X; gc.collect()
F = np.column_stack([F, ref, F[:, cols.index('gap_lobt')] - ref, F[:, cols.index('gap_sched')] - ref])
snow = (F[:, cols.index('wx_snow')] == 1) | (F[:, cols.index('wx_freezing')] == 1)
res = yf - ref; months = np.array(MONTHS); mi = np.searchsorted(months, monf); ks_ = TE.KS(Kall[:Nf])
mem = VW.load_fold(fid, monf, "ctrl", VW.MEM); qq = VW.load_fold(fid, monf, "ctrl", VW.QS); tt = VW.load_fold(fid, monf, "treat", VW.MEM)
EX = {}
ARM = "AQCEHSBR"
for L_, kk in VW.EXTRA.items():
    if L_ in ARM:
        for k_, col in zip(kk, VW.load_fold(fid, monf, "ctrl", kk).T): EX[k_] = col
def _align(Z, key="fold"):
    if len(Z["fold_ids"]) == len(fid) and (Z["fold_ids"] == fid).all(): return Z[key]
    pos = {v: i for i, v in enumerate(Z["fold_ids"].tolist())}; return Z[key][np.array([pos[v] for v in fid.tolist()])]
HO = np.load(I + "agent_lv_handover.npz", allow_pickle=True); EX["R"] = _align(HO)
VIEW, vn = VW.make_views(ref, mem, qq, tt, ARM, EX); P("view columns", len(vn))
TRN = np.ones(Nf, bool)
for _x in (mem, qq, tt, *[v_ for k_, v_ in EX.items() if k_ not in ("R", "W", "F", "T")]):
    TRN &= np.isfinite(_x).all(1) if _x.ndim == 2 else np.isfinite(_x)
P("trainable rows", int(TRN.sum()), "of", Nf)

# static per-row info (label-free): month / day of month / runway from the raw 2025 file
raw = pl.concat([pl.read_parquet(f, columns=[ID, "MVT_TIME_UTC_mvt", "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt"]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
raw = raw.with_columns(C("MVT_TIME_UTC_mvt").dt.day().alias("dom"))
J = pl.DataFrame({ID: fid}).join(raw, on=ID, how="left", maintain_order="left"); assert J["dom"].null_count() == 0
dom = J["dom"].to_numpy(); rwy_s = J["RUNWAY_mvt"].fill_null("?").to_numpy()
_, day_code = np.unique(np.array([f"{m}|{d}|{r}" for m, d, r in zip(monf.tolist(), dom.tolist(), rwy_s.tolist())]), return_inverse=True)
hour_code = np.nan_to_num(F[:, cols.index("hour")], nan=24).astype(int)
CELLS = [ks_.g[LB.KN.index("rwy_stand")][:Nf], ks_.g[LB.KN.index("rwy_hb")][:Nf], hour_code]
nm_ = len(MONTHS); Q_NEW = len(LB.NAMES)
NSH = []
for s in range(3):
    rng = np.random.default_rng(1000 + s); pvs = ref.copy()
    for m in np.unique(monf):
        ii = np.where(monf == m)[0]; pvs[ii] = ref[rng.permutation(ii)]
    NSH.append(pvs)
out = I + "agent_istarith_02_%s%s.npz" % (a.frame, a.tag)
OUTD = dict(np.load(out)) if os.path.exists(out) else {}
if a.frame == "lomo":
    BLOCKS = [("h%d" % h, [MONTHS.index(h)], [h]) for h in [int(x) for x in a.hold.split(",")]]
else:
    first = [m for m in MONTHS if m <= 4]; second = [m for m in MONTHS if m >= 7]
    BLOCKS = [("fit1234", [MONTHS.index(m) for m in second], second), ("fit78910", [MONTHS.index(m) for m in first], first)]
for name, hidx, hmon in BLOCKS:
    todo = [(arm, s) for arm in ARMS for s in SEEDS if f"{name}_{arm}_s{s}" not in OUTD]
    if not todo: continue
    pool = np.arange(Nf)
    S = np.hstack([ks_.build(None, pool, 'y', yf, mi, len(MONTHS), hidx), ks_.build(None, pool, 'r', res, mi, len(MONTHS), hidx)]).astype(np.float32)
    hold = np.isin(monf, hmon); use = ~hold
    te = np.where(hold & TRN)[0]; tr = np.where(~hold & TRN)[0]
    def new_for(pv):
        return np.hstack([LB.group_a(pv, S[:, :120]), LB.group_e(pv, hold, mi, ref, mi, use, nm_, CELLS, CELLS, day_code)])
    NEW = new_for(ref); NEWW = LB.winsor(NEW, tr)
    NS_ = [LB.winsor(new_for(NSH[s]), tr) for s in range(3)]
    P("block", name, "new cols", NEW.shape, "nan share", float(np.isnan(NEW).mean()), "winsor changed share of cells (fit rows)", float((NEW[tr] != NEWW[tr]).mean()))
    if name == (BLOCKS[0][0]) and not os.environ.get("ISTARITH_SMOKE"): np.savez(I + "agent_istarith_02_%s_cols_%s.npz" % (a.frame, name), NEW=NEW[:, :], names=np.array(LB.NAMES))
    M0 = np.hstack([F.astype(np.float32), S, VIEW]); del S; gc.collect()
    OUTD[f"{name}_idx"] = te
    for arm, s in todo:
        E = {"ctrl": None, "wins": NEWW, "unp": NEW, "shuf": NS_[s], "rand": np.random.default_rng(2000 + s).normal(0, 1000, (Nf, Q_NEW)).astype(np.float32)}[arm]
        M = M0 if E is None else np.hstack([M0, E])
        g = lgb.LGBMRegressor(**dict(objective='huber', alpha=300, min_child_samples=500, learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=s, n_jobs=a.threads, verbose=-1), **CFG)
        g.fit(M[tr], res[tr]); OUTD[f"{name}_{arm}_s{s}"] = np.clip(g.predict(M[te]), -600, 600).astype(np.float32)
        P("block", name, "arm", arm, "seed", s, "train", len(tr), "test", len(te), "cols", M.shape[1]); del M, g; gc.collect()
        np.savez(out, **OUTD)
    del M0; gc.collect()
OUTD["aux_dom"] = dom; OUTD["aux_snow"] = snow; OUTD["aux_y"] = yf; OUTD["aux_ref"] = ref; OUTD["aux_mon"] = monf
np.savez(out, **OUTD); P("done", out)
# reproducibility check of ctrl against the plan-clock experiment's ctrl fits (same recipe, same seeds)
old = I + "agent_istpcd_02_%s.npz" % a.frame
if os.path.exists(old):
    O = np.load(old)
    for k in sorted(OUTD):
        if "_ctrl_s" in k and k in O.files: P("ctrl vs istpcd ctrl", k, "max abs diff", float(np.abs(OUTD[k] - O[k]).max()))
