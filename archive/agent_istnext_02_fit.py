"""istnext 02 (experiment "istnext", 2026-10-06; designs frozen in docs/LAB-NOTES.md section 2): Istanbul (LTFM) views corrector, shipped recipe, with the designs' extra columns.
Front end copied unchanged from agent_istarith_02_fold.py (served recipe: views arm AQCEHSBR, 16 leaves, 300 trees, huber 300, min 500 per leaf, rate 0.03, bagging 0.7, feature fraction 0.7,
key statistics of label and residual out of month, target = label - ref, clip +-600, snow rows in the fit as shipped, n_jobs pinned).
usage: python archive/agent_istnext_02_fit.py JOB1,JOB2,...    JOB = ARM:lomo:HELD:SEED | ARM:temp:SIDE:SEED   (SIDE a = fit months 1-4 judge 7-10; b = the reverse)
ARM: ctrl | D1 | D1s | D2 | D2s | D3 | D3s | G<letters> (descriptive split of D1 by group, added after the gate was read)   (s = shuffled-block control: the design's new block permuted jointly among rows of the same month, draw = seed)
  D1 = A (7 keys) + E + K + Y + Q + S + R, 57 columns, winsorised 0.5 / 99.5 pct of the fit rows.   D3 = nine standardised positions clipped to [-3, 0].
  D2 = ONE column: output of the pooled response model fitted on the nine-airport rows (agent_istnext_01_pool.py), month-out; needs ARITH_I/agent_istnext_pool/*.txt.
New-column cell statistics use NON-SNOW pool rows only; a pool row leaves out its own month; a held row uses the fitted months; missing when the cell has 20 rows or fewer.
Inputs: ARITH_I (default data/interim/): agent_lv_keys8.npz, agent_lv_handover.npz, agent_istpcd_cols.npz, train_features.parquet, agent_istnext_pool/; also data/raw/training_2025-*.parquet, results/ltfmm/.
Outputs: ARITH_OUT (default data/interim/agent_istnext_out/): one file per fit <ARM>_s<seed>_<m<held>|t<side>>.npz (pred, rows, train / held diagnostics) + aux.npz. No 2026 label read."""
import gc, glob, os, sys, time
import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "archive")
import agent_lks_te as TE  # noqa: E402
import agent_lv_views as VW  # noqa: E402
import agent_arith3_02_cols as C3  # noqa: E402
import agent_istnext_lib as LB  # noqa: E402

I = os.environ.get("ARITH_I", "data/interim/"); OUT = os.environ.get("ARITH_OUT", I + "agent_istnext_out/"); os.makedirs(OUT, exist_ok=True)
POOL = I + "agent_istnext_pool/"
ID = "MVT_ID_mvt"; C = pl.col; THREADS = int(os.environ.get("BIGC_THREADS", 6))
CFG = dict(num_leaves=16, n_estimators=int(os.environ.get("ISTNEXT_NTREES", 300)))
A_KEYS = ["rwy_stand", "op", "op_rwy", "op_stand", "rwy_hb", "ty_stand", "op_hb"]
A_NAMES = [f"A_{p}_{k}" for k in A_KEYS for p in ("lvl", "z")]
t0 = time.time()
def P(*x): print("[%6.0fs]" % (time.time() - t0), *x, flush=True)
def fname(arm, kind, blk, seed): return OUT + "%s_s%d_%s%s.npz" % (arm, seed, "m" if kind == "lomo" else "t", blk)
JOBS = []
for j in sys.argv[1].split(","):
    arm, kind, blk, seed = j.split(":")
    if not os.path.exists(fname(arm, kind, blk, int(seed))): JOBS.append((arm, kind, blk, int(seed)))
if not JOBS:
    P("nothing to do"); sys.exit(0)
K = np.load(I + "agent_lv_keys8.npz", allow_pickle=True); fid = K["fid"]; yf = K["yf"]; ref = K["ref"]; monf = K["monf"]; Kall = K["Kall"]; KN8 = [str(x) for x in K["kn"]]; Nf = len(fid)
MONTHS = sorted(set(int(m) for m in monf)); P("months", MONTHS)
cols = ['hour', 'dow', 'minute_of_day', 'gap_sched', 'gap_lobt', 'gap_eobt', 'gap_aobt', 'dep_delay', 'sched_vs_eobt', 'lobt_minus_aobt', 'turn_sched_slack', 'turn_in_delay', 'turn_age', 'turn_taxi_in',
        'dep_30min', 'arr_30min', 'dep_60min', 'arr_60min', 'dep_rwy_headway', 'rwy_dep_active', 'rwy_dep_share', 'rwy_arr_active', 'rwy_mixed_mode', 'wake_heavy_60min', 'wake_prev_heavy', 'prev_rwy_gap',
        'wx_temp_c', 'wx_wind_kt', 'wx_visibility_km', 'wx_precip', 'wx_lowvis', 'wx_snow', 'wx_freezing', 'has_flight_plan', 'sched_demand_30min', 'late_apt_past60_mean', 'late_apt_past15_mean']
T = pl.read_parquet(I + 'train_features.parquet', columns=[ID, 'RUNWAY_mvt', 'WK_TBL_CAT_flt', 'is_weekend'] + cols)
X = pl.DataFrame({ID: fid}).join(T.select([ID, 'RUNWAY_mvt', 'WK_TBL_CAT_flt', 'is_weekend'] + cols), on=ID, how='left', maintain_order='left')
X = X.with_columns(pl.col('RUNWAY_mvt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('rwy'), pl.col('WK_TBL_CAT_flt').cast(pl.Categorical).to_physical().cast(pl.Float64).alias('wk'))
F = X.select(cols + ['rwy', 'wk']).to_numpy().astype(float); wkend = X["is_weekend"].to_numpy().astype(float); del T, X; gc.collect()
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
P("trainable rows", int(TRN.sum()), "of", Nf, "snow rows", int(snow.sum()))

# ---- label-free per-row inputs of the new columns: take-off time, day of month, Istanbul cells in the builder's layout ----
raw = pl.concat([pl.read_parquet(f, columns=[ID, "MVT_TIME_UTC_mvt", "PHASE_mvt", "ADEP_mvt"]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
raw = raw.unique(subset=[ID], keep="first").with_columns(C("MVT_TIME_UTC_mvt").dt.day().alias("dom"), C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"))
J = pl.DataFrame({ID: fid}).join(raw.select(ID, "dom", "t"), on=ID, how="left", maintain_order="left"); assert J["dom"].null_count() == 0
dom = J["dom"].to_numpy(); tko = J["t"].to_numpy().astype(np.int64); del raw, J
PC = np.load(I + "agent_istpcd_cols.npz", allow_pickle=True); assert (PC["fid"] == fid).all() and [str(n) for n in PC["names"]][:4] == ["d_SCHED", "d_LOBT", "d_IOBT", "d_EOBT"], [str(n) for n in PC["names"]]
kc = lambda k: Kall[:Nf, KN8.index(k)]  # noqa: E731
rwyc = np.nan_to_num(F[:, len(cols)], nan=-1)
D = dict(pv=ref.astype(np.float32), y=yf.astype(np.float64), mon=monf.astype(np.int64), apt=np.array(["LTFM"] * Nf), hour=F[:, cols.index("hour")], RUNWAY_mvt=rwyc, is_weekend=wkend,
         Kf=np.column_stack([kc("op"), kc("op_stand"), C3.factor(kc("type"), rwyc), kc("rwy_stand"), kc("cty")]).astype(np.int64), KN=["op_apt", "op_stand", "ty_rwy", "st_rwy", "cty"], Dpcd=PC["D"][:, :4].astype(np.float32), t=tko)
DAY = C3.day_block(D); nm_ = len(MONTHS)
aux = OUT + "aux.npz"
if not os.path.exists(aux): np.savez(aux, y=yf, ref=ref, mon=monf, snow=snow, dom=dom, trn=TRN, fid=fid)
LIM9 = BOOST = None
def pool_models():
    global LIM9, BOOST
    if BOOST is None:
        L = np.load(POOL + "limits.npz", allow_pickle=True); assert [str(n) for n in L["names"]] == LB.POS_NAMES
        LIM9 = (L["lo"], L["hi"]); BOOST = {}
    return LIM9, BOOST
def booster(j):
    if j not in BOOST: BOOST[j] = lgb.Booster(model_file=POOL + j + ".txt")
    return BOOST[j]

blocks = []
for arm, kind, blk, seed in JOBS:
    if (kind, blk) not in blocks: blocks.append((kind, blk))
for kind, blk in blocks:
    todo = [(a_, s_) for a_, k_, b_, s_ in JOBS if (k_, b_) == (kind, blk)]
    if kind == "lomo": hmon = [int(blk)]
    else: hmon = [m for m in MONTHS if m >= 7] if blk == "a" else [m for m in MONTHS if m <= 4]
    hidx = [MONTHS.index(m) for m in hmon]
    pool = np.arange(Nf)
    S = np.hstack([ks_.build(None, pool, 'y', yf, mi, nm_, hidx), ks_.build(None, pool, 'r', res, mi, nm_, hidx)]).astype(np.float32)
    hold = np.isin(monf, hmon); te = np.where(hold & TRN)[0]; tr = np.where(~hold & TRN)[0]
    use = ~hold & ~snow; pmask = np.array([m not in hmon for m in MONTHS])
    POS, pn = LB.position_block(D, use, pmask, day=DAY)
    A = []
    for k in A_KEYS:
        G = np.unique(kc(k), return_inverse=True)[1]
        ym = C3.loo_cell(G, yf.astype(np.float64), use, mi, nm_, pmask)[0]; sd = C3.loo_cell(G, res.astype(np.float64), use, mi, nm_, pmask)[1]
        A += [ref - ym, (ref - ym) / np.maximum(sd, 30.0)]
    B1 = np.column_stack([np.column_stack(A).astype(np.float32), POS]); lo, hi = LB.limits(B1, tr); B1 = np.clip(B1, lo, hi)
    B3 = np.clip(POS[:, [pn.index(n) for n in LB.Z9]], -3.0, 0.0)
    B2 = None
    if any(a_.startswith("D2") for a_, _ in todo):
        (l9, h9), _ = pool_models(); Xp = np.clip(POS, l9, h9); B2 = np.full((Nf, 1), np.nan, np.float32)
        for m in MONTHS:
            rr = np.where(monf == m)[0]; j = ("ha" if blk == "a" else "hb") if (kind == "temp" and m in hmon) else "m%d" % m
            B2[rr, 0] = booster(j).predict(Xp[rr])
        P("block", kind, blk, "D2 column: mean %.2f sd %.2f; corr with residual on fit rows %.4f, on held non-snow rows %.4f" % (B2.mean(), B2.std(), np.corrcoef(B2[tr, 0], res[tr])[0, 1], np.corrcoef(B2[te, 0][~snow[te]], res[te][~snow[te]])[0, 1]))
    P("block", kind, blk, "held months", hmon, "D1 cols", B1.shape[1], "nan share %.3f" % float(np.isnan(B1).mean()), "D3 nan share %.3f, share below 0: %.3f" % (float(np.isnan(B3).mean()), float((B3 < 0).mean())))
    cdump = OUT + "cols_%s%s.npz" % ("m" if kind == "lomo" else "t", blk)
    if blk in ("1", "10", "a") and not os.path.exists(cdump): np.savez(cdump, B1=B1[te], B3=B3[te], rows=te, names1=np.array(A_NAMES + pn), names3=np.array(LB.Z9), **({"B2": B2[te]} if B2 is not None else {}))
    M0 = np.hstack([F.astype(np.float32), S, VIEW]); del S; gc.collect()
    ns_tr = ~snow[tr]; ns_te = ~snow[te]
    for arm, seed in todo:
        base = arm.rstrip("s")
        if base.startswith("G"):   # descriptive only (chooses nothing): the D1 columns of the named groups, e.g. GAE, GKYQSR, GY
            E = B1[:, [i for i, n in enumerate(A_NAMES + pn) if n[0] in base[1:]]]
        else:
            E = {"ctrl": None, "D1": B1, "D2": B2, "D3": B3}[base]
        if E is not None and arm.endswith("s"):
            rng = np.random.default_rng(5000 + seed); E = E.copy()
            for m in MONTHS:
                ii = np.where(monf == m)[0]; E[ii] = E[rng.permutation(ii)]
        M = M0 if E is None else np.hstack([M0, E])
        g = lgb.LGBMRegressor(**dict(objective='huber', alpha=300, min_child_samples=500, learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=seed, n_jobs=THREADS, verbose=-1), **CFG)
        g.fit(M[tr], res[tr]); pte = np.clip(g.predict(M[te]), -600, 600).astype(np.float32); ptr = np.clip(g.predict(M[tr]), -600, 600)
        rm = lambda v: float(np.sqrt((v ** 2).mean()))  # noqa: E731
        dg = np.array([rm(res[tr][ns_tr]), rm((res[tr] - ptr)[ns_tr]), rm(res[te][ns_te]), rm((res[te] - pte)[ns_te])])
        imp = g.booster_.feature_importance("gain"); newshare = float(imp[M0.shape[1]:].sum() / max(imp.sum(), 1e-9))
        p = fname(arm, kind, blk, seed); np.savez(p + ".tmp.npz", pred=pte, rows=te, diag=dg, newshare=newshare); os.replace(p + ".tmp.npz", p)
        P("fit", arm, kind, blk, "seed", seed, "cols", M.shape[1], "non-snow RMSE fit rows %.2f -> %.2f, held %.2f -> %.2f; gain share of new columns %.3f" % (dg[0], dg[1], dg[2], dg[3], newshare)); del M, g; gc.collect()
    del M0; gc.collect()
P("done")
