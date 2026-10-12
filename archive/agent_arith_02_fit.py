"""arith step 2 (experiment "arith", 2026-10-04): corrector replica (same recipe as agent_pcd_02_fit.py, full capacity unless the arm says reduced) with the nine pcd columns ALWAYS present and
a fixed list of arithmetic columns (groups A-E, see docs/LAB-NOTES.md section 1) added on top. Our own copy; agent_pcd_* files are read only.
usage: python agent_arith_02_fit.py JOBS   JOBS = comma list of ARM:lomo:HELD:SEED or ARM:temp:SIDE:SEED
ARM = "<groups>-p<prot>-s<shuf>-r<reduced>" e.g. "ABCDE-pn-s0-r1": groups = letters from ABCDE or "0" for none (pcd columns only); prot = n none | w winsorise 0.5/99.5 pct of the non-held rows |
  c null the new columns on input-defined catastrophic-possible rows (gap_sched > 3600, or gap_lobt > 3600, or no flight plan); shuf = 1: the new block permuted jointly within airport-month
  (pcd columns stay true), draw = seed; reduced = 1: learning rate 0.10 instead of 0.03 (cheap screen, the control is then also reduced).
Output data/interim/agent_arith_out/<ARM>_s<seed>_m<held>.npz (pred, rows, bi, gate (|pv - out-of-month stand x runway label mean| < 900, label-free), cmask (the (c) mask), plus names).
Group definitions are in col_block() below; every column is a pure function of the corrector's own row inputs plus out-of-month statistics built the corrector's way."""
import os
import sys
import json
import gc
import time
import warnings
import numpy as np
import polars as pl
import lightgbm as lgb

warnings.filterwarnings("ignore")
sys.path.insert(0, "archive")
import agent_bigc_te as TE  # noqa: E402

I = os.environ.get("ARITH_I", "data/interim/")
OUT = os.environ.get("ARITH_OUT", I + "agent_arith_out/")
NR = int(os.environ.get("ARITH_NR", 1500))   # max rounds (1500 = the plan; smaller only for wiring checks)
os.makedirs(OUT, exist_ok=True)
THREADS = int(os.environ.get("BIGC_THREADS", 6))
t0 = time.time()


def P(*a):
    print("[%5.0fs]" % (time.time() - t0), *a, flush=True)


names = json.load(open(I + "agent_yearfp_names.json"))["names"]
NI = {n: i for i, n in enumerate(names)}
z = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True)
cols = [str(c) for c in z["cols"]]
catidx = [int(i) for i in z["cat"]]
y = z["y"].astype(float)
pv = z["pv"].astype(np.float32)
mon = z["mon"]
apt = np.array([str(a) for a in z["apt"]])
ids = z["ids"].astype(float)
N = len(y)
months = sorted(set(mon.tolist()))
r = np.clip(y - pv, -900, 900)
kcols = ["ADEP_mvt"] + TE.KEYCOLS
Xk = np.column_stack([z["X"][:, cols.index(c)] for c in kcols])
kf = TE.keys_from(Xk, kcols)
hour = z["X"][:, cols.index("hour")].astype(np.int64)
gap_sched = z["X"][:, cols.index("gap_sched")].astype(np.float32)
gap_lobt = z["X"][:, cols.index("gap_lobt")].astype(np.float32)
hasfp = z["X"][:, cols.index("has_flight_plan")].astype(np.float32)
del Xk
kk = np.load(I + "agent_bigc13_keys2.npz", allow_pickle=True)
KN = [str(k) for k in kk["names"]]
k0 = np.load(I + "agent_bigc12_keys.npz", allow_pickle=True)
vf = np.load(I + "agent_bigc12_vfac_clean5.npz", allow_pickle=True)
ex = np.load(I + "agent_bigc12_extra.npz", allow_pickle=True)
e2 = np.load(I + "agent_bigc12_extra2.npz", allow_pickle=True)
ymu_apt = {a: y[k0["apt_f"] == a].mean() for a in set(k0["apt_f"].tolist())}
ymu_f = np.array([ymu_apt[a] for a in k0["apt_f"]])
Kf = kk["Kf"]
C = np.load(I + os.environ.get("ARITH_PCD", "agent_pcd_cols.npz"), allow_pickle=True)
Dpcd = C["D"]
assert Dpcd.shape == (N, 9) and np.array_equal(C["ids"].astype(float), ids), "pcd columns are not in fold row order"
fl = pl.read_parquet(I + "agent_aobtflag_flags.parquet").filter(pl.col("yr") == 2025).select("id", "K")
Km = dict(zip(fl["id"].to_list(), fl["K"].to_list()))
Kdef = np.array([Km.get(v, np.nan) if Km.get(v, None) is not None else np.nan for v in ids.tolist()], np.float32)
del fl, Km
CFG = dict(num_leaves=511, min_data_in_leaf=150, lambda_l2=100.0, feature_fraction=0.7)

# ----- fixed candidate list (written before any scoring) -----
KS_A = ["st_rwy", "st_op_rwy", "op_apt", "op_rwy", "st_hb", "ty_rwy", "op_stand"]          # 7 keys: stand x runway, stand x operator x runway, operator x airport, operator x runway, stand x hour bucket, type x runway, operator x stand
VPOINT = ["S1", "SEQ", "a_cap", "a_nobx", "a_noosm", "a_noseq", "a_notrk", "cap15", "cap255", "cb_sa", "dart_sa", "hub100", "hub30", "knn", "l1_sa", "logt_sa", "mlp", "rf", "ridge", "xgb_sa"]
VQ = ["q05", "q10", "q25", "q40", "q60", "q75", "q90", "q95", "xq10", "xq50", "xq90", "cq10", "cq50", "cq90"]
PAIRS = [("mlp", "rf"), ("xgb_sa", "rf"), ("knn", "mlp"), ("ridge", "xgb_sa"), ("logt_sa", "l1_sa"), ("cb_sa", "xgb_sa"), ("S1", "SEQ"), ("hub30", "hub100"), ("cap15", "cap255"), ("a_noosm", "a_nobx")]
ADSB = ["m_e_stand", "m_e_appear", "m_e_ci20", "m_e_dend", "m_e_mv10", "m_e_pushend", "m_e_start", "m_e_revany", "ci_150", "appear_raw"]
ADSB_SET = ["m_e_stand", "m_e_appear", "m_e_ci20", "m_e_dend", "m_e_mv10", "m_e_pushend", "m_e_start", "m_e_revany"]
ADSB_PAIRS = [("m_e_start", "m_e_appear"), ("m_e_pushend", "m_e_stand"), ("m_e_ci20", "m_e_appear")]
PVI = NI["pv"]
KI = {k: i for i, k in enumerate(KN)}
GNAMES = {"A": [], "B": [], "C": [], "D": [], "E": []}
for k in KS_A:
    GNAMES["A"] += ["A_pv-ym_" + k, "A_z_" + k]
GNAMES["B"] = ["B_mean_pt", "B_sd_pt", "B_range_pt", "B_pos_share", "B_med_all", "B_mean_over_sd", "B_q50_minus_meanpt"] + ["B_%s-%s" % p for p in PAIRS]
GNAMES["C"] = ["C_pv-" + a for a in ADSB] + ["C_max-min", "C_sd"] + ["C_%s-%s" % p for p in ADSB_PAIRS]
GNAMES["D"] = ["D_pv-Kdef", "D_Kdef-ym_st_rwy"]
GNAMES["E"] = ["E_pvdev_st_rwy", "E_pvz_st_rwy", "E_pvdev_apt_hour", "E_pvz_apt_hour"]
# cell codes for the pv-within-cell columns (E)
_ah = np.array([a + "_%02d" % h for a, h in zip(apt, hour)])
_, G_AH = np.unique(_ah, return_inverse=True)
del _ah
G_SR = np.unique(Kf[:, KI["st_rwy"]], return_inverse=True)[1]
MI = np.searchsorted(months, mon)
_ag = np.array([a + "_%d" % m for a, m in zip(apt, mon.tolist())])
SHUF_GROUPS = [np.where(_ag == g)[0] for g in np.unique(_ag)]
del _ag


def pv_cell_block(use, G, pvv, pool_months_mask):
    """pv deviation from the cell's mean pv over pooled rows `use` (own month excluded when the row's month is in the pool): (dev, z). pool_months_mask: bool per month index in the pool."""
    ng = G.max() + 1
    nm = len(months)
    idx = G * nm + MI
    S = np.bincount(idx[use], weights=pvv[use], minlength=ng * nm).reshape(ng, nm)
    Q = np.bincount(idx[use], weights=pvv[use].astype(np.float64) ** 2, minlength=ng * nm).reshape(ng, nm)
    Nn = np.bincount(idx[use], minlength=ng * nm).reshape(ng, nm).astype(np.float64)
    St, Qt, Nt = S.sum(1), Q.sum(1), Nn.sum(1)
    inpool = pool_months_mask[MI]
    s = St[G] - np.where(inpool, S[G, MI], 0.0)
    q = Qt[G] - np.where(inpool, Q[G, MI], 0.0)
    n = Nt[G] - np.where(inpool, Nn[G, MI], 0.0)
    mean = np.where(n > 20, s / np.maximum(n, 1), np.nan)
    sd = np.sqrt(np.maximum(q / np.maximum(n, 1) - (s / np.maximum(n, 1)) ** 2, 0))
    dev = pvv - mean
    return dev.astype(np.float32), (dev / np.maximum(sd, 30.0)).astype(np.float32)


def pv_cell_serve(Gp, pvp, Gs, pvs):
    """serve rows: deviation of pv from the mean pv of the pooled 2025 rows in its cell (same formula as pv_cell_block with no own month to exclude)"""
    ng = max(Gp.max(), Gs.max()) + 1
    n = np.bincount(Gp, minlength=ng).astype(np.float64)
    s_ = np.bincount(Gp, weights=pvp, minlength=ng)
    q = np.bincount(Gp, weights=pvp.astype(np.float64) ** 2, minlength=ng)
    mean = np.where(n > 20, s_ / np.maximum(n, 1), np.nan)[Gs]
    sd = np.sqrt(np.maximum(q / np.maximum(n, 1) - (s_ / np.maximum(n, 1)) ** 2, 0))[Gs]
    dev = pvs - mean
    return dev.astype(np.float32), (dev / np.maximum(sd, 30.0)).astype(np.float32)


def col_block(groups, Xv, s112, use, pool_mask, pv=pv, serve=None):
    """Xv: fold-space accessor dict of the base inputs (full N rows); s112: (N,112) key statistics; returns (N, ncols) float32 in fold-row order and the list of names"""
    out, nm_ = [], []
    f = lambda n: Xv[n]  # noqa: E731
    stat = lambda k, j: s112[:, KI[k] * 7 + j].astype(np.float64)  # noqa: E731
    for g in groups:
        if g == "A":
            for k in KS_A:
                ym, sd = stat(k, 6), stat(k, 2)
                out += [pv - ym, (pv - ym) / np.maximum(sd, 30.0)]
        elif g == "B":
            Dp = np.column_stack([f("vf_d_" + v) for v in VPOINT]).astype(np.float64)
            Dq = np.column_stack([f("vf_d_" + v) for v in VQ]).astype(np.float64)
            Da = np.column_stack([Dp, Dq])
            mp = np.nanmean(Dp, 1)
            sdp = np.nanstd(Dp, 1)
            out += [mp, sdp, np.nanmax(Dp, 1) - np.nanmin(Dp, 1), np.nanmean(Dp > 0, 1), np.nanmedian(Da, 1), mp / np.maximum(sdp, 5.0), f("vf_d_xq50") - mp]
            out += [f("vf_d_" + a) - f("vf_d_" + b) for a, b in PAIRS]
        elif g == "C":
            Ds = np.column_stack([f(a) for a in ADSB_SET]).astype(np.float64)
            out += [pv - f(a) for a in ADSB]
            out += [np.nanmax(Ds, 1) - np.nanmin(Ds, 1), np.nanstd(Ds, 1)]
            out += [f(a) - f(b) for a, b in ADSB_PAIRS]
        elif g == "D":
            out += [pv - Kdef, Kdef - stat("st_rwy", 6)]
        elif g == "E":
            if serve is None:
                d1, z1 = pv_cell_block(use, G_SR, pv, pool_mask)
                d2, z2 = pv_cell_block(use, G_AH, pv, pool_mask)
            else:   # serve rows against the pooled 2025 rows (serve["pool"] = (Gsr_pool, Gah_pool, pv_pool, Gsr_s, Gah_s))
                d1, z1 = pv_cell_serve(serve[0], serve[2], serve[3], pv)
                d2, z2 = pv_cell_serve(serve[1], serve[2], serve[4], pv)
            out += [d1, z1, d2, z2]
        nm_ += GNAMES[g]
    return np.column_stack(out).astype(np.float32), nm_


def params(seed, reduced):
    return dict(objective="l2", learning_rate=0.10 if reduced else 0.03, bagging_fraction=0.5, bagging_freq=1, max_bin=63, num_threads=THREADS, verbose=-1, seed=seed, **CFG)


def parse(arm):
    t = arm.split("-")
    return dict(nopcd=(t[0] == "N"), groups=[] if t[0] in ("0", "N") else list(t[0]), prot=t[1][1:], shuf=int(t[2][1:]), red=int(t[3][1:]))


def run(arm, perm, ntr, nva, seed, stat, use, pool_mask, h_rows, path, extra_save):
    a = parse(arm)
    s4, s112 = stat
    # fold-space accessor of the label-free inputs
    Xv = {}
    need = set()
    for g in a["groups"]:
        if g == "B":
            need |= {"vf_d_" + v for v in VPOINT + VQ}
        if g == "C":
            need |= set(ADSB)
    for n in need:
        src = vf["Xf"] if n.startswith("vf_") else (z["X"] if n in cols else ex["Xf"])
        nmlist = ([str(c) for c in ex["cols"]] if (not n.startswith("vf_") and n not in cols) else None)
        if n.startswith("vf_"):
            j = NI[n] - 318
        elif n in cols:
            j = cols.index(n)
        else:
            j = nmlist.index(n)
        Xv[n] = src[:, j].astype(np.float32)
    blk, bn = (None, [])
    if a["groups"]:
        blk, bn = col_block(a["groups"], Xv, s112, use, pool_mask)
        if a["shuf"]:
            rng = np.random.default_rng(5000 + seed)
            for ii in SHUF_GROUPS:
                blk[ii] = blk[rng.permutation(ii)]
        if a["prot"] == "w":
            sub = np.where(use)[0][::7]
            lo, hi = np.nanpercentile(blk[sub], [0.5, 99.5], axis=0)
            blk = np.clip(blk, lo, hi)
        elif a["prot"] == "c":
            blk[(gap_sched > 3600) | (gap_lobt > 3600) | (hasfp == 0)] = np.nan
    del Xv
    nx = 0 if blk is None else blk.shape[1]
    npc = 0 if a["nopcd"] else 9   # arm "N": plain shipped control replica (no pcd columns)
    X = np.empty((N, 370 + npc + nx), np.float32)
    X[:, :140] = z["X"][perm]
    X[:, 140:192] = ex["Xf"][perm]
    X[:, 192:194] = e2["Xf"][perm]
    X[:, 194:198] = s4[perm]
    X[:, 198:310] = s112[perm]
    X[:, 310:318] = kk["Df"][perm]
    X[:, 318:370] = vf["Xf"][perm]
    if npc:
        X[:, 370:379] = Dpcd[perm]
    if nx:
        X[:, 370 + npc:] = blk[perm]
    del blk
    rp = r[perm].astype(np.float32)
    dt = lgb.Dataset(X[:ntr], rp[:ntr], categorical_feature=catidx, params=dict(max_bin=63, verbose=-1))
    dv = lgb.Dataset(X[ntr:ntr + nva], rp[ntr:ntr + nva], categorical_feature=catidx, reference=dt)
    dt.construct()
    dv.construct()
    Xe = X[ntr + nva:].copy()
    del X
    gc.collect()
    bst = lgb.train(params(seed, a["red"]), dt, NR, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = bst.predict(Xe, num_iteration=bst.best_iteration)
    gate = np.abs(pv[h_rows] - s112[h_rows, KI["st_rwy"] * 7 + 6]) < 900
    cm = (gap_sched[h_rows] > 3600) | (gap_lobt[h_rows] > 3600) | (hasfp[h_rows] == 0)
    np.savez(path + ".tmp.npz", pred=pred, rows=h_rows, bi=bst.best_iteration, gate=gate, cmask=cm, bnames=np.array(bn), **extra_save)
    os.replace(path + ".tmp.npz", path)
    P("fit", arm, "seed", seed, "rounds", bst.best_iteration, "ncols", Xe.shape[1], "gate share %.4f" % gate.mean())


def lomo(arm, h, seed):
    path = OUT + "%s_s%d_m%d.npz" % (arm, seed, h)
    if os.path.exists(path):
        P("skip", path)
        return
    nxt = [k for k in months if k != h]
    val = min([k for k in nxt if k > h] or nxt)
    tr_i = np.where((mon != h) & (mon != val))[0]
    va_i = np.where(mon == val)[0]
    te_i = np.where(mon == h)[0]
    perm = np.concatenate([tr_i, va_i, te_i])
    T2 = TE.Te2(Kf)
    s4 = TE.te_fold(kf, r, mon, h, months)
    s112 = T2.fold(r, y, ymu_f, mon, h, months)
    use = mon != h
    pool_mask = np.array([m != h for m in months])
    run(arm, perm, len(tr_i), len(va_i), seed, (s4, s112), use, pool_mask, te_i, path, {})


def temporal(arm, side, seed):
    path = OUT + "%s_s%d_t%s.npz" % (arm, seed, side)
    if os.path.exists(path):
        P("skip", path)
        return
    fit_m = [1, 2, 3, 4, 5, 6] if side == "a" else [7, 8, 9, 10, 11, 12]
    val = 6 if side == "a" else 7
    Ts = np.where(np.isin(mon, fit_m))[0]
    Hs = np.where(~np.isin(mon, fit_m))[0]
    tr_i = Ts[mon[Ts] != val]
    va_i = Ts[mon[Ts] == val]
    perm = np.concatenate([tr_i, va_i, Hs])
    T2 = TE.Te2(Kf[Ts])
    s4 = np.empty((N, 4), np.float32)
    s112 = np.empty((N, 112), np.float32)
    s4[Ts] = TE.te_loo(kf[Ts], r[Ts], mon[Ts], fit_m)
    s112[Ts] = T2.fold(r[Ts], y[Ts], ymu_f[Ts], mon[Ts], None, fit_m)
    s4[Hs] = TE.te_serve(kf[Ts], r[Ts], kf[Hs])
    s112[Hs] = T2.serve(r[Ts], y[Ts], mon[Ts], fit_m, Kf[Hs], ymu_f[Hs])
    use = np.isin(mon, fit_m)
    pool_mask = np.array([m in fit_m for m in months])
    run(arm, perm, len(tr_i), len(va_i), seed, (s4, s112), use, pool_mask, Hs, path, {})


def serve(arm, seed, nr):
    """SERVING fit (as modal serve_one of the pcd experiment): all 12 months of 2025 with leave-own-month-out key statistics, predicts the January/July 2026 rows.
    Group blocks for the serve rows use statistics from all 2025 months. Returns (pred, names, importance). Groups D, shuffled blocks are not supported."""
    a = parse(arm)
    assert "D" not in a["groups"] and not a["shuf"] and a["prot"] in ("n", "w", "c"), arm
    si = np.load(I + os.environ.get("ARITH_SERVE", "agent_bigc13_serve_in.npz"), allow_pickle=True)
    colsS = [str(c) for c in si["cols_pre"]]
    Xs = si["Xs_pre"]
    assert colsS == cols + [str(c) for c in ex["cols"]] + ["d_NEW", "d_ET"], "serve/fold column mismatch"
    assert len(Xs) == len(si["ids"])
    cS = lambda n: Xs[:, colsS.index(n)]   # noqa: E731
    Xks = np.column_stack([cS(c) for c in kcols])
    ks = TE.keys_from(Xks, kcols)
    pvs = si["pv26"].astype(np.float32)
    aptS = np.array([str(v) for v in si["apt"]])
    hourS = cS("hour").astype(np.int64)
    ymu_s = np.array([ymu_apt.get(v, y.mean()) for v in k0["apt_s"]])
    T2 = TE.Te2(Kf)
    s4f = TE.te_loo(kf, r, mon, months)
    s4s = TE.te_serve(kf, r, ks)
    s112f = T2.fold(r, y, ymu_f, mon, None, months)
    s112s = T2.serve(r, y, mon, months, kk["Ks"], ymu_s)
    use = np.ones(N, bool)
    pool_all = np.ones(len(months), bool)
    names_x = []
    blk = blks = None
    if a["groups"]:
        need = set()
        for g in a["groups"]:
            if g == "B":
                need |= {"vf_d_" + v for v in VPOINT + VQ}
            if g == "C":
                need |= set(ADSB)
        XvF, XvS = {}, {}
        for n in need:
            if n.startswith("vf_"):
                j = NI[n] - 318
                XvF[n], XvS[n] = vf["Xf"][:, j].astype(np.float32), vf["Xs"][:, j].astype(np.float32)
            else:
                XvS[n] = cS(n).astype(np.float32)
                XvF[n] = (z["X"][:, cols.index(n)] if n in cols else ex["Xf"][:, [str(c) for c in ex["cols"]].index(n)]).astype(np.float32)
        blk, names_x = col_block(a["groups"], XvF, s112f, use, pool_all)
        sv = None
        if "E" in a["groups"]:
            keyS = np.char.add(np.char.add(aptS, "_"), np.char.zfill(hourS.astype(str), 2))
            keyP = np.array([x_ + "_%02d" % h_ for x_, h_ in zip(apt, hour)])
            allk = np.unique(np.concatenate([keyP, keyS]))
            Gah_p, Gah_s = np.searchsorted(allk, keyP), np.searchsorted(allk, keyS)
            sr = np.concatenate([Kf[:, KI["st_rwy"]], kk["Ks"][:, KI["st_rwy"]]])
            _, inv = np.unique(sr, return_inverse=True)
            sv = (inv[:N], Gah_p, pv, inv[N:], Gah_s)
            sv = (sv[0], sv[1], sv[2], sv[3], sv[4])
        blks, nm2 = col_block(a["groups"], XvS, s112s, None, None, pv=pvs, serve=sv)
        assert nm2 == names_x
        if a["prot"] == "w":
            sub = np.where(use)[0][::7]
            lo, hi = np.nanpercentile(blk[sub], [0.5, 99.5], axis=0)
            blk, blks = np.clip(blk, lo, hi), np.clip(blks, lo, hi)
        elif a["prot"] == "c":
            blk[(gap_sched > 3600) | (gap_lobt > 3600) | (hasfp == 0)] = np.nan
            blks[(cS("gap_sched") > 3600) | (cS("gap_lobt") > 3600) | (cS("has_flight_plan") == 0)] = np.nan
    pieces_f = [z["X"], ex["Xf"], e2["Xf"], s4f, s112f, kk["Df"], vf["Xf"], Dpcd] + ([blk] if blk is not None else [])
    pieces_s = [Xs[:, :192], Xs[:, 192:194], s4s, s112s, si["Dsv"], vf["Xs"], C["Ds"]] + ([blks] if blks is not None else [])
    assert np.array_equal(Xs[:, 192:194].shape, (len(Xs), 2))
    XF = np.hstack(pieces_f).astype(np.float32)
    XS = np.hstack(pieces_s).astype(np.float32)
    assert XF.shape[1] == XS.shape[1], (XF.shape, XS.shape)
    assert np.array_equal(C["ids_s"].astype(float), si["ids"].astype(float)), "pcd serve columns are not in serve_in row order"
    del pieces_f
    gc.collect()
    if nr == 0:   # wiring check: return the fold and serve matrices without fitting
        return None, names_x, XF, XS
    b = lgb.train(params(seed, a["red"]), lgb.Dataset(XF, r.astype(np.float32), categorical_feature=catidx, params=dict(max_bin=63, verbose=-1)), nr)
    return b.predict(XS), names_x, b.feature_importance("gain"), None


if __name__ == "__main__":
    for j in sys.argv[1].split(","):
        if os.environ.get("JOB_PREEMPT_FILE") and os.path.exists(os.environ["JOB_PREEMPT_FILE"]):
            P("preempt requested")
            sys.exit(10)
        p = j.split(":")
        if p[1] == "lomo":
            lomo(p[0], int(p[2]), int(p[3]))
        else:
            temporal(p[0], p[2], int(p[3]))
