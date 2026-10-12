"""arith3 step 2 (experiment arith3, 2026-10-05): the FROZEN round-3 column list (docs/LAB-NOTES.md section 2), 45 columns in six groups, built for the corrector fold rows.
Every column is a pure function of the row's own inputs (pv, plan-clock distances, view differences, keys, hour, runway, take-off time) plus (i) out-of-month statistics built the corrector way
(pooled non-held months; a pool row excludes its own month) and (ii) label-free same-day / same-window statistics of OTHER departures predictions. No row's own label enters its columns.
Groups: K cells (10), Y day (6), Q quantiles (11), G geometry (6), S second differences (6), R reliability (6)."""
import json
import numpy as np

I = "data/interim/"
K_NAMES = ["K_%s_%s" % (c, t) for c in ("OA", "OSTD", "TR", "RHW", "CR") for t in ("dev", "z")]
Y_NAMES = ["Y_day_dev", "Y_day_z", "Y_day_pct", "Y_win60_dev", "Y_win60_z", "Y_nbr20_frac"]
Q_NAMES = ["Q_%s_%s" % (c, q) for c in ("SR", "AH", "OA") for q in ("p10", "p50", "p90")] + ["Q_SR_pos", "Q_OA_pos"]
G_NAMES = ["G_q_share_above", "G_near_share", "G_range_over_sd", "G_mean_over_sd", "G_min_over_sd", "G_max_over_sd"]
S_NAMES = ["S_dSCHED_SR", "S_dLOBT_SR", "S_dIOBT_SR", "S_dEOBT_SR", "S_dLOBT_AH", "S_dLOBT_OA"]
R_NAMES = ["R_SR_logn", "R_SR_msd", "R_OA_logn", "R_OA_msd", "R_OSTD_logn", "R_TR_logn"]
GROUPS3 = {"K": K_NAMES, "Y": Y_NAMES, "Q": Q_NAMES, "G": G_NAMES, "S": S_NAMES, "R": R_NAMES}
VPOINT = ["S1", "SEQ", "a_cap", "a_nobx", "a_noosm", "a_noseq", "a_notrk", "cap15", "cap255", "cb_sa", "dart_sa", "hub100", "hub30", "knn", "l1_sa", "logt_sa", "mlp", "rf", "ridge", "xgb_sa"]
VQ = ["q05", "q10", "q25", "q40", "q60", "q75", "q90", "q95", "xq10", "xq50", "xq90", "cq10", "cq50", "cq90"]
NBIN, BW = 76, 20.0


def load_inputs():
    """light loader: only the arrays the builder needs (fold row order)."""
    names = json.load(open(I + "agent_yearfp_names.json"))["names"]
    NI = {n: i for i, n in enumerate(names)}
    z = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True)
    cols = [str(c) for c in z["cols"]]
    X = z["X"]
    D = {c: np.asarray(X[:, cols.index(c)]).copy() for c in ("hour", "RUNWAY_mvt", "ADEP_mvt", "is_weekend")}
    del X
    D["y"] = z["y"].astype(np.float64)
    D["pv"] = z["pv"].astype(np.float32)
    D["mon"] = z["mon"]
    D["apt"] = np.array([str(a) for a in z["apt"]])
    D["ids"] = z["ids"].astype(float)
    kk = np.load(I + "agent_bigc13_keys2.npz", allow_pickle=True)
    D["KN"] = [str(k) for k in kk["names"]]
    D["Kf"] = kk["Kf"]
    C = np.load(I + "agent_pcd_cols.npz", allow_pickle=True)
    D["Dpcd"] = C["D"]
    vfz = np.load(I + "agent_bigc12_vfac_clean5.npz", allow_pickle=True)
    Xf = vfz["Xf"]
    D["vf"] = {n: np.asarray(Xf[:, NI["vf_d_" + n] - 318]).astype(np.float32) for n in VPOINT + VQ}
    del Xf
    t = np.load(I + "agent_arith3_time.npz")
    assert np.array_equal(t["ids"], D["ids"])
    D["t"] = t["t"]
    return D


def factor(*arrs):
    """integer cell code from several integer-like arrays"""
    code = np.zeros(len(arrs[0]), np.int64)
    for a in arrs:
        a = np.nan_to_num(np.asarray(a, np.float64), nan=-1).astype(np.int64)
        u, inv = np.unique(a, return_inverse=True)
        code = code * (len(u) + 1) + inv
        _, code = np.unique(code, return_inverse=True)
    return code


def loo_cell(G, v, use, MI, nm, pool_mask):
    """mean, sd, n, sd-of-monthly-means of v in cell G over pool rows `use`, own month excluded for pool rows. NaN-safe."""
    ng = int(G.max()) + 1
    f = use & np.isfinite(v)
    idx = G * nm + MI
    vv = np.where(f, v, 0.0).astype(np.float64)
    S = np.bincount(idx[f], weights=vv[f], minlength=ng * nm).reshape(ng, nm)
    Q = np.bincount(idx[f], weights=vv[f] ** 2, minlength=ng * nm).reshape(ng, nm)
    Nn = np.bincount(idx[f], minlength=ng * nm).reshape(ng, nm).astype(np.float64)
    St, Qt, Nt = S.sum(1), Q.sum(1), Nn.sum(1)
    inpool = pool_mask[MI]
    s = St[G] - np.where(inpool, S[G, MI], 0.0)
    q = Qt[G] - np.where(inpool, Q[G, MI], 0.0)
    n = Nt[G] - np.where(inpool, Nn[G, MI], 0.0)
    mean = np.where(n > 20, s / np.maximum(n, 1), np.nan)
    sd = np.sqrt(np.maximum(q / np.maximum(n, 1) - (s / np.maximum(n, 1)) ** 2, 0))
    vm = Nn >= 5
    mm = np.where(vm, S / np.maximum(Nn, 1), 0.0)
    tv, tm, tq = vm.sum(1), mm.sum(1), (mm ** 2).sum(1)
    ow = np.where(inpool, vm[G, MI], False)
    om = np.where(inpool, mm[G, MI], 0.0)
    nv = tv[G] - ow
    mean_m = (tm[G] - om) / np.maximum(nv, 1)
    var_m = (tq[G] - om ** 2) / np.maximum(nv, 1) - mean_m ** 2
    msd = np.where(nv >= 3, np.sqrt(np.maximum(var_m, 0)), np.nan)
    return mean, sd, n, msd


def loo_quant(G, ylab, use, MI, nm, pool_mask, qs=(0.1, 0.5, 0.9)):
    """quantiles of the label in cell G over pool rows (20 s bins, linear inside the bin), own month excluded for pool rows; one array per q (NaN if cell n <= 20)."""
    ng = int(G.max()) + 1
    b = np.minimum((np.clip(ylab, 0, None) / BW).astype(np.int64), NBIN - 1)
    Hm = np.zeros((nm, ng, NBIN), np.float32)
    for m in range(nm):
        r = np.where(use & (MI == m))[0]
        if len(r):
            Hm[m] = np.bincount(G[r] * NBIN + b[r], minlength=ng * NBIN).reshape(ng, NBIN)
    Ht = Hm.sum(0)
    out = [np.full(len(G), np.nan, np.float32) for _ in qs]
    for m in range(nm):
        r = np.where(MI == m)[0]
        if not len(r):
            continue
        H = Ht[G[r]] - (Hm[m][G[r]] if pool_mask[m] else 0.0)
        cs = np.cumsum(H, 1)
        n = cs[:, -1]
        ar = np.arange(len(r))
        for k, q in enumerate(qs):
            tgt = q * n
            j = (cs < tgt[:, None]).sum(1).clip(0, NBIN - 1)
            prev = np.where(j > 0, cs[ar, np.maximum(j - 1, 0)], 0.0)
            cnt = np.maximum(H[ar, j], 1e-9)
            val = (j + np.clip((tgt - prev) / cnt, 0, 1)) * BW
            out[k][r] = np.where(n > 20, val, np.nan)
    del Hm, Ht
    return out


def day_block(D):
    """label-free day / window position of each row's prediction among the OTHER departures of the same airport (pv and take-off time only; no labels)."""
    pv, t, apt = D["pv"].astype(np.float64), D["t"], D["apt"]
    N = len(pv)
    ac = np.unique(apt, return_inverse=True)[1]
    day = (t // 86400).astype(np.int64)
    dk = ac.astype(np.int64) * 100000 + day
    out = np.full((N, 6), np.nan, np.float32)
    _, dc = np.unique(dk, return_inverse=True)
    n = np.bincount(dc).astype(float)
    s = np.bincount(dc, weights=pv)
    q = np.bincount(dc, weights=pv ** 2)
    nn = n[dc] - 1
    m = (s[dc] - pv) / np.maximum(nn, 1)
    v = ((q[dc] - pv ** 2) / np.maximum(nn, 1)) - m ** 2
    ok = nn >= 5
    out[:, 0] = np.where(ok, pv - m, np.nan)
    out[:, 1] = np.where(ok, (pv - m) / np.maximum(np.sqrt(np.maximum(v, 0)), 30.0), np.nan)
    o = np.lexsort((pv, dc))
    start = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
    rank = np.empty(N)
    rank[o] = np.arange(N) - start[dc[o]]
    out[:, 2] = np.where(n[dc] >= 6, rank / np.maximum(n[dc] - 1, 1), np.nan)
    o = np.lexsort((t, ac))
    pt, tt, aa = pv[o], t[o], ac[o]
    key = aa.astype(np.float64) * 1e10 + tt
    lo = np.searchsorted(key, key - 1800, "left")
    hi = np.searchsorted(key, key + 1800, "right")
    cs = np.concatenate([[0], np.cumsum(pt)])
    cq = np.concatenate([[0], np.cumsum(pt ** 2)])
    cn = (hi - lo) - 1
    mw = (cs[hi] - cs[lo] - pt) / np.maximum(cn, 1)
    vw = (cq[hi] - cq[lo] - pt ** 2) / np.maximum(cn, 1) - mw ** 2
    okw = cn >= 3
    w = np.empty((N, 3))
    w[:, 0] = np.where(okw, pt - mw, np.nan)
    w[:, 1] = np.where(okw, (pt - mw) / np.maximum(np.sqrt(np.maximum(vw, 0)), 30.0), np.nan)
    below = np.zeros(N)
    cnt = np.zeros(N)
    ar = np.arange(N)
    for off in list(range(-10, 0)) + list(range(1, 11)):
        j = ar + off
        okj = (j >= 0) & (j < N)
        jj = np.clip(j, 0, N - 1)
        val = okj & (aa[jj] == aa) & (np.abs(tt[jj] - tt) <= 3600)
        cnt += val
        below += val & (pt[jj] < pt)
    w[:, 2] = np.where(cnt >= 3, below / np.maximum(cnt, 1), np.nan)
    out[o, 3:6] = w.astype(np.float32)
    return out


def build3(D, groups, use, pool_mask, s112, KI, day=None):
    """(N, ncols) float32 and names for the requested round-3 groups. D from load_inputs(); s112 = key statistics of the corrector for this fold (column 7*KI[k]+2 = sd of key k)."""
    pv = D["pv"].astype(np.float64)
    mon = D["mon"]
    months = sorted(set(mon.tolist()))
    nm = len(months)
    MI = np.searchsorted(months, mon)
    Kf, KN = D["Kf"], D["KN"]
    kc = lambda k: np.unique(Kf[:, KN.index(k)], return_inverse=True)[1]  # noqa: E731
    ac = np.unique(D["apt"], return_inverse=True)[1]
    cells = {"OA": kc("op_apt"), "OSTD": kc("op_stand"), "TR": kc("ty_rwy"), "SR": kc("st_rwy")}
    cells["AH"] = factor(ac, D["hour"])
    cells["RHW"] = factor(ac, D["RUNWAY_mvt"], D["hour"], D["is_weekend"])
    cells["CR"] = factor(kc("cty"), D["RUNWAY_mvt"], ac)
    out, nms = [], []
    cache = {}

    def pvst(c):
        if c not in cache:
            cache[c] = loo_cell(cells[c], pv, use, MI, nm, pool_mask)
        return cache[c]
    sdy = s112[:, KI["st_rwy"] * 7 + 2].astype(np.float64)
    for g in groups:
        if g == "K":
            for c in ("OA", "OSTD", "TR", "RHW", "CR"):
                m, sd, n, _ = pvst(c)
                out += [(pv - m).astype(np.float32), ((pv - m) / np.maximum(sd, 30.0)).astype(np.float32)]
        elif g == "Y":
            out += [day[:, i] for i in range(6)]
        elif g == "Q":
            pos = {}
            for c in ("SR", "AH", "OA"):
                p10, p50, p90 = loo_quant(cells[c], D["y"], use, MI, nm, pool_mask)
                out += [(pv - p10).astype(np.float32), (pv - p50).astype(np.float32), (pv - p90).astype(np.float32)]
                pos[c] = ((pv - p10) / np.maximum(p90 - p10, 30.0)).astype(np.float32)
            out += [pos["SR"], pos["OA"]]
        elif g == "G":
            sdc = np.maximum(sdy, 30.0)
            Ncur = len(pv)
            Gc = np.full((Ncur, 6), np.nan, np.float32)
            for a in range(0, Ncur, 200000):
                b_ = min(a + 200000, Ncur)
                Dq = np.column_stack([D["vf"][v][a:b_] for v in VQ])
                Da = np.column_stack([np.column_stack([D["vf"][v][a:b_] for v in VPOINT]), Dq])
                sc = sdc[a:b_]
                Gc[a:b_, 0] = np.nanmean(Dq > 0, 1)
                Gc[a:b_, 1] = np.nanmean(np.abs(Da) < 5, 1)
                Gc[a:b_, 2] = (np.nanmax(Da, 1) - np.nanmin(Da, 1)) / sc
                Gc[a:b_, 3] = np.nanmean(Da, 1) / sc
                Gc[a:b_, 4] = np.nanmin(Da, 1) / sc
                Gc[a:b_, 5] = np.nanmax(Da, 1) / sc
            out += [Gc[:, i] for i in range(6)]
        elif g == "S":
            Dp = D["Dpcd"]
            for j, c in ((0, "SR"), (1, "SR"), (2, "SR"), (3, "SR"), (1, "AH"), (1, "OA")):
                v = Dp[:, j]
                m, _, _, _ = loo_cell(cells[c], v, use, MI, nm, pool_mask)
                out.append((v - m).astype(np.float32))
        elif g == "R":
            for c in ("SR", "OA"):
                m, sd, n, msd = pvst(c)
                out += [np.log1p(n).astype(np.float32), msd.astype(np.float32)]
            for c in ("OSTD", "TR"):
                m, sd, n, msd = pvst(c)
                out.append(np.log1p(n).astype(np.float32))
        nms += GROUPS3[g]
    return np.column_stack([np.asarray(o, np.float32) for o in out]), nms
