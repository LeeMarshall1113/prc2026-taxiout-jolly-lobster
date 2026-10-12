"""arith4 step 0 (experiment "arith4", 2026-10-06): where does the residual error of the v150 chain still sit on 2025 held-out rows, AFTER the ABCDE-pw corrector?
Reads only the stored held-out predictions of ABCDE-pw-s0-r0 (seeds 0-2, months 1,2,3,7,8,9,10,11; data/interim/agent_arith_out/) and the judge frame of
archive/agent_arith_03_judge.py (served chain, nine airports). Decomposes the chain-level squared error by airport, month, coverage class, plan, label band,
corrector magnitude, seed disagreement and base-member disagreement; then scans EVERY corrector input column (and the ABCDE blocks, and the same columns
recomputed against the CORRECTED prediction) for structure left in the corrected residual, with an OUT-OF-MONTH binned-mean oracle (bin means from the other
seven months applied to the held month; a placebo with the residual permuted within airport-month gives the noise bar). Also: out-of-month key means of the
corrected residual (the information a second-stage corrector would add) and the dose slope by airport x class.
usage: python archive/agent_arith4_00_decomp.py   (writes logs/agent_arith4_decomp.txt and data/interim/agent_arith4_decomp.npz). Reads 2025 labels only."""
import os
import sys
import gc
import json
import time
import numpy as np

sys.path.insert(0, "archive")
import agent_bigc_te as TE   # noqa: E402
I = "data/interim/"
os.makedirs("logs", exist_ok=True)
LOG = open("logs/agent_arith4_decomp.txt", "w", encoding="utf-8")
t0 = time.time()


def P(*a):
    s = " ".join(str(x) for x in a)
    print("[%5.0fs] %s" % (time.time() - t0, s), flush=True)
    LOG.write(s + "\n")
    LOG.flush()


ALLM = (1, 2, 3, 7, 8, 9, 10, 11)
SEEDS = (0, 1, 2)
CAND = "ABCDE-pw-s0-r0"
# ---- lean copies of what agent_basefix_lib (judge frame) and agent_arith_02_fit (corrector inputs) load, selected columns only (memory: the run stays under 4 GB)
import pyarrow.parquet as pq   # noqa: E402
fr = pq.read_table(I + "agent_ablate_frame.parquet", columns=["id", "y", "day", "apt", "mon", "M1", "k5", "tailB", "rule", "eb_mid", "eb_P2", "eb_gs_r", "eb_pred"]).to_pandas()
fr = fr[fr["M1"].notna() & (fr["apt"] != "LTFM")].reset_index(drop=True)
aux = pq.read_table(I + "agent_basefix_aux.parquet", columns=["id", "cov", "plan"]).to_pandas().set_index("id")
fr["cov"] = aux.loc[fr["id"].values, "cov"].values
fr["plan"] = aux.loc[fr["id"].values, "plan"].values
del aux


class L:   # the judge frame's names
    ids = fr["id"].to_numpy().astype(float); y = fr["y"].to_numpy().astype(float); apt = fr["apt"].to_numpy().astype(str); mon = fr["mon"].to_numpy().astype(int)
    day = fr["day"].to_numpy().astype(str); cov = fr["cov"].to_numpy().astype(str); plan = fr["plan"].to_numpy().astype(bool)
    K5 = np.nan_to_num(fr["k5"].to_numpy().astype(float)); TAIL = np.nan_to_num(fr["tailB"].to_numpy().astype(float)); rule = fr["rule"].to_numpy().astype(str)
    mid = np.nan_to_num(fr["eb_mid"].to_numpy().astype(float)) > 0.5; P2 = np.nan_to_num(fr["eb_P2"].to_numpy().astype(float)); gsr = np.nan_to_num(fr["eb_gs_r"].to_numpy().astype(float))
    ebpred = fr["eb_pred"].to_numpy().astype(float); W3 = np.where(gsr < 7200, 0.5, 0.85); MIX = P2 * gsr + (1 - P2) * 1025.5
    JAN = mon == 1; JUL = mon == 7; ALL = np.ones(len(y), bool)


del fr
gc.collect()


import types   # noqa: E402
F = types.SimpleNamespace()
F.names = json.load(open(I + "agent_yearfp_names.json"))["names"]
F.NI = {n: i for i, n in enumerate(F.names)}
F.z = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True)
F.cols = [str(c) for c in F.z["cols"]]
F.catidx = [int(i) for i in F.z["cat"]]
F.y = F.z["y"].astype(float)
F.pv = F.z["pv"].astype(np.float32)
F.mon = F.z["mon"]
F.apt = np.array([str(a) for a in F.z["apt"]])
F.ids = F.z["ids"].astype(float)
F.N = len(F.y)
F.months = sorted(set(F.mon.tolist()))
F.r = np.clip(F.y - F.pv, -900, 900)
F.kk = np.load(I + "agent_bigc13_keys2.npz", allow_pickle=True)
F.KN = [str(k) for k in F.kk["names"]]
F.Kf = F.kk["Kf"]
F.KI = {k: i for i, k in enumerate(F.KN)}
F.k0 = np.load(I + "agent_bigc12_keys.npz", allow_pickle=True)
F.ymu_apt = {a: F.y[F.k0["apt_f"] == a].mean() for a in set(F.k0["apt_f"].tolist())}
F.ymu_f = np.array([F.ymu_apt[a] for a in F.k0["apt_f"]])
F.Dpcd = np.load(I + "agent_pcd_cols.npz", allow_pickle=True)["D"]
F.vf = np.load(I + "agent_bigc12_vfac_clean5.npz", allow_pickle=True)
F.ex = np.load(I + "agent_bigc12_extra.npz", allow_pickle=True)
F.e2 = np.load(I + "agent_bigc12_extra2.npz", allow_pickle=True)
MM = I + "agent_arith4_mmap/"
os.makedirs(MM, exist_ok=True)


def mm(name, npz, key):
    """one-time .npy copy of a big npz member, then a memory map (page cache, not commit)"""
    f = MM + name + ".npy"
    if not os.path.exists(f):
        a = npz[key]
        np.save(f + ".tmp.npy", a)
        del a
        gc.collect()
        os.replace(f + ".tmp.npy", f)
    return np.load(f, mmap_mode="r")


F.X = mm("fold_X", F.z, "X")
F.vfX = mm("vf_Xf", F.vf, "Xf")
F.exX = mm("ex_Xf", F.ex, "Xf")
F.VPOINT = ["S1", "SEQ", "a_cap", "a_nobx", "a_noosm", "a_noseq", "a_notrk", "cap15", "cap255", "cb_sa", "dart_sa", "hub100", "hub30", "knn", "l1_sa", "logt_sa", "mlp", "rf", "ridge", "xgb_sa"]
F.VQ = ["q05", "q10", "q25", "q40", "q60", "q75", "q90", "q95", "xq10", "xq50", "xq90", "cq10", "cq50", "cq90"]
F.ADSB = ["m_e_stand", "m_e_appear", "m_e_ci20", "m_e_dend", "m_e_mv10", "m_e_pushend", "m_e_start", "m_e_revany", "ci_150", "appear_raw"]
F.KS_A = ["st_rwy", "st_op_rwy", "op_apt", "op_rwy", "st_hb", "ty_rwy", "op_stand"]
fold_ids = F.ids
pos = {v: i for i, v in enumerate(fold_ids.tolist())}
ix = np.array([pos[v] for v in L.ids.tolist()])          # frame row -> fold row
NB = np.load(I + "agent_bigc13_base.npz")["nb"][ix]
y = L.y
mon = L.mon
apt = L.apt
Nf = len(y)
P("frame rows %d (fold rows %d, fold months %s); frame airports %s; frame months %s" % (Nf, F.N, F.months, sorted(set(apt)), sorted(set(mon))))
P("pv (fold) == nb (base) on frame rows: max abs diff %.3f" % np.abs(F.pv[ix] - NB).max())


def load(arm, seed, what="pred", folder="agent_arith_out/"):
    out = np.full(F.N, np.nan)
    for m in ALLM:
        d = np.load(I + folder + "%s_s%d_m%d.npz" % (arm, seed, m))
        out[d["rows"]] = d[what]
    return out[ix]


def served(base, corr):
    x = np.maximum(base + np.where(np.isfinite(corr), corr, 0) * L.K5, 0)
    x = np.maximum(x + 0.5 * L.TAIL, 0)
    out = x.copy()
    out[L.mid] = x[L.mid] + L.W3[L.mid] * (L.MIX[L.mid] - x[L.mid])
    for rr in ("R1", "R2", "R3b"):
        s = L.rule == rr
        out[s] = L.ebpred[s]
    return np.rint(np.maximum(out, 0))


rm = lambda e, m: float(np.sqrt((e[m] ** 2).mean()))   # noqa: E731

# ---------------------------------------------------------------- 1. levels
Fs = np.stack([load(CAND, s) for s in SEEDS])            # (3, Nf) corrector out-of-fold predictions (residual scale)
assert np.isfinite(Fs).all()
Fm = Fs.mean(0)
Fsd = Fs.std(0)
term = L.K5 * Fm                                          # the served term
Pc = served(NB, Fm)
e = y - Pc                                                # corrected residual (chain level), positive = under-prediction
Fd = np.mean([load("dist", s, folder="agent_pcd_out/") for s in SEEDS], 0)
Fk = np.mean([load("ctrl", s, folder="agent_pcd_out/") for s in SEEDS], 0)
e_pcd = y - served(NB, Fd)
e_ctrl = y - served(NB, Fk)
e_base = y - served(NB, np.zeros(Nf))
P("\n== 1. chain RMSE (nine airports, eight months): no corrector %.3f | ctrl %.3f | pcd %.3f | cand ABCDE-pw %.3f" % (rm(e_base, L.ALL), rm(e_ctrl, L.ALL), rm(e_pcd, L.ALL), rm(e, L.ALL)))
P("   cand Jan %.3f Jul %.3f; mean residual (bias) all %+.2f s, Jan %+.2f, Jul %+.2f" % (rm(e, L.JAN), rm(e, L.JUL), e.mean(), e[L.JAN].mean(), e[L.JUL].mean()))
SSE = (e ** 2).sum()
ALL = L.ALL


def classes(name, code, labels=None):
    """share of rows, share of SSE, mean residual, in-month bias share (n*mean^2 / SSE), OUT-OF-MONTH bias gain (class mean from other months applied; RMSE seconds)"""
    u = np.unique(code)
    P("   -- %s: %d classes" % (name, len(u)))
    rows = []
    adj = loo_mean(code, e, prior=100.0)
    for c in u:
        m = code == c
        lab = labels[c] if labels is not None else str(c)
        g_oom = rm(e, ALL) - rm(np.where(m, e - adj, e), ALL)
        rows.append((lab, m.mean(), (e[m] ** 2).sum() / SSE, e[m].mean(), np.sqrt((e[m] ** 2).mean()), m.sum() * e[m].mean() ** 2 / SSE, g_oom))
    rows.sort(key=lambda r: -r[2])
    for r in rows[:40]:
        P("      %-28s rows %6.2f%%  SSE %6.2f%%  bias %+7.1f s  rmse %7.1f  bias/SSE %.4f  out-of-month bias gain %+.4f s" % (r[0], 100 * r[1], 100 * r[2], r[3], r[4], r[5], r[6]))
    return rows


MI = np.searchsorted(np.array(ALLM), mon)
NM = len(ALLM)


def loo_mean(code, v, prior=30.0):
    """leave-own-month-out shrunk class mean of v (frame rows), prior toward 0"""
    _, g = np.unique(code, return_inverse=True)
    G = g.max() + 1
    idx = g * NM + MI
    S = np.bincount(idx, weights=v, minlength=G * NM).reshape(G, NM)
    n = np.bincount(idx, minlength=G * NM).reshape(G, NM).astype(float)
    return (S.sum(1)[g] - S[g, MI]) / (n.sum(1)[g] - n[g, MI] + prior)


def oracle(v, cat=False, nbins=20, prior=30.0, ev=None):
    """RMSE gain (seconds, positive = structure found) from correcting the residual by the out-of-month binned mean of v. NaN is its own bin."""
    ev = e if ev is None else ev
    if cat:
        code = np.nan_to_num(v.astype(np.float64), nan=-1.0).astype(np.int64)
    else:
        f = np.isfinite(v)
        if f.sum() < 1000:
            return 0.0, 0.0, 0.0
        q = np.unique(np.nanpercentile(v[f], np.linspace(0, 100, nbins + 1)[1:-1]))
        code = np.where(f, np.searchsorted(q, np.where(f, v, 0), side="right"), nbins + 1)
    adj = loo_mean(code, ev, prior=prior)
    e2 = ev - adj
    return rm(ev, ALL) - rm(e2, ALL), rm(ev, L.JAN) - rm(e2, L.JAN), rm(ev, L.JUL) - rm(e2, L.JUL)


# ---------------------------------------------------------------- 2. decomposition by class
P("\n== 2. squared error of the corrected residual by class (share of rows, share of SSE, bias, in-month bias share, out-of-month bias gain)")
_, acode = np.unique(apt, return_inverse=True)
classes("airport", acode, labels=sorted(set(apt)))
classes("month", mon)
_, ccode = np.unique(L.cov, return_inverse=True)
classes("coverage class", ccode, labels=sorted(set(L.cov)))
classes("plan", L.plan.astype(int), labels={0: "no plan", 1: "plan"})
lab_band = np.digitize(y, [300, 900, 3600, 14400])
classes("label band", lab_band, labels=["<300", "300-900", "900-3600", "3600-14400", ">14400"])
pb = np.digitize(NB, [300, 600, 900, 1500, 3600])
classes("base prediction band", pb, labels=["<300", "300-600", "600-900", "900-1500", "1500-3600", ">3600"])
tb = np.digitize(term, [-120, -60, -20, 20, 60, 120, 300])
classes("served corrector term", tb, labels=["<-120", "-120..-60", "-60..-20", "-20..20", "20..60", "60..120", "120..300", ">300"])
sb = np.digitize(Fsd, np.percentile(Fsd, [50, 80, 95, 99]))
classes("seed disagreement (sd over 3 seeds) percentile band", sb, labels=["<p50", "p50-80", "p80-95", "p95-99", ">p99"])
# base-member disagreement = sd over the 20 point-view differences (group B's B_sd_pt), built from vf
vfX = F.vfX
Dp = np.column_stack([vfX[:, F.NI["vf_d_" + v] - 318] for v in F.VPOINT]).astype(np.float32)[ix]
bsd = np.nanstd(Dp, 1)
del Dp
bb = np.digitize(bsd, np.nanpercentile(bsd, [50, 80, 95, 99]))
classes("base-member disagreement (sd of 20 point views) percentile band", bb, labels=["<p50", "p50-80", "p80-95", "p95-99", ">p99"])
ab = np.digitize(np.abs(e), [30, 100, 300, 900])
classes("|corrected residual| band (descriptive only)", ab, labels=["<30", "30-100", "100-300", "300-900", ">900"])
# what the corrector did vs what was left: gain of the corrector by label band and by term band
gain_c = (e_pcd ** 2 - e ** 2)
P("   -- where the ABCDE gain over pcd sits (share of gain) by label band: " + " ".join("%s %.2f" % (l, gain_c[lab_band == i].sum() / gain_c.sum()) for i, l in enumerate(["<300", "300-900", "900-3600", "3600-14400", ">14400"])))
P("   -- and the REMAINING SSE by label band: " + " ".join("%s %.2f" % (l, (e[lab_band == i] ** 2).sum() / SSE) for i, l in enumerate(["<300", "300-900", "900-3600", "3600-14400", ">14400"])))

# ---------------------------------------------------------------- 3. dose slopes by airport x class (is the served dose right by class?)
P("\n== 3. dose slope of the corrected residual on the served term, out-of-month, by airport x class (slope b: e ~ b * term; gain = RMSE change if the class dose were (1+b) x, b fitted on the other months)")


def dose_table(name, code, labels):
    P("   -- %s" % name)
    tot_adj = np.zeros(Nf)
    for a in sorted(set(apt)):
        line = []
        for c in np.unique(code):
            m = (apt == a) & (code == c)
            if m.sum() < 2000:
                continue
            adj = np.zeros(Nf)
            for h in ALLM:
                tr = m & (mon != h)
                te_ = m & (mon == h)
                b = (e[tr] * term[tr]).sum() / max((term[tr] ** 2).sum(), 1e-9)
                adj[te_] = b * term[te_]
            bfull = (e[m] * term[m]).sum() / max((term[m] ** 2).sum(), 1e-9)
            tot_adj += adj
            line.append("%s b %+.2f gain %+.3f" % (labels[c], bfull, rm(e, ALL) - rm(e - adj, ALL)))
        P("      %s: %s" % (a, " | ".join(line)))
    P("      all classes together: pooled gain %+.4f s (Jan %+.4f, Jul %+.4f)" % (rm(e, ALL) - rm(e - tot_adj, ALL), rm(e, L.JAN) - rm(e - tot_adj, L.JAN), rm(e, L.JUL) - rm(e - tot_adj, L.JUL)))


dose_table("airport (per-airport dose re-check)", np.zeros(Nf, int), {0: "all"})
dose_table("airport x coverage class", ccode, dict(enumerate(sorted(set(L.cov)))))
dose_table("airport x plan", L.plan.astype(int), {0: "no plan", 1: "plan"})
dose_table("airport x label-free base band", np.digitize(NB, [600, 1200]), {0: "base<600", 1: "600-1200", 2: ">1200"})

# ---------------------------------------------------------------- 4. column scan
P("\n== 4. out-of-month binned-mean oracle over every corrector input (gain in RMSE seconds, pooled / Jan / Jul; placebo = residual permuted within airport-month)")
rng = np.random.default_rng(4)
e_perm = e.copy()
_ag = np.array([a + "_%d" % m for a, m in zip(apt, mon.tolist())])
for g in np.unique(_ag):
    ii = np.where(_ag == g)[0]
    e_perm[ii] = e[rng.permutation(ii)]
res = []


def scan(name, v, cat=False, group=""):
    g = oracle(v, cat)
    pl = oracle(v, cat, ev=e_perm)[0]
    res.append((group, name, g[0], g[1], g[2], pl))


catset = set(F.cols[i] for i in F.catidx)
X = F.X
for j, c in enumerate(F.cols):
    scan(c, X[:, j].astype(np.float32)[ix], cat=(c in catset), group="fold140")
Xv = {n: X[:, F.cols.index(n)].astype(np.float32) for n in F.ADSB if n in F.cols}
rwy = X[:, F.cols.index("RUNWAY_mvt")][ix]
hour_f = X[:, F.cols.index("hour")].astype(np.int64)
hour = hour_f[ix]
del X
gc.collect()
exX = F.exX
for j, c in enumerate([str(c) for c in F.ex["cols"]]):
    scan(c, exX[:, j].astype(np.float32)[ix], group="extra52")
    if c in F.ADSB:
        Xv[c] = exX[:, j].astype(np.float32)
del exX
e2X = F.e2["Xf"]
for j in range(e2X.shape[1]):
    scan(F.names[192 + j], e2X[:, j].astype(np.float32)[ix], group="extra2")
del e2X
# key statistics, leave-own-month-out (serve-style; the fold version differs per held month only in the pool)
kcols = ["ADEP_mvt"] + TE.KEYCOLS
X = F.X
kf = TE.keys_from(np.column_stack([X[:, F.cols.index(c)] for c in kcols]), kcols)
del X
s4 = TE.te_loo(kf, F.r, F.mon, F.months)
del kf
for j in range(4):
    scan(F.names[194 + j], s4[:, j][ix], group="te4")
del s4
T2 = TE.Te2(F.Kf)
s112 = T2.fold(F.r, F.y, F.ymu_f, F.mon, None, F.months)
for j in range(112):
    scan(F.names[198 + j], s112[:, j][ix], group="keystat112")
Dfz = F.kk["Df"]
for j in range(8):
    scan(F.names[310 + j], Dfz[:, j].astype(np.float32)[ix], group="drift8")
del Dfz
for j in range(52):
    scan(F.names[318 + j], vfX[:, j].astype(np.float32)[ix], group="views52")
for j in range(9):
    scan("pcd_%d" % j, F.Dpcd[:, j].astype(np.float32)[ix], group="pcd9")
# the ABCDE blocks (serve-style pool: all months, own month excluded) and the same columns against the CORRECTED prediction
for n in ["vf_d_" + v for v in F.VPOINT + F.VQ]:
    Xv[n] = vfX[:, F.NI[n] - 318].astype(np.float32)
del vfX
gc.collect()
def cell_block(G, pvv):
    """pv deviation from the cell mean of pv over the other months (leave-own-month-out), and over the cell sd (floor 30): as pv_cell_block in the fit with every month in the pool"""
    nm = len(F.months); MIf = np.searchsorted(F.months, F.mon); ng = G.max() + 1; idx = G * nm + MIf
    S = np.bincount(idx, weights=pvv, minlength=ng * nm).reshape(ng, nm); Q = np.bincount(idx, weights=pvv.astype(np.float64) ** 2, minlength=ng * nm).reshape(ng, nm)
    Nn = np.bincount(idx, minlength=ng * nm).reshape(ng, nm).astype(np.float64)
    s_ = S.sum(1)[G] - S[G, MIf]; q = Q.sum(1)[G] - Q[G, MIf]; n = Nn.sum(1)[G] - Nn[G, MIf]
    mean = np.where(n > 20, s_ / np.maximum(n, 1), np.nan); sd = np.sqrt(np.maximum(q / np.maximum(n, 1) - (s_ / np.maximum(n, 1)) ** 2, 0))
    dev = pvv - mean
    return dev.astype(np.float32), (dev / np.maximum(sd, 30.0)).astype(np.float32)


G_SR = np.unique(F.Kf[:, F.KI["st_rwy"]], return_inverse=True)[1]
_, G_AH = np.unique(np.array([a + "_%02d" % h for a, h in zip(F.apt, hour_f)]), return_inverse=True)


def ace_block(pvv, tag):
    stat = lambda k, j: s112[:, F.KI[k] * 7 + j].astype(np.float64)   # noqa: E731
    for k in F.KS_A:
        ym, sd = stat(k, 6), stat(k, 2)
        scan(tag + "A_pv-ym_" + k, (pvv - ym)[ix], group=tag + "ACE"); scan(tag + "A_z_" + k, ((pvv - ym) / np.maximum(sd, 30.0))[ix], group=tag + "ACE")
    for a in F.ADSB:
        scan(tag + "C_pv-" + a, (pvv - Xv[a])[ix], group=tag + "ACE")
    for G, nm_ in ((G_SR, "st_rwy"), (G_AH, "apt_hour")):
        d, zz = cell_block(G, pvv)
        scan(tag + "E_pvdev_" + nm_, d[ix], group=tag + "ACE"); scan(tag + "E_pvz_" + nm_, zz[ix], group=tag + "ACE")


ace_block(F.pv.astype(np.float64), "")
# corrected prediction in fold space: pv + served term (frame rows; other fold rows keep pv)
pvc = F.pv.astype(np.float32).copy()
pvc[ix] = (NB + term).astype(np.float32)
ace_block(pvc.astype(np.float64), "corrected_")
del Xv
gc.collect()
# interactions
scan("pv_band x cov", (np.digitize(NB, [300, 600, 900, 1500, 3600]) * 10 + ccode).astype(np.float32), cat=True, group="interaction")
scan("hour x runway x airport", (acode * 1000000 + np.nan_to_num(rwy, nan=-1).astype(np.int64) * 100 + hour).astype(np.float64), cat=True, group="interaction")
scan("term band x cov", (tb * 10 + ccode).astype(np.float32), cat=True, group="interaction")
scan("term band x airport", (tb * 100 + acode).astype(np.float32), cat=True, group="interaction")
scan("seed sd x airport", (sb * 100 + acode).astype(np.float32), cat=True, group="interaction")
scan("member sd x airport", (bb * 100 + acode).astype(np.float32), cat=True, group="interaction")
scan("seed sd (continuous)", Fsd, group="uncertainty")
scan("member sd (continuous)", bsd, group="uncertainty")
scan("corrector term (continuous)", term, group="uncertainty")
scan("corrector term x pv band", (tb * 10 + pb).astype(np.float32), cat=True, group="interaction")

res.sort(key=lambda r: -r[2])
P("   placebo (permuted residual) over all %d columns: max %.4f, 95th pct %.4f, mean %.4f" % (len(res), max(r[5] for r in res), np.percentile([r[5] for r in res], 95), np.mean([r[5] for r in res])))
P("   top 45 columns by out-of-month oracle gain (pooled / Jan / Jul / placebo):")
for r in res[:45]:
    P("      %-18s %-34s %+.4f  %+.4f  %+.4f   placebo %+.4f" % (r[0], r[1], r[2], r[3], r[4], r[5]))
P("   per group: max gain and count above 0.02 s:")
for g in sorted(set(r[0] for r in res)):
    rr = [r for r in res if r[0] == g]
    P("      %-18s n %3d  max %+.4f (%s)  n>0.02: %d" % (g, len(rr), rr[0][2], rr[0][1], sum(1 for r in rr if r[2] > 0.02)))

# ---------------------------------------------------------------- 5. key means of the CORRECTED residual (second-stage information)
P("\n== 5. out-of-month key means of the corrected residual applied to the held month (what a nested second stage could add; prior 30 / 300)")
KN = F.KN
tot = {}
for j, k in enumerate(KN):
    code = F.Kf[:, j][ix]
    for pr in (30.0, 300.0):
        adj = loo_mean(code, e, prior=pr)
        g = rm(e, ALL) - rm(e - adj, ALL)
        tot[(k, pr)] = (g, rm(e, L.JAN) - rm(e - adj, L.JAN), rm(e, L.JUL) - rm(e - adj, L.JUL), rm(e_perm, ALL) - rm(e_perm - loo_mean(code, e_perm, prior=pr), ALL))
    P("      %-12s prior30 %+.4f (Jan %+.4f Jul %+.4f, placebo %+.4f) | prior300 %+.4f (Jan %+.4f Jul %+.4f, placebo %+.4f)" % ((k,) + tot[(k, 30.0)] + tot[(k, 300.0)]))
# all keys stacked greedily (each key's loo mean on the residual left by the previous ones): a cheap ceiling for a linear second stage
e_left = e.copy()
order = sorted(KN, key=lambda k: -tot[(k, 300.0)][0])
for k in order:
    code = F.Kf[:, KN.index(k)][ix]
    e_left = e_left - loo_mean(code, e_left, prior=300.0)
P("      greedy stack of all %d keys (prior 300): pooled %+.4f  Jan %+.4f  Jul %+.4f" % (len(KN), rm(e, ALL) - rm(e_left, ALL), rm(e, L.JAN) - rm(e_left, L.JAN), rm(e, L.JUL) - rm(e_left, L.JUL)))
e_left = e_perm.copy()
for k in order:
    code = F.Kf[:, KN.index(k)][ix]
    e_left = e_left - loo_mean(code, e_left, prior=300.0)
P("      same on the placebo: pooled %+.4f" % (rm(e_perm, ALL) - rm(e_left, ALL)))
# the same for the residual left by pcd (how much of this the ABCDE step already took)
e_left = e_pcd.copy()
for k in order:
    code = F.Kf[:, KN.index(k)][ix]
    e_left = e_left - loo_mean(code, e_left, prior=300.0)
P("      same on the pcd residual (before ABCDE): pooled %+.4f" % (rm(e_pcd, ALL) - rm(e_left, ALL)))

# ---------------------------------------------------------------- 6. seed / member disagreement vs error
P("\n== 6. disagreement vs error: by seed-sd band and member-sd band, rmse of the corrected residual, mean residual, and the 3-seed-mean-vs-single-seed loss")
e_single = y - served(NB, Fs[0])
for nm, code, labs in (("seed sd", sb, ["<p50", "p50-80", "p80-95", "p95-99", ">p99"]), ("member sd", bb, ["<p50", "p50-80", "p80-95", "p95-99", ">p99"])):
    for c in range(5):
        m = code == c
        P("      %-10s %-7s rows %5.1f%%  rmse(3-seed) %7.1f  rmse(seed0) %7.1f  bias %+6.1f  SSE share %.3f" % (nm, labs[c], 100 * m.mean(), rm(e, m), rm(e_single, m), e[m].mean(), (e[m] ** 2).sum() / SSE))
P("      seed noise: 3-seed mean vs seed 0: %+.4f s pooled; implied infinite-seed gain over 3 seeds (var/k law): %+.4f s; over 12 seeds %+.4f s; 24 seeds %+.4f s" % (
    rm(e, ALL) - rm(e_single, ALL), *[(rm(e, ALL) - rm(e_single, ALL)) * (1 - 1.0 / k) / (1 - 1.0 / 3) - (rm(e, ALL) - rm(e_single, ALL)) for k in (1e9, 12, 24)]))
# direct: per-row seed variance of the served term and its share of SSE
sv = (L.K5 ** 2) * Fs.var(0, ddof=1)
P("      mean per-seed variance of the served term %.1f s^2 (= %.3f of SSE per seed); 3-seed mean leaves 1/3 of it: %.1f s^2 -> RMSE %.4f; 12 seeds %.4f; 24 seeds %.4f" % (
    sv.mean(), sv.mean() / (SSE / Nf), sv.mean() / 3, np.sqrt((e ** 2).mean()) - np.sqrt((e ** 2).mean() - sv.mean() / 3 + sv.mean() / 1e9),
    np.sqrt((e ** 2).mean()) - np.sqrt((e ** 2).mean() - sv.mean() / 3 + sv.mean() / 12), np.sqrt((e ** 2).mean()) - np.sqrt((e ** 2).mean() - sv.mean() / 3 + sv.mean() / 24)))

np.savez(I + "agent_arith4_decomp.npz", e=e, term=term, Fsd=Fsd, bsd=bsd, ids=L.ids, scan=np.array([(r[0], r[1]) for r in res]), scan_val=np.array([(r[2], r[3], r[4], r[5]) for r in res]))
P("\ndone")
LOG.close()
