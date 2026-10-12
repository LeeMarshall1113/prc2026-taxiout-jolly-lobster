"""arith3 judge (copy of agent_arith_03_judge.py logic; where() also reads agent_arith3_out): judge arms of agent_arith_02_fit.py / agent_arith3_02_fit.py (and the pcd experiment's ctrl / dist / shuf fits, read only) THROUGH THE CHAIN as served (nb + per-airport dose x 3-seed-mean correction, 0.5 x tail, Rome rules)
on the audit frame (nine airports, months 1,2,3,7,8,9,10,11; 2025 labels only).
usage: python agent_arith_03_judge.py screen MONTHS SEEDS BASE ARM1,ARM2,...      cheap screen: RMSE difference arm minus BASE per month and pooled, corrector level and chain level
       python agent_arith_03_judge.py full SEEDS CAND PCDARM SHUFARM [GATEELSE]    full table: CAND vs ctrl, vs PCDARM, vs SHUFARM (all-eight, Jan+Jul, bootstrap, body/tail/top20, best-50, airports, rules)
       python agent_arith_03_judge.py temp CAND PCDARM SEEDS                         temporal frame (fit 1-6 -> 7-12 and reverse)
arm names: ctrl, dist, shuf (pcd experiment fits in agent_pcd_out) or an arith arm name (agent_arith_out). Negative = first arm better."""
import sys
import glob
import numpy as np

sys.path.insert(0, "archive")
import agent_basefix_lib as L   # noqa: E402

I = "data/interim/"
ALLM = (1, 2, 3, 7, 8, 9, 10, 11)
z = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True)
fold_ids = z["ids"].astype(float)
pos = {v: i for i, v in enumerate(fold_ids.tolist())}
ix = np.array([pos[v] for v in L.ids.tolist()])
NB = np.load(I + "agent_bigc13_base.npz")["nb"][ix]
y = L.y
NBNR = np.maximum(np.maximum(L.p, 0) + L.M1 + L.M2 + L.M3, 0) + L.SIL + L.META
rm = lambda p, m: float(np.sqrt(((p - y)[m] ** 2).mean()))   # noqa: E731


def where(arm):
    """round-3 arms live in agent_arith3_out; round-2 arms (e.g. the base ABCDE-pw-s0-r1) in agent_arith_out; pcd experiment arms in agent_pcd_out (all read only)"""
    if arm in ("ctrl", "dist", "shuf", "noise"):
        return I + "agent_pcd_out/"
    import os
    return I + ("agent_arith3_out/" if glob.glob(I + "agent_arith3_out/%s_s*_m*.npz" % arm) else "agent_arith_out/")


def load(arm, seed, mons, frame="lomo", what="pred"):
    F = np.full(len(fold_ids), np.nan)
    pat = {"lomo": ["m%d"], "temp": ["ta", "tb"]}[frame]
    for m in (mons if frame == "lomo" else [0]):
        for p in pat:
            f = where(arm) + "%s_s%d_%s.npz" % (arm, seed, p % m if "%d" in p else p)
            try:
                d = np.load(f)
            except FileNotFoundError:
                continue
            F[d["rows"]] = d["pred"] if what == "pred" else d[what].astype(float)
    return F[ix]


def avg(arm, seeds, mons, frame="lomo"):
    return np.mean([load(arm, s, mons, frame) for s in seeds], 0)


def served(base, corr):
    x = np.maximum(base + np.where(np.isfinite(corr), corr, 0) * L.K5, 0)
    x = np.maximum(x + 0.5 * L.TAIL, 0)
    out = x.copy()
    out[L.mid] = x[L.mid] + L.W3[L.mid] * (L.MIX[L.mid] - x[L.mid])
    for rr in ("R1", "R2", "R3b"):
        s = L.rule == rr
        out[s] = L.ebpred[s]
    return np.rint(np.maximum(out, 0))


days = np.array([str(d) for d in L.day])
ud = sorted(set(days.tolist()))
di = {d: i for i, d in enumerate(ud)}
did = np.array([di[d] for d in days])
nd = len(ud)
dm = np.array([int(d[5:7]) for d in ud])
rng = np.random.default_rng(11)
B = 2000
CNT = np.zeros((B, nd))
for m in set(dm.tolist()):
    idx = np.where(dm == m)[0]
    dr = rng.integers(0, len(idx), (B, len(idx)))
    for b in range(B):
        np.add.at(CNT[b], idx[dr[b]], 1)


def boot(Ea, Eb, mask):
    sa = np.bincount(did[mask], Ea[mask], minlength=nd)
    sb = np.bincount(did[mask], Eb[mask], minlength=nd)
    n = np.bincount(did[mask], minlength=nd).astype(float)
    c = CNT * (np.isin(dm, sorted(set(L.mon[mask].tolist()))))[None, :]
    d = np.sqrt((c @ sa) / (c @ n)) - np.sqrt((c @ sb) / (c @ n))
    return np.percentile(d, [2.5, 97.5]), (d < 0).mean()


def pair_report(nm, Pa, Pb, mask, months_):
    Ea, Eb = (Pa - y) ** 2, (Pb - y) ** 2
    mm = " ".join("m%d %+.3f" % (m, rm(Pa, mask & (L.mon == m)) - rm(Pb, mask & (L.mon == m))) for m in months_)
    ci, p = boot(Ea, Eb, mask)
    jj = mask & L.JJ
    s = "%s: pooled %+.3f [%+.3f, %+.3f] P %.2f | Jan %+.3f Jul %+.3f JJ %+.3f | %s" % (
        nm, rm(Pa, mask) - rm(Pb, mask), ci[0], ci[1], p,
        rm(Pa, mask & L.JAN) - rm(Pb, mask & L.JAN) if (mask & L.JAN).any() else np.nan,
        rm(Pa, mask & L.JUL) - rm(Pb, mask & L.JUL) if (mask & L.JUL).any() else np.nan,
        rm(Pa, jj) - rm(Pb, jj) if jj.any() else np.nan, mm)
    print(s)
    return Ea, Eb


def detail(nm, Pa, Pb, mask, months_):
    Ea, Eb = (Pa - y) ** 2, (Pb - y) ** 2
    for lab, mm_ in (("Jan", L.JAN), ("Jul", L.JUL)):
        print("   %s per airport %s: %s" % (nm, lab, " ".join("%s %+.2f" % (a, rm(Pa, mask & mm_ & (L.apt == a)) - rm(Pb, mask & mm_ & (L.apt == a))) for a in L.AIR if (mask & mm_ & (L.apt == a)).any())))
    imp = Eb - Ea
    body = y <= 3600
    top = np.argsort(-np.where(mask, imp, -1e18))[:20]
    tot = imp[mask].sum()
    print("   %s split (share of second arm's SSE): body %+.5f tail %+.5f; top-20 rows hold %.2f of the gain" % (
        nm, imp[mask & body].sum() / Eb[mask].sum(), imp[mask & ~body].sum() / Eb[mask].sum(), imp[top].sum() / tot if tot > 0 else float("nan")))
    keep = np.zeros(len(y), bool)
    for m in months_:
        ii = np.where(mask & (L.mon == m))[0]
        keep[ii[np.argsort(-imp[ii])[50:]]] = True
    print("   %s best-50-per-month removed: %+.3f" % (nm, np.sqrt(Ea[keep].mean()) - np.sqrt(Eb[keep].mean())))
    rome = L.apt == "LIRF"
    print("   %s Rome rows %+.3f non-Rome %+.3f | per airport: %s" % (
        nm, rm(Pa, mask & rome) - rm(Pb, mask & rome), rm(Pa, mask & ~rome) - rm(Pb, mask & ~rome),
        " ".join("%s %+.2f" % (a, rm(Pa, mask & (L.apt == a)) - rm(Pb, mask & (L.apt == a))) for a in L.AIR if (mask & (L.apt == a)).any())))


mode = sys.argv[1]
if mode == "screen":
    mons = [int(m) for m in sys.argv[2].split(",")]
    seeds = [int(s) for s in sys.argv[3].split(",")]
    base = sys.argv[4]
    arms = sys.argv[5].split(",")
    mask = np.isin(L.mon, mons)
    Fb = avg(base, seeds, mons)
    mask &= np.isfinite(Fb)
    Pb = served(NB, Fb)
    print("screen months %s seeds %s base %s rows %d base RMSE %.3f (corrector level %.3f)" % (mons, seeds, base, mask.sum(), rm(Pb, mask), rm(np.maximum(NB + np.where(np.isfinite(Fb), Fb, 0) * L.K5, 0), mask)))
    for a in arms:
        Fa = avg(a, seeds, mons)
        if not np.isfinite(Fa[mask]).all():
            print(a, "incomplete")
            continue
        Pa = served(NB, Fa)
        lv = rm(np.maximum(NB + Fa * L.K5, 0), mask) - rm(np.maximum(NB + Fb * L.K5, 0), mask)
        pair_report("%-18s corr-level %+.3f chain" % (a, lv), Pa, Pb, mask, mons)
        detail(a, Pa, Pb, mask, mons)
elif mode == "full":
    seeds = [int(s) for s in sys.argv[2].split(",")]
    cand, pcd, shuf = sys.argv[3], sys.argv[4], sys.argv[5]
    mask = np.ones(len(y), bool)
    F = {a: avg(a, seeds, ALLM) for a in ("ctrl", pcd, shuf, cand)}
    for a in F:
        if not np.isfinite(F[a]).all():
            print(a, "incomplete: rows missing", int((~np.isfinite(F[a])).sum()))
    P_ = {a: served(NB, F[a]) for a in F}
    print("RMSE all eight: " + " ".join("%s %.3f" % (a, rm(P_[a], mask)) for a in P_))
    for nm, a, b in (("cand-ctrl", cand, "ctrl"), ("cand-pcd", cand, pcd), ("cand-shuf", cand, shuf), ("pcd-ctrl", pcd, "ctrl"), ("shuf-pcd (placebo)", shuf, pcd), ("shuf-ctrl", shuf, "ctrl")):
        pair_report(nm, P_[a], P_[b], mask, ALLM)
    for nm, a, b in (("cand-pcd", cand, pcd), ("cand-shuf", cand, shuf), ("cand-ctrl", cand, "ctrl")):
        detail(nm, P_[a], P_[b], mask, ALLM)
    # rules earned
    for a in (pcd, cand):
        print("rules earned %s: %+.3f" % (a, rm(served(NBNR, F[a]), mask) - rm(P_[a], mask)))
    # gate arm (d): new output only where the label-free gate passes
    G = np.mean([load(cand, s, ALLM, what="gate") for s in seeds], 0) > 0.5
    for else_ in ("ctrl", pcd):
        Fd = np.where(G, F[cand], F[else_])
        Pd = served(NB, Fd)
        pair_report("gated(d) [cand where gate else %s] minus pcd" % else_, Pd, P_[pcd], mask, ALLM)
        pair_report("gated(d) [cand where gate else %s] minus ctrl" % else_, Pd, P_["ctrl"], mask, ALLM)
    print("gate share of rows %.4f" % G[mask].mean())
elif mode == "temp":
    cand, pcd = sys.argv[2], sys.argv[3]
    seeds = [int(s) for s in sys.argv[4].split(",")]
    Fc = avg(cand, seeds, [0], "temp")
    Fp = avg(pcd, seeds, [0], "temp")
    Fk = avg("ctrl", seeds, [0], "temp")
    mask = np.isfinite(Fc) & np.isfinite(Fp) & np.isfinite(Fk)
    mons = sorted(set(L.mon[mask].tolist()))
    print("temporal frame rows %d months %s" % (mask.sum(), mons))
    Pc, Pp, Pk = served(NB, Fc), served(NB, Fp), served(NB, Fk)
    pair_report("cand-pcd", Pc, Pp, mask, mons)
    pair_report("cand-ctrl", Pc, Pk, mask, mons)
    pair_report("pcd-ctrl", Pp, Pk, mask, mons)
    detail("cand-pcd", Pc, Pp, mask, mons)
