"""refit2 experiment, second pass (docs/LAB-NOTES.md): local estimates on arrays on disk. New file only.
    python archive/agent_refit2_11_est.py
Same frame, served chain, bootstrap as agent_refit2_10_full.py (nine airports, eight old months, 3-seed refit arrays vs shipped arrays).
A: airports not refitted (EHAM, LFPG; LTFM is outside the corrector). B: seed curve. C: dose grid. D: row groups."""
import sys, json, itertools
import numpy as np
sys.path.insert(0, "archive")
import agent_basefix_lib as L   # noqa: E402
I = "data/interim/"
z = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
fold_ids = z["ids"].astype(float); pos = {v: i for i, v in enumerate(fold_ids.tolist())}
ix = np.array([pos[v] for v in L.ids.tolist()])
NB = np.load(I + "agent_bigc13_base.npz")["nb"][ix]
N_ = {s: np.load(I + f"agent_bigc13_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"][ix] for s in (0, 1, 2)}
C_ = {s: np.load(I + f"agent_bigc12_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"][ix] for s in (0, 1, 2)}
y, mon, apt, AIR = L.y, L.mon, L.apt, L.AIR
MONS = sorted(set(mon.tolist())); N = len(y)
DOSE7 = {"EDDF": 1.25, "EDDM": 1.25, "EGLL": 1.0, "LEBL": 1.0, "LEMD": 1.0, "LIRF": 0.75, "LSZH": 1.25}
SHIPK = json.load(open(I + "agent_m12_kapt_C4ff7V5.json"))
K5 = L.K5; assert np.allclose(K5, np.array([SHIPK[a] for a in apt]))
K41 = np.array([DOSE7.get(a, SHIPK[a]) for a in apt])
is7 = np.isin(apt, list(DOSE7))
avg = lambda D, S: np.mean([D[s] for s in S], 0)   # noqa: E731


def served(corr, kvec):
    x = np.maximum(NB + corr * kvec, 0); x = np.maximum(x + 0.5 * L.TAIL, 0); out = x.copy()
    out[L.mid] = x[L.mid] + L.W3[L.mid] * (L.MIX[L.mid] - x[L.mid])
    for rr in ("R1", "R2", "R3b"):
        s_ = L.rule == rr; out[s_] = L.ebpred[s_]
    return np.rint(np.maximum(out, 0))


days = sorted(set(L.day.tolist())); did = {d: i for i, d in enumerate(days)}; D = np.array([did[d] for d in L.day.tolist()]); nd = len(days)
dm = np.zeros(nd, int)
for d, i in did.items():
    dm[i] = int(d[5:7])
rng = np.random.default_rng(11); B = 2000; CNT = np.zeros((B, nd))
for m_ in MONS:
    idx = np.where(dm == m_)[0]; draw = rng.integers(0, len(idx), size=(B, len(idx)))
    for b in range(B):
        np.add.at(CNT[b], idx[draw[b]], 1)


def daysum(e2, sel):
    return np.bincount(D[sel], weights=e2[sel], minlength=nd), np.bincount(D[sel], minlength=nd).astype(float)


def cmp(Pv, Av, months, extra=None):
    sel = np.isin(mon, months)
    if extra is not None:
        sel = sel & extra
    e2p, e2a = (Pv - y) ** 2, (Av - y) ** 2
    sp, n = daysum(e2p, sel); sa, _ = daysum(e2a, sel)
    c = CNT * np.isin(dm, months)[None, :]
    dd = np.sqrt((c @ sp) / (c @ n)) - np.sqrt((c @ sa) / (c @ n))
    lo, hi = np.percentile(dd, [5, 95])
    return float(np.sqrt(e2p[sel].mean()) - np.sqrt(e2a[sel].mean())), float((dd < 0).mean()), float(lo), float(hi)


def line(Pv, Av, tag):
    o = [f"{tag:30s}"]
    for nm, ms in (("Jan", [1]), ("Jul", [7]), ("J+J", [1, 7]), ("all8", MONS)):
        g, p, lo, hi = cmp(Pv, Av, ms)
        o.append(f"{nm} {g:+.3f} (P {p:.2f}" + (f", 90% [{lo:+.3f},{hi:+.3f}]" if nm in ("J+J", "all8") else "") + ")")
    e2p, e2a = (Pv - y) ** 2, (Av - y) ** 2; jj_ = np.isin(mon, [1, 7]); g = (e2a - e2p)[jj_]; keep = np.ones(jj_.sum(), bool); keep[np.argsort(-g)[:50]] = False
    o.append(f"m50 {np.sqrt(e2p[jj_][keep].mean()) - np.sqrt(e2a[jj_][keep].mean()):+.3f}")
    return "  ".join(o)


S3 = (0, 1, 2)
jj = np.isin(mon, [1, 7])
SH = served(avg(C_, S3), K5)
print("frame rows %d; served-chain RMSE shipped: J+J %.3f all8 %.3f" % (N, np.sqrt(((SH - y)[jj] ** 2).mean()), np.sqrt(((SH - y) ** 2).mean())))
V141 = served(np.where(is7, avg(N_, S3), avg(C_, S3)), K41)
print("\n== 0. v141 recipe re-scored (7 airports refit at the mega doses, EHAM/LFPG shipped) vs shipped, 3 seeds ==")
print(line(V141, SH, "v141 mix avg3"))
for s in S3:
    print(line(served(np.where(is7, N_[s], C_[s]), K41), served(C_[s], K5), f"v141 mix seed {s}"))

print("\n== A. airports not refitted ==")
for a in ("EHAM", "LFPG"):
    m = apt == a
    print(f"-- {a}: scored 2026 rows {dict(EHAM=38182, LFPG=39872)[a]:,} of 344,841; frame rows {m.sum():,}")
    for dose_tag, kv in (("shipped dose", K5), ("dose 1.0", np.ones(N)), ("dose 0.75", np.full(N, .75)), ("dose 1.25", np.full(N, 1.25))):
        R = served(avg(N_, S3), np.where(m, kv, K5))
        R = np.where(m, R, SH)
        print("  " + line(R, SH, f"refit avg3, {dose_tag}"))
    for s in S3:
        R = served(np.where(m, N_[s], C_[s]), K5); Rs = served(C_[s], K5)
        print("  " + line(R, Rs, f"refit seed {s}, shipped dose"))
    Rm = served(np.where(m, avg(N_, S3), avg(C_, S3)), K5)
    print("  per month avg3 (refit - ship RMSE at this airport):", {mm: round(cmp(Rm, SH, [mm], m)[0], 3) for mm in MONS})
    print("  corr(refit raw, shipped raw) avg3: %.3f; rms diff of dosed term %.2f s" % (np.corrcoef(avg(N_, S3)[m], avg(C_, S3)[m])[0, 1], np.sqrt(((avg(N_, S3) - avg(C_, S3))[m] ** 2).mean()) * K5[m][0]))
both = np.isin(apt, ["EHAM", "LFPG"])
print("-- EHAM+LFPG pooled, refit at shipped doses:", line(served(np.where(both, avg(N_, S3), avg(C_, S3)), K5), SH, "refit EHAM+LFPG"))
print("-- ALL NINE refit at shipped doses:", line(served(avg(N_, S3), K5), SH, "refit all nine, shipped"))
print("-- nine refit with v141 doses at seven, shipped at two (EHAM/LFPG refit at shipped dose):", line(served(avg(N_, S3), K41), SH, "refit nine, K41"))

print("\n== B. seed curve (served chain; mean over subsets of seeds; v141 mix arm vs shipped arm) ==")
rows = {}
for k in (1, 2, 3):
    r = {"ship": [], "mix": []}
    for S in itertools.combinations(S3, k):
        r["ship"].append(served(avg(C_, S), K5)); r["mix"].append(served(np.where(is7, avg(N_, S), avg(C_, S)), K41))
    rows[k] = r


def mse(P, sel):
    return float(((P - y)[sel] ** 2).mean())


for nm, sel in (("J+J", jj), ("all8", np.ones(N, bool))):
    print(f"[{nm}]")
    fit = {}
    for arm in ("ship", "mix"):
        M = {k: np.mean([mse(P, sel) for P in rows[k][arm]]) for k in (1, 2, 3)}
        s_ = (M[1] - M[3]) / (1 - 1 / 3); Minf = M[1] - s_
        fit[arm] = (M, s_, Minf)
        print(f"  {arm}: MSE k=1 {M[1]:.2f} k=2 {M[2]:.2f} k=3 {M[3]:.2f}; model MSE = Minf + s/k with s {s_:.2f}; k=2 predicted {Minf + s_ / 2:.2f}; RMSE k=1,2,3: " + " ".join(f"{np.sqrt(M[k]):.3f}" for k in (1, 2, 3)))
        print(f"       extrapolated RMSE k=6 {np.sqrt(Minf + s_ / 6):.3f}, k=12 {np.sqrt(Minf + s_ / 12):.3f}, inf {np.sqrt(Minf):.3f}; gain 3->6 {np.sqrt(M[3]) - np.sqrt(Minf + s_ / 6):.3f}, 6->12 {np.sqrt(Minf + s_ / 6) - np.sqrt(Minf + s_ / 12):.3f}")
    fm, fs = fit["mix"], fit["ship"]
    print("  mix minus ship RMSE by k: " + " ".join(f"k={k} {np.sqrt(fm[0][k]) - np.sqrt(fs[0][k]):+.3f}" for k in (1, 2, 3))
          + f"; extrapolated k=6 {np.sqrt(fm[2] + fm[1] / 6) - np.sqrt(fs[2] + fs[1] / 6):+.3f}, k=12 {np.sqrt(fm[2] + fm[1] / 12) - np.sqrt(fs[2] + fs[1] / 12):+.3f}, k=inf {np.sqrt(fm[2]) - np.sqrt(fs[2]):+.3f}")
print("row-level single-seed sd of the dosed term (rms over rows, unbiased over 3 seeds), refit arm / shipped arm:")
Nst = np.stack([N_[s] for s in S3]); Cst = np.stack([C_[s] for s in S3])
for a in AIR:
    m = apt == a
    sn = np.sqrt(Nst[:, m].var(0, ddof=1).mean()) * (DOSE7.get(a, SHIPK[a])); sc = np.sqrt(Cst[:, m].var(0, ddof=1).mean()) * SHIPK[a]
    print(f"  {a}: refit {sn:.2f} s, shipped {sc:.2f} s (dose-scaled; n {m.sum():,})")
for kk in ((3, 6), (6, 12), (6, 24)):
    d_mse = 0.0
    for a in DOSE7:
        m = (apt == a) & jj
        d_mse += (Nst[:, m].var(0, ddof=1).mean() * DOSE7[a] ** 2) * (1 / kk[0] - 1 / kk[1]) * m.sum()
    rm = np.sqrt(mse(V141, jj))
    print(f"  analytic seven-airport noise saving seeds {kk[0]}->{kk[1]}: {d_mse / jj.sum() / (2 * rm):.4f} s RMSE on J+J (nine-airport frame)")

print("\n== C. dose on the refitted corrector (avg3) ==")
GR = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
SSE = {}
for k in GR:
    e2 = (served(avg(N_, S3), k) - y) ** 2
    for a in AIR:
        for m_ in MONS:
            SSE[(k, a, m_)] = float(e2[(apt == a) & (mon == m_)].sum())
print("all-8 SSE-best dose per airport (refit):")
for a in AIR:
    tot = {k: sum(SSE[(k, a, m_)] for m_ in MONS) for k in GR}
    jjs = {k: sum(SSE[(k, a, m_)] for m_ in (1, 7)) for k in GR}
    kb = min(tot, key=tot.get)
    print(f"  {a}: all8 best {kb} (SSE rel to dose1.0: " + " ".join(f"{k}:{tot[k] / tot[1.0] - 1:+.4f}" for k in GR) + f"); J+J best {min(jjs, key=jjs.get)}; v141 dose {K41[apt == a][0]}")
kn = np.zeros(N)
chosen = {}
for a in AIR:
    for h in MONS:
        kb = min(GR, key=lambda k: sum(SSE[(k, a, m_)] for m_ in MONS if m_ != h)); chosen[(a, h)] = kb
        kn[(apt == a) & (mon == h)] = kb
for a in AIR:
    print(f"  nested choice {a}: " + " ".join(str(chosen[(a, h)]) for h in MONS))
RN = served(avg(N_, S3), kn); RV = V141
print(line(RN, SH, "refit nine, nested dose vs ship"))
print(line(RN, RV, "refit nine nested dose vs v141"))
kn2 = np.where(is7, kn, K5)
RN2 = served(np.where(is7, avg(N_, S3), avg(C_, S3)), kn2)
print(line(RN2, SH, "seven refit, nested dose vs ship"))
print(line(RN2, RV, "seven refit, nested dose vs v141"))
print("smooth (parabola) all-8 SSE optimum with k seeds averaged, refit arm (does the optimum rise with k?)")
for a in AIR:
    out = []
    for k_ in (1, 2, 3):
        cnt = {k: 0.0 for k in GR}
        for S in itertools.combinations(S3, k_):
            for k in GR:
                cnt[k] += float(((served(avg(N_, S), k) - y)[apt == a] ** 2).sum())
        xs = np.array(GR); ys = np.array([cnt[k] for k in GR]); c = np.polyfit(xs, ys, 2); out.append(-c[1] / (2 * c[0]))
    print(f"  {a}: k=1 {out[0]:.2f}, k=2 {out[1]:.2f}, k=3 {out[2]:.2f}  (v141 dose {K41[apt == a][0]})")

print("\n== D. row groups (served chain, J+J rows of the frame) ==")
cov, plan, rule = L.cov, L.plan, L.rule
NOC = served(np.zeros(N), 0.0)
T = ((SH - y) ** 2)[jj].sum()
groups = {}
groups["rule rows R1/R2/R3b"] = np.isin(rule, ["R1", "R2", "R3b"])
groups["Rome mixture rows (mid)"] = L.mid
groups["no plan (plan False)"] = ~plan
groups["points_no_liftoff"] = cov == "points_no_liftoff"
groups["no_adsb_points"] = cov == "no_adsb_points"
groups["other cov classes"] = ~np.isin(cov, ["matched", "points_no_liftoff", "no_adsb_points"])
groups["matched"] = cov == "matched"
for a in AIR:
    groups[f"{a} no-plan"] = (apt == a) & ~plan
groups["all"] = np.ones(N, bool)
for nm, g in groups.items():
    gj = g & jj
    if gj.sum() == 0:
        print(nm, "no J+J rows"); continue
    e2s = ((SH - y) ** 2)[gj].sum(); e2n = ((NOC - y) ** 2)[gj].sum(); e2v = ((V141 - y) ** 2)[gj].sum()
    print(f"{nm:26s} rows {g.sum():>9,} ({g.sum() / N:6.2%}) J+J rows {gj.sum():>8,} SSE share {e2s / T:6.2%} RMSE {np.sqrt(e2s / gj.sum()):7.1f} corrector gain (SSE share pts) {(e2n - e2s) / T * 100:+6.2f} v141-ship dSSE/row(all J+J) {(e2v - e2s) / jj.sum():+8.3f}")
