"""pcdx step 1: per-airport choice and dose for the plan-clock-distance (pcd) corrector, same procedure as agent_refit2_10_full.py / agent_refit2_11_est.py.
Frame: nine airports x eight old months (audit frame), served chain, 3 seeds averaged, 2000 day-bootstrap draws within month. Negative = first arm better.
Arms: OLD = shipped term (agent_bigc12_r1x), V141 = refit term (agent_bigc13_r1x), PCD = pcd refit (agent_pcd_modal). Currently served (v143): PCD at the seven at v141 doses, OLD at EHAM/LFPG at shipped doses.
Local arrays only; 2025 labels only."""
import sys, json, itertools
from collections import Counter
import numpy as np
sys.path.insert(0, "archive")
import agent_basefix_lib as L   # noqa: E402
I = "data/interim/"
z = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True)
fold_ids = z["ids"].astype(float); pos = {v: i for i, v in enumerate(fold_ids.tolist())}
ix = np.array([pos[v] for v in L.ids.tolist()]); monf = z["mon"]
assert np.array_equal(np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)["ids"].astype(float), fold_ids)
NB = np.load(I + "agent_bigc13_base.npz")["nb"][ix]
Nf = len(fold_ids)


def pcdarr(s):
    arr = np.zeros(Nf); cnt = np.zeros(Nf, int)
    for m in range(1, 13):
        d = np.load(I + f"agent_pcd_modal/C4ff7V5_s{s}_m{m}.npz"); assert np.array_equal(d["rows"], np.where(monf == m)[0]); arr[d["rows"]] = d["pred"]; cnt[d["rows"]] += 1
    assert (cnt == 1).all()
    return arr


S3 = (0, 1, 2)
P_ = {s: pcdarr(s)[ix] for s in S3}
V_ = {s: np.load(I + f"agent_bigc13_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"][ix] for s in S3}
O_ = {s: np.load(I + f"agent_bigc12_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"][ix] for s in S3}
y, mon, apt, AIR = L.y, L.mon, L.apt, L.AIR
MONS = sorted(set(mon.tolist())); N = len(y); jj = np.isin(mon, [1, 7])
DOSE7 = {"EDDF": 1.25, "EDDM": 1.25, "EGLL": 1.0, "LEBL": 1.0, "LEMD": 1.0, "LIRF": 0.75, "LSZH": 1.25}
SHIPK = json.load(open(I + "agent_m12_kapt_C4ff7V5.json")); K5 = L.K5; assert np.allclose(K5, np.array([SHIPK[a] for a in apt]))
K41 = np.array([DOSE7.get(a, SHIPK[a]) for a in apt]); is7 = np.isin(apt, list(DOSE7))
avg = lambda D: np.mean([D[s] for s in S3], 0)   # noqa: E731
P, V, O = avg(P_), avg(V_), avg(O_)
ARM = {"old": O, "pcd": P, "v141": V}
print("rms of term: pcd %.2f v141 %.2f old %.2f; corr(pcd,old) %.3f corr(pcd,v141) %.3f" % (np.sqrt((P**2).mean()), np.sqrt((V**2).mean()), np.sqrt((O**2).mean()), np.corrcoef(P, O)[0, 1], np.corrcoef(P, V)[0, 1]))


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
daysum = lambda e2, sel: (np.bincount(D[sel], weights=e2[sel], minlength=nd), np.bincount(D[sel], minlength=nd).astype(float))   # noqa: E731


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


def line(Pv, Av, tag, extra=None):
    o = [f"{tag:30s}"]
    for nm, ms in (("Jan", [1]), ("Jul", [7]), ("J+J", [1, 7]), ("all8", MONS)):
        g, p, lo, hi = cmp(Pv, Av, ms, extra)
        o.append(f"{nm} {g:+.3f} (P {p:.2f}" + (f", 90% [{lo:+.3f},{hi:+.3f}]" if nm in ("J+J", "all8") else "") + ")")
    sel = jj if extra is None else jj & extra
    e2p, e2a = (Pv - y) ** 2, (Av - y) ** 2; g = (e2a - e2p)[sel]; keep = np.ones(sel.sum(), bool); keep[np.argsort(-g)[:50]] = False
    o.append(f"m50 {np.sqrt(e2p[sel][keep].mean()) - np.sqrt(e2a[sel][keep].mean()):+.3f}")
    return "  ".join(o)


def split(Pv, Av, sel, tag):
    ec, ea = (Av - y) ** 2, (Pv - y) ** 2; imp = (ec - ea)[sel]; yy = y[sel]; tot = imp.sum(); top = np.argsort(-imp)[:20]
    print(f"   split {tag}: SSE change share body(y<=3600) {imp[yy <= 3600].sum() / ec[sel].sum():+.5f} tail {imp[yy > 3600].sum() / ec[sel].sum():+.5f}; top-20 rows hold {imp[top].sum() / tot if tot > 0 else float('nan'):.2f} of gain; total share {tot / ec[sel].sum():+.5f}")


CUR = served(np.where(is7, P, O), K41)          # v143 as served on the frame
print("\n== 0. v143's corrector change on the frame: pcd (3 seeds) at the seven at v141 doses vs the v142 term (v141 refit at the seven, EHAM/LFPG old) ==")
V142 = served(np.where(is7, V, O), K41)
print(line(CUR, V142, "v143 vs v142"))
for s in S3:
    print(line(served(np.where(is7, P_[s], O_[s]), K41), served(np.where(is7, V_[s], O_[s]), K41), f"  seed {s}"))
print(line(CUR, served(O, K5), "v143 vs shipped-old (all)"))
split(CUR, V142, jj, "J+J v143 vs v142"); split(CUR, V142, np.ones(N, bool), "all8 v143 vs v142")

print("\n== 1. per airport: pcd term vs term currently served, airport rows only ==")
for a in AIR:
    m = apt == a
    print(f"-- {a} (frame rows {m.sum():,}, current = {'pcd@' + str(DOSE7[a]) if a in DOSE7 else 'old@' + str(SHIPK[a])})")
    R = served(np.where(m, P, np.where(is7, P, O)), K41)
    if a in DOSE7:
        print("   " + line(CUR, served(np.where(is7, V, O), K41), "pcd vs v141refit (same dose)", m))
        print("   " + line(CUR, served(np.where(m, O, np.where(is7, P, O)), np.where(m, K5, K41)), "pcd@v141dose vs old@shipped", m))
    else:
        print("   " + line(R, CUR, "pcd@shipped dose vs old", m))
        for s in S3:
            print("   " + line(served(np.where(m, P_[s], O_[s]), K5), served(O_[s], K5), f"  seed {s}", m))
        print("   " + line(served(np.where(m, P, O), K5), served(np.where(m, V, O), K5), "pcd vs v141refit (same dose)", m))
        split(R, CUR, jj & m, "J+J"); split(R, CUR, m, "all8")
        print("   per month (pcd - old):", {mm: round(cmp(R, CUR, [mm], m)[0], 3) for mm in MONS})

GR = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
OPT = [(arm, k) for arm in ("old", "pcd") for k in GR]
T = {}
for arm, k in OPT:
    e2 = (served(ARM[arm], k) - y) ** 2
    for a in AIR:
        for m in MONS:
            T[(arm, k, a, m)] = float(e2[(apt == a) & (mon == m)].sum())


def nested(optsfn):
    pred = np.zeros(N); ch = {}
    for a in AIR:
        opts = optsfn(a)
        for h in MONS:
            best = min(opts, key=lambda o: sum(T[(o[0], o[1], a, m)] for m in MONS if m != h)); ch[(a, h)] = best
            s_ = (apt == a) & (mon == h); pred[s_] = served(ARM[best[0]], best[1])[s_]
    return pred, ch


print("\n== 1b. NESTED per-airport arm choice (old at shipped dose vs pcd at served dose; each held month chooses on the other seven) ==")
prA, chA = nested(lambda a: [("old", SHIPK[a]), ("pcd", DOSE7.get(a, SHIPK[a]))])
print(line(prA, CUR, "nested arm vs v143"))
for a in AIR:
    o2 = [("old", SHIPK[a]), ("pcd", DOSE7.get(a, SHIPK[a]))]
    print(f"  {a}: " + " ".join(chA[(a, h)][0][0] + str(chA[(a, h)][1]) for h in MONS) + "   all-8 SSE-best: " + str(min(o2, key=lambda o: sum(T[(o[0], o[1], a, m)] for m in MONS))))
prB, chB = nested(lambda a: OPT)
print(line(prB, CUR, "nested arm+dose (old/pcd x grid)"))
prC, chC = nested(lambda a: [("pcd", k) for k in GR])
print(line(prC, CUR, "pcd everywhere, nested dose"))
print("\n== 2. DOSE grid on the pcd corrector, all nine airports (3-seed avg) ==")
chosen = {}; kn = np.zeros(N)
for a in AIR:
    tot = {k: sum(T[("pcd", k, a, m)] for m in MONS) for k in GR}; jjs = {k: sum(T[("pcd", k, a, m)] for m in (1, 7)) for k in GR}
    kb = min(tot, key=tot.get)
    print(f"  {a}: all8 best {kb} (SSE rel dose1.0 " + " ".join(f"{k}:{tot[k] / tot[1.0] - 1:+.4f}" for k in GR) + f"); J+J best {min(jjs, key=jjs.get)}; served dose {DOSE7.get(a, SHIPK[a])} ({'pcd' if a in DOSE7 else 'old'})")
    for h in MONS:
        chosen[(a, h)] = min(GR, key=lambda k: sum(T[("pcd", k, a, m)] for m in MONS if m != h)); kn[(apt == a) & (mon == h)] = chosen[(a, h)]
    kc = DOSE7.get(a, SHIPK[a]); agree = sum(chosen[(a, h)] == kc for h in MONS); mc = Counter(chosen[(a, h)] for h in MONS).most_common(1)[0]
    print("     nested pcd dose per held month: " + " ".join(str(chosen[(a, h)]) for h in MONS) + f"  | agreeing with served dose {kc}: {agree}/8 | modal {mc[0]} in {mc[1]}/8")
RN7 = served(np.where(is7, P, O), np.where(is7, kn, K5)); print(line(RN7, CUR, "seven, nested pcd dose vs v143"))
RN9 = served(P, kn); print(line(RN9, CUR, "nine, nested pcd dose vs v143"))
print("smooth parabola optimum vs seeds averaged k (does it rise with k?)")
for a in AIR:
    out = []
    for k_ in (1, 2, 3):
        cnt = {k: 0.0 for k in GR}
        for S in itertools.combinations(S3, k_):
            for k in GR:
                cnt[k] += float(((served(np.mean([P_[s] for s in S], 0), k) - y)[apt == a] ** 2).sum())
        c = np.polyfit(GR, [cnt[k] for k in GR], 2); out.append(-c[1] / (2 * c[0]))
    print(f"  {a}: k=1 {out[0]:.2f} k=2 {out[1]:.2f} k=3 {out[2]:.2f}")
