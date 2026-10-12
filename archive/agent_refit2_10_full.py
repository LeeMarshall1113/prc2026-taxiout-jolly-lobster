"""refit2 step 10: judge the full run (36 fits: seeds 0-2, held months 1-12) of the refitted corrector against the shipped corrector as served.
    python archive/agent_refit2_10_full.py
Writes data/interim/agent_bigc13_r1x_s{0,1,2}_C4ff7V5.npz (assembled as the app's fetch does). New file only.
Served chain (nine airports, the eight old months of the audit frame; shipped doses K5 both arms unless stated) for January, July, pooled, every old month, all eight.
New months (Apr, May, Jun, Dec): no served frame; reported on the plain base chain max(nb + K F, 0) only (no tail, no Rome rules, no bootstrap: no day column), never gating.
Day-bootstrap: 2000 draws, days resampled within month. Negative = refit better."""
import sys
import json
import numpy as np
sys.path.insert(0, "archive")
import agent_basefix_lib as L   # noqa: E402
I = "data/interim/"
z = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
fold_ids = z["ids"].astype(float); pos = {v: i for i, v in enumerate(fold_ids.tolist())}
ix = np.array([pos[v] for v in L.ids.tolist()])
Nf = len(z["y"]); monf = z["mon"]; aptf = z["apt"].astype(str); yf = z["y"].astype(float)
NBf = np.load(I + "agent_bigc13_base.npz")["nb"]; NB = NBf[ix]

# assemble per-seed arrays exactly as fetch does
NEWF = {}
for s in (0, 1, 2):
    arr = np.zeros(Nf); cnt = np.zeros(Nf, int); bis = []
    for m in range(1, 13):
        d = np.load(I + f"refit2_out_new/C4ff7V5_s{s}_m{m}.npz"); arr[d["rows"]] = d["pred"]; cnt[d["rows"]] += 1; bis.append(int(d["bi"]))
    assert (cnt == 1).all(), (s, np.bincount(cnt))
    np.savez(I + f"agent_bigc13_r1x_s{s}_C4ff7V5.npz", C4ff7V5=arr, C4ff7V5_bi=np.array(bis)); NEWF[s] = arr
CTLF = {s: np.load(I + f"agent_bigc12_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"] for s in (0, 1, 2)}
for s in (0, 1, 2):
    print(f"seed {s}: refit pred rms {np.sqrt((NEWF[s]**2).mean()):.2f}  shipped rms {np.sqrt((CTLF[s]**2).mean()):.2f}  corr {np.corrcoef(NEWF[s], CTLF[s])[0,1]:.3f}", flush=True)
N_ = {s: NEWF[s][ix] for s in (0, 1, 2)}; C_ = {s: CTLF[s][ix] for s in (0, 1, 2)}
N_["avg3"] = np.mean([N_[s] for s in (0, 1, 2)], 0); C_["avg3"] = np.mean([C_[s] for s in (0, 1, 2)], 0)
MONS = sorted(set(L.mon.tolist())); print("served-frame months", MONS)
y, mon, apt, AIR = L.y, L.mon, L.apt, L.AIR
KS = [0.75, 1.0, 1.25]


def served(corr, kvec):
    x = np.maximum(NB + corr * kvec, 0); x = np.maximum(x + 0.5 * L.TAIL, 0); out = x.copy()
    out[L.mid] = x[L.mid] + L.W3[L.mid] * (L.MIX[L.mid] - x[L.mid])
    for rr in ("R1", "R2", "R3b"):
        s_ = L.rule == rr; out[s_] = L.ebpred[s_]
    return np.rint(np.maximum(out, 0))


# bootstrap over all days of the frame, resampled within month
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


def cmp(Pv, Av, months, sel_extra=None):
    """(refit - ship RMSE, P refit better) on the given months"""
    sel = np.isin(mon, months)
    if sel_extra is not None:
        sel = sel & sel_extra
    e2p, e2a = (Pv - y) ** 2, (Av - y) ** 2
    sp, n = daysum(e2p, sel); sa, _ = daysum(e2a, sel)
    msk = np.isin(dm, months)
    c = CNT * msk[None, :]
    dd = np.sqrt((c @ sp) / (c @ n)) - np.sqrt((c @ sa) / (c @ n))
    return float(np.sqrt(e2p[sel].mean()) - np.sqrt(e2a[sel].mean())), float((dd < 0).mean())


def line(Pv, Av, tag):
    out = [f"{tag:22s}"]
    for nm, ms in (("Jan", [1]), ("Jul", [7]), ("J+J", [1, 7]), ("all8", MONS)):
        g, p = cmp(Pv, Av, ms); out.append(f"{nm} {g:+.3f} (P {p:.2f})")
    e2p, e2a = (Pv - y) ** 2, (Av - y) ** 2; jj = np.isin(mon, [1, 7]); g = (e2a - e2p)[jj]; keep = np.ones(jj.sum(), bool); keep[np.argsort(-g)[:50]] = False
    out.append(f"m50 {np.sqrt(e2p[jj][keep].mean()) - np.sqrt(e2a[jj][keep].mean()):+.3f}")
    return "  ".join(out)


Kship = L.K5
print("\n== 1. REFIT vs SHIPPED, served chain, shipped doses both arms (negative = refit better) ==")
SH = {k: served(C_[k], Kship) for k in C_}; RF = {k: served(N_[k], Kship) for k in N_}
for k in ("seed0", "seed1", "seed2") if False else (0, 1, 2, "avg3"):
    print(line(RF[k], SH[k], f"seed {k}" if k != "avg3" else "3-seed average"))
print("\nevery old month (refit - shipped RMSE): per seed and average")
for m in MONS:
    r = [cmp(RF[k], SH[k], [m]) for k in (0, 1, 2, "avg3")]
    print(f"  month {m:2d}: " + "  ".join(f"{t} {g:+.3f}" for t, (g, p) in zip(("s0", "s1", "s2", "avg3"), r)) + f"   avg3 P {r[3][1]:.2f}   shipped RMSE {np.sqrt(((SH['avg3'] - y)[mon == m] ** 2).mean()):.2f}")
print("\nper airport, J+J and all8 (refit - shipped), per-airport day-bootstrap P of refit better, 3-seed avg; per-seed J+J in brackets")
APTROW = {}
for a in AIR:
    ma = apt == a
    gj, pj = cmp(RF["avg3"], SH["avg3"], [1, 7], ma); ga, pa = cmp(RF["avg3"], SH["avg3"], MONS, ma)
    ps = [cmp(RF[s], SH[s], [1, 7], ma)[0] for s in (0, 1, 2)]; APTROW[a] = (gj, pj, ga, pa)
    print(f"  {a}: J+J {gj:+.3f} (P {pj:.2f})  all8 {ga:+.3f} (P {pa:.2f})  seeds [{ps[0]:+.2f} {ps[1]:+.2f} {ps[2]:+.2f}]")

print("\nnew months (Apr May Jun Dec), plain base chain max(nb+K F,0) over the nine airports, refit - shipped RMSE, avg3 and per seed (no tail/Rome rules, no bootstrap; NOT gating):")
Kf = np.array([json.load(open(I + "agent_m12_kapt_C4ff7V5.json"))[a] for a in aptf])
pl = lambda F: np.maximum(NBf + F * Kf, 0)  # noqa: E731
Fa = {k: (np.mean([NEWF[s] for s in (0, 1, 2)], 0), np.mean([CTLF[s] for s in (0, 1, 2)], 0)) if k == "avg3" else (NEWF[k], CTLF[k]) for k in (0, 1, 2, "avg3")}
for m in (4, 5, 6, 12, 1, 7):
    s_ = monf == m
    print(f"  month {m:2d}: " + "  ".join(f"{t} {np.sqrt(((pl(Fa[k][0])[s_] - yf[s_]) ** 2).mean()) - np.sqrt(((pl(Fa[k][1])[s_] - yf[s_]) ** 2).mean()):+.3f}" for t, k in zip(("s0", "s1", "s2", "avg3"), (0, 1, 2, "avg3"))))

print("\n== 2. NESTED per-airport choice (leave-one-month-out over the 8 old months; options chosen by served-chain SSE on the other 7 months) ==")
# options: (arm, dose); SSE per airport per month
OPT = [(arm, k) for arm in ("ship", "refit") for k in KS]
SSE = {}
for arm, k in OPT:
    pr = served((C_ if arm == "ship" else N_)["avg3"], k)
    e2 = (pr - y) ** 2
    for a in AIR:
        for m in MONS:
            SSE[(arm, k, a, m)] = float(e2[(apt == a) & (mon == m)].sum())


def nested(opts):
    """returns served prediction where each (airport, held month) uses the option minimising SSE over the other months; also the choices"""
    pred = np.zeros(len(y)); ch = {}
    for a in AIR:
        for h in MONS:
            best = min(opts, key=lambda o: sum(SSE[(o[0], o[1], a, m)] for m in MONS if m != h)); ch[(a, h)] = best
            s_ = (apt == a) & (mon == h)
            pred[s_] = served((C_ if best[0] == "ship" else N_)["avg3"], best[1])[s_]
    return pred, ch


def nested_vs(opts, tag):
    pr, ch = nested(opts)
    print(line(pr, SH["avg3"], tag))
    return pr, ch


prA, chA = nested_vs(OPT, "choose arm + dose")
prB, chB = nested_vs([o for o in OPT if o[0] == "refit"], "refit everywhere, dose")
prC, chC = nested_vs([("ship", k) for k in KS], "ship, nested dose (ref)")
prD, chD = nested_vs([("ship", 1.0), ("refit", 1.0)], "choose arm, dose 1.0")
print("arm choice share (arm+dose rule): refit chosen in %.0f%% of airport-month cells; by airport (of 8 months):" % (100 * np.mean([c[0] == "refit" for c in chA.values()])))
for a in AIR:
    print(f"  {a}: " + " ".join(f"{chA[(a, h)][0][0]}{chA[(a, h)][1]}" for h in MONS) + "   all-8 SSE-best: " + str(min(OPT, key=lambda o: sum(SSE[(o[0], o[1], a, m)] for m in MONS))))
print("per month nested (arm + dose) minus shipped:", {m: round(cmp(prA, SH["avg3"], [m])[0], 3) for m in MONS})
print("per month nested (refit, dose) minus shipped:", {m: round(cmp(prB, SH["avg3"], [m])[0], 3) for m in MONS})
print("\n== 3. bar: pooled J+J gain >= 0.03 s, Jan and Jul each not worse, P >= 0.9 ==")
for tag, pr in (("refit, shipped doses", RF["avg3"]), ("choose arm + dose", prA), ("refit everywhere, dose", prB), ("choose arm, dose 1.0", prD)):
    gj, pj = cmp(pr, SH["avg3"], [1, 7]); gJ, _ = cmp(pr, SH["avg3"], [1]); gU, _ = cmp(pr, SH["avg3"], [7])
    print(f"  {tag:24s} J+J {gj:+.3f} P {pj:.2f}  Jan {gJ:+.3f} Jul {gU:+.3f}  PASS = {gj <= -0.03 and gJ <= 0 and gU <= 0 and pj >= 0.9}")
