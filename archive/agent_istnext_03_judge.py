"""istnext 03: judge the Istanbul designs as the Istanbul experiments did (copy of the agent_istarith_03_judge.py logic). No 2026 label.
Chain = served-style reference + 1.0 x (mean over seeds of the clipped corrector) on non-snow rows; snow rows stay at the reference (change asserted to be exactly 0).
Gain = RMSE(other arm) - RMSE(design) on Istanbul non-snow rows (positive = the design helps); total-equivalent = x0.142. Day bootstrap 2,000 draws.
GATE (fixed, docs/LAB-NOTES.md): pooled >= 0.352 s vs ctrl; months 1-4 and 7-10 both not worse; P >= 0.9; better than the shuffled control; positive with the best 50 rows per month removed; temporal not worse.
usage: python archive/agent_istnext_03_judge.py D1,D3[,D2] [seeds]"""
import os
import sys
import numpy as np

I = "data/interim/"; OUT = os.environ.get("ARITH_OUT", I + "agent_istnext_out/")
DES = sys.argv[1].split(","); SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2").split(",")]; CONV = 0.142; BAR = 0.05 / CONV
A = np.load(OUT + "aux.npz", allow_pickle=True); y, ref, mon, snow, dom = A["y"], A["ref"], A["mon"].astype(int), A["snow"], A["dom"]
N = len(y); DAY = dom + 100 * mon; NS = ~snow; E0 = (y - ref) ** 2
LOMO = [1, 2, 3, 4, 7, 8, 9, 10]


def load(arm, frame):
    """seed-mean corrector over all rows (NaN where no fit), diagnostics (mean over fits), per-seed arrays"""
    per = []; dg = []; ns = []
    for s in SEEDS:
        c = np.full(N, np.nan)
        for b in ([("m%d" % h) for h in LOMO] if frame == "lomo" else ["ta", "tb"]):
            p = OUT + "%s_s%d_%s.npz" % (arm, s, b)
            if not os.path.exists(p): return None
            z = np.load(p); c[z["rows"]] = z["pred"]; dg.append(z["diag"]); ns.append(float(z["newshare"]))
        per.append(c)
    return np.mean(per, 0), np.mean(dg, 0), float(np.mean(ns)), per


def err(c): return (y - (ref + np.where(snow, 0.0, c))) ** 2
def rm(e, s): return float(np.sqrt(e[s].mean()))


def boot(sel, ea, eb, nb=2000):
    d = DAY[sel]; u, inv = np.unique(d, return_inverse=True); n = np.bincount(inv).astype(float)
    sa = np.bincount(inv, weights=ea[sel]); sb = np.bincount(inv, weights=eb[sel]); out = np.empty(nb); r = np.random.default_rng(0)
    for i in range(nb):
        k = r.integers(0, len(u), len(u)); out[i] = np.sqrt(sb[k].sum() / n[k].sum()) - np.sqrt(sa[k].sum() / n[k].sum())
    return out


def drop50(d, mm):
    keep = np.ones(len(d), bool)
    for m in sorted(set(mm.tolist())):
        ii = np.where(mm == m)[0]; keep[ii[np.argsort(-d[ii])[:50]]] = False
    return keep


RES = {}
for frame in ("lomo", "temporal"):
    L = {a: load(a, frame) for a in ["ctrl"] + DES + [d + "s" for d in DES]}
    if L["ctrl"] is None:
        print("frame", frame, ": control fits missing"); continue
    cc = L["ctrl"][0]; ok = np.isfinite(cc); sel = ok & NS
    print(f"\n================ frame {frame}: rows judged {int(ok.sum())}, non-snow {int(sel.sum())}, seeds {SEEDS} ================")
    # reproduction of the stored control fits of the istarith experiment
    old = I + "agent_istarith_02_%s.npz" % frame
    if os.path.exists(old):
        O = np.load(old); mx = 0.0
        for s in SEEDS:
            for b in [k[:-4] for k in O.files if k.endswith("_idx")]:
                k = f"{b}_ctrl_s{s}"
                if k in O.files: mx = max(mx, float(np.abs(O[k] - L["ctrl"][3][SEEDS.index(s)][O[b + "_idx"]]).max()))
        print("control reproduces the stored istarith control fits: max abs difference %.6f s" % mx)
    Ec = err(cc)
    print("control: ref %.3f -> ctrl %.3f (non-snow); fit-row RMSE %.2f -> %.2f, held %.2f -> %.2f (mean over fits; corrector level, non-snow)" % (rm(E0, sel), rm(Ec, sel), *L["ctrl"][1]))
    for d in DES:
        if L[d] is None or L[d + "s"] is None:
            print("design", d, ": fits missing in frame", frame); continue
        cd, cs = L[d][0], L[d + "s"][0]; Ed, Es = err(cd), err(cs)
        assert np.array_equal(Ed[ok & snow], E0[ok & snow]) and np.array_equal(Es[ok & snow], E0[ok & snow]), "snow rows changed"
        print(f"\n---- design {d} ({frame}) ---- snow rows changed by exactly 0 (asserted, n {int((ok & snow).sum())}); corrector move vs ctrl rms {np.sqrt(((cd - cc)[sel] ** 2).mean()):.2f} s, corr {np.corrcoef(cd[sel], cc[sel])[0, 1]:.4f}")
        print("  capacity: fit-row RMSE ctrl %.2f / %s %.2f / shuffled %.2f; held-out ctrl %.2f / %s %.2f / shuffled %.2f; gain share of the new columns %.3f (shuffled %.3f)" % (
            L["ctrl"][1][1], d, L[d][1][1], L[d + "s"][1][1], L["ctrl"][1][3], d, L[d][1][3], L[d + "s"][1][3], L[d][2], L[d + "s"][2]))
        G = {"all": sel, "Jan-like 1-4": sel & (mon <= 4), "Jul-like 7-10": sel & (mon >= 7), "Jan": sel & (mon == 1), "Jul": sel & (mon == 7), "month 10": sel & (mon == 10),
             "27 Oct only": sel & (mon == 10) & (dom == 27), "all without 27 Oct": sel & ~((mon == 10) & (dom == 27)), "all without February": sel & (mon != 2), "all without Feb and 27 Oct": sel & (mon != 2) & ~((mon == 10) & (dom == 27)), "Jul without 5-26 Jul": sel & (mon == 7) & ~((dom >= 5) & (dom <= 26))}
        out = {}
        for gname, g in G.items():
            if not g.any(): continue
            gc_, gs_ = rm(Ec, g) - rm(Ed, g), rm(Es, g) - rm(Ed, g); pl_ = rm(Ec, g) - rm(Es, g)
            bs = boot(g, Ed, Ec); bs2 = boot(g, Ed, Es)
            out[gname] = (gc_, gs_, float((bs > 0).mean()), float((bs2 > 0).mean()))
            print(f"  {gname:22s} n {int(g.sum()):6d} vs ctrl {gc_:+.3f} (total {gc_ * CONV:+.3f}) P {(bs > 0).mean():.2f} 90% [{np.percentile(bs, 5):+.2f},{np.percentile(bs, 95):+.2f}] | vs shuffled {gs_:+.3f} P {(bs2 > 0).mean():.2f} | placebo (shuffled vs ctrl) {pl_:+.3f}")
        print("  by month vs ctrl / vs shuffled: " + "  ".join(f"{m}: {rm(Ec, sel & (mon == m)) - rm(Ed, sel & (mon == m)):+.2f}/{rm(Es, sel & (mon == m)) - rm(Ed, sel & (mon == m)):+.2f}" for m in LOMO))
        print("  per seed pooled vs ctrl: " + "  ".join(f"s{s}: {rm(err(L['ctrl'][3][i]), sel) - rm(err(L[d][3][i]), sel):+.3f}" for i, s in enumerate(SEEDS)))
        dd = (Ec - Ed)[sel]; keep = drop50(dd, mon[sel]); b50 = float(np.sqrt(Ec[sel][keep].mean()) - np.sqrt(Ed[sel][keep].mean()))
        dd2 = (Es - Ed)[sel]; keep2 = drop50(dd2, mon[sel]); b50s = float(np.sqrt(Es[sel][keep2].mean()) - np.sqrt(Ed[sel][keep2].mean()))
        dp = (Ec - Es)[sel]; keep3 = drop50(dp, mon[sel]); b50p = float(np.sqrt(Ec[sel][keep3].mean()) - np.sqrt(Es[sel][keep3].mean()))
        tot = dd.sum(); yy = y[sel]
        print(f"  best 50 rows per month removed: vs ctrl {b50:+.3f}, vs shuffled {b50s:+.3f} (placebo under the same removal {b50p:+.3f}); SSE gain {tot:.3e}, top-20 rows hold {np.sort(dd)[::-1][:20].sum() / tot:+.2f}, label >= 1800 hold {dd[yy >= 1800].sum() / tot:+.2f}")
        RES[(d, frame)] = (out, b50)
print("\n================ GATE (bar %.3f s Istanbul non-snow = 0.05 s total) ================" % BAR)
for d in DES:
    if (d, "lomo") not in RES: continue
    o, b50 = RES[(d, "lomo")]; t = RES.get((d, "temporal"))
    c1 = o["all"][0] >= BAR; c2 = o["Jan-like 1-4"][0] >= 0 and o["Jul-like 7-10"][0] >= 0; c3 = o["all"][2] >= 0.9; c4 = o["all"][1] > 0; c5 = b50 > 0; c6 = (t is not None) and t[0]["all"][0] >= 0
    print(f"{d}: (1) pooled {o['all'][0]:+.3f} [{'PASS' if c1 else 'FAIL'}]  (2) Jan-like {o['Jan-like 1-4'][0]:+.3f} / Jul-like {o['Jul-like 7-10'][0]:+.3f} (Jan {o['Jan'][0]:+.3f}, Jul {o['Jul'][0]:+.3f}) [{'PASS' if c2 else 'FAIL'}]  (3) P {o['all'][2]:.2f} [{'PASS' if c3 else 'FAIL'}]  "
          f"(4) vs shuffled {o['all'][1]:+.3f} [{'PASS' if c4 else 'FAIL'}]  (5) best-50 removed {b50:+.3f} [{'PASS' if c5 else 'FAIL'}]  (6) temporal {('%+.3f' % t[0]['all'][0]) if t else 'n/a'} [{'PASS' if c6 else 'FAIL'}]  ==> {'PASSES' if all([c1, c2, c3, c4, c5, c6]) else 'DOES NOT PASS'}")
