"""istarith 03: judge the Istanbul corrector arms (ctrl, wins, shuf, rand) through the served-style chain (experiment "istarith", 2026-10-05). No labels from 2026.
Chain = served-style reference + dose x (mean over seeds of the clipped corrector value) on non-snow rows; snow rows stay at the reference (as served).
Dose 1.0 = as shipped since v142 (0.75 also printed). Gain = RMSE(other arm) - RMSE(wins) on Istanbul rows, seconds (positive = the columns help).
Total-RMSE conversion x0.142 (13.8% of rows, Istanbul RMSE about 225, total 218.7). Day bootstrap (days resampled, 2000 draws, fixed seed).
Groups: all 8 months, January-like 1-4, July-like 7-10, January, July; with / without the 5-26 July 2025 anomaly weeks.
usage: python agent_istwins_03_judge.py lomo|temporal"""
import sys
import numpy as np

frame = sys.argv[1]; I = "data/interim/"
Z = np.load(I + "agent_istarith_02_%s.npz" % frame)
y, ref, mon, snow, dom = Z["aux_y"], Z["aux_ref"], Z["aux_mon"], Z["aux_snow"], Z["aux_dom"]
ARMS = ["ctrl", "wins", "unp", "shuf", "rand"]; SEEDS = [0, 1, 2]; CONV = 0.142
blocks = sorted({k.rsplit("_idx", 1)[0] for k in Z.files if k.endswith("_idx")})
idx = np.concatenate([Z[b + "_idx"] for b in blocks])
corr = {arm: np.concatenate([np.mean([Z[f"{b}_{arm}_s{s}"] for s in SEEDS], 0) for b in blocks]) for arm in ARMS}
cseed = {(arm, s): np.concatenate([Z[f"{b}_{arm}_s{s}"] for b in blocks]) for arm in ARMS for s in SEEDS}
Y, R, M, SN, DOM = y[idx], ref[idx], mon[idx].astype(int), snow[idx], dom[idx]
DAY = DOM + 100 * M
anom = (M == 7) & (DOM >= 5) & (DOM <= 26)
print("frame", frame, "blocks", blocks, "rows judged", len(idx), "months", sorted(set(M.tolist())), "snow rows", int(SN.sum()), "anomaly-week rows", int(anom.sum()))


def err(c, dose=1.0):
    return (Y - (R + np.where(SN, 0.0, dose * c))) ** 2


E = {arm: err(corr[arm]) for arm in ARMS}
E0 = (Y - R) ** 2


def boot(sel, ea, eb, nb=2000):
    d = DAY[sel]; u, inv = np.unique(d, return_inverse=True); n = np.bincount(inv).astype(float)
    sa = np.bincount(inv, weights=ea[sel]); sb = np.bincount(inv, weights=eb[sel]); out = np.empty(nb)
    r = np.random.default_rng(0)
    for i in range(nb):
        k = r.integers(0, len(u), len(u)); out[i] = np.sqrt(sb[k].sum() / n[k].sum()) - np.sqrt(sa[k].sum() / n[k].sum())   # eb - ea: positive = a better
    return out


def rm(e, sel):
    return float(np.sqrt(e[sel].mean()))


NS = ~SN
groups = {"all8": np.isin(M, [1, 2, 3, 4, 7, 8, 9, 10]), "Jan-like 1-4": np.isin(M, [1, 2, 3, 4]), "July-like 7-10": np.isin(M, [7, 8, 9, 10]), "Jan": M == 1, "Jul": M == 7}


def line(name, sel, a, b, ea=None, eb=None, with_boot=True):
    ea = E[a] if ea is None else ea; eb = E[b] if eb is None else eb
    g = rm(eb, sel) - rm(ea, sel)
    if with_boot:
        bs = boot(sel, ea, eb)
        return f"{name:30s} n {int(sel.sum()):6d} {b}-{a}: {g:+.3f} s  (total {g*CONV:+.3f})  P {float((bs>0).mean()):.2f}  90% [{np.percentile(bs,5):+.2f},{np.percentile(bs,95):+.2f}]"
    return f"{name:30s} n {int(sel.sum()):6d} {b}-{a}: {g:+.3f} s"


print("\n== per month (non-snow rows, chain dose 1.0): RMSE ref / ctrl / wins / shuf / rand; gain of wins over each (positive = wins better); placebo = shuf or rand over ctrl ==")
for m in sorted(set(M.tolist())):
    s = (M == m) & NS
    print(f"month {m:2d} n {int(s.sum()):6d} ref {rm(E0,s):7.2f} ctrl {rm(E['ctrl'],s):7.2f} wins {rm(E['wins'],s):7.2f} unp {rm(E['unp'],s):7.2f} shuf {rm(E['shuf'],s):7.2f} rand {rm(E['rand'],s):7.2f} | gain vs ctrl {rm(E['ctrl'],s)-rm(E['wins'],s):+.3f} vs shuf {rm(E['shuf'],s)-rm(E['wins'],s):+.3f} vs rand {rm(E['rand'],s)-rm(E['wins'],s):+.3f} | placebo shuf {rm(E['ctrl'],s)-rm(E['shuf'],s):+.3f} rand {rm(E['ctrl'],s)-rm(E['rand'],s):+.3f}")
print("\n== groups, non-snow rows, dose 1.0, WITH the anomaly weeks ==")
for gname, gsel in groups.items():
    for b in ("ctrl", "shuf", "rand", "unp"):
        print(line(gname, gsel & NS, "wins", b))
    print(line(gname + " (placebo)", gsel & NS, "shuf", "ctrl"))
print("\n== same, WITHOUT the anomaly weeks (5-26 July 2025 rows removed) ==")
for gname, gsel in groups.items():
    if gname in ("Jan", "Jan-like 1-4"):
        continue
    for b in ("ctrl", "shuf", "rand"):
        print(line(gname + " noanom", gsel & NS & ~anom, "wins", b))
print("\n== dose 0.75 (the dose before v142), vs ctrl and shuf ==")
E75 = {arm: err(corr[arm], 0.75) for arm in ARMS}
for gname in ("all8", "Jan", "Jul"):
    for b in ("ctrl", "shuf"):
        print(line(gname + " d.75", groups[gname] & NS, "wins", b, E75["wins"], E75[b], with_boot=False))
print("\n== per seed (single-seed corrector, dose 1.0), non-snow ==")
for s in SEEDS:
    ee = {arm: err(cseed[(arm, s)]) for arm in ARMS}; sel = groups["all8"] & NS; j1 = (M == 1) & NS; j7 = (M == 7) & NS
    print(f"seed {s}: all8 vs ctrl {rm(ee['ctrl'],sel)-rm(ee['wins'],sel):+.3f} vs shuf {rm(ee['shuf'],sel)-rm(ee['wins'],sel):+.3f} vs rand {rm(ee['rand'],sel)-rm(ee['wins'],sel):+.3f} | Jan vs ctrl {rm(ee['ctrl'],j1)-rm(ee['wins'],j1):+.3f} Jul vs ctrl {rm(ee['ctrl'],j7)-rm(ee['wins'],j7):+.3f}")
print("\n== snow rows: the served chain leaves them at the reference (change exactly 0). Raw corrector applied on them, information only ==")
if SN.any():
    print(f"snow rows n {int(SN.sum())}: ref RMSE {np.sqrt(E0[SN].mean()):.1f}; with corrector ctrl {np.sqrt(((Y-(R+corr['ctrl']))**2)[SN].mean()):.1f}, wins {np.sqrt(((Y-(R+corr['wins']))**2)[SN].mean()):.1f}, shuf {np.sqrt(((Y-(R+corr['shuf']))**2)[SN].mean()):.1f}; rms(wins-ctrl corrector) {np.sqrt(((corr['wins']-corr['ctrl'])[SN]**2).mean()):.1f} s")
print("\n== where the gain sits (all8 non-snow, dose 1.0): share of SSE gain by label (body < 1500 s, tail >= 1800 s, 20 best rows); best 50 rows of each month removed ==")
sel = groups["all8"] & NS; mm = M[sel]
def drop50(d):
    keep = np.ones(len(d), bool)
    for m in sorted(set(mm.tolist())):
        ii = np.where(mm == m)[0]; keep[ii[np.argsort(-d[ii])[:50]]] = False
    return keep
for b in ("ctrl", "shuf", "rand", "unp"):
    d = (E[b] - E["wins"])[sel]; yy = Y[sel]; tot = d.sum()
    body = d[yy < 1500].sum() / tot; tail = d[yy >= 1800].sum() / tot; top20 = np.sort(d)[::-1][:20].sum() / tot
    keep = drop50(d)
    print(f"wins vs {b}: SSE gain {tot:.3e}; body {body:+.2f} mid {1-body-tail:+.2f} tail {tail:+.2f} top20 {top20:+.2f}; best-50-per-month removed: RMSE gain {np.sqrt(E[b][sel][keep].mean())-np.sqrt(E['wins'][sel][keep].mean()):+.3f} s")
d = (E["ctrl"] - E["shuf"])[sel]; keep = drop50(d)
print(f"placebo shuf vs ctrl under the same removal: RMSE gain {np.sqrt(E['ctrl'][sel][keep].mean())-np.sqrt(E['shuf'][sel][keep].mean()):+.3f} s")
print("corrector change (wins - ctrl), all rows: rms %.1f s, corr %.3f" % (float(np.sqrt(((corr['wins'] - corr['ctrl']) ** 2).mean())), float(np.corrcoef(corr['wins'], corr['ctrl'])[0, 1])))
if frame == "lomo":
    sel = NS & groups["all8"]
    g_ctrl = rm(E["ctrl"], sel) - rm(E["wins"], sel); g_sh = rm(E["shuf"], sel) - rm(E["wins"], sel); g_rn = rm(E["rand"], sel) - rm(E["wins"], sel)
    jan = (M == 1) & NS; jul = (M == 7) & NS
    P_ = float((boot(sel, E["wins"], E["ctrl"]) > 0).mean()); Pj = [float((boot(sel, E["wins"], E[b]) > 0).mean()) for b in ("shuf", "rand")]
    jj = [rm(E[b], jan) - rm(E["wins"], jan) for b in ("ctrl", "shuf", "rand")]; jl = [rm(E[b], jul) - rm(E["wins"], jul) for b in ("ctrl", "shuf", "rand")]
    print("\nGATE (>= 0.5 s Istanbul pooled over held months vs ctrl; Jan and Jul not worse; P >= 0.9; better than both controls):")
    print(f"  pooled vs ctrl {g_ctrl:+.3f} (P {P_:.2f}); vs shuf {g_sh:+.3f} (P {Pj[0]:.2f}); vs rand {g_rn:+.3f} (P {Pj[1]:.2f}); Jan vs ctrl/shuf/rand {jj[0]:+.3f}/{jj[1]:+.3f}/{jj[2]:+.3f}; Jul {jl[0]:+.3f}/{jl[1]:+.3f}/{jl[2]:+.3f}")
    ok = g_ctrl >= 0.5 and min(jj + jl) >= 0 and P_ >= 0.9 and min(Pj) >= 0.9 and g_sh > 0 and g_rn > 0
    print("  GATE", "PASSED" if ok else "NOT PASSED")
