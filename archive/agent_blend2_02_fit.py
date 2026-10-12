"""experiment blend2 02: leak-free per-airport NNLS blend over the v114-analogue control (docs/LAB-NOTES.md).

Scheme = archive/agent_blend_02_fit.py (day-grouped 10-fold OOF, fold-seeds A/B, per-airport NNLS on corrections relative
to the control, no sum-to-one), run for member-seed replicates r = 0, 1, 2 with CTRL_r = RS_E9S_s{r} + Dnbr. Verdict =
4 clauses at all 6 (r, fold-seed) + the CV stability gate on (0, A); per-airport fallback once if WHOLE fails.
Writes data/interim/agent_blend2_verdict.json, agent_blend2_weights_final.json, agent_blend2_ctrl_r0.npz,
agent_blend2_oof_r0_A.npz (lens fold-treat) and agent_blend2_foldfinal_r0.npz (final weights on fold, guard (a)).
"""
import json

import numpy as np
from scipy.optimize import nnls

I = "data/interim/"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
M = ["EDDF", "EDDM", "EHAM", "LEBL", "LSZH"]
N_FOLDS, CV_GATE = 10, 0.75
SEEDED = {"E9": "agent_stagecap_RS_E9_s{s}.npz", "E9f": "agent_stagecap_RS_E9f_s{s}.npz", "T": "agent_perapt_T_s{s}.npz"}
FIXED = {"V9TOP1": "agent_restack5_V9TOP1.npz", "S1": "agent_stagecap_RS_S1_s0.npz", "E3": "agent_stagecap_RS_E3_s0.npz",
         "nbrB": "agent_neighbour_B.npz", "RS_v2": "modal/RS_v2.npz", "CAB": "agent_restack3_CAB.npz",
         "ctrl4": "agent_restack4_ctrl.npz", "v109a": "agent_blend_v109analogue.npz"}
OLD = {"nbrB", "RS_v2", "CAB", "ctrl4", "v109a"}
MEMBERS = ["V9TOP1", "E9", "E9f", "S1", "E3", "T", "T9", "nbrB", "RS_v2", "CAB", "ctrl4", "v109a"]
SERVE_AT = {"V9TOP1": set(M), "T9": set(M), "E9f": {"EGLL", "LEMD", "LFPG", "LIRF"}}  # s0.2 table; others: all 9


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys}


c0 = load(I + "agent_stagecap_RS_E9S_s0.npz")
ids, y, days, apts = c0["ids"], c0["y"].astype(float), c0["days"].astype(str), c0["apts"].astype(str)
N = len(y)
mon = np.array([d[5:7] for d in days])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])
mk = np.isin(apts, M)


def P(path):
    z = load(path, ("ids", "both_guard"))
    assert np.array_equal(z["ids"], ids), f"{path}: id mismatch"
    return z["both_guard"].astype(float)


rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean())) if m.any() else float("nan")  # noqa: E731


def clauses(pt, base_p, base_name):  # verbatim from archive/agent_blend_02_fit.py
    d = lambda m: rm(pt, m) - rm(base_p, m)  # noqa: E731
    gain = (base_p - y) ** 2 - (pt - y) ** 2
    keep = np.ones(N, bool)
    keep[np.argsort(-gain)[:50]] = False
    ch = pt != base_p
    h = [d(m) if (ch & m).any() else 0.0 for m in (H1, H2)]
    per = {a: d(apts == a) for a in AIRPORTS}
    c = {"vs": base_name, "all": d(np.ones(N, bool)), "worst_apt": max(per.values()), "m50": d(keep),
         "h1": h[0], "h2": h[1], "per": per, "rows_changed": int(ch.sum())}
    c["pass"] = c["all"] <= -0.10 and c["worst_apt"] <= 0.10 and c["m50"] < 0 and c["h1"] < 0 and c["h2"] < 0
    return c


def show(name, c):
    print(f"{name}: all {c['all']:+.4f}  worst {c['worst_apt']:+.3f}  m50 {c['m50']:+.3f}  Jan-Mar {c['h1']:+.3f}  "
          f"Jul-Sep {c['h2']:+.3f}  rows changed {c['rows_changed']:,}  4-clause {'PASS' if c['pass'] else 'FAIL'}", flush=True)
    print("    per airport: " + "  ".join(f"{a} {v:+.3f}" for a, v in c["per"].items()), flush=True)


dnbr = P(I + "agent_blend_oof_seedA.npz") - P(I + "agent_blend_v109analogue.npz")
assert (dnbr[~np.isin(apts, AIRPORTS)] == 0).all()
fixed = {m: P(I + f) for m, f in FIXED.items()}


def replicate(r):
    e9s = P(I + f"agent_stagecap_RS_E9S_s{r}.npz")
    ctrl = e9s + dnbr
    mem = dict(fixed)
    for m, f in SEEDED.items():
        mem[m] = P(I + f.format(s=r))
    mem["T9"] = np.where(mk, P(I + f"agent_perapt_bx_apt_E9f_s{r}.npz"), e9s)
    X = {m: (mem[m] - ctrl) if m in OLD else (mem[m] - e9s) for m in MEMBERS}
    elig = {}
    for a in AIRPORTS:
        k = apts == a
        elig[a] = [m for m in MEMBERS if a in SERVE_AT.get(m, set(AIRPORTS)) and (X[m][k] != 0).any()]
        for m in MEMBERS:  # a member not offered at A must not move A
            if m not in elig[a]:
                X[m][k] = 0.0
    return ctrl, X, elig


udays = np.unique(days)
fmaps = {"A": {d: i % N_FOLDS for i, d in enumerate(np.sort(udays))},
         "B": {d: i % N_FOLDS for i, d in enumerate(np.random.default_rng(20260925).permutation(np.sort(udays)))}}
fold_of = {s: np.array([fm[d] for d in days]) for s, fm in fmaps.items()}


def oof(ctrl, X, elig, seed, wlog=None):
    fold = fold_of[seed]
    out, best = ctrl.copy(), ctrl.copy()
    res = y - ctrl
    for a in AIRPORTS:
        E = elig[a]
        ma = apts == a
        if not E:
            continue
        for k in range(N_FOLDS):
            tr, te = ma & (fold != k), ma & (fold == k)
            if not te.any():
                continue
            Xtr = np.column_stack([X[m][tr] for m in E])
            w, _ = nnls(Xtr, res[tr])
            out[te] = ctrl[te] + np.column_stack([X[m][te] for m in E]) @ w
            if wlog is not None:
                wlog.setdefault(a, []).append(dict(zip(E, w.tolist())))
            # context only: best single member (or the control) by TRAIN rmse at A
            cand = [(float(np.mean(res[tr] ** 2)), None)] + [(float(np.mean((res[tr] - X[m][tr]) ** 2)), m) for m in E]
            bm = min(cand, key=lambda t: t[0])[1]
            best[te] = ctrl[te] + (X[bm][te] if bm else 0.0)
    return out, best


res_all = {"members": MEMBERS, "replicates": {}}
keep = {}
all_pass = True
for r in (0, 1, 2):
    ctrl, X, elig = replicate(r)
    if r == 0:
        print("eligibility (r=0): " + "; ".join(f"{a}: {','.join(elig[a])}" for a in AIRPORTS), flush=True)
    for seed in ("A", "B"):
        wlog = {} if (r == 0 and seed == "A") else None
        o, b = oof(ctrl, X, elig, seed, wlog)
        c = clauses(o, ctrl, f"CTRL_{r}")
        cb = clauses(b, ctrl, f"CTRL_{r}")
        show(f"WHOLE r={r} seed {seed}", c)
        print(f"    (context) best-single-member OOF: all {cb['all']:+.4f} worst {cb['worst_apt']:+.3f} m50 {cb['m50']:+.3f} "
              f"4-clause {'PASS' if cb['pass'] else 'FAIL'}", flush=True)
        res_all["replicates"][f"r{r}_{seed}"] = {"nnls": c, "best_single": cb}
        all_pass = all_pass and c["pass"]
        if r == 0 and seed == "A":
            keep = {"ctrl": ctrl, "X": X, "elig": elig, "oof": o, "wlog": wlog}
        del o, b

# stability gate on (0, A)
stab, worst_cv, worst_at = {}, 0.0, None
for a in AIRPORTS:
    E = keep["elig"][a]
    if not E:
        continue
    W = np.array([[fw.get(m, 0.0) for m in E] for fw in keep["wlog"][a]])
    mean, std = W.mean(0), W.std(0)
    cv = np.where(mean > 1e-9, std / np.where(mean > 1e-9, mean, 1.0), 0.0)
    dom = int(np.argmax(mean))
    stab[a] = {"members": E, "mean": mean.tolist(), "cv": cv.tolist(), "dominant": E[dom], "dom_mean": float(mean[dom]),
               "dom_cv": float(cv[dom])}
    print(f"  stability {a}: dominant {E[dom]} mean {mean[dom]:.3f} cv {cv[dom]:.3f}  | "
          + ", ".join(f"{m}={mv:.3f}" for m, mv in zip(E, mean) if mv > 0.0005), flush=True)
    if mean[dom] > 0.1 and cv[dom] > worst_cv:
        worst_cv, worst_at = float(cv[dom]), (a, E[dom])
stab_ok = worst_cv <= CV_GATE
print(f"STABILITY GATE: worst dominant CV {worst_cv:.3f} at {worst_at} -> {'PASS' if stab_ok else 'FAIL'}", flush=True)
res_all.update(stability=stab, stability_pass=bool(stab_ok), worst_cv=worst_cv)
whole_pass = bool(all_pass and stab_ok)
print(f"WHOLE VERDICT: {'PASS' if whole_pass else 'FAIL'} (6 clause checks pass={all_pass}, stability={stab_ok})", flush=True)

ctrl0, o0 = keep["ctrl"], keep["oof"]
IN = list(AIRPORTS)
verdict = "PASS WHOLE" if whole_pass else None
if not whole_pass:
    # per-airport fallback, ONCE, on (r=0, seed A)
    IN = []
    for a in AIRPORTS:
        k = apts == a
        d = rm(o0, k) - rm(ctrl0, k)
        dh1, dh2 = rm(o0, k & H1) - rm(ctrl0, k & H1), rm(o0, k & H2) - rm(ctrl0, k & H2)
        g = (ctrl0 - y) ** 2 - (o0 - y) ** 2
        g = np.where(k, g, -np.inf)
        kk = k.copy()
        kk[np.argsort(-g)[:10]] = False
        dt = rm(o0, kk) - rm(ctrl0, kk)
        ok = d <= -0.10 and dh1 < 0 and dh2 < 0 and dt < 0
        print(f"  IN test {a}: d {d:+.3f} Jan-Mar {dh1:+.3f} Jul-Sep {dh2:+.3f} minus-own-top10 {dt:+.3f} -> {'IN' if ok else 'out'}", flush=True)
        if ok:
            IN.append(a)
    res_all["fallback_IN"] = IN
    print(f"FALLBACK IN = {IN}", flush=True)
    if not IN or len(IN) == len(AIRPORTS):
        verdict = "FAIL"
    else:
        fb_ok = True
        for r in (0, 1, 2):
            ctrl, X, elig = replicate(r)
            elig = {a: (elig[a] if a in IN else []) for a in AIRPORTS}
            for seed in ("A", "B"):
                o, _ = oof(ctrl, X, elig, seed)
                c = clauses(o, ctrl, f"CTRL_{r}")
                show(f"FALLBACK r={r} seed {seed}", c)
                res_all["replicates"][f"fb_r{r}_{seed}"] = c
                fb_ok = fb_ok and c["pass"]
                if r == 0 and seed == "A":
                    o0 = o
        fb_stab = all(not (stab[a]["dom_mean"] > 0.1 and stab[a]["dom_cv"] > CV_GATE) for a in IN if a in stab)
        print(f"FALLBACK stability at IN: {'PASS' if fb_stab else 'FAIL'}", flush=True)
        verdict = f"PASS FALLBACK {','.join(IN)}" if (fb_ok and fb_stab) else "FAIL"
res_all["verdict"] = verdict
print(f"BLEND2_VERDICT {verdict}", flush=True)

# final all-data weights (replicate 0; recorded whatever the verdict), at IN only
X, elig = keep["X"], keep["elig"]
fw, foldfinal = {}, ctrl0.copy()
res0 = y - ctrl0
for a in AIRPORTS:
    E = elig[a] if a in IN else []
    k = apts == a
    if not E:
        fw[a] = {}
        continue
    w, _ = nnls(np.column_stack([X[m][k] for m in E]), res0[k])
    fw[a] = dict(zip(E, w.tolist()))
    foldfinal[k] = ctrl0[k] + np.column_stack([X[m][k] for m in E]) @ w
    print(f"FINAL weights {a}: " + ", ".join(f"{m}={v:.4f}" for m, v in fw[a].items() if v > 0) , flush=True)
cf = clauses(foldfinal, ctrl0, "CTRL_0")
show("(in-sample, final weights on fold, r=0 -- NOT a verdict)", cf)
dpf = foldfinal - ctrl0
print(f"fold footprint (final weights): sum dp^2/row {float((dpf ** 2).mean()):.2f}; by airport "
      + ", ".join(f"{a} {float((dpf[apts == a] ** 2).mean()):.1f}" for a in AIRPORTS), flush=True)
res_all["fold_dp2_final"] = float((dpf ** 2).mean())
res_all["fold_dp2_final_apt"] = {a: float((dpf[apts == a] ** 2).mean()) for a in AIRPORTS}
res_all["fold_rows_final_apt"] = {a: int((apts == a).sum()) for a in AIRPORTS}
json.dump({"weights": fw, "IN": IN, "verdict": verdict}, open(I + "agent_blend2_weights_final.json", "w"), indent=1)
json.dump(res_all, open(I + "agent_blend2_verdict.json", "w"), indent=1, default=float)
sv = lambda f, p: np.savez(I + f, ids=ids, y=y, days=days, apts=apts, both_guard=p)  # noqa: E731
sv("agent_blend2_ctrl_r0.npz", ctrl0)
sv("agent_blend2_oof_r0_A.npz", o0)
sv("agent_blend2_foldfinal_r0.npz", foldfinal)
print("wrote agent_blend2_{ctrl_r0,oof_r0_A,foldfinal_r0}.npz, agent_blend2_{verdict,weights_final}.json\nDONE", flush=True)
