"""experiment blend 02: day-grouped OOF fit of the per-airport NNLS blend (docs/LAB-NOTES.md).

Loads the five servable members + the v109-analogue base (s0.3), builds two independent day-fold assignments (seed A =
sorted round-robin, seed B = shuffled round-robin, s0.4), and for each (seed, fold, airport) fits
    w = argmin_{w>=0} || X_train w - r_train ||^2 ,   X_train[:, i] = member_i(train) - base(train), r_train = y(train) - base(train)
via scipy.optimize.nnls (s0.5), applies it out-of-fold, and pools the OOF predictions into one full-length vector per
seed. Judges the pooled OOF blend vs the v109 analogue with the seqsplice 4-clause code (s0.7), reports weight
stability (s0.7), and also reports the (b) "best single member per airport, OOF" variant for comparison only.
Finally refits the SAME NNLS on ALL rows (no held-out fold) per airport -- the weights the build step (03) will use.

Writes data/interim/agent_blend_verdict.json and data/interim/agent_blend_weights_final.json.
"""
import json

import numpy as np
from scipy.optimize import nnls

I = "data/interim/"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
MEMBERS = ["RS_v2", "CAB", "ctrl4", "nbrB", "BXnest_s0"]
MEMBER_FILE = {
    "RS_v2": I + "modal/RS_v2.npz",
    "CAB": I + "agent_restack3_CAB.npz",
    "ctrl4": I + "agent_restack4_ctrl.npz",
    "nbrB": I + "agent_neighbour_B.npz",
    "BXnest_s0": I + "agent_backx_BXnest_s0.npz",
}
EXCLUDE = {"BXnest_s0": {"LFPG", "LIRF"}}  # s0.6: no servable realization there
N_FOLDS = 10
CV_GATE = 0.75  # s0.7: dominant member's fold-to-fold CV must be <= this


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys}


base_d = load(I + "agent_blend_v109analogue.npz")
ids, y, days, apts = base_d["ids"], base_d["y"].astype(float), base_d["days"].astype(str), base_d["apts"].astype(str)
base = base_d["both_guard"].astype(float)
N = len(y)
mon = np.array([d[5:7] for d in days])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])

pred = {}
for m in MEMBERS:
    d = load(MEMBER_FILE[m])
    assert np.array_equal(d["ids"], ids), f"{m} id mismatch"
    pred[m] = d["both_guard"].astype(float)

# also TX_S6, smallB as context-only diagnostics (s0.2), never fit into the design matrix
ctx = {}
for m, f in (("TX_S6", I + "modal/TX_S6.npz"), ("smallB", I + "agent_smallbundle_B.npz")):
    d = load(f)
    assert np.array_equal(d["ids"], ids)
    ctx[m] = d["both_guard"].astype(float)


def eligible(a):
    return [m for m in MEMBERS if a not in EXCLUDE.get(m, set())]


def day_folds_seedA(udays):
    order = np.sort(udays)
    return {d: i % N_FOLDS for i, d in enumerate(order)}


def day_folds_seedB(udays, seed=20260925):
    order = np.sort(udays)
    perm = np.random.default_rng(seed).permutation(order)
    return {d: i % N_FOLDS for i, d in enumerate(perm)}


udays = np.unique(days)
fold_maps = {"A": day_folds_seedA(udays), "B": day_folds_seedB(udays)}
fold_of_row = {s: np.array([fm[d] for d in days]) for s, fm in fold_maps.items()}
for s, fm in fold_maps.items():
    sizes = np.bincount(list(fm.values()), minlength=N_FOLDS)
    print(f"seed {s}: {len(fm)} days over {N_FOLDS} folds, size range {sizes.min()}-{sizes.max()}")

rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean())) if m.any() else float("nan")  # noqa: E731


def clauses(pt, base_p, base_name):
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


results = {}
oof_by_seed = {}
weight_log = {s: {a: [] for a in AIRPORTS} for s in ("A", "B")}
for seed in ("A", "B"):
    fold = fold_of_row[seed]
    # LTFM (and anything else outside the 9-airport ADS-B dump scope) is not part of this blend; those rows keep the
    # v109-analogue base value untouched (weight 0, exactly like the build step's handling of unlisted airports).
    oof_nnls = base.copy()
    oof_best = base.copy()
    for A in AIRPORTS:
        elig = eligible(A)
        m_apt = apts == A
        for k in range(N_FOLDS):
            tr = m_apt & (fold != k)
            te = m_apt & (fold == k)
            if not te.any():
                continue
            Xtr = np.column_stack([pred[m][tr] - base[tr] for m in elig])
            rtr = y[tr] - base[tr]
            w, _ = nnls(Xtr, rtr)
            Xte = np.column_stack([pred[m][te] - base[te] for m in elig])
            oof_nnls[te] = base[te] + Xte @ w
            if seed == "A":
                weight_log["A"][A].append(dict(zip(elig, w.tolist())))
            # (b) best single member by TRAIN rmse
            best_m, best_r = None, np.inf
            for m in elig:
                r = rm(pred[m], tr)
                if r < best_r:
                    best_m, best_r = m, r
            oof_best[te] = pred[best_m][te]
    assert not np.isnan(oof_nnls).any() and not np.isnan(oof_best).any()
    oof_by_seed[seed] = oof_nnls
    c_nnls = clauses(oof_nnls, base, "v109 analogue")
    c_best = clauses(oof_best, base, "v109 analogue")
    print(f"== seed {seed}: NNLS blend OOF vs v109 analogue ==")
    print(f"  all {c_nnls['all']:+.4f}  worst {c_nnls['worst_apt']:+.3f}  m50 {c_nnls['m50']:+.3f}  "
          f"h1 {c_nnls['h1']:+.3f}  h2 {c_nnls['h2']:+.3f}  4-clause {'PASS' if c_nnls['pass'] else 'FAIL'}")
    print(f"  per airport: " + "  ".join(f"{a} {v:+.3f}" for a, v in c_nnls["per"].items()))
    print(f"== seed {seed}: (b) best-single-member-per-airport OOF vs v109 analogue (context only) ==")
    print(f"  all {c_best['all']:+.4f}  worst {c_best['worst_apt']:+.3f}  m50 {c_best['m50']:+.3f}  "
          f"h1 {c_best['h1']:+.3f}  h2 {c_best['h2']:+.3f}  4-clause {'PASS' if c_best['pass'] else 'FAIL'}")
    results[seed] = {"nnls": c_nnls, "best_single": c_best}

# weight stability (seed A only, per s0.7)
stability = {}
worst_cv, worst_apt_member = 0.0, None
for A in AIRPORTS:
    elig = eligible(A)
    W = np.array([[fw.get(m, 0.0) for m in elig] for fw in weight_log["A"][A]])  # (10, |elig|)
    mean, std = W.mean(axis=0), W.std(axis=0)
    cv = np.where(mean > 1e-9, std / np.where(mean > 1e-9, mean, 1.0), 0.0)
    dom = int(np.argmax(mean))
    stability[A] = {"members": elig, "mean": mean.tolist(), "std": std.tolist(), "cv": cv.tolist(),
                    "dominant_member": elig[dom], "dominant_mean": float(mean[dom]), "dominant_cv": float(cv[dom])}
    print(f"  stability {A}: dominant={elig[dom]} mean={mean[dom]:.3f} cv={cv[dom]:.3f}  (all means: "
          + ", ".join(f"{m}={mv:.3f}" for m, mv in zip(elig, mean)) + ")")
    if mean[dom] > 0.1 and cv[dom] > worst_cv:
        worst_cv, worst_apt_member = float(cv[dom]), (A, elig[dom])

stability_gate_pass = bool(worst_cv <= CV_GATE)
print(f"STABILITY GATE: worst dominant-member CV = {worst_cv:.3f} at {worst_apt_member} "
      f"({'PASS' if stability_gate_pass else 'FAIL'}, threshold {CV_GATE})")

overall_pass = bool(results["A"]["nnls"]["pass"] and results["B"]["nnls"]["pass"] and stability_gate_pass)
print(f"\nOVERALL PRE-REGISTERED VERDICT: {'PASS' if overall_pass else 'FAIL'} "
      f"(seedA pass={results['A']['nnls']['pass']}, seedB pass={results['B']['nnls']['pass']}, stability={stability_gate_pass})")

EXCLUDE_JSON = {k: sorted(v) for k, v in EXCLUDE.items()}
verdict = {"results": results, "stability": stability, "stability_gate_pass": stability_gate_pass,
           "worst_dominant_cv": worst_cv, "worst_apt_member": worst_apt_member, "overall_pass": overall_pass,
           "members": MEMBERS, "eligibility_exclusions": EXCLUDE_JSON, "n_folds": N_FOLDS, "cv_gate": CV_GATE}
json.dump(verdict, open(I + "agent_blend_verdict.json", "w"), indent=1)

# ---- final (all-data) weights for the build step, regardless of verdict (recorded either way) ----
final_weights = {}
for A in AIRPORTS:
    elig = eligible(A)
    m_apt = apts == A
    X = np.column_stack([pred[m][m_apt] - base[m_apt] for m in elig])
    r = y[m_apt] - base[m_apt]
    w, _ = nnls(X, r)
    final_weights[A] = dict(zip(elig, w.tolist()))
    print(f"FINAL (all-data) weights {A}: " + ", ".join(f"{m}={wv:.4f}" for m, wv in zip(elig, w)))
json.dump({"members": MEMBERS, "eligibility_exclusions": EXCLUDE_JSON, "weights": final_weights}, open(I + "agent_blend_weights_final.json", "w"), indent=1)

# ---- save the pooled seed-A OOF blend prediction (already computed above) as an npz for agent_lens_09_predict.py ----
np.savez(I + "agent_blend_oof_seedA.npz", ids=ids, y=y, days=days, apts=apts, both_guard=oof_by_seed["A"])
print("wrote data/interim/agent_blend_oof_seedA.npz (pooled seed-A OOF blend prediction, for the lens predictor)")
print("DONE")
