"""experiment restack6 chain: the PRE-REGISTERED capacity grid of docs/LAB-NOTES.md, run end to end with
no human choice in between. Resumable: a fit whose npz exists is not refitted. Clause code (rm / clauses / show) copied
verbatim from archive/agent_restack5_v109chain.py (== agent_restack5_chain.py == agent_seqsplice_02_judge.py). WHOLE
domain only -- no per-airport fallback, no second selection (unlike restack5's stage-1 candidates). Seed 0 judged first;
seed 0 FAIL -> candidate dropped without fitting seeds 1/2. Best passer picked by seed-0 all-rows d (pre-declared
tie-break: smaller n_estimators, then id order r1 < r2 < r3).

    python archive/agent_restack6_chain.py      (inside a scheduler slot, under archive/capped_run.py --cap-gb 6)

Writes data/interim/agent_restack6_verdict.json and prints WINNER=... / FINAL=... for the build driver.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")
import numpy as np

I = "data/interim/"
PY = sys.executable
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
SBX = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LSZH"]
S2 = ["LFPG", "LIRF"]
CANDIDATES = ["r1", "r2", "r3"]  # pre-registered order; NOT the selection order (that is by seed-0 d, s0.4)
T0 = time.time()
VJ = I + "agent_restack6_verdict.json"
res = json.load(open(VJ)) if os.path.exists(VJ) else {}


def say(m):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z +{time.time() - T0:5.0f}s] {m}", flush=True)


def save():
    json.dump(res, open(VJ, "w"), indent=1, default=float)


def run(args, log):
    say("RUN " + " ".join(args) + f"  > {log}")
    with open(log, "w") as fh:
        rc = subprocess.call([PY, "-u"] + args, stdout=fh, stderr=subprocess.STDOUT)
    for ln in open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-2:]:
        print("    | " + ln, flush=True)
    return rc


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys if k in z.files}


TOP_NPZ = I + "agent_restack5_V9TOP1.npz"
B9_NPZ = I + "agent_restack5_B9.npz"
c0 = load(TOP_NPZ)
y, apts, days = c0["y"].astype(float), c0["apts"].astype(str), c0["days"].astype(str)
mon = np.array([d[5:7] for d in days])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])
top = c0["both_guard"].astype(float)
N = len(y)
rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean())) if m.any() else float("nan")  # noqa: E731


def clauses(pt, base, base_name):
    d = lambda m: rm(pt, m) - rm(base, m)  # noqa: E731
    gain = (base - y) ** 2 - (pt - y) ** 2
    keep = np.ones(N, bool)
    order = np.argsort(-gain)[:50]
    keep[order] = False
    ch = pt != base
    h = [d(m) if (ch & m).any() else 0.0 for m in (H1, H2)]
    per = {a: d(apts == a) for a in AIRPORTS}
    c = {"vs": base_name, "all": d(np.ones(N, bool)), "worst_apt": max(per.values()), "m50": d(keep),
         "h1": h[0], "h2": h[1], "per": per, "rows_changed": int(ch.sum())}
    c["pass"] = c["all"] <= -0.10 and c["worst_apt"] <= 0.10 and c["m50"] < 0 and c["h1"] < 0 and c["h2"] < 0
    return c


def show(name, c):
    print(f"{name} (vs {c['vs']}): all {c['all']:+.4f}  worst airport {c['worst_apt']:+.3f}  minus-top-50 {c['m50']:+.3f}  "
          f"Jan-Mar {c['h1']:+.3f}  Jul-Sep {c['h2']:+.3f}  rows changed {c['rows_changed']:,}  "
          f"4-clause {'PASS' if c['pass'] else 'FAIL'}", flush=True)
    print("   by airport: " + "  ".join(f"{a} {v:+.3f}" for a, v in c["per"].items()), flush=True)


def P(path):
    z = load(path, ("ids", "both_guard"))
    assert (z["ids"] == c0["ids"]).all(), f"{path}: id mismatch"
    return z["both_guard"].astype(float)


def nest(recipe, seed):
    """cap,evt,trk,bx,<tok> at SBX, nested leak-fixed via agent_restack6_bxnest.py."""
    out = I + f"agent_restack6_bxn_{recipe.replace(',', '_')}_s{seed}.npz"
    if not os.path.exists(out):
        rc = run(["-W", "ignore", "archive/agent_restack6_bxnest.py", "--recipe", recipe, "--seed", str(seed), "--out", out],
                 f"logs/agent_restack6_bxn_{recipe.replace(',', '_')}_s{seed}.log")
        assert rc == 0, f"nested fit {recipe} s{seed} failed rc={rc}"
    return out


def seqfold(recipe, seed):
    """cap,evt,trk,seq,<tok> at LFPG,LIRF, via agent_restack6_fold4.py."""
    out = I + f"agent_restack6_f_{recipe.replace(',', '_')}_s{seed}.npz"
    if not os.path.exists(out):
        rc = run(["archive/agent_restack6_fold4.py", "--recipe", recipe, "--seed", str(seed), "--out", out],
                 f"logs/agent_restack6_fold4_{recipe.replace(',', '_')}_s{seed}.log")
        assert rc == 0, f"fold4 fit {recipe} s{seed} failed rc={rc}"
    return out


def sp(base, arm, aa):
    return np.where(np.isin(apts, aa), arm, base)


# ---------------------------------------------------------------- 0.1 identity checks (before any r1/r2/r3 fit)
d1 = float(np.abs(P(nest("cap,evt,trk,bx,big", 0)) - P(I + "agent_restack5_bxn_cap_evt_trk_bx_big_s0.npz")).max())
say(f"IDENTITY: cap,evt,trk,bx,big seed 0 through agent_restack6_bxnest.py vs "
    f"agent_restack5_bxn_cap_evt_trk_bx_big_s0.npz: max |diff| {d1}")
res["identity_bxnest_big"] = d1
save()
assert d1 == 0.0, "the bxnest copy does not reproduce round 5's big fit -- stop before any r1/r2/r3 fit"

d2 = float(np.abs(P(seqfold("cap,evt,trk,seq,big", 0)) - P(I + "agent_restack5_f_cap_evt_trk_seq_big_s0.npz")).max())
say(f"IDENTITY: cap,evt,trk,seq,big seed 0 through agent_restack6_fold4.py vs "
    f"agent_restack5_f_cap_evt_trk_seq_big_s0.npz: max |diff| {d2}")
res["identity_fold4_big"] = d2
save()
assert d2 == 0.0, "the fold4 copy does not reproduce round 5's big fit -- stop before any r1/r2/r3 fit"

# ---- sanity: TOP (V9TOP1) vs B9, for context and the capacity curve (s0.5)
b9 = P(B9_NPZ)
c_top_vs_b9 = clauses(top, b9, "B9(900 trees)")
show("(context) TOP=V9TOP1 (1500 trees) vs B9 (900 trees)", c_top_vs_b9)
res["TOP_vs_B9"] = c_top_vs_b9
save()

# ---------------------------------------------------------------- grid (0.2 / 0.3)
res.setdefault("grid", {})


def construct(tok, seed):
    p = top.copy()
    p = sp(p, P(nest(f"cap,evt,trk,bx,{tok}", seed)), SBX)
    p = sp(p, P(seqfold(f"cap,evt,trk,seq,{tok}", seed)), S2)
    return p


def judge(tok):
    out = {}
    p0 = None
    for rep in (0, 1, 2):
        p = construct(tok, rep)
        c = clauses(p, top, "TOP(V9TOP1)")
        show(f"{tok} replicate {rep}", c)
        out[f"rep{rep}"] = c
        if rep == 0:
            p0 = p
            if not c["pass"]:
                say(f"{tok} FAILS at seed 0 (whole domain, no fallback) -> dropped, seeds 1/2 not fit")
                out["pass"] = False
                res["grid"][tok] = out
                save()
                return False, None, c["all"]
        if not c["pass"]:
            say(f"{tok} FAILS at replicate {rep} -> dropped")
            out["pass"] = False
            res["grid"][tok] = out
            save()
            return False, None, out["rep0"]["all"]
    out["pass"] = True
    res["grid"][tok] = out
    save()
    say(f"{tok}: PASS at all 3 seeds")
    return True, p0, out["rep0"]["all"]


passers = {}
for tok in CANDIDATES:
    ok, p0, d_seed0 = judge(tok)
    if ok:
        passers[tok] = (p0, d_seed0)

res["passers"] = {k: v[1] for k, v in passers.items()}
save()
say(f"STAGE PASSERS: {res['passers'] or 'none'}")

# ---------------------------------------------------------------- 0.4 selection (pre-declared tie-break)
NTREES = {"r1": 2500, "r2": 3000, "r3": 1500}
ORDER_IDX = {"r1": 0, "r2": 1, "r3": 2}
winner = None
if passers:
    winner = min(passers, key=lambda k: (passers[k][1], NTREES[k], ORDER_IDX[k]))
    say(f"WINNER: {winner} (seed-0 all-rows d {passers[winner][1]:+.4f})")
    out_npz = I + f"agent_restack6_{winner}_TOP.npz"
    np.savez(out_npz, ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=passers[winner][0])
    res["winner"] = winner
    res["winner_npz"] = out_npz
else:
    say("NO CANDIDATE PASSES -- nothing to build")
    res["winner"] = None
save()

# ---------------------------------------------------------------- 0.5 capacity curve
rmse_b9 = rm(b9, np.ones(N, bool))
rmse_top = rm(top, np.ones(N, bool))
curve = {"B9_900trees": rmse_b9, "TOP_1500trees": rmse_top}
if "r1" in res["grid"]:
    r1_rmse = rm(passers["r1"][0], np.ones(N, bool)) if "r1" in passers else None
    if r1_rmse is None:
        # r1 was fit at seed 0 even if it failed (judge() always fits seed 0); reconstruct it for the curve
        p_r1 = construct("r1", 0)
        r1_rmse = rm(p_r1, np.ones(N, bool))
    curve["r1_2500trees_seed0"] = r1_rmse
say(f"CAPACITY CURVE (all-rows RMSE): 900 trees (B9) {rmse_b9:.4f}  ->  1500 trees (TOP) {rmse_top:.4f}"
    + (f"  ->  2500 trees (r1 seed0) {curve.get('r1_2500trees_seed0', float('nan')):.4f}" if "r1_2500trees_seed0" in curve else ""))
say(f"  deltas: 900->1500 {rmse_top - rmse_b9:+.4f}"
    + (f"; 1500->2500 {curve.get('r1_2500trees_seed0', float('nan')) - rmse_top:+.4f}" if "r1_2500trees_seed0" in curve else ""))
res["capacity_curve"] = curve
save()

print(f"WINNER={winner or 'NONE'}")
print(f"FINAL={'1' if winner else '0'}")
