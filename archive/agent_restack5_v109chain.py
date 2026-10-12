"""experiment restack5 v109 chain: the INTERACTION CHECK of docs/LAB-NOTES.md (base change to v109), run end to end
with no human choice. Resumable (a fit whose npz exists is not refitted). Clause code copied verbatim from
archive/agent_restack5_chain.py (== agent_seqsplice_02_judge.py). Nested-bx fits via archive/agent_restack5_bxnest.py.
Writes data/interim/agent_restack5_v109_verdict.json, composites agent_restack5_V9TOP{k}.npz, prints STEP= lines.
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
R6 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH"]
T0 = time.time()
VJ = I + "agent_restack5_v109_verdict.json"
res = {}


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


B9 = I + "agent_restack5_B9.npz"
if not os.path.exists(B9):
    assert run(["archive/splice_npz.py", I + "agent_restack4_ctrl.npz", I + "agent_backx_BXnest_s0.npz", ",".join(SBX), B9],
               "logs/agent_restack5_B9.log") == 0


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys if k in z.files}


c0 = load(B9)
y, apts, days = c0["y"].astype(float), c0["apts"].astype(str), c0["days"].astype(str)
mon = np.array([d[5:7] for d in days])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])
pq = c0["both_guard"].astype(float)
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
    out = I + f"agent_restack5_bxn_{recipe.replace(',', '_')}_s{seed}.npz"
    if not os.path.exists(out):
        rc = run(["-W", "ignore", "archive/agent_restack5_bxnest.py", "--recipe", recipe, "--seed", str(seed), "--out", out],
                 f"logs/agent_restack5_bxn_{recipe.replace(',', '_')}_s{seed}.log")
        assert rc == 0, f"nested fit {recipe} s{seed} failed rc={rc}"
    return out


def sp(base, arm, aa):
    return np.where(np.isin(apts, aa), arm, base)


# ---- sanity: B9 vs CTRL (context), identity of the nested copy
ctrl = P(I + "agent_restack4_ctrl.npz")
cb9 = clauses(pq, ctrl, "CTRL(v108 analogue)")
show("(context) B9 = v109 analogue", cb9)
res["B9_vs_CTRL"] = cb9
dI = float(np.abs(P(nest("cap,evt,trk,bx", 0)) - P(I + "agent_backx_BXnest_s0.npz")).max())
say(f"IDENTITY: cap,evt,trk,bx seed 0 through agent_restack5_bxnest.py vs agent_backx_BXnest_s0.npz: max |diff| {dI}")
res["identity_bxnest"] = dI
save()
assert dI == 0.0, "nested copy does not reproduce BXnest_s0 -- stop"

steps = []


def judge_step(tag, base, base_name, groups):
    """groups: [(airports, recipe_fn(seed) -> path)]; returns (pass, p0, verdicts)."""
    out, p0 = {}, None
    ok = True
    for rep in (0, 1, 2):
        p = base.copy()
        for aa, fn in groups:
            p = sp(p, P(fn(rep)), aa)
        c = clauses(p, base, base_name)
        show(f"{tag} replicate {rep}", c)
        out[f"rep{rep}"] = c
        if rep == 0:
            p0 = p
            cc = clauses(p, pq, "B9")
            show(f"(cumulative) {tag} replicate 0", cc)
            out["cumulative_vs_B9"] = cc
        ok = ok and c["pass"]
        if not ok:
            bad = [a for a, v in c["per"].items() if v > 0.10]
            out["failing_airports"] = bad
            say(f"{tag} FAILS at replicate {rep}; airports above +0.10: {bad or 'none'}; clauses: all {c['all']:+.3f}, "
                f"m50 {c['m50']:+.3f}, Jan-Mar {c['h1']:+.3f}, Jul-Sep {c['h2']:+.3f}")
            break
    out["pass"] = ok
    res[tag] = out
    save()
    return ok, p0


def seqbig(s):
    return I + f"agent_restack5_f_cap_evt_trk_seq_big_s{s}.npz"


say("== STEP 1: big on B9 (cap,evt,trk,bx,big nested at S_bx; cap,evt,trk,seq,big at LFPG, LIRF)")
ok1, p1 = judge_step("big_on_B9", pq, "B9", [(SBX, lambda s: nest("cap,evt,trk,bx,big", s)), (["LFPG", "LIRF"], seqbig)])
top, top_npz = pq, B9
if ok1:
    out = I + "agent_restack5_V9TOP1.npz"
    np.savez(out, ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=p1)
    steps.append({"k": 1, "tag": "big", "groups": [{"recipe": "cap,evt,trk,bx,big", "airports": R6},
                                                   {"recipe": "cap,evt,trk,seq,bx,big", "airports": ["LEMD"]},
                                                   {"recipe": "cap,evt,trk,seq,big", "airports": ["LFPG", "LIRF"]}],
                  "npz_before": B9, "npz_after": out})
    top, top_npz = p1, out
    say("== STEP 2: nbr on step 1 (cap,evt,trk,nbr,bx,big nested at R6)")
    ok2, p2 = judge_step("nbr_on_big", top, "step1", [(R6, lambda s: nest("cap,evt,trk,nbr,bx,big", s))])
    rec2 = "cap,evt,trk,nbr,bx,big"
else:
    say("== STEP 1 failed -> nbr on B9 (cap,evt,trk,nbr,bx nested at R6), s6.2")
    ok2, p2 = judge_step("nbr_on_B9", top, "B9", [(R6, lambda s: nest("cap,evt,trk,nbr,bx", s))])
    rec2 = "cap,evt,trk,nbr,bx"
if ok2:
    k = len(steps) + 1
    out = I + f"agent_restack5_V9TOP{k}.npz"
    np.savez(out, ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=p2)
    steps.append({"k": k, "tag": "nbr", "groups": [{"recipe": rec2, "airports": R6}], "npz_before": top_npz, "npz_after": out})
res["steps"] = steps
save()
if not steps:
    print("FINAL=NONE")
    sys.exit(0)
for st in steps:
    print("STEP=" + json.dumps(st))
print(f"FINAL={len(steps)}")
