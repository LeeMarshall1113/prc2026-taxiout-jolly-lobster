"""experiment expensive 06: PRE-REGISTERED judge for the re-stack arms (docs/LAB-NOTES.md), written before any
238-day number exists.

    python archive/agent_expensive_06_judge.py CONTROL.npz TREAT.npz

d = RMSE(T) - RMSE(Q) on `both_guard`, all rows of the npz. PASS = all four:
  1. all rows <= -0.10 s; 2. no airport worse than +0.10 s; 3. negative after deleting the 50 rows T helps most;
  4. negative on the Jan-Mar days AND on the Jul-Sep days.
If T fails: SPLICE FALLBACK (same as docs/LAB-NOTES.md). Airport a is eligible iff W_a (control with T's
rows at a) has all-rows d < 0, own-airport d <= -0.10 s, minus-top-50 < 0, Jan-Mar and Jul-Sep both < 0 (a half in which
W_a changes zero rows counts as passed). The splice over ALL eligible airports must then pass clauses 1-4, else nothing
ships. No second selection. Prints the --skip list that realises the splice in archive/agent_expensive_04_restack_serve.py.
"""
import sys

import numpy as np


def load(p):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in ("ids", "y", "days", "apts", "both_guard")}


q, t = load(sys.argv[1]), load(sys.argv[2])
assert (q["ids"] == t["ids"]).all()
y, apts = q["y"].astype(float), q["apts"].astype(str)
mon = np.array([d[5:7] for d in q["days"].astype(str)])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])
pq = q["both_guard"].astype(float)
rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean()))  # noqa: E731
ALL = np.ones(len(y), bool)


def clauses(pt):
    d = lambda m: rm(pt, m) - rm(pq, m)  # noqa: E731
    gain = (pq - y) ** 2 - (pt - y) ** 2
    keep = ALL.copy()
    keep[np.argsort(-gain)[:50]] = False
    ch = pt != pq
    h = [d(m) if (ch & m).any() else 0.0 for m in (H1, H2)]
    per = {a: d(apts == a) for a in sorted(set(apts))}
    c = {"all": d(ALL), "worst_apt": max(per.values()), "m50": d(keep), "h1": h[0], "h2": h[1], "per": per,
         "h_changed": [bool((ch & H1).any()), bool((ch & H2).any())]}
    c["pass"] = c["all"] <= -0.10 and c["worst_apt"] <= 0.10 and c["m50"] < 0 and c["h1"] < 0 and c["h2"] < 0
    return c


def show(name, c):
    print(f"{name}: all {c['all']:+.3f}  worst airport {c['worst_apt']:+.3f}  minus-top-50 {c['m50']:+.3f}  "
          f"Jan-Mar {c['h1']:+.3f}  Jul-Sep {c['h2']:+.3f}  -> {'PASS' if c['pass'] else 'FAIL'}")
    print("   by airport: " + "  ".join(f"{a} {v:+.3f}" for a, v in c["per"].items()))


c = clauses(t["both_guard"].astype(float))
show("T whole", c)
if c["pass"]:
    print("VERDICT: PASS (whole). Serve with no --skip.")
    sys.exit(0)
elig = []
for a in sorted(set(apts)):
    w = np.where(apts == a, t["both_guard"], pq).astype(float)
    ca = clauses(w)
    h1 = ca["h1"] < 0 or not ca["h_changed"][0]
    h2 = ca["h2"] < 0 or not ca["h_changed"][1]
    ok = ca["all"] < 0 and ca["per"][a] <= -0.10 and ca["m50"] < 0 and h1 and h2
    if (w != pq).any():
        print(f"  splice {a}: all {ca['all']:+.3f} own {ca['per'][a]:+.3f} m50 {ca['m50']:+.3f} "
              f"JanMar {ca['h1']:+.3f} JulSep {ca['h2']:+.3f} -> {'eligible' if ok else 'no'}")
    if ok:
        elig.append(a)
if not elig:
    print("VERDICT: FAIL (no eligible airport). Nothing ships.")
    sys.exit(1)
w = np.where(np.isin(apts, elig), t["both_guard"], pq).astype(float)
cs = clauses(w)
show(f"SPLICE {','.join(elig)}", cs)
skip = sorted(set(apts) - set(elig) - {"LTFM"})
print(f"VERDICT: {'PASS (splice) -- serve with --skip ' + ','.join(skip) if cs['pass'] else 'FAIL (splice). Nothing ships.'}")
