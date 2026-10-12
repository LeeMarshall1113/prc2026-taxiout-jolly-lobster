"""experiment frontier 04: judge the 60-day stage-1 check of --floor-stack-feats quality_anch (archive/agent_frontier_s1.sh).

    .venv/Scripts/python.exe archive/agent_frontier_04_judge.py [A.npz B.npz]

Arms: Q = what v97/v98 serve (--stack-feats quality_anch --floor-stack-feats quality); F = the same with the floor stack also fed
the anchor columns (--floor-stack-feats quality_anch). Blend both_guard, after hard floor + floor stack, leave-one-day-out.
PASS RULE, fixed in docs/LAB-NOTES.md BEFORE the run (d = RMSE(F) - RMSE(Q)):
  1. Oct-Nov rows: d <= -0.10 s
  2. Jan-Sep rows: d <= 0
  3. Oct-Nov rows: d < 0 after deleting the 50 rows F helps most
  4. Oct-Nov: no airport worse than +0.10 s
  5. Oct-Nov: F better on >= 60% of days
"""
import sys

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in z.keys()}


fa, fb = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("data/interim/frontier_s1_Q.npz", "data/interim/frontier_s1_F.npz")
q, f = load(fa), load(fb)
assert (q["ids"] == f["ids"]).all(), "arms score different rows"
y, days, apts = q["y"], q["days"].astype(str), q["apts"].astype(str)
pq, pf = q["both_guard"], f["both_guard"]
on = np.array([d[5:7] in ("10", "11") for d in days])
rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean()))  # noqa: E731
dsq = (pf - y) ** 2 - (pq - y) ** 2
print(f"rows {len(y):,} ({int(on.sum()):,} Oct-Nov on {len(set(days[on]))} days, {int((~on).sum()):,} Jan-Sep on {len(set(days[~on]))} days)")
res = {}
for name, m in (("Oct-Nov", on), ("Jan-Sep", ~on)):
    d = rm(pf, m) - rm(pq, m)
    dd = dsq[m]
    top = np.sort(dd)[:50].sum()
    d50 = np.sqrt(((pq[m] - y[m]) ** 2).sum() / m.sum() + (dd.sum() - top) / m.sum()) - rm(pq, m)
    byday = {dy: dsq[m & (days == dy)].sum() for dy in sorted(set(days[m]))}
    better = np.mean([v < 0 for v in byday.values()])
    byapt = {a: rm(pf, m & (apts == a)) - rm(pq, m & (apts == a)) for a in sorted(set(apts[m]))}
    res[name] = (d, d50, better, byapt)
    print(f"  {name}: RMSE Q {rm(pq, m):.3f}  F {rm(pf, m):.3f}  d {d:+.3f} s | minus top-50 {d50:+.3f} | days better {better:.0%} | "
          f"dSSE {dd.sum():+.3e}")
    print("     by airport: " + "  ".join(f"{a} {v:+.2f}" for a, v in byapt.items()))
d_on, d50_on, better_on, apt_on = res["Oct-Nov"]
checks = {"1. Oct-Nov d <= -0.10 s": d_on <= -0.10, "2. Jan-Sep d <= 0": res["Jan-Sep"][0] <= 0,
          "3. Oct-Nov minus top-50 < 0": d50_on < 0, "4. Oct-Nov no airport worse than +0.10 s": max(apt_on.values()) <= 0.10,
          "5. Oct-Nov days better >= 60%": better_on >= 0.60}
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"  VERDICT: {'PASS' if all(checks.values()) else 'FAIL'}")
