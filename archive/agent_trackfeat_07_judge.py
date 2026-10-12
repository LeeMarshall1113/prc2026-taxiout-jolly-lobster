"""experiment trackfeat 07: judge the full-day (177/264-day) confirmation of whatever passed the 60-day screen
(archive/agent_trackfeat_s1.sh -> archive/agent_trackfeat_04_judge.py). PRE-REGISTERED before any confirmation
run: this file is written and the rule fixed in docs/LAB-NOTES.md section 3, BEFORE the screen's
results are read.

    .venv/Scripts/python.exe archive/agent_trackfeat_07_judge.py [Q.npz T.npz]

Reads data/interim/trackfeat_c{N}_{Q,T}.npz (blend both_guard, after hard floor + floor stack, leave-one-day-out
over all scored days). Q = production (v99/v100 as shipped: --stack-feats quality_anch --floor-stack-feats
quality_anch --floor-stack-anch-skip LIRF). T = Q plus --floor-stack-track on --track-cols <the arm that
passed the screen>. d = RMSE(T) - RMSE(Q).
PASS RULE, FIXED before the confirmation runs (same four clauses the plan asked for, and the same form as
archive/agent_frontier_07_judge.py's lead-set rule for the anchors' own confirmation):
  1. all rows d <= -0.10 s
  2. no airport worse than +0.10 s
  3. d < 0 after deleting the 50 rows T helps most
  4. d < 0 on the Jan-Mar days AND on the Jul-Sep days
Reported, not gating: Oct-Nov d, core rows (y <= 10,000 s), per airport, per month, days better, first-order
board conversion.
"""
import sys

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in z.keys()}


fa, fb = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("data/interim/trackfeat_c238_Q.npz", "data/interim/trackfeat_c238_T.npz")
q, f = load(fa), load(fb)
assert (q["ids"] == f["ids"]).all(), "arms score different rows"
y, days, apts = q["y"], q["days"].astype(str), q["apts"].astype(str)
pq, pf = q["both_guard"], f["both_guard"]
mon = np.array([d[5:7] for d in days])
rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean()))  # noqa: E731
d_of = lambda m: rm(pf, m) - rm(pq, m)  # noqa: E731
allm = np.ones(len(y), bool)
dsq = (pf - y) ** 2 - (pq - y) ** 2
top = np.sort(dsq)[:50].sum()
d50 = float(np.sqrt(((pq - y) ** 2).mean() + (dsq.sum() - top) / len(y)) - rm(pq, allm))
byapt = {a: d_of(apts == a) for a in sorted(set(apts))}
jm, js, on = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"]), np.isin(mon, ["10", "11"])
byday = np.array([dsq[days == dd].sum() for dd in sorted(set(days))])
print(f"full-day confirmation, T (trackfeat, the arm the 60-day screen passed) against Q (production): {len(y):,} rows on {len(set(days))} days")
print(f"  RMSE Q {rm(pq, allm):.3f}  T {rm(pf, allm):.3f}; rows changed {int((pf != pq).sum()):,}")
checks = {"1. all rows <= -0.10 s": (d_of(allm), d_of(allm) <= -0.10),
          "2. no airport worse than +0.10 s": (max(byapt.values()), max(byapt.values()) <= 0.10),
          "3. negative minus top-50": (d50, d50 < 0),
          "4a. Jan-Mar days negative": (d_of(jm), d_of(jm) < 0),
          "4b. Jul-Sep days negative": (d_of(js), d_of(js) < 0)}
for k, (v, ok) in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k:<36} {v:+.3f}")
print(f"  (reported) Oct-Nov days: {d_of(on):+.3f}   core rows y <= 10,000 s: {d_of(y <= 10000):+.3f}")
print("  (reported) by airport: " + "  ".join(f"{a} {v:+.2f}" for a, v in byapt.items()))
print("  (reported) by month:   " + "  ".join(f"{m} {d_of(mon == m):+.2f}" for m in sorted(set(mon))))
print(f"  (reported) days better {int((byday < 0).sum())}/{len(byday)}; dSSE {dsq.sum():+.3e}; "
      f"first-order board (dSSE per departure x 344,841 / (2 x 238.1 x 344,841)) {dsq.sum() / len(y) / (2 * 238.1):+.3f} s")
print(f"  VERDICT: {'PASS' if all(ok for _, ok in checks.values()) else 'FAIL'}")
