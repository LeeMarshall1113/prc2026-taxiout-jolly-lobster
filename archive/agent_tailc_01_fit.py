"""tailc fit (docs/LAB-NOTES.md): LOMO tail-excess models A (two-part) and B (direct) on e = r - clip(r,+-900), r = y - current prediction.
usage: agent_tailc_01_fit.py SEED   -> data/interim/agent_tailc_s{SEED}.npz (ehatA, ehatB over the 12-month fold; filled for held months of the original 8 only)"""
import json
import sys
import numpy as np
import lightgbm as lgb

S = int(sys.argv[1])
I = "data/interim/"
z = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
y, pv, apt, mon = z["y"], z["pv"], z["apt"].astype(str), z["mon"]
cols = list(z["cols"])
N = len(y)
F = np.mean([np.load(I + f"agent_bigc12_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"] for s in (0, 1, 2)], 0)
kap = json.load(open(I + "agent_m12_kapt_C4ff7V5.json"))
dose = np.array([kap[a] for a in apt])
cur = pv + dose * F
nx = z["X"].shape[1]
X = np.empty((N, nx + 2), np.float32)
X[:, :nx] = z["X"]
X[:, nx] = F
X[:, nx + 1] = cur - pv
plan = z["X"][:, cols.index("has_flight_plan")] == 1
del z
r = y - cur
rw = np.clip(r, -7200, 7200)
e = rw - np.clip(rw, -900, 900)
ep, en = np.maximum(e, 0), np.minimum(e, 0)
print("events: pos", int((ep > 0).sum()), "neg", int((en < 0).sum()), "plan rows", int(plan.sum()), flush=True)
MON8 = [1, 2, 3, 7, 8, 9, 10, 11]
rng = np.random.default_rng(100 + S)
keepu = rng.random(N) < 0.2
base = dict(objective="binary", verbosity=-1, num_threads=6, seed=S, bagging_seed=S, feature_fraction_seed=S, feature_fraction=0.5, bagging_fraction=0.5, bagging_freq=1,
            max_bin=63, lambda_l2=20.0)
outA, outB = np.zeros(N), np.zeros(N)
import os
for m in ([int(os.environ["TAILC_M"])] if "TAILC_M" in os.environ else MON8):
    te = (mon == m) & plan
    trm = (mon != m) & plan
    ev = (ep > 0) | (en < 0)
    tr = np.where(trm & (ev | keepu))[0]
    w = np.where(ev[tr], 1.0, 5.0)
    Xt = X[tr]
    # B direct
    mb = lgb.train({**base, "objective": "regression", "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 200, "lambda_l2": 50.0},
                   lgb.Dataset(Xt, e[tr], weight=w), 300)
    outB[te] = mb.predict(X[te])
    # A two-part
    pa = np.zeros(te.sum())
    for sgn, tgt in ((1, ep), (-1, en)):
        yb = (tgt[tr] != 0).astype(float)
        mc = lgb.train({**base, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 100}, lgb.Dataset(Xt, yb, weight=w), 200)
        evr = tr[tgt[tr] != 0]
        mr = lgb.train({**base, "objective": "regression", "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 100},
                       lgb.Dataset(X[evr], tgt[evr]), 150)
        pa += mc.predict(X[te]) * mr.predict(X[te])
    outA[te] = pa
    print(f"seed {S} month {m} done: rows {int(te.sum())} mean ehatA {pa.mean():+.2f} B {outB[te].mean():+.2f} realised e {e[te].mean():+.2f}", flush=True)
np.savez(I + (f"agent_tailc_smoke.npz" if "TAILC_M" in os.environ else f"agent_tailc_s{S}.npz"), A=outA, B=outB, e=e, cur=cur, plan=plan)
