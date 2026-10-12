"""tailc serving twin (model B direct, trained on all 12 months of the fold, plan rows, 3 seeds averaged) -> data/interim/agent_tailc_serve_ehat.npz (ehat for the January/July serve rows, 0 for no-plan / non-nine-airport rows)"""
import json
import numpy as np
import lightgbm as lgb
I = "data/interim/"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
z = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
y, pv, apt = z["y"], z["pv"], z["apt"].astype(str)
cols = [str(c) for c in z["cols"]]
N, nx = len(y), len(cols)
F = np.mean([np.load(I + f"agent_bigc12_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"] for s in (0, 1, 2)], 0)
kap = json.load(open(I + "agent_m12_kapt_C4ff7V5.json"))
cur = pv + np.array([kap[a] for a in apt]) * F
X = np.empty((N, nx + 2), np.float32)
X[:, :nx] = z["X"]
X[:, nx] = F
X[:, nx + 1] = cur - pv
plan = X[:, cols.index("has_flight_plan")] == 1
del z
rw = np.clip(y - cur, -7200, 7200)
e = rw - np.clip(rw, -900, 900)
ev = e != 0
si = np.load(I + "agent_bigc12_serve_in.npz", allow_pickle=True)
sa = si["apt"].astype(str)
cs = [str(c) for c in si["cols_pre"]]
Xs = np.empty((len(sa), nx + 2), np.float32)
Xs[:, :nx] = si["Xs_pre"][:, [cs.index(c) for c in cols]]
Fs = np.mean([np.load(I + f"bigc12_serve/C4ff7V5_s{s}.npz")["pred"] for s in (0, 1, 2)], 0)
Xs[:, nx] = Fs
Xs[:, nx + 1] = np.array([kap.get(a, 0.0) for a in sa]) * Fs
use = np.isin(sa, AIR) & (Xs[:, cols.index("has_flight_plan")] == 1)
print("serve rows", len(sa), "with plan, nine airports", int(use.sum()), flush=True)
out = []
for S in (0, 1, 2):
    rng = np.random.default_rng(100 + S)
    tr = np.where(plan & (ev | (rng.random(N) < 0.2)))[0]
    w = np.where(ev[tr], 1.0, 5.0)
    p = dict(objective="regression", verbosity=-1, num_threads=6, seed=S, bagging_seed=S, feature_fraction_seed=S, feature_fraction=0.5, bagging_fraction=0.5,
             bagging_freq=1, max_bin=63, learning_rate=0.03, num_leaves=15, min_child_samples=200, lambda_l2=50.0)
    m = lgb.train(p, lgb.Dataset(X[tr], e[tr], weight=w), 300)
    h = np.zeros(len(sa))
    h[use] = m.predict(Xs[use])
    out.append(h)
    print(f"seed {S}: serve mean ehat {h[use].mean():+.2f}, sd {h[use].std():.1f}", flush=True)
np.savez(I + "agent_tailc_serve_ehat.npz", ehat=np.mean(out, 0), ids=si["ids"], apt=sa, use=use)
print("seed corr", np.round(np.corrcoef([o[use] for o in out])[0, 1:], 3))
