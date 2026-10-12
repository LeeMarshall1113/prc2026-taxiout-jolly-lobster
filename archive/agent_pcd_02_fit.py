"""pcd step 2: local replica of the real corrector (config C4ff7V5, full capacity: 511 leaves, 150 min rows, l2 100, feature fraction 0.7, rate 0.03, bagging 0.5, early stopping 50 on the next month,
target clip +-900, base = the refit's fit base nb) leave-one-month-out, plus the plan-clock distance columns / controls. Our own copy of the recipe in modal/agent_bigc13_app.py (read only).
usage: python agent_pcd_02_fit.py JOBS    JOBS = comma list of ARM:lomo:HELD:SEED or ARM:temp:SIDE:SEED   (SIDE a = fit months 1-6 -> predict 7-12; b = fit 7-12 -> predict 1-6)
ARM: ctrl | dist | shuf (distance columns computed with the prediction shuffled within airport-month, draw = seed) | noise (9 random columns, draw = seed).
Output data/interim/agent_pcd_out/<ARM>_s<seed>_m<held>.npz or <ARM>_s<seed>_t<side>.npz  (pred = correction on the predicted rows, rows = fold row index, bi = best iteration).
Temporal frame: key statistics of the fit rows are leave-own-month-out inside the fit months; predicted rows (the other half-year) get statistics from all fit months (serve-style); the label-free drift
and variant-view columns are the shipped precomputed ones. Needs about 5 GB: run under a 5 GB slot."""
import os
import sys
import time
import numpy as np
import lightgbm as lgb

sys.path.insert(0, "archive")
import agent_bigc_te as TE

I = "data/interim/"
OUT = I + "agent_pcd_out/"
os.makedirs(OUT, exist_ok=True)
THREADS = int(os.environ.get("BIGC_THREADS", 6))
t0 = time.time()


def P(*a):
    print("[%5.0fs]" % (time.time() - t0), *a, flush=True)


z = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True)
cols = [str(c) for c in z["cols"]]
catidx = [int(i) for i in z["cat"]]
y = z["y"].astype(float)
pv = z["pv"].astype(float)
mon = z["mon"]
N = len(y)
months = sorted(set(mon.tolist()))
r = np.clip(y - pv, -900, 900)
kcols = ["ADEP_mvt"] + TE.KEYCOLS
Xk = np.column_stack([z["X"][:, cols.index(c)] for c in kcols])
kf = TE.keys_from(Xk, kcols)
del Xk
kk = np.load(I + "agent_bigc13_keys2.npz", allow_pickle=True)
k0 = np.load(I + "agent_bigc12_keys.npz", allow_pickle=True)
vf = np.load(I + "agent_bigc12_vfac_clean5.npz", allow_pickle=True)
ex = np.load(I + "agent_bigc12_extra.npz", allow_pickle=True)
e2 = np.load(I + "agent_bigc12_extra2.npz", allow_pickle=True)
ymu_apt = {a: y[k0["apt_f"] == a].mean() for a in set(k0["apt_f"].tolist())}
ymu_f = np.array([ymu_apt[a] for a in k0["apt_f"]])
Kf = kk["Kf"]
C = np.load(I + "agent_pcd_cols.npz", allow_pickle=True)
CFG = dict(num_leaves=511, min_data_in_leaf=150, lambda_l2=100.0, feature_fraction=0.7)


def params(seed):
    return dict(objective="l2", learning_rate=0.03, bagging_fraction=0.5, bagging_freq=1, max_bin=63, num_threads=THREADS, verbose=-1, seed=seed, **CFG)


def extra_cols(arm, seed):
    if arm == "ctrl":
        return None
    if arm == "dist":
        return C["D"]
    if arm == "shuf":
        return C["Dshuf"][seed]
    if arm == "noise":
        return C["Dnoise"][seed]
    raise ValueError(arm)


def assemble(perm, ntr_stat, held_stat, arm, seed):
    """X in perm order. ntr_stat / held_stat: callables giving the (4, 112) key statistic blocks for fit rows and predicted rows (in perm order)."""
    ec = extra_cols(arm, seed)
    nx = 0 if ec is None else ec.shape[1]
    X = np.empty((N, 370 + nx), np.float32)
    X[:, :140] = z["X"][perm]
    X[:, 140:140 + 52] = ex["Xf"][perm]
    X[:, 192:194] = e2["Xf"][perm]
    s4, s112 = ntr_stat(perm)
    X[:, 194:198] = s4
    X[:, 198:310] = s112
    X[:, 310:318] = kk["Df"][perm]
    X[:, 318:370] = vf["Xf"][perm]
    if nx:
        X[:, 370:] = ec[perm]
    return X


def train_predict(X, rp, ntr, nva, seed, ncols_arm):
    Xt, Xv, Xe = X[:ntr], X[ntr:ntr + nva], X[ntr + nva:]
    dt = lgb.Dataset(Xt, rp[:ntr], categorical_feature=catidx, params=dict(max_bin=63, verbose=-1))
    dv = lgb.Dataset(Xv, rp[ntr:ntr + nva], categorical_feature=catidx, reference=dt)
    bst = lgb.train(params(seed), dt, 1500, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    return bst.predict(Xe, num_iteration=bst.best_iteration), bst.best_iteration


def lomo(arm, h, seed):
    path = OUT + "%s_s%d_m%d.npz" % (arm, seed, h)
    if os.path.exists(path):
        P("skip", path)
        return
    nxt = [k for k in months if k != h]
    val = min([k for k in nxt if k > h] or nxt)
    tr_i = np.where((mon != h) & (mon != val))[0]
    va_i = np.where(mon == val)[0]
    te_i = np.where(mon == h)[0]
    perm = np.concatenate([tr_i, va_i, te_i])
    ntr, nva = len(tr_i), len(va_i)
    T2 = TE.Te2(Kf)

    def stat(pm):
        return TE.te_fold(kf, r, mon, h, months)[pm], T2.fold(r, y, ymu_f, mon, h, months)[pm]
    X = assemble(perm, stat, None, arm, seed)
    pred, bi = train_predict(X, r[perm].astype(np.float32), ntr, nva, seed, X.shape[1])
    np.savez(path + ".tmp.npz", pred=pred, rows=te_i, bi=bi)
    os.replace(path + ".tmp.npz", path)
    P("lomo", arm, "held", h, "seed", seed, "rounds", bi, "ncols", X.shape[1])


def temporal(arm, side, seed):
    path = OUT + "%s_s%d_t%s.npz" % (arm, seed, side)
    if os.path.exists(path):
        P("skip", path)
        return
    fit_m = [1, 2, 3, 4, 5, 6] if side == "a" else [7, 8, 9, 10, 11, 12]
    val = 6 if side == "a" else 7
    Ts = np.where(np.isin(mon, fit_m))[0]
    Hs = np.where(~np.isin(mon, fit_m))[0]
    tr_i = Ts[mon[Ts] != val]
    va_i = Ts[mon[Ts] == val]
    perm = np.concatenate([tr_i, va_i, Hs])
    ntr, nva = len(tr_i), len(va_i)
    T2 = TE.Te2(Kf[Ts])
    ymu_s = ymu_f

    def stat(pm):
        s4 = np.empty((N, 4), np.float32)
        s112 = np.empty((N, 112), np.float32)
        s4[Ts] = TE.te_loo(kf[Ts], r[Ts], mon[Ts], fit_m)
        s112[Ts] = T2.fold(r[Ts], y[Ts], ymu_s[Ts], mon[Ts], None, fit_m)
        s4[Hs] = TE.te_serve(kf[Ts], r[Ts], kf[Hs])
        s112[Hs] = T2.serve(r[Ts], y[Ts], mon[Ts], fit_m, Kf[Hs], ymu_s[Hs])
        return s4[pm], s112[pm]
    X = assemble(perm, stat, None, arm, seed)
    pred, bi = train_predict(X, r[perm].astype(np.float32), ntr, nva, seed, X.shape[1])
    np.savez(path + ".tmp.npz", pred=pred, rows=Hs, bi=bi)
    os.replace(path + ".tmp.npz", path)
    P("temporal", arm, "side", side, "seed", seed, "rounds", bi, "ncols", X.shape[1])


for j in sys.argv[1].split(","):
    if os.environ.get("JOB_PREEMPT_FILE") and os.path.exists(os.environ["JOB_PREEMPT_FILE"]):
        P("preempt requested")
        sys.exit(10)
    p = j.split(":")
    if p[1] == "lomo":
        lomo(p[0], int(p[2]), int(p[3]))
    else:
        temporal(p[0], p[2], int(p[3]))
