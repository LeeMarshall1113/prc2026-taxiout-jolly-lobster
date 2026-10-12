"""pcd step 1: prediction-minus-plan-clock distance columns for the corrector (experiment "pcd", 2026-10-04). Our own construction; idea credited to the public repository
https://github.com/EnioAguiar/prc-taxiout-2026 (GPL-3.0), no code copied.
Fold rows: the refit corrector inputs (agent_bigc13_fold.npz: pv = nb, the out-of-fold serve-like base the corrector already sees). Serve rows: agent_bigc13_serve_in.npz (pv = served base, no 2026 label).
Columns (9): d_SCHED, d_LOBT, d_IOBT, d_EOBT, d_AOBT3 = pv - (take-off - clock) (= pv - gap_k; gap_k = take-off minus clock is already an input);
clamp = gap_sched - pv where gap_sched in [5000, 50000) (distance to the gap clamp, NaN elsewhere); wmin = 3606 - |d_LOBT| (distance to the NEARER edge of the plan-clock window; the two one-sided edge
distances are monotone copies of d_LOBT, identical for trees, so not added); d_AOBT3 split by the label-free AOBT_3 default/report flag (agent_aobtflag_flags.parquet): d_AOBT3_ind (independent), d_AOBT3_der (derived).
Controls: shuf = the same 9 columns with pv permuted within airport-month (3 shuffles, one per seed), noise = 9 N(0,1000) columns (3 draws).
Output data/interim/agent_pcd_cols.npz ; drift AUC (same month, within airport, row-weighted, symmetric) data/interim/agent_pcd_auc.csv. No 2026 departure label read."""
import time
import numpy as np
import pandas as pd
import polars as pl

I = "data/interim/"
t0 = time.time()
NAMES = ["d_SCHED", "d_LOBT", "d_IOBT", "d_EOBT", "d_AOBT3", "clamp", "wmin", "d_AOBT3_ind", "d_AOBT3_der"]


def P(*a):
    print("[%5.0fs]" % (time.time() - t0), *a, flush=True)


def build(pv, gap, flag):
    """gap: dict of gap_sched, gap_lobt, gap_iobt, gap_eobt, gap_aobt (take-off minus clock; NaN missing); flag: 0/1/2 or nan"""
    d = {k: pv - gap[k] for k in ("sched", "lobt", "iobt", "eobt", "aobt")}
    gs = gap["sched"]
    clamp = np.where((gs >= 5000) & (gs < 50000), gs - pv, np.nan)
    wmin = 3606.0 - np.abs(d["lobt"])
    ind = np.where(flag == 1, d["aobt"], np.nan)
    der = np.where(flag == 2, d["aobt"], np.nan)
    return np.column_stack([d["sched"], d["lobt"], d["iobt"], d["eobt"], d["aobt"], clamp, wmin, ind, der]).astype(np.float32)


def gaps(X, cols):
    g = lambda n: X[:, cols.index(n)].astype(np.float64)  # noqa: E731
    return dict(sched=g("gap_sched"), lobt=g("gap_lobt"), iobt=g("gap_lobt") - g("clk_iobt_minus_lobt"), eobt=g("gap_eobt"), aobt=g("gap_aobt")), g("pv")


z = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True)
cols = [str(c) for c in z["cols"]]
X = z["X"]
mon = z["mon"]
apt = np.array([str(a) for a in z["apt"]])
ids = z["ids"].astype(float)
gap, pv = gaps(X, cols)
assert np.allclose(pv, z["pv"], atol=1e-3), "pv column is not the fold base"
fl = pl.read_parquet(I + "agent_aobtflag_flags.parquet").filter(pl.col("yr") == 2025).select("id", "flag")
fm = dict(zip(fl["id"].to_list(), fl["flag"].to_list()))
flag = np.array([fm.get(v, np.nan) for v in ids.tolist()], float)
P("fold flag coverage", np.isfinite(flag).mean(), "derived share", np.nanmean(flag == 2))
D = build(pv, gap, flag)
# shuffled-prediction controls: permute pv within airport-month (3 draws)
Dshuf = np.empty((3,) + D.shape, np.float32)
grp = pd.Series(apt) + "_" + pd.Series(mon.astype(str))
gi = [np.where(grp.values == g_)[0] for g_ in grp.unique()]
for s in range(3):
    rng = np.random.default_rng(1000 + s)
    pvs = pv.copy()
    for ii in gi:
        pvs[ii] = pv[rng.permutation(ii)]
    Dshuf[s] = build(pvs, gap, flag)
Dnoise = np.stack([np.random.default_rng(2000 + s).normal(0, 1000, D.shape).astype(np.float32) for s in range(3)])
# serve rows
si = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True)
scol = [str(c) for c in si["cols_pre"]]
gs_, pvs_ = gaps(si["Xs_pre"], scol)
fs = pl.read_parquet(I + "agent_aobtflag_flags.parquet").filter(pl.col("yr") == 2026).select("id", "flag")
fsm = dict(zip(fs["id"].to_list(), fs["flag"].to_list()))
flag_s = np.array([fsm.get(v, np.nan) for v in si["ids"].astype(float).tolist()], float)
P("serve flag coverage", np.isfinite(flag_s).mean())
Ds = build(pvs_, gs_, flag_s)
np.savez(I + "agent_pcd_cols.npz", D=D, Dshuf=Dshuf, Dnoise=Dnoise, Ds=Ds, names=np.array(NAMES), ids=ids)
del Dshuf, Dnoise, gi, grp, gap, gs_, flag, fm, fsm, z
P("saved; fold NaN share per column", np.round(np.isnan(D).mean(0), 3).tolist())
P("serve NaN share per column", np.round(np.isnan(Ds).mean(0), 3).tolist())

# ---- drift: same month, within airport, fold rows (2025, out-of-fold base) against served 2026 rows; row-weighted symmetric AUC
mon_s = si["mon_s"]
apt_s = np.array([str(a) for a in si["apt"]])
is9 = si["is9"].astype(bool)
Xs = si["Xs_pre"]
ref = {n: (X[:, cols.index(n)].copy(), Xs[:, scol.index(n)].copy()) for n in ["pv", "gap_sched", "gap_lobt", "gap_eobt", "gap_aobt", "lobt_minus_aobt"]}
del X, Xs
allc = {n: (D[:, i], Ds[:, i]) for i, n in enumerate(NAMES)}
allc.update(ref)


def auc(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return np.nan
    r = pd.Series(np.concatenate([a, b])).rank().values
    u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    v = u / (len(a) * len(b))
    return max(v, 1 - v)


rows = []
for n, (f, s) in allc.items():
    out = {"col": n}
    for m in (1, 7):
        num = den = 0.0
        for a in sorted(set(apt.tolist())):
            sel_f = (mon == m) & (apt == a)
            sel_s = (mon_s == m) & is9 & (apt_s == a)
            if sel_s.sum() < 50:
                continue
            v = auc(f[sel_f], s[sel_s])
            w = sel_s.sum()
            if np.isfinite(v):
                num += v * w
                den += w
        out["auc_m%d" % m] = num / den if den else np.nan
    # NaN-rate shift
    out["nan_f"] = float(np.isnan(f[np.isin(mon, (1, 7))]).mean())
    out["nan_s"] = float(np.isnan(s[np.isin(mon_s, (1, 7)) & is9]).mean())
    rows.append(out)
    P(out)
pd.DataFrame(rows).to_csv(I + "agent_pcd_auc.csv", index=False)
