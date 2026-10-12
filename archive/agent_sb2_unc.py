"""experiment sb2 unc (docs/LAB-NOTES.md, C4): predictive-uncertainty width unc_w = q90 - q10 of the
re-stack target y - fin, from two quantile LightGBM heads on the recipe evt,trk matrix (est_v2 base + evt + trk).

Fold (leak-free, NESTED over the re-stack's own 10 day-folds, per fin base):
    python archive/agent_sb2_unc.py --fin-npz data/interim/modal/TX_S6.npz --out data/interim/agent_sb2_unc_TX.npz
  -> ids (the c238 row order) and w0..w9: for outer fold k, rows of fold j != k carry the width from heads fitted on folds
     not in {j, k}; rows of fold k carry the width from heads fitted on folds != k.
Serving helpers (imported by scripts/agent_sb2_{bx,}serve.py): oof_widths() (10-fold OOF on the training rows) and
full_heads() (heads on all training rows).
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import lightgbm as lgb
import numpy as np
import polars as pl

import archive.agent_sb2_restack as R

HP = dict(n_estimators=200, learning_rate=0.08, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=0)
ALPHAS = (0.1, 0.9)
RC = "evt,trk"
EVT, TRK = "data/interim/agent_restack3_evt_2025.parquet", "data/interim/agent_restack3_trk_2025.parquet"


def heads(X, t):
    return [lgb.LGBMRegressor(objective="quantile", alpha=a, **HP).fit(X, t, categorical_feature=[0]) for a in ALPHAS]


def width(hs, X):
    return hs[1].predict(X) - hs[0].predict(X)


def day_folds(days, k=10):
    ud = sorted(set(days))
    f = {d: i % k for i, d in enumerate(ud)}
    return np.array([f[d] for d in days])


def oof_widths(X, t, days):
    fo = day_folds(days)
    w = np.full(len(t), np.nan)
    for k in range(10):
        te = fo == k
        w[te] = width(heads(X[~te], t[~te]), X[te])
    return w


def full_heads(X, t):
    return heads(X, t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fin-npz", required=True)
    ap.add_argument("--rows", default="data/interim/agent_offblock_rows_c238.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T0 = time.time()
    z = np.load(a.fin_npz, allow_pickle=True)
    pos_of = {k: i for i, k in enumerate(z["ids"].astype(float))}
    r = R.load_rows(a.rows, RC, EVT, TRK)
    pos = np.array([pos_of.get(m, -1) for m in r["MVT_ID_mvt"].to_list()])
    assert (pos >= 0).all()
    fin = z["both_guard"][pos].astype(float)
    t = z["y"][pos].astype(float) - fin
    days = z["days"][pos].astype(str)
    assert (days == np.asarray(r["day"].to_list())).all()
    X = R.features(r, fin, RC)
    fo = day_folds(days)
    print(f"rows {len(t):,}; X {X.shape}", flush=True)
    inner = {}  # (j, k) with j != k: width on fold j from heads fitted on folds not in {j, k}
    for k in range(10):
        for j in range(10):
            if j == k:
                continue
            tr = (fo != k) & (fo != j)
            inner[(j, k)] = width(heads(X[tr], t[tr]), X[fo == j])
        print(f"  outer {k}: inner done ({time.time() - T0:.0f}s)", flush=True)
    out = {"ids": r["MVT_ID_mvt"].to_numpy()}
    for k in range(10):
        w = np.full(len(t), np.nan)
        te = fo == k
        w[te] = width(heads(X[~te], t[~te]), X[te])
        for j in range(10):
            if j != k:
                w[fo == j] = inner[(j, k)]
        out[f"w{k}"] = w
        print(f"  outer {k}: width mean {np.nanmean(w):.1f} s, test-fold mean {np.nanmean(w[te]):.1f} s "
              f"({time.time() - T0:.0f}s)", flush=True)
    np.savez(a.out, **out)
    print(f"-> {a.out} ({time.time() - T0:.0f}s)")
