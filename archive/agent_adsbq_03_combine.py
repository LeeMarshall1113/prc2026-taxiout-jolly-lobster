"""experiment adsbq 03: can a better estimate (multi-anchor combination, per-stand offset) cut the ADS-B error,
and does the cut survive production's second stage?

    .venv/Scripts/python.exe archive/agent_adsbq_03_combine.py [--trees 300]

Reads data/interim/agent_adsbq_rows.parquet (agent_adsbq_01_dump.py). Plan rows with a production
estimate kind, everything leave-one-day-out:

  E0  production: taxi - median bias per (airport, kind)
  E1  + per-(airport, kind, stand) offset, shrunk to the airport offset (k rows of prior weight)
  E2  per-(airport, kind) least squares on the other days: y ~ taxi + sum_k mask_k * (alt_k - taxi)
      over alternative anchors from the same track (tight radii, interpolated crossing, run start)
  E3  E2 fitted on the stand-offset-corrected target (E1 + E2)

For each: the estimate's own error variance (winsorised) and the inverse-variance blend it implies,
then a replica of the production second stage (scripts/adsb_apply.stack_features, --stack-feats
quality: x - p, nm - p, p, x, airport, kind, hour, QUALITY, stand, type; LightGBM 300/15/depth 4,
lr 0.05, min_child 50, target clip +-3600, correction clip +-1800) with x taken from each estimate.
Fold scale: ~45 days' rows train each day's stack, production trains on 177 days.
"""
import argparse
import sys

import numpy as np
import polars as pl

sys.path.insert(0, ".")
from prc.adsb_match import QUALITY  # noqa: E402

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--trees", type=int, default=300)
ap.add_argument("--shrink", type=float, default=20.0, help="prior rows for the per-stand offset")
ap.add_argument("--folds", type=int, default=0, help="0 = leave-one-day-out; else grouped-by-day folds")
ap.add_argument("--save", default="data/interim/agent_adsbq_03_preds.parquet")
args = ap.parse_args()

df = (pl.read_parquet("data/interim/agent_adsbq_rows.parquet")
        .filter(C("plan") & C("kind").is_not_null() & C("nm_taxi").is_finite() & C("y_true").is_finite() & C("model").is_finite())
        .with_columns(C("model").clip(lower_bound=0).alias("p")))
days = sorted(df["day"].unique().to_list())
if args.folds:
    fold_of = {d: i % args.folds for i, d in enumerate(days)}
else:
    fold_of = {d: i for i, d in enumerate(days)}
df = df.with_columns(C("day").replace_strict(fold_of, return_dtype=pl.Int32).alias("fold"))
NF = max(fold_of.values()) + 1
print(f"plan rows with a production kind: {df.height:,} on {len(days)} days, {NF} folds")

ALTS = ["ci_20", "ci_30", "ci_50", "ci_75", "ci_100", "ci_150", "ci_300", "appear_raw"]
# consistency gates on (alt - taxi): a tighter circle is left EARLIER than the 150 m one (alt >= taxi), and
# by little; a wider one later. Outside the gate the anchor is some other visit to the stand, not this one.
GATE = {"ci_20": (-60, 600), "ci_30": (-60, 600), "ci_50": (-60, 600), "ci_75": (-60, 600), "ci_100": (-60, 600),
        "ci_150": (-120, 120), "ci_300": (-900, 60), "appear_raw": (-60, 1800)}
APTS = sorted(df["apt"].unique().to_list())
KINDS = ("stand", "appear", "start", "tight")
STANDS = {k: i for i, k in enumerate(sorted({(a, s) for a, s in zip(df["apt"].to_list(), df["stand"].to_list()) if s}))}
TYPES = {k: i for i, k in enumerate(sorted({t for t in df["type"].to_list() if t}))}


def wvar(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = np.percentile(x, [1, 99])
    return float(np.var(np.clip(x, lo, hi)))


y = df["y_true"].to_numpy()
taxi = df["taxi"].to_numpy()
p = df["p"].to_numpy()
fold = df["fold"].to_numpy()
akey = np.array([f"{a}|{k}" for a, k in zip(df["apt"].to_list(), df["kind"].to_list())])
skey = np.array([f"{a}|{k}|{s}" for a, k, s in zip(df["apt"].to_list(), df["kind"].to_list(), df["stand"].to_list())])
A = {}
for c in ALTS:
    v = df[c].to_numpy().astype(float)
    d = v - taxi
    A[c] = np.where(np.isfinite(d) & (d >= GATE[c][0]) & (d <= GATE[c][1]), v, np.nan)
print("gated anchor availability: " + ", ".join(f"{c} {100 * np.isfinite(A[c]).mean():.0f}%" for c in ALTS))

E = {name: np.full(len(y), np.nan) for name in ("E0", "E1", "E2", "E3")}
for f in range(NF):
    te = fold == f
    tr = ~te
    err = taxi - y
    # E0: per (airport, kind) median bias
    b0 = {}
    for k in np.unique(akey[tr]):
        m = tr & (akey == k)
        if m.sum() >= 30:
            b0[k] = float(np.median(err[m]))
    base_b = np.array([b0.get(k, np.nan) for k in akey])
    E["E0"][te] = (taxi - base_b)[te]
    # E1: per-stand offset, shrunk: b_s = (n * med_s + k * b_a) / (n + k), medians of the other days
    e_tr = err - base_b
    st = pl.DataFrame({"s": skey[tr], "e": e_tr[tr]}).filter(C("e").is_finite()).group_by("s").agg(
        C("e").median().alias("m"), pl.len().alias("n"))
    so = {s: (n * m_) / (n + args.shrink) for s, m_, n in st.iter_rows()}
    stand_b = np.array([so.get(s, 0.0) for s in skey])
    E["E1"][te] = (taxi - base_b - stand_b)[te]
    # E2 / E3: per (airport, kind) least squares of the target on anchors, winsorised target
    for tgt_name, off in (("E2", np.zeros(len(y))), ("E3", stand_b)):
        for k in np.unique(akey):
            mtr, mte = tr & (akey == k), te & (akey == k)
            if mtr.sum() < 200 or not mte.any():
                E[tgt_name][mte] = (taxi - base_b - off)[mte]
                continue
            cols = [np.ones(len(y)), taxi]
            for c in ALTS:
                ok = np.isfinite(A[c])
                cols.append(np.where(ok, A[c] - taxi, 0.0))
                cols.append(ok.astype(float))
            X = np.column_stack(cols)
            tgt = y + off                              # the anchor combination predicts y + stand offset
            r0 = tgt[mtr] - (taxi[mtr] - base_b[mtr])
            lo, hi = np.percentile(r0, [1, 99])
            tclip = (taxi[mtr] - base_b[mtr]) + np.clip(r0, lo, hi)
            beta, *_ = np.linalg.lstsq(X[mtr], tclip, rcond=None)
            E[tgt_name][mte] = (X[mte] @ beta) - off[mte]

print("\nESTIMATE ERROR (winsorised SD of estimate - y, leave-one-day-out), and the inverse-variance blend it implies")
print(f"  {'cell':<14}{'n':>7}{'model':>7}" + "".join(f"{n:>7}" for n in E) + "   blend RMSE " + " ".join(f"{n:>6}" for n in E))
vm_all = {}
saved_blend = {n: 0.0 for n in E}
for k in sorted(np.unique(akey)):
    m = (akey == k) & np.isfinite(E["E0"])
    if m.sum() < 100:
        continue
    vm = wvar(p[m] - y[m])
    vm_all[k] = vm
    cells, bl = [], []
    for n, e in E.items():
        va = wvar(e[m] - y[m])
        cells.append(f"{np.sqrt(va):>7.0f}")
        w = vm / (vm + va)
        pb = w * np.clip(e[m], 0, None) + (1 - w) * p[m]
        bl.append(float(np.sqrt(((pb - y[m]) ** 2).mean())))
        saved_blend[n] += float(((pb - y[m]) ** 2).sum())
    print(f"  {k:<14}{m.sum():>7,}{np.sqrt(vm):>7.0f}" + "".join(cells) + "   " + " ".join(f"{b:>6.0f}" for b in bl))
m_all = np.isfinite(E["E0"])
print("  pooled estimate wSD: " + ", ".join(f"{n} {np.sqrt(wvar(e[m_all] - y[m_all])):.1f}" for n, e in E.items())
      + f";  pooled variance cut vs E0: " + ", ".join(f"{n} {100 * (1 - wvar(e[m_all] - y[m_all]) / wvar(E['E0'][m_all] - y[m_all])):+.1f}%" for n, e in E.items()))
print("  inverse-variance blend SSE vs E0 (per corrected row, s^2): " + ", ".join(
    f"{n} {(saved_blend[n] - saved_blend['E0']) / m_all.sum():+.0f}" for n in E))

# ---------- the production second stage, replicated ----------
import lightgbm as lgb  # noqa: E402

nm = df["nm_taxi"].to_numpy().astype(float)
apt_i = np.array([APTS.index(a) for a in df["apt"].to_list()], float)
kind_i = np.array([KINDS.index(k) for k in df["kind"].to_list()], float)
hour = df["hour"].to_numpy().astype(float)
Q = [df[q].to_numpy().astype(float) for q in QUALITY]
stand_i = np.array([STANDS.get((a, s), -1) for a, s in zip(df["apt"].to_list(), df["stand"].to_list())], float)
type_i = np.array([TYPES.get(t, -1) for t in df["type"].to_list()], float)
CATS = [4, 5, 7 + len(QUALITY), 8 + len(QUALITY)]


def feats(x, extra=()):
    x = np.clip(x, 0, None)
    return np.column_stack([x - p, nm - p, p, x, apt_i, kind_i, hour, *Q, stand_i, type_i, *extra])


EXTRA_ANCH = [np.where(np.isfinite(A[c]), A[c] - p, np.nan) for c in ALTS] + \
             [df[c].to_numpy().astype(float) for c in ("gp_20", "gp_150", "t_in150", "dens", "start_d")]
arms = {"S0 production (E0)": feats(E["E0"]),
        "S1 x = E1 (stand offset)": feats(E["E1"]),
        "S2 x = E2 (anchor combo)": feats(E["E2"]),
        "S3 x = E3 (both)": feats(E["E3"]),
        "S4 E0 + raw anchors as features": feats(E["E0"], EXTRA_ANCH),
        "S5 E3 + raw anchors as features": feats(E["E3"], EXTRA_ANCH)}
target = np.clip(y - p, -3600, 3600)
base_sse = ((p - y) ** 2)
PRED = {}
print(f"\nSECOND STAGE replica, leave-one-day-out, {m_all.sum():,} rows; model-only RMSE on them {np.sqrt(base_sse[m_all].mean()):.1f}s")
for name, X in arms.items():
    out = p.copy()
    for f in range(NF):
        te, tr = (fold == f) & m_all, (fold != f) & m_all
        if not te.any():
            continue
        mdl = lgb.LGBMRegressor(n_estimators=args.trees, max_depth=4, num_leaves=15, learning_rate=0.05, min_child_samples=50,
                                verbose=-1, n_jobs=4).fit(X[tr], target[tr], categorical_feature=CATS)
        out[te] = np.maximum(p[te] + np.clip(mdl.predict(X[te]), -1800, 1800), 0.0)
    PRED[name] = out
    sse = (out - y) ** 2
    by = "  ".join(f"{a} {np.sqrt(sse[m_all & (apt_i == i)].mean()):.1f}" for i, a in enumerate(APTS) if (m_all & (apt_i == i)).sum())
    ref = PRED["S0 production (E0)"]
    dsse = float((sse[m_all] - ((ref - y) ** 2)[m_all]).sum())
    dayd = []
    for d in days:
        md = m_all & (df["day"].to_numpy() == d)
        dayd.append(float((sse[md] - ((ref - y) ** 2)[md]).sum()))
    dayd = np.array(dayd)
    top50 = np.sort(sse[m_all] - ((ref - y) ** 2)[m_all])[:50].sum()
    print(f"  {name:<34} RMSE {np.sqrt(sse[m_all].mean()):7.2f}  dSSE/row vs S0 {dsse / m_all.sum():+8.0f} s^2  "
          f"days better {int((dayd < 0).sum())}/{len(days)}  minus-top-50 {(dsse - top50) / m_all.sum():+8.0f}")
    print(f"      by airport RMSE: {by}")
    for i, a in enumerate(APTS):
        ma = m_all & (apt_i == i)
        if ma.sum():
            print(f"      {a} dSSE/row vs S0 {float((sse[ma] - ((ref - y) ** 2)[ma]).sum()) / ma.sum():+8.0f}  n={ma.sum():,}", end="")
    print()
if args.save:
    pl.DataFrame({"MVT_ID_mvt": df["MVT_ID_mvt"], "day": df["day"], "apt": df["apt"], "kind": df["kind"], "y": y, "p": p,
                  **{n: E[n] for n in E}, **{f"stk{i}": v for i, v in enumerate(PRED.values())}}).write_parquet(args.save)
    print(f"saved {args.save}")
