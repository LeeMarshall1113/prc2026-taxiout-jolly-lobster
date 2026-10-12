"""Post-process a submission: a schedule-copy gate for Rome's flight-plan rows.

    .venv/Scripts/python.exe archive/apply_rome_plan_gate.py --in submissions/X.parquet --out submissions/Y.parquet

About 20% of Rome's flight-plan departures have their off-block copied from the schedule
(y = MVT - SCHED exactly). The global model predicts one conditional mean over copies and
genuine taxis. This trains, on all twelve 2025 months of Rome plan rows and every model feature:
  clf   LightGBM classifier for copy (|BLOCK - SCHED| <= 60s)
  reg   LightGBM regressor for the taxi, fitted on non-copy rows only
  mix   P(copy) * gap + (1 - P(copy)) * reg, then linearly calibrated (a + b * mix) on a
        random 20% of rows held out from both learners
and replaces the prediction of every 2026 Rome plan row with schedule gap <= 7,200s by
    0.5 * prediction + 0.5 * calibrated mix.

EVIDENCE (scratch directory rome_plan_gate*.py). Uncapped, the half blend failed its rule: fold (1,7)
+0.25s, all of it from 27 rows with gaps over 20,000s where a wrong copy call costs P * gap.
Capped at 7,200s -- copies are ~1% above two hours -- per fold: (1,7) -0.53s, (2,8) -1.19s,
(3,9) -1.15s. The cap was chosen after seeing the band table, so it was confirmed on a fold
never looked at, (5,11) with its older control model: -0.35s. Rule fixed before that run.
"""
import argparse
import sys

sys.path.insert(0, ".")

import lightgbm as lgb
import numpy as np
import polars as pl

from prc.config import RAW_DIR
from prc.features import CATEGORICAL, FEATURES, TARGET, build

C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
GAP_CAP = 7200.0

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gap-cap", type=float, default=7200.0,
                help="apply the gate to LIRF plan rows with schedule gap <= this. "
                     "7200 is SHIPPED but was chosen against RAW model predictions with "
                     "no gap clamp and no LOBT box; the chain now bounds the rows that "
                     "objection rested on. See the 2026-09-20 23:00 ledger entry.")
ap.add_argument("--blend-weight", type=float, default=0.5,
                help="weight on the calibrated mixture: pred*(1-w) + mix*w. 0.5 is SHIPPED; the out-of-fold least-squares weight is 0.724 and 0.60 is the concentration-safe value.")
ap.add_argument("--seed", type=int, default=None,
                help="bagging seed for the hold/fit split and both LightGBM learners. Unset = SHIPPED behaviour, bit-identical (rng 0, LightGBM default seed).")
ap.add_argument("--copy-threshold", type=float, default=60.0,
                help="|BLOCK-SCHED| <= this counts as a copy for the classifier and the "
                     "regressor exclusion. 60 is SHIPPED and measured optimal; 6 is the "
                     "true substituted class. See the 2026-09-20 ledger entry.")
args = ap.parse_args()

feat = pl.scan_parquet("data/interim/train_features.parquet")
use = [f for f in FEATURES if f in set(feat.collect_schema().names()) and f != TARGET]
cats = [c for c in use if c in CATEGORICAL]
raw = (pl.scan_parquet("data/raw/training_*.parquet")
       .filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_not_null())
       .select(C(ID), (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap_s"),
               ((C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= args.copy_threshold).alias("copy"))
       .collect())
tr = (feat.filter(C(ID).is_in(raw[ID].implode())).select([ID, TARGET, *use]).collect().join(raw, on=ID))


def pandas(frame):
    return frame.with_columns([C(c).cast(pl.Utf8).cast(pl.Categorical) for c in cats]).select(use).to_pandas()


params = dict(learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1)
rng = np.random.default_rng(0 if args.seed is None else args.seed)
if args.seed is not None:
    params = dict(params, random_state=args.seed)
idx = rng.permutation(tr.height)
hold, fit = idx[: tr.height // 5], idx[tr.height // 5:]
Xall = pandas(tr)
copy = tr["copy"].to_numpy()
y = tr[TARGET].to_numpy().astype(float)
gap = tr["gap_s"].to_numpy()

# calibration of the mixture on the held-out 20%
clf_h = lgb.LGBMClassifier(n_estimators=400, **params).fit(Xall.iloc[fit], copy[fit])
nfit = np.intersect1d(fit, np.flatnonzero(~copy))
reg_h = lgb.LGBMRegressor(n_estimators=600, **params).fit(Xall.iloc[nfit], np.clip(y[nfit], 0, 7200))
ph = clf_h.predict_proba(Xall.iloc[hold])[:, 1]
mix_h = ph * gap[hold] + (1 - ph) * reg_h.predict(Xall.iloc[hold])
coef = np.linalg.lstsq(np.column_stack([np.ones(len(hold)), mix_h]), np.clip(y[hold], -3600, 20000), rcond=None)[0]
# final learners on every 2025 Rome plan row
clf = lgb.LGBMClassifier(n_estimators=400, **params).fit(Xall, copy)
reg = lgb.LGBMRegressor(n_estimators=600, **params).fit(Xall[~copy], np.clip(y[~copy], 0, 7200))
print(f"copy threshold {args.copy_threshold:g}s")
print(f"trained on {tr.height:,} Rome plan rows (copy {copy.mean():.1%}); calibration a={coef[0]:+.1f} b={coef[1]:.3f}")

rank = build(pl.read_parquet(RAW_DIR / "ranking.parquet"), with_target=False)
rk_raw = (pl.scan_parquet(RAW_DIR / "ranking.parquet")
          .filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_not_null())
          .select(C(ID), (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap_s")).collect())
te = rank.join(rk_raw, on=ID, how="inner").filter(C("gap_s") <= args.gap_cap)
Xte = pandas(te)
pc = clf.predict_proba(Xte)[:, 1]
mix = coef[0] + coef[1] * (pc * te["gap_s"].to_numpy() + (1 - pc) * reg.predict(Xte))

sub = pl.read_parquet(args.inp)
j = sub.join(te.select(ID).with_columns(pl.Series("mix", mix)), on=ID, how="left", maintain_order="left")
if j.height != sub.height:
    raise SystemExit("join changed the row count")
p = j[T].to_numpy().astype(float)
m = j["mix"].to_numpy().astype(float)
has = ~np.isnan(m)
w = args.blend_weight
new = np.where(has, np.clip((1.0 - w) * p + w * m, 0, None), p)
d = (new - p)[has]
print(f"gap cap {args.gap_cap:,.0f}s, blend weight {w:.2f}")
print(f"Rome plan rows with gap <= {args.gap_cap:,.0f}s: {int(has.sum()):,}; mean P(copy) {pc.mean():.3f}; "
      f"move median {np.median(d):+.0f}s, mean {d.mean():+.0f}s, median size {np.median(np.abs(d)):.0f}s, p99 size {np.percentile(np.abs(d), 99):.0f}s")
if has.sum() < 15000 or has.sum() > 30000:
    raise SystemExit(f"refusing: {int(has.sum())} Rome plan rows in scope, expected ~20,000-25,000")
sub.with_columns(pl.Series(T, np.rint(new).astype(np.int64)).cast(sub[T].dtype)).write_parquet(args.out)
print(f"wrote {args.out}")
