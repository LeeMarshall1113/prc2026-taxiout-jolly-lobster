"""Post-process a submission: never predict above the schedule gap where truth cannot exceed it.

    .venv/Scripts/python.exe archive/apply_gap_clamp.py --in submissions/X.parquet --out submissions/Y.parquet

RULE. For every departure with 5,000s <= gap < 50,000s (gap = MVT - SCHED):
    prediction = min(prediction, gap)

WHY IT IS SAFE. Below the day-early regime a departure record is one of two things: a schedule
copy, BLOCK = SCHED, so y = gap exactly; or an ordinary taxi, far below a gap of 5,000s or
more. Either way truth <= gap, so lowering a prediction that sits above the gap down to the gap
can only shrink its error. Day-early records (y ~ 86,400 + taxi) need gaps over ~50,000s by
their mechanism -- the off-block time of day is stamped on the date nearest the schedule, which
lands a day early only for delays past ~13.7 hours -- and apply_rome_dayearly.py owns that band.

EVIDENCE (2025, scratch directory gap_clamp.py, pass rule fixed before the run: premise on >= 99.9% of
rows in scope and no fold worse):
    premise   truth > gap + 60s on 58 of 93,993 rows with gap 5,000-50,000s (0.062%), all but
              one by a few hundred seconds
    folds     n_feat (1,7) -0.60s, n_feat (2,8) -0.70s, n3_bulk (3,9) -5.58s; 96 rows moved
    2026      v18 had 29 predictions above the gap in scope (28 Rome, 1 Heathrow), squared
              excess 6.1e7 s^2: at least -0.33s on the board if every one is a copy, more if not

The model lands above the gap because the Rome no-plan gate blends a copy (y = gap) with the
airport's day-early and long-delay tail, and the no-plan model extrapolates on rare huge gaps.
"""
import argparse

import numpy as np
import polars as pl

C = pl.col
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
LO, HI = 5000.0, 50000.0
MAX_CHANGED = 200  # a submission-wide drift here would mean the rule is misapplied

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

sub = pl.read_parquet(args.inp)
gap = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C(ID), C("ADEP_mvt").alias("apt"),
               (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap"))
       .collect())
j = sub.join(gap, on=ID, how="left", maintain_order="left")
if j.height != sub.height:
    raise SystemExit(f"join changed the row count: {j.height:,} vs {sub.height:,}")

p = j[TARGET].to_numpy().astype(float)
g = j["gap"].fill_null(-1.0).to_numpy().astype(float)
use = (g >= LO) & (g < HI) & (p > g)
new = np.where(use, np.floor(g), p)
n = int(use.sum())
if n > MAX_CHANGED:
    raise SystemExit(f"refusing: {n} rows would change, more than {MAX_CHANGED}")
excess = (p - np.floor(g))[use]
by_apt = j.filter(pl.Series(use)).group_by("apt").len().sort("apt")
print(f"gap clamp {LO:,.0f}-{HI:,.0f}s: lowered {n} predictions to their schedule gap; excess removed "
      f"total {excess.sum():,.0f}s, squared {float((excess ** 2).sum()):,.0f} s^2; by airport "
      + ", ".join(f"{a} {k}" for a, k in by_apt.iter_rows()))
out = sub.with_columns(pl.Series(TARGET, np.rint(new).astype(np.int64)).cast(sub[TARGET].dtype))
out.write_parquet(args.out)
print(f"wrote {args.out}")
