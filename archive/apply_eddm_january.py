"""Munich, January 2026: shrink the model's year-over-year jump on the rows where we can measure it.

    .venv/Scripts/python.exe archive/apply_eddm_january.py --in X.parquet --out Y.parquet

WHAT THIS CORRECTS. Our model trains on 2025 and predicts 2026, so a year-over-year level change is
invisible to every fold we run. Three instruments say Munich's January is over-predicted:
  * EUROCONTROL's published monthly taxi-out (open data), after fitting their coverage effect on 2025
    (60s per 10 points of coverage lost): -76s;
  * our own 2026 ADS-B estimates, bias-calibrated on 2025 and independent of EUROCONTROL: -16s, with a
    selection bias toward short taxis that makes it a floor rather than a point estimate;
  * the flight plan's own taxi proxy (MVT - AOBT_3), measured the same way in both years: our predicted
    shift is +78s on small-gap rows where Munich's proxy moved +2s, an excess of +76s.
SCOPE. EDDM, January 2026, schedule gap < 1,800s -- the population the clean measurement was made on.
Large-gap rows are excluded because there a schedule copy's y IS the gap, and January 2026's gaps are
348s larger than January 2025's, which inflates the true mean legitimately.
SIZE. Half the measured 76s. The instruments agree on sign and disagree on magnitude (76 vs 16), and
the arithmetic of a level shift is forgiving in one direction only: correcting c against a true bias b
gains 2bc - c^2, so c = 38s gains about -0.18s if b = 76, roughly nothing if b = 16, and costs about
+0.06s if there is no bias at all.
NOT FOLD-VALIDATED, and it cannot be: the effect is 2026-specific. That is why it is shrunk, scoped to
one airport-month, and applied to 7,062 rows rather than a population where it could do real damage.
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
SHIFT, GAP_CUT, APT, MONTH = -38.0, 1800.0, "EDDM", 1
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
scored = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
          .select(C(ID), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
                  (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs")).collect())
hit = set(scored.filter((C("apt") == APT) & (C("mo") == MONTH) & (C("gs") < GAP_CUT))[ID].to_list())
sub = pl.read_parquet(args.inp)
old = sub[TARGET].to_numpy().astype(float)
new = np.array([max(v + SHIFT, 0.0) if m in hit else v for m, v in zip(sub[ID].to_list(), old)])
ch = new != old
print(f"{APT} month {MONTH}, gap < {GAP_CUT:,.0f}s: {len(hit):,} scored rows in scope, {int(ch.sum()):,} moved by {SHIFT:+.0f}s")
print(f"  mean prediction there {old[[m in hit for m in sub[ID].to_list()]].mean():,.0f}s -> {new[[m in hit for m in sub[ID].to_list()]].mean():,.0f}s")
out = sub.with_columns(pl.Series(TARGET, np.rint(new)).cast(sub.schema[TARGET]))
assert out.height == sub.height and out[ID].equals(sub[ID])
out.write_parquet(args.out)
print(f"wrote {args.out}")
