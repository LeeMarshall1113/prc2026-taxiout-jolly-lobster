"""Per-airport minimum-taxi floor, applied on top of a built submission.

    .venv/Scripts/python.exe archive/apply_min_floor.py --in X.parquet --out Y.parquet

pred = max(pred, q01(airport)), q01 = the 1st percentile of y among 2025's NON-copy departures at that
airport; for plan rows the floor is capped at the LOBT box edge (MVT - LOBT + 3,606s) so it can never
push a prediction outside a bound we know holds. The chain otherwise floors only at zero, which leaves
a few hundred scored rows below any taxi ever observed at their airport.
MEASURED (archive/screen_rule.py, out-of-fold floors): folds -0.107 / -0.044 / -0.061 / -0.024 /
-0.018, pooled -0.055s, -0.032s after deleting the 20 rows it helps most. Every fold negative with a
mechanism, but BELOW the project's -0.10s pooled bar for adopting a rule: ship only as a disclosed
rider on a larger change, never on its own.
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, TARGET, W = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 3606.0
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

gen = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C("ADEP_mvt").alias("apt"),
               (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
               (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y"))
       .collect().filter((C("y") - C("gs")).abs() > 60))
floors = dict(gen.group_by("apt").agg(C("y").quantile(0.01).alias("f")).sort("apt").iter_rows())
print("floors: " + ", ".join(f"{a} {f:.0f}s" for a, f in sorted(floors.items())))
scored = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
          .select(C(ID), C("ADEP_mvt").alias("apt"), C("AOBT_3_flt").is_not_null().alias("plan"),
                  (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl")).collect())
sub = pl.read_parquet(args.inp)
j = sub.select(ID).join(scored, on=ID, how="left")
f = np.array([floors.get(a, 0.0) for a in j["apt"].to_list()], float)
gl = j["gl"].fill_null(np.nan).to_numpy().astype(float)
box = j["plan"].fill_null(False).to_numpy() & np.isfinite(gl)
f = np.where(box, np.minimum(f, gl + W), f)
old = sub[TARGET].to_numpy().astype(float)
new = np.maximum(old, np.clip(f, 0, None))
ch = new != old
print(f"minimum-taxi floor: {int(ch.sum()):,} of {len(old):,} scored rows raised")
if ch.any():
    mv = (new - old)[ch]
    print(f"  moves: median {np.median(mv):+,.0f}s, mean {mv.mean():+,.0f}s, max {mv.max():+,.0f}s")
out = sub.with_columns(pl.Series(TARGET, np.rint(new)).cast(sub.schema[TARGET]))
assert out.height == sub.height and out[ID].equals(sub[ID])
out.write_parquet(args.out)
print(f"wrote {args.out}")
