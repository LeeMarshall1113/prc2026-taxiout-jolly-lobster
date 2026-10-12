"""What did the 295-day calibration actually change? Run before recommending an upload.

The day curve says the bigger calibration base is better on a held-out ten days. That is the
evidence. This is the plausibility check on top of it: a refined calibration should move many rows
by a little, roughly centred, concentrated on ADS-B-matched rows, and it should not reintroduce any
bound violation that the chain enforces. A large one-sided shift would mean something other than
calibration changed, and that is the shape that has cost this project a board second before.

    .venv/Scripts/python.exe archive/compare_v39_v40.py [new_submission.parquet]
"""
import sys

import numpy as np
import polars as pl

C = pl.col
ID, W = "MVT_ID_mvt", 3606.0
NEW = sys.argv[1] if len(sys.argv) > 1 else "submissions/jolly-lobster_v40.parquet"
OLD = "submissions/jolly-lobster_v39.parquet"

a = pl.read_parquet(OLD).rename({"TAXITIME_SEC_mvt": "old"})
b = pl.read_parquet(NEW).rename({"TAXITIME_SEC_mvt": "new"})
r = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
     .select(C(ID), C("ADEP_mvt").alias("apt"), C("AOBT_3_flt").is_not_null().alias("plan"),
             (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
             (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl")).collect())
d = a.join(b, on=ID, how="inner").join(r, on=ID, how="left")
print(f"{OLD} vs {NEW}: {d.height:,} rows joined "
      f"(v39 {a.height:,}, new {b.height:,})")
if d.height != a.height:
    print("  WARNING: the two files do not cover the same rows")

o = d["old"].to_numpy().astype(float)
n = d["new"].to_numpy().astype(float)
delta = n - o
moved = np.abs(delta) > 0.5
print(f"\nrows moved: {int(moved.sum()):,} ({100 * moved.mean():.1f}%)")
print(f"  mean shift over ALL rows   {delta.mean():+8.2f}s")
print(f"  mean shift over moved rows {delta[moved].mean():+8.2f}s")
print(f"  median |shift| on moved    {np.median(np.abs(delta[moved])):8.2f}s")
for q in (1, 25, 50, 75, 99):
    print(f"  p{q:<3} of the shift          {np.percentile(delta[moved], q):+8.2f}s")
print(f"  max absolute shift         {np.abs(delta).max():8.0f}s")
up, dn = (delta[moved] > 0).sum(), (delta[moved] < 0).sum()
print(f"  raised {int(up):,} / lowered {int(dn):,}   ratio {up / max(dn, 1):.2f}")

print("\nby airport, mean shift and rows moved:")
apt = d["apt"].to_numpy()
for ap in sorted(set(apt.tolist())):
    m = apt == ap
    mm = m & moved
    if not mm.sum():
        continue
    print(f"  {ap}: {int(mm.sum()):7,} moved of {int(m.sum()):7,}"
          f"   mean {delta[mm].mean():+8.2f}s   median |{np.median(np.abs(delta[mm])):.0f}|s")

print("\nbound audit on the NEW file (the chain must still hold after every rider):")
gs = d["gs"].fill_null(np.nan).to_numpy().astype(float)
gl = d["gl"].fill_null(np.nan).to_numpy().astype(float)
plan = d["plan"].fill_null(False).to_numpy()
clampable = np.isfinite(gs) & (gs >= 5000) & (gs < 50000)
box = plan & np.isfinite(gl)
print(f"  non-finite predictions            {int((~np.isfinite(n)).sum()):,}")
print(f"  negative predictions              {int((n < 0).sum()):,}")
print(f"  gap-clamp violations (pred > gs)  {int((clampable & (n > gs + 1e-6)).sum()):,}")
outside = box & ((n < gl - W - 14.001) | (n > gl + W + 14.001))
print(f"  LOBT-box violations beyond slack  {int(outside.sum()):,}")
print("\nREAD IT AS: many rows moved by a little, roughly balanced up and down, concentrated where")
print("ADS-B reaches, and zero bound violations. A large one-sided mean shift is the warning sign.")
