"""v36's rule: the hidden-plan UPPER bound at LIRF and LTFM, applied on top of a built submission.

    .venv/Scripts/python.exe archive/apply_upper_bound.py --in X.parquet --out Y.parquet

The other half of v35's deduction (docs/LAB-NOTES.md). For a no-plan departure whose
plan is recovered from the destination's arrival row (same flight number, landing 0-8h later, plan
origin = our ADEP), with gl = MVT - LOBT and gs = MVT - SCHED: if gl >= 3,606s AND |gs - gl| < 3,606s
then a schedule copy is impossible -- BLOCK = SCHED would sit inside the join window and the plan
would have joined. So BLOCK lies outside LOBT +/- 3,606s, and on 2025 every one of 132 such rows at
LIRF+LTFM falls on the SAME side: y < gl - 3,606s. Zero violations, zero on the other side, zero
inside the window. An UPPER bound, applied LAST: a prediction above it is lowered to it. Gate-inclusive
folds: -0.272 / -0.172s, no fold positive. It is certain where it holds, so immune to the v32 trap.
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
W = 3606.0
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FL = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase()
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
R = "data/raw/ranking.parquet"
dep = (pl.scan_parquet(R).filter((C("PHASE_mvt") == "DEP") & C("AOBT_3_flt").is_null() & C("ADEP_mvt").is_in(["LIRF", "LTFM"])
                                 & C("ADES_mvt").is_in(APTS) & C("FLIGHT_mvt").is_not_null())
       .select(C(ID), C("ADEP_mvt").alias("adep"), C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("mvt"),
               (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"))
       .collect().sort("mvt"))
arr = (pl.scan_parquet(R).filter((C("PHASE_mvt") == "ARR") & C("LOBT_flt").is_not_null())
       .select(C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("arr_t"),
               C("LOBT_flt").alias("lobt"), C("ADEP_flt").alias("plan_adep")).collect().sort("arr_t"))
rec = (dep.join_asof(arr, left_on="mvt", right_on="arr_t", by=["ades", "flight"], strategy="forward", tolerance="8h")
          .filter(C("lobt").is_not_null() & (C("plan_adep") == C("adep")))
          .with_columns((C("mvt") - C("lobt")).dt.total_seconds().alias("gl"))
          .filter((C("gl") >= W) & ((C("gs") - C("gl")).abs() < W)))
cap = dict(zip(rec[ID].to_list(), (rec["gl"] - W).to_list()))
sub = pl.read_parquet(args.inp)
old = sub[TARGET].to_numpy().astype(float)
new = np.array([min(v, cap[m]) if m in cap else v for m, v in zip(sub[ID].to_list(), old)])
new = np.clip(new, 0, None)
ch = new != old
print(f"upper bound: {len(cap)} scored rows in scope; {int(ch.sum())} lowered")
if ch.any():
    mv = (new - old)[ch]
    print(f"  moves: median {np.median(mv):+,.0f}s, mean {mv.mean():+,.0f}s, min {mv.min():+,.0f}s, max {mv.max():+,.0f}s")
out = sub.with_columns(pl.Series(TARGET, np.rint(new)).cast(sub.schema[TARGET]))
assert out.height == sub.height and out[ID].equals(sub[ID])
out.write_parquet(args.out)
print(f"wrote {args.out}")
