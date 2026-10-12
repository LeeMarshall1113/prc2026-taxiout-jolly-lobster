"""v35's two post-processing rules, applied on top of a BUILT submission.

    .venv/Scripts/python.exe archive/apply_v35_rules.py --in submissions/X.parquet --out submissions/Y.parquet

Both passed rules declared before they ran (docs/LAB-NOTES.md, 2026-09-19):

  1. ROME HABITUAL-COPY AIRLINES. LIRF no-plan departures, schedule gap < 50,000s, whose FLIGHT_mvt
     operator prefix copied the schedule >= 97% of the time over >= 8 rows of 2025 -> prediction :=
     gap. 97.9% right out of fold; fired 23 / 22 rows at 100% copies on the gate-inclusive folds,
     -0.158 / -0.171s. A prediction, not a bound, so it is applied FIRST.
  2. HIDDEN-PLAN LOWER BOUND at LIRF and LTFM. For a no-plan departure whose plan is recovered from
     the destination's arrival row (same flight number, landing 0-8h later, plan origin = our ADEP),
     with gl = MVT - LOBT < 3,606s, y > gl + 3,606s. Zero violations on 50 rows of 2025; every fold
     non-positive. A true BOUND, so it is applied LAST and wins any conflict: a prediction below it
     is raised to it, nothing above it is touched. Immune to the v32 trap by construction.

The script reports every row it changes so the move can be read before anything is uploaded.
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
BOUND_APTS = ["LIRF", "LTFM"]
FL = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase()
OP = FL.str.extract(r"^([A-Z]{2,3})", 1)

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

# --- rule 1: habitual-copy airlines at Rome, learned from all of 2025 ---
tr = (pl.scan_parquet("data/raw/training_*.parquet")
      .filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_null())
      .select(OP.alias("op"), ((C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60).alias("copy"))
      .collect())
habitual = set(tr.group_by("op").agg(pl.len().alias("n"), C("copy").mean().alias("r"))
                 .filter((C("n") >= 8) & (C("r") >= 0.97))["op"].to_list())
print(f"rule 1 habitual-copy operators at LIRF: {sorted(habitual)}")

RANK = "data/raw/ranking.parquet"
dep = (pl.scan_parquet(RANK).filter((C("PHASE_mvt") == "DEP") & C("AOBT_3_flt").is_null())
       .select(C(ID), C("ADEP_mvt").alias("adep"), C("ADES_mvt").alias("ades"), FL.alias("flight"), OP.alias("op"),
               C("MVT_TIME_UTC_mvt").alias("mvt"),
               (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"))
       .collect())

# --- rule 2: recover the hidden plan from the destination's arrival row ---
arr = (pl.scan_parquet(RANK).filter((C("PHASE_mvt") == "ARR") & C("LOBT_flt").is_not_null())
       .select(C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("arr_t"),
               C("LOBT_flt").alias("lobt"), C("ADEP_flt").alias("plan_adep"))
       .collect().sort("arr_t"))
rec = (dep.filter(C("adep").is_in(BOUND_APTS) & C("ades").is_in(APTS) & C("flight").is_not_null()).sort("mvt")
          .join_asof(arr, left_on="mvt", right_on="arr_t", by=["ades", "flight"], strategy="forward", tolerance="8h")
          .filter(C("lobt").is_not_null() & (C("plan_adep") == C("adep")))
          .with_columns((C("mvt") - C("lobt")).dt.total_seconds().alias("gl"))
          .filter(C("gl") < W))
bound = dict(zip(rec[ID].to_list(), (rec["gl"] + W).to_list()))
airline = dict(zip(*[dep.filter((C("adep") == "LIRF") & C("op").is_in(list(habitual)) & (C("gs") < 50000))[c].to_list()
                     for c in (ID, "gs")]))

sub = pl.read_parquet(args.inp)
ids = sub[ID].to_list()
old = sub[TARGET].to_numpy().astype(float)
new = old.copy()
n1 = n2 = 0
for i, m in enumerate(ids):
    if m in airline and abs(new[i] - max(airline[m], 0.0)) > 0.5:
        new[i] = max(airline[m], 0.0)
        n1 += 1
    if m in bound and new[i] < bound[m]:
        new[i] = bound[m]
        n2 += 1
ch = new != old
print(f"rule 1 (airline -> gap): {n1} rows set; rule 2 (hidden-plan bound): {n2} rows raised; {int(ch.sum())} rows changed")
if ch.any():
    mv = (new - old)[ch]
    print(f"  moves: median {np.median(mv):+,.0f}s, mean {mv.mean():+,.0f}s, min {mv.min():+,.0f}s, max {mv.max():+,.0f}s")
out = sub.with_columns(pl.Series(TARGET, np.rint(new)).cast(sub.schema[TARGET]))
assert out.height == sub.height and out[ID].equals(sub[ID])
out.write_parquet(args.out)
print(f"wrote {args.out}")
