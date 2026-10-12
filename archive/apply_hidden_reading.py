"""Rome's copy-impossible no-plan departures: predict the hidden plan's own off-block reading.

    .venv/Scripts/python.exe archive/apply_hidden_reading.py --in X.parquet --out Y.parquet

Scope is v36's (archive/apply_upper_bound.py), narrowed to LIRF: a no-plan departure whose plan is
recovered from the destination's arrival row (same flight number, landing 0-8h later, plan origin =
our ADEP), with gl = MVT - LOBT >= 3,606s and |gs - gl| < 3,606s. A schedule copy is impossible there,
yet the model still hedges toward it. The same recovered plan carries AOBT_3, and a = MVT - AOBT_3
sits within 600s of the truth on 94 per cent of 2025's LIRF rows in scope (RMSE 285s).
RULE, declared before its test (archive/hidden_reading_rule.py, S1 / R): prediction := clip(a, 0,
gl - 3,606s), applied LAST. Gate-inclusive folds: (1,7) -0.517s on 19 rows, (2,8) -0.063s on 17;
pooled -0.139s, -0.029s after deleting the five rows it helps most. PASS. The all-airport version
failed (three folds positive): elsewhere the model already predicts a normal taxi on these rows.
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
dep = (pl.scan_parquet(R).filter((C("PHASE_mvt") == "DEP") & C("AOBT_3_flt").is_null() & (C("ADEP_mvt") == "LIRF")
                                 & C("ADES_mvt").is_in(APTS) & C("FLIGHT_mvt").is_not_null())
       .select(C(ID), C("ADEP_mvt").alias("adep"), C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("mvt"),
               (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"))
       .collect().sort("mvt"))
arr = (pl.scan_parquet(R).filter((C("PHASE_mvt") == "ARR") & C("LOBT_flt").is_not_null())
       .select(C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("arr_t"),
               C("LOBT_flt").alias("lobt"), C("ADEP_flt").alias("plan_adep"), C("AOBT_3_flt").alias("aobt"))
       .collect().sort("arr_t"))
rec = (dep.join_asof(arr, left_on="mvt", right_on="arr_t", by=["ades", "flight"], strategy="forward", tolerance="8h")
          .filter(C("lobt").is_not_null() & (C("plan_adep") == C("adep")) & C("aobt").is_not_null())
          .with_columns((C("mvt") - C("lobt")).dt.total_seconds().alias("gl"),
                        (C("mvt") - C("aobt")).dt.total_seconds().alias("rd"))
          .filter((C("gl") >= W) & ((C("gs") - C("gl")).abs() < W)))
target = dict(zip(rec[ID].to_list(), np.clip(rec["rd"].to_numpy(), 0, (rec["gl"] - W).to_numpy()).tolist()))
sub = pl.read_parquet(args.inp)
old = sub[TARGET].to_numpy().astype(float)
new = np.array([target.get(m, v) for m, v in zip(sub[ID].to_list(), old)])
ch = np.abs(new - old) > 0.5
print(f"hidden-plan reading at LIRF: {len(target)} scored rows in scope; {int(ch.sum())} change")
if ch.any():
    mv = (new - old)[ch]
    print(f"  moves: median {np.median(mv):+,.0f}s, mean {mv.mean():+,.0f}s, min {mv.min():+,.0f}s, max {mv.max():+,.0f}s")
out = sub.with_columns(pl.Series(TARGET, np.rint(new)).cast(sub.schema[TARGET]))
assert out.height == sub.height and out[ID].equals(sub[ID])
out.write_parquet(args.out)
print(f"wrote {args.out}")
