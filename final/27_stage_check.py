"""Check of the ADS-B stage output on chosen days: share of rows the stage changes, per airport, against its input; optionally
against another run of the same stage (cloud versus local) on the same rows.

    python final/27_stage_check.py --ranking data/raw/final_ranking.parquet --inp data/final/J5_s0ship.parquet \
        --out data/final/J5_v105_adsb_1.parquet --days 2026-02-15,2026-06-10 [--other data/final/_t2x/J5y_105.parquet]
"""
import argparse
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--ranking", required=True)
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--days", required=True)
ap.add_argument("--other", default=None)
a = ap.parse_args()
days = a.days.split(",")
dep = (pl.scan_parquet(a.ranking).filter(C("PHASE_mvt") == "DEP")
       .select("MVT_ID_mvt", C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day"))
       .filter(C("day").is_in(days)).collect())
r = lambda p, n: pl.read_parquet(p).select("MVT_ID_mvt", C("TAXITIME_SEC_mvt").cast(pl.Float64).alias(n))  # noqa: E731
d = dep.join(r(a.inp, "i"), on="MVT_ID_mvt").join(r(a.out, "o"), on="MVT_ID_mvt")
if a.other:
    d = d.join(r(a.other, "x"), on="MVT_ID_mvt")
agg = [(C("i") != C("o")).mean().round(3).alias("changed_share"), pl.len().alias("rows")]
if a.other:
    agg += [(C("x") != C("i")).mean().round(3).alias("other_changed"), ((C("x") - C("o")) ** 2).mean().sqrt().round(1).alias("rms_run_vs_run")]
print(d.group_by("day", "apt").agg(*agg).sort("day", "apt"))
