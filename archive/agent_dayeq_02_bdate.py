"""day-equation 02: which calendar day does the portal file a flight under? Regroup our sum of max(B-S,0) by the OFF-BLOCK date (B = take-off - label)
instead of the take-off date and compare the identity residual (published minus ours) per airport. Also the schedule-date grouping."""
import glob
import numpy as np
import polars as pl

C = pl.col
R = ""
E = R + "data/external/eurocontrol/"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
tr = (pl.scan_parquet(files).filter(C("PHASE_mvt") == "DEP").filter(C("TAXITIME_SEC_mvt").is_not_null() & C("SCHED_TIME_UTC_mvt").is_not_null())
      .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"), C("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y"),
              (C("MVT_TIME_UTC_mvt") - pl.duration(seconds=C("TAXITIME_SEC_mvt"))).alias("B"), C("BLOCK_TIME_UTC_mvt").alias("BLK"))
      .with_columns(((C("B") - C("S")).dt.total_seconds() / 60.0).clip(lower_bound=0).alias("dm"),
                    C("T").dt.date().cast(pl.Utf8).alias("tday"), C("B").dt.date().cast(pl.Utf8).alias("bday"), C("S").dt.date().cast(pl.Utf8).alias("sday"),
                    C("BLK").dt.date().cast(pl.Utf8).alias("blkday"), (C("BLK") == C("B")).alias("blk_eq_b"))
      .collect())
print("rows", tr.height, "BLOCK_TIME == T - y share", tr["blk_eq_b"].mean(), "BLOCK null", tr["BLK"].null_count())
pub = pl.read_csv(f"{E}all_pre_departure_delays_2025.csv", infer_schema_length=0).filter(C("APT_ICAO").is_in(APTS)).select(
    C("APT_ICAO").alias("apt"), C("FLT_DATE").str.slice(0, 10).alias("day"), C("DLY_ALL_PRE_2").cast(pl.Float64).alias("pub"), C("FLT_DEP_IFR_2").cast(pl.Float64).alias("nifr"), C("FLT_DEP_1").cast(pl.Float64).alias("n1"))
for key in ["tday", "bday", "sday", "blkday"]:
    o = tr.group_by("apt", C(key).alias("day")).agg(s=C("dm").sum(), n=pl.len())
    j = pub.join(o, on=["apt", "day"], how="inner").with_columns((C("pub") - C("s")).alias("res"), (C("n1") - C("n")).alias("dn"))
    g = j.group_by("apt").agg(days=pl.len(), med=C("res").abs().median(), p90=C("res").abs().quantile(0.9), p99=C("res").abs().quantile(0.99), sd=C("res").std(), mean=C("res").mean(),
                              dn_med=C("dn").median(), dn_p90=C("dn").abs().quantile(0.9), big=(C("res").abs() > 30).sum()).sort("apt")
    print(f"\n== grouping by {key}: identity residual published - sum(max(B-S,0)) in minutes")
    print(g.with_columns(pl.col(pl.Float64).round(1)))
    if key == "bday":
        print("worst days by |res| (bday):")
        print(j.sort(C("res").abs(), descending=True).head(25).with_columns(pl.col(pl.Float64).round(1)))
