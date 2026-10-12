"""Build LIRF departure + arrival frames (2025, 2026) for the state analysis."""
import polars as pl, numpy as np
C=pl.col
R=""
a=pl.read_parquet(R+"data/interim/agent_rome2_all.parquet")
OP=C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().str.extract(r"^([A-Z]{2,3})",1)
arr=a.filter((C("PHASE_mvt")=="ARR")&(C("ADES_mvt")=="LIRF")).select(
  C("MVT_ID_mvt").alias("aid"),"yr",OP.alias("op"),C("FLIGHT_mvt").alias("fl"),C("STAND_mvt").alias("stand"),
  C("MVT_TIME_UTC_mvt").alias("t_land"),C("BLOCK_TIME_UTC_mvt").alias("t_in"),C("SCHED_TIME_UTC_mvt").alias("t_sched"),C("LOBT_flt").is_null().alias("np"))
print(arr.group_by("yr").agg(pl.len(),C("t_in").null_count().alias("in_null"),C("stand").null_count().alias("st_null")))
arr=arr.with_columns(((C("t_in")-C("t_sched")).dt.total_seconds().abs()<=60).alias("copy"))
arr.write_parquet(R+"data/interim/agent_rome2_arr.parquet")
d=pl.read_parquet(R+"data/interim/agent_rome2_depall.parquet").filter(C("ADEP_mvt")=="LIRF")
d=d.with_columns(C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().str.extract(r"^([A-Z]{2,3})",1).alias("op"),
  (C("MVT_TIME_UTC_mvt")-C("LOBT_flt")).dt.total_seconds().alias("gl"),(C("MVT_TIME_UTC_mvt")-C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
  (C("MVT_TIME_UTC_mvt")-C("EOBT_1_flt")).dt.total_seconds().alias("ge"),(C("MVT_TIME_UTC_mvt")-C("IOBT_flt")).dt.total_seconds().alias("gi"),
  C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"))
d=d.with_columns(C("gs").cast(pl.Float64))
d=d.with_columns(((C("y")-C("gs")).abs()<=60).alias("copy_raw"))
d.write_parquet(R+"data/interim/agent_rome2_depL.parquet")
print(d.group_by("yr","np").len())
