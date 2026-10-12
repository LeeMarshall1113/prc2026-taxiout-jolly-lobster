"""v158 = v156 with the plan-clock window re-applied last: a plan row's taxi-out lies within 3,606 s of take-off minus LOBT
(2,062,577 of 2,062,577 plan departures in 2025). The day-level calibration of v153 moved a few rows just past that edge.
Out: data/interim/agent_bundle_v158_final.parquet"""
import polars as pl
C = pl.col; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; D = "data/interim/"
b = pl.read_parquet(D + "agent_bundle_v156_final.parquet")
r = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
     .select(ID, "ADEP_mvt", ((C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds()).alias("gl")).collect())
j = b.join(r, on=ID, how="left")
lo = pl.max_horizontal(C("gl") - 3606, pl.lit(0)); hi = C("gl") + 3606
j = j.with_columns(pl.when(C("gl").is_null()).then(C(T)).otherwise(pl.min_horizontal(pl.max_horizontal(C(T), lo), hi)).alias("n"))
ch = j.filter(C("n") != C(T)).with_columns((C("n") - C(T)).alias("d"))
print("rows moved", ch.height, "max", ch["d"].max(), "min", ch["d"].min()); print(ch.group_by("ADEP_mvt").agg(pl.len(), C("d").abs().max()).sort("ADEP_mvt"))
out = b.select(ID).join(j.select(ID, C("n").round().cast(b.schema[T]).alias(T)), on=ID, how="left"); assert (out[ID] == b[ID]).all()
out.write_parquet(D + "agent_bundle_v158_final.parquet")
