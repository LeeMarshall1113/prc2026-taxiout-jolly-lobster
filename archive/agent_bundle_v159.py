"""v159 = v158 + the day-level calibration of delayed flights at Istanbul from the published daily delay identity, with stop days excluded
(piece agent_dayeq2_ltfm_level_final, built on v156; Istanbul rows only), plan-clock window re-applied last.
Out: data/interim/agent_bundle_v159_final.parquet"""
import polars as pl
C = pl.col; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; D = "data/interim/"
b = pl.read_parquet(D + "agent_bundle_v158_final.parquet"); p = pl.read_parquet(D + "agent_dayeq2_ltfm_level_final.parquet").rename({T: "p"})
r = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
     .select(ID, "ADEP_mvt", ((C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds()).alias("gl")).collect())
j = b.join(p, on=ID).join(r, on=ID, how="left").with_columns(pl.when(C("ADEP_mvt") == "LTFM").then(C("p")).otherwise(C(T)).alias("n"))
lo = pl.max_horizontal(C("gl") - 3606, pl.lit(0)); hi = C("gl") + 3606
j = j.with_columns(pl.when(C("gl").is_null()).then(C("n")).otherwise(pl.min_horizontal(pl.max_horizontal(C("n"), lo), hi)).alias("n"))
out = b.select(ID).join(j.select(ID, C("n").round().cast(b.schema[T]).alias(T)), on=ID, how="left"); assert (out[ID] == b[ID]).all()
out.write_parquet(D + "agent_bundle_v159_final.parquet"); d = out[T] - b[T]
ch = j.with_columns((C("n") - C(T)).alias("d")).filter(C("d") != 0)
print("rows changed", ch.height, "by airport", dict(ch.group_by("ADEP_mvt").len().iter_rows()), "mean", float(d.mean()), "max", int(d.max()), "min", int(d.min()))
