"""Build agent_detrules_final.parquet = v136 with the hidden-plan UPPER bound (y < gl - 3,606 s) applied at every airport with a clean 2025 record,
reached through the destination arrival record by flight string OR by the plan's own callsign. Only rows whose v136 value sits above the bound change."""
import polars as pl, numpy as np
C=pl.col
ID,T="MVT_ID_mvt","TAXITIME_SEC_mvt"; W=3606
KEEP=["LEBL","LEMD","LFPG","LTFM","EDDM","EGLL","LIRF"]   # 2025: >=128 rows, 0-1 violations each (Wilson lower >= 0.97)
F = pl.read_parquet("data/interim/agent_detrules_frame.parquet").filter((C("yr")==2026)&C("lobt").is_null())
cs = pl.read_parquet("data/interim/agent_detrules_cs.parquet").filter(C("yr")==2026).select(ID,"hc_lobt")
F = F.join(cs,on=ID,how="left").with_columns(pl.coalesce("h_lobt","hc_lobt").alias("L"), pl.when(C("h_lobt").is_not_null()).then(pl.lit("str")).otherwise(pl.lit("cs")).alias("route"))
F = F.filter(C("L").is_not_null()&C("adep").is_in(KEEP)).with_columns((C("mvt")-C("sched")).alias("gs"),(C("mvt")-C("L")).alias("gl"),(C("L")-C("sched")).alias("l_s"))
F = F.filter((C("gl")>=W)&(C("l_s").abs()<W)).with_columns((C("gl")-W).cast(pl.Float64).alias("U"))
ref = pl.read_parquet("data/interim/agent_bundle_v136_final.parquet")
v = ref.select(ID, C(T).cast(pl.Float64).alias("old"))
R = F.select(ID,"adep","route","U").join(v,on=ID).filter(C("old")>C("U")+0.5).with_columns(C("U").round(0).alias("new"))
print(R)
rows = R.select(ID, pl.lit("hidden_upper_bound").alias("rule"), "old", "new", "adep", "route")
rows.write_parquet("data/interim/agent_detrules_rows.parquet")
mp = dict(zip(R[ID].to_list(), R["new"].to_list()))
vals = [mp.get(i, x) for i,x in zip(ref[ID].to_list(), ref[T].to_list())]
out = ref.with_columns(pl.Series(T, np.rint(vals)).cast(ref.schema[T]))
assert out[ID].equals(ref[ID]); out.write_parquet("data/interim/agent_detrules_final.parquet")
print("changed", int((out[T]!=ref[T]).sum()))
