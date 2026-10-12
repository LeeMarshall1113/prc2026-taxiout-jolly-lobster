"""Hidden-plan recovery keyed on the plan's own callsign (ICAO form), per year. Output rows: dep ids with no own plan whose plan sits on a destination arrival record."""
import polars as pl, numpy as np, glob, gc
C = pl.col
ID = "MVT_ID_mvt"
APTS = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH","LTFM"]
FL = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase()
cols = [ID,"FLIGHT_mvt","ADEP_mvt","ADES_mvt","PHASE_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","AOBT_3_flt","EOBT_1_flt","CALLSIGN_flt","ADEP_flt","TAXITIME_SEC_mvt"]
def T(c): return C(c).dt.epoch("s").cast(pl.Int64)
def build(al, yr):
    al = al.with_columns(FL.alias("flight"), T("MVT_TIME_UTC_mvt").alias("mvt"), T("SCHED_TIME_UTC_mvt").alias("sched"), T("LOBT_flt").alias("lobt"), T("AOBT_3_flt").alias("aobt"), T("EOBT_1_flt").alias("eobt"),
                       C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("cs"))
    dep = al.filter((C("PHASE_mvt")=="DEP") & C("lobt").is_null() & C("ADES_mvt").is_in(APTS)).select(ID, C("ADEP_mvt").alias("adep"), C("ADES_mvt").alias("ades"), "flight", "mvt", "sched").sort("mvt")
    arr = (al.filter((C("PHASE_mvt")=="ARR") & C("lobt").is_not_null() & C("cs").is_not_null())
           .select(C("ADES_mvt").alias("ades"), C("cs").alias("flight"), C("mvt").alias("arr_t"), C("lobt").alias("hc_lobt"), C("ADEP_flt").alias("hc_adep"), C("aobt").alias("hc_aobt"), C("eobt").alias("hc_eobt")).sort("arr_t"))
    r = (dep.join_asof(arr, left_on="mvt", right_on="arr_t", by=["ades","flight"], strategy="forward", tolerance=8*3600)
         .filter(C("hc_lobt").is_not_null() & (C("hc_adep")==C("adep")) & (C("arr_t")>C("mvt"))).with_columns(pl.lit(yr).alias("yr")))
    return r.select(ID,"yr","adep","ades","flight","mvt","sched","hc_lobt","hc_aobt","hc_eobt","arr_t")
tr = pl.concat([pl.scan_parquet(f).select(cols).collect() for f in sorted(glob.glob("data/raw/training_*.parquet"))])
o1 = build(tr, 2025); del tr; gc.collect()
rk = pl.scan_parquet("data/raw/ranking.parquet").select(cols).collect()
o2 = build(rk, 2026)
out = pl.concat([o1,o2]); print(out.group_by("yr","adep").len().sort("yr","adep"))
out.write_parquet("data/interim/agent_detrules_cs.parquet")
