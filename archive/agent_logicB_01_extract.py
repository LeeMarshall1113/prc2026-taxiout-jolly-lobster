"""logicB 01: compact DEP table (2025 train + 2026 scored) with epoch-second clocks and local dates per airport."""
import glob, polars as pl
C = pl.col
TZ = {"EDDF":"Europe/Berlin","EDDM":"Europe/Berlin","EGLL":"Europe/London","EHAM":"Europe/Amsterdam","LEBL":"Europe/Madrid",
      "LEMD":"Europe/Madrid","LFPG":"Europe/Paris","LIRF":"Europe/Rome","LSZH":"Europe/Zurich","LTFM":"Europe/Istanbul"}
R = ""
clk = {"sch":"SCHED_TIME_UTC_mvt","mvt":"MVT_TIME_UTC_mvt","blk":"BLOCK_TIME_UTC_mvt","lobt":"LOBT_flt","iobt":"IOBT_flt","eobt":"EOBT_1_flt",
       "aobt":"AOBT_3_flt","arv1":"ARVT_1_flt","arv3":"ARVT_3_flt"}
def one(path, yr):
    df = pl.read_parquet(path).filter(C("PHASE_mvt")=="DEP")
    parts = []
    for ap, tz in TZ.items():
        d = df.filter(C("ADEP_mvt")==ap)
        if d.height==0: continue
        e = [C(v).dt.epoch("s").alias(k) for k,v in clk.items()]
        ld = [C(v).dt.convert_time_zone(tz).dt.date().cast(pl.Int32).alias("ld_"+k) for k,v in clk.items() if k in("sch","mvt","blk","lobt")]
        parts.append(d.select(C("MVT_ID_mvt").cast(pl.Int64).alias("id"), C("FLIGHT_ID_mvt").cast(pl.Int64).alias("fid"), C("FLIGHT_mvt").alias("fl"),
            C("CALLSIGN_flt").alias("cs"), C("ADEP_mvt").alias("apt"), C("ADES_mvt").alias("ades"), C("STAND_mvt").alias("stand"), C("RUNWAY_mvt").alias("rwy"),
            C("TAXITIME_SEC_mvt").alias("y"), *e, *ld, pl.lit(yr).alias("yr")))
    return pl.concat(parts)
tr = pl.concat([one(f,2025) for f in sorted(glob.glob(R+"data/raw/training_2025-*.parquet"))])
tr.write_parquet(R+"data/interim/local/logic_rows/dep2025.parquet"); print(tr.shape)
rk = one(R+"data/raw/ranking.parquet", 2026)
sv = pl.read_parquet(R+"data/interim/agent_bundle_v145_final.parquet").select(C("MVT_ID_mvt").cast(pl.Int64).alias("id"), C("TAXITIME_SEC_mvt").alias("served"))
rk = rk.join(sv, on="id", how="left"); print(rk.shape, rk["served"].null_count(), rk["y"].null_count())
rk.write_parquet(R+"data/interim/local/logic_rows/dep2026.parquet")
print(tr.select([C(c).null_count() for c in ["sch","mvt","blk","lobt","iobt","eobt","aobt","arv1","arv3","y"]]))
print(tr.group_by("apt").agg(n=pl.len(), noplan=C("lobt").is_null().sum()).sort("apt"))
print(rk.group_by("apt").agg(n=pl.len(), noplan=C("lobt").is_null().sum()).sort("apt"))
