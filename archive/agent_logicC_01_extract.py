"""logicC 01: compact extract, nine airports (no Rome), 2025 labelled and 2026 inputs + served v145 value.
Writes data/interim/local/logic_rows/C_dep2025.parquet, C_dep2026.parquet. No 2026 label exists in the ranking file."""
import os
import polars as pl
C = pl.col
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, ".local", "logic_rows")
TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}
def secs(a, b):
    return (C(a) - C(b)).dt.total_seconds().cast(pl.Int64)
def build(src, labelled):
    sel = [C("MVT_ID_mvt").alias("id"), C("FLIGHT_ID_mvt").alias("fid"), C("ADEP_mvt").alias("apt"), C("FLIGHT_mvt").alias("flight"),
           C("CALLSIGN_flt").alias("callsign"), C("AIRCRAFT_OPERATOR_flt").alias("op"),
           C("AOBT_3_flt").is_not_null().alias("plan"), C("SCHED_TIME_UTC_mvt").alias("sched"), C("MVT_TIME_UTC_mvt").alias("mvt"),
           C("LOBT_flt").alias("lobt"), C("EOBT_1_flt").alias("eobt1"), C("IOBT_flt").alias("iobt"), C("AOBT_3_flt").alias("aobt3"),
           C("STAND_mvt").alias("stand"), C("RUNWAY_mvt").alias("runway"), C("ADES_mvt").alias("ades"), C("ADES_flt").alias("ades_flt"),
           C("AIRCRAFT_TYPE_mvt").alias("acft"), C("FLIGHT_RULE_mvt").alias("rule"), C("FLIGHT_TYPE_flt").alias("ftype"),
           C("MARKET_SEGMENT_flt").alias("seg"), C("WK_TBL_CAT_flt").alias("wk"),
           secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("gap"), secs("MVT_TIME_UTC_mvt", "LOBT_flt").alias("gl"),
           secs("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("ga"), secs("MVT_TIME_UTC_mvt", "EOBT_1_flt").alias("ge"),
           secs("MVT_TIME_UTC_mvt", "IOBT_flt").alias("gi")]
    if labelled:
        sel += [C("BLOCK_TIME_UTC_mvt").alias("block"), C("TAXITIME_SEC_mvt").cast(pl.Int64).alias("y")]
    parts = []
    for apt, tz in TZ.items():
        lf = pl.scan_parquet(src).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == apt)).select(sel)
        d = [C("sched").dt.convert_time_zone(tz).dt.date().alias("sched_d"), C("mvt").dt.convert_time_zone(tz).dt.date().alias("mvt_d"),
             C("sched").dt.convert_time_zone(tz).dt.strftime("%H:%M").alias("sched_hm"), C("mvt").dt.convert_time_zone(tz).dt.strftime("%H:%M").alias("mvt_hm")]
        if labelled:
            d += [C("block").dt.convert_time_zone(tz).dt.date().alias("block_d"), C("block").dt.convert_time_zone(tz).dt.strftime("%H:%M").alias("block_hm")]
        parts.append(lf.with_columns(d).collect())
    return pl.concat(parts)
tr = build(os.path.join(ROOT, "data", "raw", "training_*.parquet"), True).with_columns(C("mvt").dt.month().alias("month"))
print("2025", tr.height); tr.write_parquet(os.path.join(OUT, "C_dep2025.parquet"))
rk = build(os.path.join(ROOT, "data", "raw", "ranking.parquet"), False)
sub = pl.read_parquet(os.path.join(ROOT, "data", "interim", "agent_bundle_v145_final.parquet")).rename({"MVT_ID_mvt": "id", "TAXITIME_SEC_mvt": "served"})
rk = rk.join(sub, on="id", how="left").with_columns(C("mvt").dt.month().alias("month"))
print("2026", rk.height, "served nulls", rk["served"].null_count()); rk.write_parquet(os.path.join(OUT, "C_dep2026.parquet"))
