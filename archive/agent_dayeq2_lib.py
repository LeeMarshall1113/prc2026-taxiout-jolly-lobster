"""day-equation round two: shared loaders. 2025 frame = raw departures joined to the held-out chain value (agent_eb136_tbl pred, frame
months 1,2,3,7-11) and the compact clock table (data/interim/local/logic_rows/C_dep2025_typed: gl, plan, ades, y); 2026 frame = ranking departures
joined to a base file's values and C_dep2026. Published series: all_pre_departure_delays_{year}.csv (DLY_ALL_PRE_2, FLT_DEP_1)."""
import glob
import numpy as np
import polars as pl

C = pl.col
R = ""
E = R + "data/external/eurocontrol/"
I = R + "data/interim/"
L = R + "data/interim/local/logic_rows/"
FRAME_MONTHS = [1, 2, 3, 7, 8, 9, 10, 11]
HEDGED = ["WMTT3508", "SVII3006", "ITYTY672", "NOSOS431", "QFAA7006", "TAY9LA"]
SE_SCALE = 1.5e8
DN_MAX = 15


def load_pub(year, apts):
    d = pl.read_csv(f"{E}all_pre_departure_delays_{year}.csv", infer_schema_length=0).filter(C("APT_ICAO").is_in(apts))
    d = d.select(C("APT_ICAO").alias("apt"), C("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"),
                 C("DLY_ALL_PRE_2").cast(pl.Float64, strict=False).alias("pub"), C("FLT_DEP_1").cast(pl.Float64, strict=False).alias("n_pub"))
    p7 = d.select("apt", (C("day") - pl.duration(days=7)).alias("day"), C("pub").alias("pub7"), C("n_pub").alias("n_pub7"))
    return d.join(p7, on=["apt", "day"], how="left")


def frame_2025(apts):
    files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
    raw = (pl.scan_parquet(files).filter((C("PHASE_mvt") == "DEP") & C("ADEP_mvt").is_in(apts))
           .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"), C("FLIGHT_ID_mvt").is_not_null().alias("hasplan"), C("FLIGHT_mvt").alias("fs"),
                   C("ADES_mvt").alias("ades"), (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("gl_raw")).collect())
    eb = pl.read_parquet(I + "agent_eb136_tbl.parquet").select("id", "mon", "pred").filter(C("mon").is_in(FRAME_MONTHS))
    ck = pl.read_parquet(L + "C_dep2025_typed.parquet", columns=["id", "gl", "ga", "plan", "tbig", "rule"]).rename({"tbig": "kind"})
    f = raw.join(eb, on="id", how="inner").join(ck, on="id", how="left").with_columns(pl.coalesce(C("gl").cast(pl.Float64), C("gl_raw")).alias("gl")).drop("gl_raw")
    return _common(f)


def frame_2026(apts, base):
    raw = (pl.scan_parquet(R + "data/raw/ranking.parquet").filter((C("PHASE_mvt") == "DEP") & C("ADEP_mvt").is_in(apts))
           .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   C("FLIGHT_ID_mvt").is_not_null().alias("hasplan"), C("FLIGHT_mvt").alias("fs"), C("ADES_mvt").alias("ades"),
                   (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("gl_raw")).collect())
    v = pl.read_parquet(base).select(C("MVT_ID_mvt").alias("id"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred"))
    ck = pl.read_parquet(L + "C_dep2026.parquet", columns=["id", "gl", "ga", "plan", "rule"])
    f = raw.join(v, on="id", how="left").join(ck, on="id", how="left").with_columns(C("T").dt.month().alias("mon"), pl.lit(None, pl.Float64).alias("y"),
                                                                                    pl.coalesce(C("gl"), C("gl_raw")).alias("gl")).drop("gl_raw")
    return _common(f)


def _common(f):
    f = f.with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs"), C("gl").cast(pl.Float64))
    f = f.filter(C("gs").is_not_null() & C("pred").is_not_null())
    taxi = f.filter(C("gs").abs() < 3600).group_by("apt", "mon").agg(taxi=C("pred").median())
    f = f.join(taxi, on=["apt", "mon"], how="left")
    # copy possible on a plan row only when the schedule lies inside the plan-clock window; no-plan rows: always
    f = f.with_columns(pl.when(C("hasplan")).then((C("gs") - C("gl")).abs() <= 3606).otherwise(True).alias("copy_ok"),
                       (C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day_pred"))
    return f


def day_bootstrap(gain_by_day, n=2000, seed=0):
    rng = np.random.default_rng(seed); g = np.asarray(gain_by_day, float)
    bs = np.array([rng.choice(g, len(g)).sum() for _ in range(n)]) / SE_SCALE
    return float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95)), float((bs > 0).mean())


def solve_delta(gs, yh, target_min):
    """delta (s) with sum max(gs - yh - delta, 0)/60 = target_min; bisection on [-7200, 7200]."""
    def S(d):
        return np.clip(gs - yh - d, 0, None).sum() / 60.0
    lo, hi = -7200.0, 7200.0
    if S(lo) < target_min:
        return lo
    if S(hi) > target_min:
        return hi
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if S(m) > target_min:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)
