"""experiment ltreg (docs/LAB-NOTES.md): LTFM-only extra features for the regime / snow arms.

SNOW_FEATURES (pre-registered s1b) from data/raw/weather/LTFM.csv METAR present-weather codes; an observation is a snow
observation iff wxcodes contains 'SN' or 'FZRA' or 'FZDZ' (= wxev_snowfz of archive/agent_wxev_01_screen.py). Only
observations at or before the departure's MVT_TIME_UTC_mvt are used (prc.weather.attach's backward rule).
  lt_snow_now    wxev_snowfz of the latest observation at/before MVT (no tolerance, as the screen)
  lt_snow_3h     1 if any snow observation in [MVT-3h, MVT]
  lt_snow_12h    1 if any snow observation in [MVT-12h, MVT]
  lt_snow_n24h   number of snow observations in [MVT-24h, MVT]   (duration / accumulation proxy; METAR ~2/h at LTFM)
  lt_snow_since  hours since the latest snow observation at/before MVT, capped at 96 (96 = none within 96 h)
REGIME_FEATURES: lt_regime = 1 iff MVT date >= 2025-04-17 (Istanbul triple-runway operations); serve = 1.
Both are NaN at the other nine airports (bulk model only), like the SEQ family.
"""
import numpy as np
import polars as pl

SNOW_FEATURES = ["lt_snow_now", "lt_snow_3h", "lt_snow_12h", "lt_snow_n24h", "lt_snow_since"]
REGIME_FEATURES = ["lt_regime"]
CUT = (2025, 4, 17)


def metar_snow(raw_dir):
    r = pl.read_csv(raw_dir / "weather" / "LTFM.csv", null_values=["M", "", "None"], infer_schema_length=0)
    r = r.select(pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False).dt.replace_time_zone("UTC").alias("t"),
                 pl.col("wxcodes").fill_null("").alias("wx")).drop_nulls("t")
    r = r.sort("t").unique(subset=["t"], keep="last").sort("t")
    r = r.with_columns(sn=(pl.col("wx").str.contains("SN") | pl.col("wx").str.contains("FZRA") | pl.col("wx").str.contains("FZDZ")))
    return r


def snow_features(dep: pl.DataFrame, raw_dir) -> pl.DataFrame:
    """dep: MVT_ID_mvt, MVT_TIME_UTC_mvt (LTFM departures). Returns MVT_ID_mvt + SNOW_FEATURES."""
    w = metar_snow(raw_dir)
    tw = w["t"].dt.epoch("s").to_numpy().astype(np.int64)
    sn = w["sn"].to_numpy().astype(bool)
    ts = tw[sn]
    t = dep["MVT_TIME_UTC_mvt"].dt.replace_time_zone("UTC").dt.epoch("s").to_numpy().astype(np.int64) \
        if dep["MVT_TIME_UTC_mvt"].dtype.time_zone is None else dep["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    i = np.searchsorted(tw, t, side="right") - 1          # latest obs at/before t
    now = np.where(i >= 0, sn[np.clip(i, 0, None)], False).astype(float)
    now[i < 0] = np.nan
    hi = np.searchsorted(ts, t, side="right")             # snow obs <= t
    def cnt(h):
        return hi - np.searchsorted(ts, t - h * 3600, side="left")
    last = np.where(hi > 0, ts[np.clip(hi - 1, 0, None)], -10**12)
    since = np.minimum((t - last) / 3600.0, 96.0)
    return pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], "lt_snow_now": now,
                         "lt_snow_3h": (cnt(3) > 0).astype(float), "lt_snow_12h": (cnt(12) > 0).astype(float),
                         "lt_snow_n24h": cnt(24).astype(float), "lt_snow_since": since})


def regime_features(dep: pl.DataFrame, serve: bool) -> pl.DataFrame:
    if serve:
        v = np.ones(dep.height)
    else:
        v = dep.select((pl.col("MVT_TIME_UTC_mvt").dt.date() >= pl.date(*CUT)).cast(pl.Float64)).to_series().to_numpy()
    return pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], "lt_regime": v})
