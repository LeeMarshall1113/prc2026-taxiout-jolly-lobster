"""experiment eusnow (docs/LAB-NOTES.md): the v117 LTFM snow features (archive/agent_ltreg_feat.py, SAME 5 column names and
definitions, reused verbatim) generalised to any airport: METAR file data/raw/weather/<ADEP>.csv of the departure's OWN airport.
Only change vs agent_ltreg_feat.snow_features is the airport (per-ADEP METAR). Observations at/before MVT only. No label."""
import os
import numpy as np
import polars as pl
from agent_ltreg_feat import SNOW_FEATURES, snow_features as _sf  # noqa: F401

NEW_APTS = ["EHAM", "LSZH", "EDDM", "EDDF"]
ALL_APTS = ["LTFM"] + NEW_APTS


def metar_snow_apt(raw_dir, apt):
    r = pl.read_csv(os.path.join(os.environ.get("EUSNOW_WX", str(raw_dir / "weather")), f"{apt}.csv"), null_values=["M", "", "None"], infer_schema_length=0)
    r = r.select(pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False).dt.replace_time_zone("UTC").alias("t"),
                 pl.col("wxcodes").fill_null("").alias("wx")).drop_nulls("t")
    r = r.sort("t").unique(subset=["t"], keep="last").sort("t")
    return r.with_columns(sn=(pl.col("wx").str.contains("SN") | pl.col("wx").str.contains("FZRA") | pl.col("wx").str.contains("FZDZ")))


def snow_features_apts(dep: pl.DataFrame, raw_dir, apts) -> pl.DataFrame:
    """dep: MVT_ID_mvt, ADEP_mvt, MVT_TIME_UTC_mvt. Returns MVT_ID_mvt + SNOW_FEATURES for rows whose ADEP is in apts (others absent)."""
    import agent_ltreg_feat as L
    out = []
    for a in apts:
        d = dep.filter(pl.col("ADEP_mvt") == a)
        if d.height == 0:
            continue
        L.metar_snow = lambda rd, _a=a: metar_snow_apt(rd, _a)     # verbatim snow_features(), per-airport METAR
        out.append(L.snow_features(d.select("MVT_ID_mvt", "MVT_TIME_UTC_mvt"), raw_dir))
    return pl.concat(out)
