"""experiment neighbour: build the NBR columns for the 2026 scored departures of ONE build path (docs/LAB-NOTES.md
s2.1/s2.4), mirroring archive/agent_neighbour_03_build2025.py: p0 = the path's ADS-B-stage INPUT (v10x_s0ship or
v10x_gatebag_s0), fin = the path's stage OUTPUT (v10x_s1 / v10x_gatebag_s1). Label-free (2026 has no labels anyway).
"""
import numpy as np
import polars as pl

from archive.agent_neighbour_feat import features, sources

C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def build(s0_path, s1_path, rows_2026="data/interim/agent_offblock_rows_2026.parquet", ranking="data/raw/ranking.parquet"):
    s0 = pl.read_parquet(s0_path).select(C(ID).cast(pl.Float64), C(T).cast(pl.Float64).alias("p0"))
    s1 = pl.read_parquet(s1_path).select(C(ID).cast(pl.Float64), C(T).cast(pl.Float64).alias("fin"))
    assert s0[ID].n_unique() == s0.height and s1[ID].n_unique() == s1.height and s0.height == s1.height
    Z = s0.join(s1, on=ID, how="inner")
    assert Z.height == s0.height
    rk = (pl.read_parquet(ranking, columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
          .filter(C("PHASE_mvt") == "DEP").with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
          .select(ID, C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t"),
                  C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d").alias("day"), C("RUNWAY_mvt").alias("rwy")))
    Z = Z.join(rk, on=ID, how="left")
    assert Z["t"].null_count() == 0, "2026 rows without a ranking departure"
    R = pl.read_parquet(rows_2026, columns=[ID, "src", "prod_kind", "prod_t", "hour", "day", "apt"]).with_columns(C(ID).cast(pl.Float64))
    assert R[ID].n_unique() == R.height
    Z = Z.join(R.rename({"day": "rday", "apt": "rapt"}), on=ID, how="left")
    chk = Z.filter(C("src").is_not_null())
    assert (chk["rday"] == chk["day"]).all() and (chk["rapt"] == chk["apt"]).all(), "dump day/apt disagree with ranking"
    in_dump = Z["src"].is_not_null().to_numpy()
    is_m = Z["prod_kind"].is_in(["stand", "appear"]).fill_null(False).to_numpy()
    tx = (Z["t"] - Z["prod_t"]).cast(pl.Float64).fill_null(np.nan).to_numpy()
    p0 = Z["p0"].to_numpy()
    m, lvl, s = sources(tx - p0, tx, Z["fin"].to_numpy() - p0, is_m, in_dump)
    F = features(Z.select("apt", "day", "t", "rwy").with_columns(pl.Series("m", m), pl.Series("lvl", lvl), pl.Series("s", s)))
    cls = np.where(~in_dump, "U", np.where(is_m, "M", "W"))
    return pl.concat([Z.select(ID, "apt", "day", C("hour").fill_null(((C("t") % 86400) // 3600).cast(pl.Int64)),
                               (C("src") == "taxinm").fill_null(False).cast(pl.Float64).alias("taxinm"))
                      .with_columns(pl.Series("cls", cls)), F], how="horizontal")
