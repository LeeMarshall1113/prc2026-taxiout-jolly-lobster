"""experiment neighbour 03: build the pre-registered NBR columns (archive/agent_neighbour_feat.py) for every row of the 238
report days (docs/LAB-NOTES.md). Label-free: sources are ADS-B measurements (mvt - prod_t) and the
stage's moves (fin - p0) of OTHER rows; y is never read.

    python archive/agent_neighbour_03_build2025.py [--fin-npz data/interim/modal/TX_S6.npz] --out data/interim/agent_neighbour_nbr_2025.parquet
"""
import argparse
import glob
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_neighbour_feat import NBR, features, sources

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--fin-npz", default="data/interim/modal/TX_S6.npz")
ap.add_argument("--out", default="data/interim/agent_neighbour_nbr_2025.parquet")
a = ap.parse_args()
T0 = time.time()
z = np.load(a.fin_npz, allow_pickle=True)
Z = pl.DataFrame({"MVT_ID_mvt": z["ids"].astype(float), "p0": z["p0"].astype(float), "fin": z["both_guard"].astype(float),
                  "day": z["days"].astype(str), "apt": z["apts"].astype(str)})
assert Z["MVT_ID_mvt"].n_unique() == Z.height
raw = []
for f in sorted(glob.glob("data/raw/training_2025-*.parquet")):
    raw.append(pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
               .filter(C("PHASE_mvt") == "DEP").drop("PHASE_mvt"))
raw = (pl.concat(raw).with_columns(C("MVT_ID_mvt").cast(pl.Float64), C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t"),
                                   C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d").alias("utcday"))
       .unique(subset="MVT_ID_mvt", keep="first").select("MVT_ID_mvt", "t", "utcday", C("RUNWAY_mvt").alias("rwy")))
Z = Z.join(raw, on="MVT_ID_mvt", how="left")
assert Z["t"].null_count() == 0
print(f"rows {Z.height:,}; npz day == UTC MVT date on {float((Z['day'] == Z['utcday']).mean()):.4f} of rows")
R = pl.read_parquet("data/interim/agent_offblock_rows_c238.parquet", columns=["MVT_ID_mvt", "src", "prod_kind", "prod_t", "hour"]
                    ).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
assert R["MVT_ID_mvt"].n_unique() == R.height
Z = Z.join(R, on="MVT_ID_mvt", how="left")
in_dump = Z["src"].is_not_null().to_numpy()
is_m = (Z["prod_kind"].is_in(["stand", "appear"]).fill_null(False)).to_numpy()
tx = (Z["t"] - Z["prod_t"]).cast(pl.Float64).fill_null(np.nan).to_numpy()
p0 = Z["p0"].to_numpy()
m, lvl, s = sources(tx - p0, tx, Z["fin"].to_numpy() - p0, is_m, in_dump)
fr = Z.select("apt", "day", "t", "rwy").with_columns(pl.Series("m", m), pl.Series("lvl", lvl), pl.Series("s", s))
F = features(fr)
cls = np.where(~in_dump, "U", np.where(is_m, "M", "W"))
out = pl.concat([Z.select("MVT_ID_mvt", "apt", "day", C("hour").fill_null(((C("t") % 86400) // 3600).cast(pl.Int64)),
                          (C("src") == "taxinm").fill_null(False).cast(pl.Float64).alias("taxinm")).with_columns(pl.Series("cls", cls)), F],
                how="horizontal")
out.write_parquet(a.out)
print(out.group_by("cls").agg(pl.len(), *[C(c).is_not_nan().mean().round(3).alias(c) for c in ("m_c30", "s_c30", "m_day")]).sort("cls"))
print(f"-> {a.out} ({time.time() - T0:.0f}s)")
