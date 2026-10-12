"""Is the January gap real, or does EUROCONTROL's offset itself have a season?

    .venv/Scripts/python.exe archive/eurocontrol_seasonal.py

Using an ANNUAL per-airport offset to translate their 2026 mean into "our" units assumes the offset has
no seasonal component. If what their metric excludes (they score ~92% of our flights) is winter tails,
January's offset will be larger than the year's -- and a January-only "bias" appears that is purely the
method's. So: compare the January-2025 and July-2025 offsets against the annual one, per airport, and
redo the 2026 comparison with MONTH-MATCHED offsets.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, APTS = "MVT_ID_mvt", ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def ec(year):
    return (pl.read_csv(f"data/external/eurocontrol/taxi_out_additional_time_{year}.csv")
            .filter(C("APT_ICAO").is_in(APTS))
            .with_columns(((C("TOTAL_REF_TIME_MIN") + C("TOTAL_ADD_TIME_MIN")) / C("TOTAL_REF_NB_FL") * 60).alias("ec_mean"))
            .select(C("APT_ICAO").alias("apt"), C("MONTH_NUM").alias("mo"), "ec_mean", C("TOTAL_REF_NB_FL").alias("ec_n")))


ours = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
        .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
                (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y")).collect()
        .group_by("apt", "mo").agg(C("y").mean().alias("our_mean"), pl.len().alias("n")))
j = ec(2025).join(ours, on=["apt", "mo"], how="inner").with_columns((C("our_mean") - C("ec_mean")).alias("off"))
print("2025: is the offset seasonal? (their coverage = their n / our n)\n")
print(f"{'apt':<6}{'annual':>8}{'Jan':>7}{'Jul':>7}{'Jan-ann':>9}{'Jul-ann':>9}{'cov Jan':>9}{'cov Jul':>9}")
jan_off, jul_off = {}, {}
for a in APTS:
    g = j.filter(C("apt") == a)
    ann = g["off"].mean()
    j1 = g.filter(C("mo") == 1)
    j7 = g.filter(C("mo") == 7)
    jan_off[a], jul_off[a] = float(j1["off"][0]), float(j7["off"][0])
    print(f"{a:<6}{ann:>8.0f}{jan_off[a]:>7.0f}{jul_off[a]:>7.0f}{jan_off[a]-ann:>9.0f}{jul_off[a]-ann:>9.0f}"
          f"{j1['ec_n'][0]/j1['n'][0]:>9.2f}{j7['ec_n'][0]/j7['n'][0]:>9.2f}")

v37 = pl.read_parquet("submissions/jolly-lobster_v37.parquet").select(ID, C("TAXITIME_SEC_mvt").alias("p"))
scored = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
          .select(C(ID), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo")).collect()
          .join(v37, on=ID, how="inner").group_by("apt", "mo").agg(C("p").mean().alias("v37_mean"), pl.len().alias("n")))
k = ec(2026).filter(C("mo").is_in([1, 7])).join(scored, on=["apt", "mo"], how="inner")
k = k.with_columns(pl.struct("apt", "mo").map_elements(
    lambda s: (jan_off if s["mo"] == 1 else jul_off)[s["apt"]], return_dtype=pl.Float64).alias("off_matched"))
k = k.with_columns((C("ec_mean") + C("off_matched")).alias("implied"), (C("ec_n") / C("n")).alias("cov"))
k = k.with_columns((C("implied") - C("v37_mean")).alias("gap")).sort("mo", "apt")
print("\n2026 with MONTH-MATCHED offsets (Jan-2025 offset for Jan-2026, Jul for Jul)\n")
print(f"{'apt':<6}{'mo':>4}{'rows':>8}{'cov':>6}{'implied true':>14}{'v37 mean':>10}{'gap':>8}")
for a, mo, n, cov, imp, v, gap in k.select("apt", "mo", "n", "cov", "implied", "v37_mean", "gap").iter_rows():
    print(f"{a:<6}{mo:>4}{n:>8,}{cov:>6.2f}{imp:>14,.0f}{v:>10,.0f}{gap:>8,.0f}")
N, base = k["n"].sum(), 242.7021
for label, col in (("annual offset", None), ("month-matched", "gap")):
    if col is None:
        continue
    g2 = float((k[col].to_numpy() ** 2 * k["n"].to_numpy()).sum() / N)
    print(f"\n{label}: {g2:,.0f} s^2 per row if every gap were pure bias -> {np.sqrt(max(base**2-g2,0))-base:+.2f}s")
print(f"Jan gaps: {k.filter(C('mo')==1)['gap'].mean():+.0f}s mean; Jul gaps: {k.filter(C('mo')==7)['gap'].mean():+.0f}s mean")
