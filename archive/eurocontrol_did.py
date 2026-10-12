"""Difference-in-differences: how much did taxi-out actually MOVE from 2025 to 2026, and did we follow?

    .venv/Scripts/python.exe archive/eurocontrol_did.py

Comparing our mean to theirs needs an offset, because they score ~92% of the flights we do and that
share moves. Their YEAR-OVER-YEAR change does not need one: the same methodology sits on both sides,
so a constant per-airport exclusion cancels.
    implied_true_2026(apt, month) = our_actual_2025(apt, month) + [ec_2026 - ec_2025](apt, month)
Our model is trained on 2025 and predicts 2026, so it cannot see a year-over-year level shift at all.
This asks whether one happened and whether v37 followed it. Residual coverage drift is the error term,
so their flight-count share is reported for both years and both months.
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
            .with_columns(((C("TOTAL_REF_TIME_MIN") + C("TOTAL_ADD_TIME_MIN")) / C("TOTAL_REF_NB_FL") * 60).alias("m"))
            .select(C("APT_ICAO").alias("apt"), C("MONTH_NUM").alias("mo"), "m", C("TOTAL_REF_NB_FL").alias("n")))


o25 = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
               (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y")).collect()
       .group_by("apt", "mo").agg(C("y").mean().alias("our25"), pl.len().alias("n25")))
v37 = pl.read_parquet("submissions/jolly-lobster_v37.parquet").select(ID, C("TAXITIME_SEC_mvt").alias("p"))
s26 = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C(ID), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo")).collect()
       .join(v37, on=ID, how="inner").group_by("apt", "mo").agg(C("p").mean().alias("v37"), pl.len().alias("n26")))
a = (ec(2025).rename({"m": "ec25", "n": "ecn25"}).join(ec(2026).rename({"m": "ec26", "n": "ecn26"}), on=["apt", "mo"])
     .join(o25, on=["apt", "mo"]).join(s26, on=["apt", "mo"]).filter(C("mo").is_in([1, 7])))
a = a.with_columns((C("ec26") - C("ec25")).alias("ec_shift"), (C("ecn25") / C("n25")).alias("cov25"),
                   (C("ecn26") / C("n26")).alias("cov26"))
a = a.with_columns((C("our25") + C("ec_shift")).alias("implied26")).with_columns((C("implied26") - C("v37")).alias("gap")).sort("mo", "apt")
print("their year-over-year shift, what it implies for 2026, and what v37 predicts (seconds)\n")
print(f"{'apt':<6}{'mo':>3}{'rows':>8}{'cov25':>7}{'cov26':>7}{'ec 2025':>9}{'ec 2026':>9}{'shift':>7}{'our 2025':>10}{'implied':>9}{'v37':>8}{'gap':>7}")
for r in a.select("apt", "mo", "n26", "cov25", "cov26", "ec25", "ec26", "ec_shift", "our25", "implied26", "v37", "gap").iter_rows():
    ap, mo, n, c5, c6, e5, e6, sh, o5, im, v, gp = r
    print(f"{ap:<6}{mo:>3}{n:>8,}{c5:>7.2f}{c6:>7.2f}{e5:>9,.0f}{e6:>9,.0f}{sh:>7,.0f}{o5:>10,.0f}{im:>9,.0f}{v:>8,.0f}{gp:>7,.0f}")
N, base = a["n26"].sum(), 242.7021
g = a["gap"].to_numpy()
n = a["n26"].to_numpy()
tot = float((g ** 2 * n).sum() / N)
print(f"\nJan mean gap {a.filter(C('mo')==1)['gap'].mean():+.0f}s, Jul {a.filter(C('mo')==7)['gap'].mean():+.0f}s")
print(f"if every gap were pure bias and corrected exactly: {tot:,.0f} s^2/row -> {np.sqrt(max(base**2-tot,0))-base:+.2f}s")
half = float(((g / 2) ** 2 * n).sum() / N)
print(f"correcting HALF of each gap (the usual hedge against the offset being wrong): "
      f"{np.sqrt(max(base**2 - (tot - half), 0)) - base:+.2f}s")
print(f"coverage drift 2025->2026: Jan {a.filter(C('mo')==1)['cov26'].mean()-a.filter(C('mo')==1)['cov25'].mean():+.3f}, "
      f"Jul {a.filter(C('mo')==7)['cov26'].mean()-a.filter(C('mo')==7)['cov25'].mean():+.3f} "
      f"-- this is the assumption the whole calculation rests on")
