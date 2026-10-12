"""Is their offset just TRIMMING? Reproduce their published mean by dropping our longest flights.

    .venv/Scripts/python.exe archive/eurocontrol_trim.py

They score a share `cov` of the flights we see (0.67-0.99, varying by airport and month). If the flights
they drop are the long ones -- no unimpeded reference exists for a disrupted or de-iced departure -- then
their mean is our TRIMMED mean, the offset is mechanical, and a year with lower coverage produces a lower
published mean with no real change in taxi time. That would make January 2026's apparent 100-second bias
an artifact of coverage, not a bias in our predictions.
Test on 2025, where we have every label: for each airport-month, trim our y distribution to their
coverage share (drop the longest 1-cov) and compare with their published mean. Compare the residual of
that against the residual of the untrimmed mean. Then re-do the 2026 comparison in trimmed space.
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
            .select(C("APT_ICAO").alias("apt"), C("MONTH_NUM").alias("mo"), "m", C("TOTAL_REF_NB_FL").alias("ecn")))


rows = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
        .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
                (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y")).collect())
e25 = ec(2025)
res_raw, res_trim = [], []
print("2025: their published mean vs ours, untrimmed and trimmed to their coverage\n")
print(f"{'apt':<6}{'mo':>3}{'cov':>6}{'theirs':>9}{'ours':>8}{'raw res':>9}{'trimmed':>9}{'trim res':>10}")
for r in e25.join(rows.group_by("apt", "mo").agg(pl.len().alias("n")), on=["apt", "mo"]).sort("apt", "mo").iter_rows(named=True):
    g = rows.filter((C("apt") == r["apt"]) & (C("mo") == r["mo"]))["y"].to_numpy()
    cov = min(r["ecn"] / r["n"], 1.0)
    trimmed = float(np.sort(g)[:max(int(round(cov * len(g))), 1)].mean())
    res_raw.append(g.mean() - r["m"])
    res_trim.append(trimmed - r["m"])
    if r["mo"] in (1, 7):
        print(f"{r['apt']:<6}{r['mo']:>3}{cov:>6.2f}{r['m']:>9,.0f}{g.mean():>8,.0f}{res_raw[-1]:>9,.0f}{trimmed:>9,.0f}{res_trim[-1]:>10,.0f}")
rr, rt = np.array(res_raw), np.array(res_trim)
print(f"\nacross all {len(rr)} airport-months of 2025:")
print(f"  untrimmed: mean residual {rr.mean():+.0f}s, sd {rr.std():.0f}, mean |residual| {np.abs(rr).mean():.0f}s")
print(f"  trimmed  : mean residual {rt.mean():+.0f}s, sd {rt.std():.0f}, mean |residual| {np.abs(rt).mean():.0f}s")
print("\nIf trimming collapses the residual, their number measures a trimmed population and the")
print("year-over-year 'bias' we computed is coverage drift, not a level shift in taxi time.")
