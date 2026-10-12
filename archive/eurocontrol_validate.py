"""Does EUROCONTROL's published taxi-out match OUR target? Validate on 2025, where we have labels.

    .venv/Scripts/python.exe archive/eurocontrol_validate.py

ansperformance.eu publishes, per airport-month, the number of reference flights, their total
"unimpeded" reference time and the total additional time (EUROCONTROL, open data, attribution
required). Mean actual taxi-out = (REF_TIME + ADD_TIME) / NB_FL minutes. Our target is y = MVT - BLOCK.
If the two agree on 2025, then their Jan and Jul 2026 figures measure the mean of the very thing we
are predicting on the scored months -- a calibration constraint on every prediction whose labels we
do not have. If they disagree, this source is not about our quantity and we drop it.
Reported per airport-month: their mean, ours, the difference, and our row count -- plus the same for
the copy-free subset, since a schedule-copy row's y is not a taxi time at all.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
ec = (pl.read_csv("data/external/eurocontrol/taxi_out_additional_time_2025.csv")
      .filter(C("APT_ICAO").is_in(APTS))
      .with_columns(((C("TOTAL_REF_TIME_MIN") + C("TOTAL_ADD_TIME_MIN")) / C("TOTAL_REF_NB_FL") * 60).alias("ec_mean"))
      .select(C("APT_ICAO").alias("apt"), C("MONTH_NUM").alias("mo"), "ec_mean", C("TOTAL_REF_NB_FL").alias("ec_n")))
ours = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
        .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
                (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y")).collect())
agg = (ours.group_by("apt", "mo").agg(pl.len().alias("n"), C("y").mean().alias("our_mean"),
                                      C("y").filter((C("y") - C("gs")).abs() > 60).mean().alias("our_mean_nocopy"),
                                      C("y").filter((C("y") - C("gs")).abs() > 60).len().alias("n_nocopy")))
j = ec.join(agg, on=["apt", "mo"], how="inner").sort("apt", "mo")
j = j.with_columns((C("our_mean") - C("ec_mean")).alias("d_all"), (C("our_mean_nocopy") - C("ec_mean")).alias("d_nocopy"))
print("2025, per airport-month: EUROCONTROL's implied mean taxi-out vs ours (seconds)\n")
print(f"{'apt':<6}{'months':>7}{'their n':>10}{'our n':>10}{'their mean':>12}{'our mean':>10}{'diff':>8}{'diff(no copies)':>17}")
for a in APTS:
    g = j.filter(C("apt") == a)
    if not g.height:
        continue
    print(f"{a:<6}{g.height:>7}{g['ec_n'].sum():>10,}{g['n'].sum():>10,}{g['ec_mean'].mean():>12,.0f}"
          f"{g['our_mean'].mean():>10,.0f}{g['d_all'].mean():>8,.0f}{g['d_nocopy'].mean():>17,.0f}")
d = j["d_all"].to_numpy()
dn = j["d_nocopy"].to_numpy()
print(f"\nacross {j.height} airport-months: mean diff {d.mean():+.0f}s (sd {d.std():.0f}), "
      f"no-copy {dn.mean():+.0f}s (sd {dn.std():.0f})")
print(f"correlation of their monthly mean with ours: all {np.corrcoef(j['ec_mean'], j['our_mean'])[0,1]:.4f}, "
      f"no-copy {np.corrcoef(j['ec_mean'], j['our_mean_nocopy'])[0,1]:.4f}")
print(f"their flight counts vs ours: ratio {j['ec_n'].sum() / j['n'].sum():.3f}")
