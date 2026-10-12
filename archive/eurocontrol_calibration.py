"""Is EUROCONTROL's per-airport offset stable enough to calibrate our 2026 predictions?

    .venv/Scripts/python.exe archive/eurocontrol_calibration.py

Step 1 showed their published mean taxi-out tracks ours at r=0.978 across 120 airport-months of 2025,
with a per-airport offset (they score ~92% of the flights we do). A per-airport offset is only useful
if it is STABLE month to month: the calibration is
    true_mean(2026, apt, month) ~= ec_mean(2026, apt, month) + offset(apt),
with offset(apt) fitted on 2025, so the month-to-month spread of that offset IS the error bar.
Then: compare the implied 2026 means against what v37 actually predicts, airport by airport, for the
two scored months. A gap there is a bias we cannot otherwise see -- our model is trained on 2025 and
predicts 2026, and nothing in our validation can detect a year-over-year level shift.
Leave-one-month-out is used for the offset so the reported spread is out-of-sample.
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
            .select(C("APT_ICAO").alias("apt"), C("MONTH_NUM").alias("mo"), "ec_mean"))


ours = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
        .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
                (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y")).collect()
        .group_by("apt", "mo").agg(C("y").mean().alias("our_mean"), pl.len().alias("n")))
j = ec(2025).join(ours, on=["apt", "mo"], how="inner").with_columns((C("our_mean") - C("ec_mean")).alias("off"))
print("2025 offset (our mean - theirs), per airport, and its month-to-month spread:\n")
print(f"{'apt':<6}{'offset':>9}{'sd':>7}{'min':>7}{'max':>7}{'LOMO |err|':>12}")
lomo = {}
for a in APTS:
    g = j.filter(C("apt") == a).sort("mo")
    o = g["off"].to_numpy()
    err = [abs(o[i] - np.delete(o, i).mean()) for i in range(len(o))]   # leave-one-month-out
    lomo[a] = float(np.mean(err))
    print(f"{a:<6}{o.mean():>9.0f}{o.std(ddof=1):>7.0f}{o.min():>7.0f}{o.max():>7.0f}{np.mean(err):>12.0f}")
off = dict(j.group_by("apt").agg(C("off").mean().alias("o")).iter_rows())
print(f"\nmean leave-one-month-out error of the offset: {np.mean(list(lomo.values())):.0f}s "
      f"-- that is the precision of any calibration built on it")

e26 = ec(2026).filter(C("mo").is_in([1, 7]))
v37 = pl.read_parquet("submissions/jolly-lobster_v37.parquet").select(ID, C("TAXITIME_SEC_mvt").alias("p"))
scored = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
          .select(C(ID), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo")).collect()
          .join(v37, on=ID, how="inner").group_by("apt", "mo").agg(C("p").mean().alias("v37_mean"), pl.len().alias("n")))
k = e26.join(scored, on=["apt", "mo"], how="inner").with_columns(
    (C("ec_mean") + C("apt").replace_strict(off, default=None)).alias("implied_true"))
k = k.with_columns((C("implied_true") - C("v37_mean")).alias("gap")).sort("apt", "mo")
print("\n2026 scored months: what their data implies the true mean is, vs what v37 predicts\n")
print(f"{'apt':<6}{'mo':>4}{'rows':>8}{'their mean':>12}{'implied true':>14}{'v37 mean':>10}{'gap':>8}")
for a, mo, ecm, v, n, imp, gap in k.select("apt", "mo", "ec_mean", "v37_mean", "n", "implied_true", "gap").iter_rows():
    print(f"{a:<6}{mo:>4}{n:>8,}{ecm:>12,.0f}{imp:>14,.0f}{v:>10,.0f}{gap:>8,.0f}")
N = k["n"].sum()
gain = float((k["gap"].to_numpy() ** 2 * k["n"].to_numpy()).sum() / N)
base = 242.7021
print(f"\nrows covered: {N:,} of 344,841")
print(f"IF every gap were pure bias and we corrected it exactly: {gain:,.0f} s^2 per row -> "
      f"{np.sqrt(max(base ** 2 - gain, 0)) - base:+.2f}s on the board")
print(f"The offset's own leave-one-month-out error is {np.mean(list(lomo.values())):.0f}s, which enters the same way: "
      f"a spurious 'bias' of that size costs {np.mean(list(lomo.values())) ** 2 / (2 * base):.2f}s if we chase it.")
