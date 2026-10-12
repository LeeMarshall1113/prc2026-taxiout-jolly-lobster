"""Does COVERAGE alone explain the apparent January-2026 bias? Fit the rate on 2025 and apply it.

    .venv/Scripts/python.exe archive/eurocontrol_coverage.py

Their published mean sits below ours by an offset that moves with how many flights they score. Fit
    offset(apt, month) = a(apt) + b * (1 - coverage)
on 2025's 120 airport-months, with b shared (the mechanism -- excluded flights are long ones -- should
not be airport-specific) and a(apt) free. Then predict 2026's offset from its own coverage and see what
is left of the January gap. If the residual gaps collapse toward zero, January 2026's apparent
100-second over-prediction is coverage drift and correcting it would have been a large, costly mistake.
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


o25 = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
               (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y")).collect()
       .group_by("apt", "mo").agg(C("y").mean().alias("our"), pl.len().alias("n")))
j = ec(2025).join(o25, on=["apt", "mo"]).with_columns((C("our") - C("m")).alias("off"), (1 - C("ecn") / C("n")).alias("miss"))
# design: airport dummies + shared slope on the missing share
A = np.zeros((j.height, len(APTS) + 1))
for i, ap in enumerate(j["apt"].to_list()):
    A[i, APTS.index(ap)] = 1.0
A[:, -1] = j["miss"].to_numpy()
y = j["off"].to_numpy()
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
pred = A @ coef
b = coef[-1]
print(f"fit on 2025 ({j.height} airport-months): offset = a(airport) + b * missing_share")
print(f"  b = {b:,.0f} s per unit missing share  (i.e. {b/10:.0f}s per 10 percentage points of coverage lost)")
print(f"  residual: mean |err| {np.abs(y - pred).mean():.0f}s, sd {(y - pred).std():.0f}s "
      f"(against {np.abs(y - y.mean()).mean():.0f}s with no model)")
a_apt = dict(zip(APTS, coef[:len(APTS)]))

v37 = pl.read_parquet("submissions/jolly-lobster_v37.parquet").select(ID, C("TAXITIME_SEC_mvt").alias("p"))
s26 = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C(ID), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo")).collect()
       .join(v37, on=ID, how="inner").group_by("apt", "mo").agg(C("p").mean().alias("v37"), pl.len().alias("n")))
k = ec(2026).filter(C("mo").is_in([1, 7])).join(s26, on=["apt", "mo"])
k = k.with_columns((1 - C("ecn") / C("n")).alias("miss"))
k = k.with_columns((C("apt").replace_strict(a_apt, default=None) + b * C("miss")).alias("off_pred"))
k = k.with_columns((C("m") + C("off_pred")).alias("implied"), ).with_columns((C("implied") - C("v37")).alias("gap")).sort("mo", "apt")
print(f"\n2026 with the coverage-aware offset:\n")
print(f"{'apt':<6}{'mo':>3}{'rows':>8}{'missing':>9}{'offset used':>13}{'implied true':>14}{'v37':>8}{'gap':>7}")
for ap, mo, n, ms, op, im, v, gp in k.select("apt", "mo", "n", "miss", "off_pred", "implied", "v37", "gap").iter_rows():
    print(f"{ap:<6}{mo:>3}{n:>8,}{ms:>9.2f}{op:>13,.0f}{im:>14,.0f}{v:>8,.0f}{gp:>7,.0f}")
N, base = k["n"].sum(), 242.7021
g, nn = k["gap"].to_numpy(), k["n"].to_numpy()
tot = float((g ** 2 * nn).sum() / N)
print(f"\nJan mean gap {k.filter(C('mo')==1)['gap'].mean():+.0f}s, Jul {k.filter(C('mo')==7)['gap'].mean():+.0f}s")
print(f"potential if every remaining gap were真 bias: {tot:,.0f} s^2/row -> {np.sqrt(max(base**2-tot,0))-base:+.2f}s")
print(f"the fit's own residual is {(y - pred).std():.0f}s per airport-month, which costs "
      f"{(y - pred).std() ** 2 / (2 * base):.2f}s if mistaken for bias -- compare that with the line above")
