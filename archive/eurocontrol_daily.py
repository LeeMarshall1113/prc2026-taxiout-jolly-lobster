"""Does EUROCONTROL's DAILY ATC pre-departure delay explain our day-level residual?

    .venv/Scripts/python.exe archive/eurocontrol_daily.py

The monthly taxi-out file gives a level per airport-month; this one is per airport-DAY and covers
2026-01-01..08-31, so it spans both scored months. ATC pre-departure delay is stand-holding time
attributed to ATC, which is NOT taxi time by definition -- but it is a direct measure of how backed up
the departure operation was that day, which is. Question: on 2025, where we have labels, does it predict
the sign and size of our day-level error?
Residuals come from the fold predictions with production's chain applied (gap clamp, LOBT box), Rome
scored only on the gate-inclusive folds, as always.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, W = "MVT_ID_mvt", 3606.0
FOLDS = [("1_7", "results/preds_q_gate_1_7.parquet", False), ("2_8", "results/preds_q_gate_2_8.parquet", False),
         ("3_9", "results/preds_n3_bulk_3_9.parquet", True), ("4_10", "results/preds_s4_ctrl_4_10.parquet", True),
         ("5_11", "results/preds_c_ctrl_5_11.parquet", True)]
raw = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C(ID), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day"),
               C("MVT_TIME_UTC_mvt").dt.month().alias("mo"), C("AOBT_3_flt").is_not_null().alias("plan"),
               (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
               (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl")).collect())
parts = []
for label, path, drop_lirf in FOLDS:
    P = pl.read_parquet(path).select(ID, "y", "bagged").join(raw, on=ID, how="inner")
    if drop_lirf:
        P = P.filter(C("apt") != "LIRF")
    y = P["y"].to_numpy().astype(float)
    gs = P["gs"].fill_null(np.nan).to_numpy().astype(float)
    gl = P["gl"].fill_null(np.nan).to_numpy().astype(float)
    plan = P["plan"].fill_null(False).to_numpy()
    p = np.clip(P["bagged"].to_numpy().astype(float), 0, None)
    p = np.where(np.isfinite(gs) & (gs >= 5000) & (gs < 50000) & (p > gs), gs, p)
    box = plan & np.isfinite(gl)
    p = np.where(box, np.clip(p, np.where(box, gl - W, -np.inf), np.where(box, gl + W, np.inf)), p)
    parts.append(P.select("apt", "day", "mo").with_columns(pl.Series("resid", y - np.clip(p, 0, None))))
res = pl.concat(parts).group_by("apt", "day", "mo").agg(pl.len().alias("n"), C("resid").mean().alias("resid"))
d25 = (pl.read_csv("data/external/eurocontrol/atc_pre_departure_delays_2025.csv")
       .select(C("APT_ICAO").alias("apt"), C("FLT_DATE").str.slice(0, 10).alias("day"),
               C("FLT_DEP_1").alias("dep"), C("DLY_ATC_PRE_2").alias("dly2"), C("DLY_ATC_PRE_3").alias("dly3")))
j = res.join(d25, on=["apt", "day"], how="inner").filter(C("dep") > 0)
j = j.with_columns((C("dly2").fill_null(0) / C("dep")).alias("dpm2"), (C("dly3").fill_null(0) / C("dep")).alias("dpm3"))
print(f"airport-days matched: {j.height:,} (of {res.height:,} in the folds)\n")
print(f"{'apt':<6}{'days':>6}{'corr(resid, dly/dep)':>22}{'mean resid':>12}{'mean dly/dep(min)':>19}")
for a in sorted(set(j["apt"].to_list())):
    g = j.filter(C("apt") == a)
    if g.height < 30:
        continue
    r2 = np.corrcoef(g["resid"], g["dpm2"])[0, 1]
    print(f"{a:<6}{g.height:>6}{r2:>22.3f}{g['resid'].mean():>12.1f}{g['dpm2'].mean():>19.2f}")
allr = np.corrcoef(j["resid"], j["dpm2"])[0, 1]
print(f"\npooled correlation: {allr:.3f}   (resid = truth - our prediction, so positive means a busy day is under-predicted)")
hi = j.filter(C("dpm2") > C("dpm2").quantile(0.95))
lo = j.filter(C("dpm2") < C("dpm2").quantile(0.5))
print(f"top 5% delay days: mean residual {hi['resid'].mean():+.0f}s over {hi.height} airport-days; "
      f"bottom half: {lo['resid'].mean():+.0f}s")
print(f"an uncaptured {hi['resid'].mean():+.0f}s on 5% of rows would be worth "
      f"{(hi['resid'].mean() ** 2 * 0.05) / (2 * 242.7):.2f}s if corrected perfectly")
