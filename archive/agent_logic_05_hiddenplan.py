"""experiment logic 05: hidden-plan bounds re-verified on raw 2025 (plan recovered from the destination's arrival record), and the
served v145 values of scored no-plan rows against those bounds.

    .venv/Scripts/python.exe archive/capped_run.py --cap-gb 3 -- .venv/Scripts/python.exe archive/agent_logic_05_hiddenplan.py

Join: no-plan departure at airport A, flight string F, destination D (one of the ten); arrival record at D with the same flight
string, plan present, ADEP_flt == A, landing 0 to 8 h after the departure's take-off. gl = MVT - LOBT, ga = MVT - AOBT_3.
Lower bound: gl < 3,606  ->  y > gl + 3,606.  Upper bound: gl >= 3,606 and |gap - gl| < 3,606  ->  y < gl - 3,606.
"""
import os
import polars as pl

C = pl.col
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(230)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, ".local", "logic_rows", "audit")
RAW = os.path.join(ROOT, "data", "raw")
W = 3606
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def arrivals(src):
    return (pl.scan_parquet(src).filter((C("PHASE_mvt") == "ARR") & C("AOBT_3_flt").is_not_null() & C("ADEP_flt").is_in(APTS))
            .select(C("FLIGHT_mvt").alias("flight"), C("ADES_mvt").alias("ades"), C("ADEP_flt").alias("apt"),
                    C("MVT_TIME_UTC_mvt").alias("land"), C("LOBT_flt").alias("lobt"), C("AOBT_3_flt").alias("aobt3"))
            .filter(C("flight").is_not_null()).collect())


def recover(dep, arr):
    j = (dep.join(arr, on=["flight", "apt", "ades"], how="inner")
         .with_columns((C("land") - C("mvt")).dt.total_seconds().alias("dt"))
         .filter((C("dt") >= 0) & (C("dt") <= 8 * 3600))
         .with_columns((C("mvt") - C("lobt")).dt.total_seconds().alias("gl"), (C("mvt") - C("aobt3")).dt.total_seconds().alias("ga")))
    # one plan per departure: the earliest landing
    return j.sort("dt").unique(subset=["id"], keep="first")


cols = ["id", "apt", "flight", "ades", "plan", "mvt", "gap", "month"]
dep = pl.read_parquet(os.path.join(D, "dep2025.parquet"), columns=cols + ["y"]).filter(~C("plan") & C("flight").is_not_null()).drop("plan")
arr = arrivals(os.path.join(RAW, "training_*.parquet"))
h = recover(dep, arr).with_columns(
    (C("gl") < W).alias("lower_scope"), ((C("gl") >= W) & ((C("gap") - C("gl")).abs() < W)).alias("upper_scope"),
    (C("y") > C("gl") + W).alias("lower_ok"), (C("y") < C("gl") - W).alias("upper_ok"), ((C("y") - C("gap")).abs() <= 60).alias("copy"))
print("=== HP1. 2025 no-plan departures with a plan recovered from the destination's arrival record (flight-string key) ===")
print(h.group_by("apt").agg(n=pl.len(), lower_n=C("lower_scope").sum(), lower_exc=(C("lower_scope") & ~C("lower_ok")).sum(),
                             upper_n=C("upper_scope").sum(), upper_exc=(C("upper_scope") & ~C("upper_ok")).sum(),
                             upper_copies=(C("upper_scope") & C("copy")).sum()).sort("apt"))
print("HP1b. Rome and Istanbul by month (lower scope n / exceptions, upper scope n / exceptions):")
print(h.filter(C("apt").is_in(["LIRF", "LTFM"])).group_by("apt", "month").agg(
    lower_n=C("lower_scope").sum(), lower_exc=(C("lower_scope") & ~C("lower_ok")).sum(),
    upper_n=C("upper_scope").sum(), upper_exc=(C("upper_scope") & ~C("upper_ok")).sum()).sort("apt", "month"))
print("HP1c. exceptions at Rome / Istanbul listed:")
print(h.filter(C("apt").is_in(["LIRF", "LTFM"]) & ((C("lower_scope") & ~C("lower_ok")) | (C("upper_scope") & ~C("upper_ok"))))
      .select("apt", "flight", "mvt", "gap", "gl", "ga", "y", "lower_scope", "upper_scope"))
print("HP1d. Rome reading: y - ga on recovered rows in the upper scope: quantiles and share within 600 s")
r = h.filter((C("apt") == "LIRF") & C("upper_scope")).with_columns((C("y") - C("ga")).alias("y_ga"))
print(r.select(n=pl.len(), p05=C("y_ga").quantile(0.05), p50=C("y_ga").median(), p95=C("y_ga").quantile(0.95), within600=(C("y_ga").abs() <= 600).mean().round(3)))
print("HP1e. lever B: Rome recovered rows with |LOBT - SCHED| <= 3,606 and AOBT_3 - LOBT >= 3,606: labels")
lb = h.filter((C("apt") == "LIRF") & ((C("gap") - C("gl")).abs() <= W) & ((C("gl") - C("ga")) >= W))
print(lb.select("flight", "mvt", "gap", "gl", "ga", "y", "copy"))

print("\n=== HP2. Scored no-plan rows with a recovered plan: served v145 against the bounds ===")
rk = pl.read_parquet(os.path.join(D, "dep2026.parquet"), columns=cols + ["served"]).filter(~C("plan") & C("flight").is_not_null()).drop("plan")
arr26 = arrivals(os.path.join(RAW, "ranking.parquet"))
h26 = recover(rk, arr26).with_columns(
    (C("gl") < W).alias("lower_scope"), ((C("gl") >= W) & ((C("gap") - C("gl")).abs() < W)).alias("upper_scope"),
    (C("served") > C("gl") + W).alias("lower_ok"), (C("served") < C("gl") - W).alias("upper_ok"))
print(h26.group_by("apt").agg(n=pl.len(), lower_n=C("lower_scope").sum(), lower_viol=(C("lower_scope") & ~C("lower_ok")).sum(),
                               upper_n=C("upper_scope").sum(), upper_viol=(C("upper_scope") & ~C("upper_ok")).sum()).sort("apt"))
print("HP2b. violations at Rome and Istanbul listed (served vs bound):")
v = h26.filter(C("apt").is_in(["LIRF", "LTFM"]) & ((C("lower_scope") & ~C("lower_ok")) | (C("upper_scope") & ~C("upper_ok"))))
print(v.select("apt", "flight", "mvt", "gap", "gl", "ga", "served", "lower_scope", "upper_scope",
               pl.when(C("lower_scope")).then(C("gl") + W - C("served")).otherwise(C("served") - (C("gl") - W)).alias("by")).sort("by", descending=True))
print("HP2c. Rome scored rows in the upper scope: served - ga quantiles (the reading says ~0):")
print(h26.filter((C("apt") == "LIRF") & C("upper_scope")).with_columns((C("served") - C("ga")).alias("s_ga")).select(
    n=pl.len(), p05=C("s_ga").quantile(0.05), p50=C("s_ga").median(), p95=C("s_ga").quantile(0.95), within600=(C("s_ga").abs() <= 600).mean().round(3)))
h26.write_parquet(os.path.join(D, "hidden26.parquet"))
