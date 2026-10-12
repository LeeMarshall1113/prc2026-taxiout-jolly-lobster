"""experiment multipush 08: the duplicate-pair rule. A departure record WITH a plan whose partner (same airport, flight string, take-off date, take-off within 1800 s)
has NO plan: is its recorded off-block a copy of the schedule? Track record, leave-one-month-out gain, 2026 footprint."""
import sys
sys.path.insert(0, ".")
import numpy as np, polars as pl
C = pl.col
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
I = "data/interim/"
base = [C("MVT_ID_mvt").cast(pl.Float64).alias("id"), C("FLIGHT_mvt").alias("fl"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("mvt"),
        C("SCHED_TIME_UTC_mvt").dt.epoch("s").alias("sched"), C("AOBT_3_flt").dt.epoch("s").alias("aobt"), C("LOBT_flt").dt.epoch("s").alias("lobt"),
        C("STAND_mvt").cast(pl.Utf8).alias("stand"), C("AIRCRAFT_TYPE_mvt").alias("type"), C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day"),
        C("MVT_TIME_UTC_mvt").dt.month().alias("mon")]
def pairs(d):
    d = d.filter(C("fl").is_not_null()).sort("apt", "fl", "day", "mvt").with_columns(C("aobt").is_not_null().alias("plan"))
    d = d.with_columns(C("mvt").diff().over("apt", "fl", "day").alias("dm_p"), (-C("mvt").diff(-1).over("apt", "fl", "day")).alias("dm_n"),
                       C("plan").shift(1).over("apt", "fl", "day").alias("pl_p"), C("plan").shift(-1).over("apt", "fl", "day").alias("pl_n"),
                       pl.len().over("apt", "fl", "day").alias("n"))
    # partner = adjacent record within 1800 s with the opposite plan flag, group size exactly 2
    return d.with_columns(((C("n") == 2) & (((C("dm_p") < 1800) & (C("pl_p") != C("plan"))) | ((C("dm_n") < 1800) & (C("pl_n") != C("plan"))))).fill_null(False).alias("pair"))
d25 = pairs(pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP").select(base + [C("BLOCK_TIME_UTC_mvt").dt.epoch("s").alias("blk"), C("TAXITIME_SEC_mvt").alias("y")]).collect())
d26 = pairs(pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP").select(base).collect())
pp = d25.filter(C("pair") & C("plan")).with_columns((C("blk") - C("sched")).alias("b_s"), (C("mvt") - C("sched")).alias("gap"))
print("2025 plan-record-of-pair rows by airport"); print(pp.group_by("apt").agg(pl.len(), C("b_s").median().alias("med_bs"), (C("b_s").abs() <= 1300).mean().alias("sh_within1300"), C("y").median().alias("med_y")).sort("len", descending=True))
npk = d25.filter(C("pair") & ~C("plan"))
print("2025 no-plan record of pair:"); print(npk.group_by("apt").agg(pl.len(), C("y").median().alias("med_y"), (C("y") > 2000).mean().alias("sh_gt2000")).sort("len", descending=True))
lf = pp.filter(C("apt") == "LFPG")
print("LFPG plan-record b_s quantiles", np.percentile(lf["b_s"].to_numpy(), [0, 5, 10, 25, 50, 75, 90, 95, 100]))
print("n LFPG", lf.height, " |b_s|<=300:", (lf["b_s"].abs() <= 300).sum(), " <=1300:", (lf["b_s"].abs() <= 1300).sum(), " <=2000:", (lf["b_s"].abs() <= 2000).sum())
print("y vs gap-600: abs err quantiles", np.percentile(np.abs(lf["y"].to_numpy() - (lf["gap"].to_numpy() - 600)), [50, 75, 90, 100]))
T = pl.read_parquet(I + "agent_eb130_tbl.parquet").select("id", "pred", "pv", "mon", "y", "category").rename({"y": "yt", "mon": "m2"})
lf = lf.join(T, on="id", how="inner")
lf = lf.with_columns((C("yt") - C("pred")).alias("r"))
print("fold residual of LFPG plan-record rows: n", lf.height, "mean r", lf["r"].mean(), "rmse", float(np.sqrt((lf["r"] ** 2).mean())), " pred mean", lf["pred"].mean(), " category counts", lf["category"].value_counts())
lf.write_parquet(I + "agent_multipush_pairs_lfpg25.parquet")
print(lf.select("fl", "day", "m2", "gap", "b_s", "yt", "pred", "pv", "category").sort("m2").head(40))
# 2026 census
for nm, dd in (("2026", d26),):
    q = dd.filter(C("pair"))
    print("\n2026 pair rows:", q.height, " by airport/plan:"); print(q.group_by("apt", "plan").len().sort("apt", "plan"))
    q.filter(C("apt") == "LFPG").write_parquet(I + "agent_multipush_pairs_lfpg26_all.parquet")
