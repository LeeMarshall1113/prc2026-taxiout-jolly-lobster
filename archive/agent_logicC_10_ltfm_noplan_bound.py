"""logicC 10: Istanbul no-plan rows that took off at or after their schedule (gap >= 0): the recorded off-block is never more than
C = 1,200 s before the schedule, i.e. label <= gap + 1,200 (2025: 0 exceptions of 3,585 rows, max 903 s; all months incl. the February snow month).
Served rows above gap + 1,200 are lowered to it. Base: data/interim/agent_bundle_v145_final.parquet.
Out: data/interim/agent_logicC_LTFM_noplan-early-bound_final.parquet"""
import numpy as np, polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
C = pl.col
CC = 1200
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
# 2025 verification (printed)
d = pl.read_parquet("data/interim/local/logic_rows/C_dep2025_typed.parquet", columns=["apt", "plan", "gap", "y", "month"]).filter((C("apt") == "LTFM") & (~C("plan")) & (C("gap") >= 0)).with_columns((C("y") - C("gap")).alias("early"))
print("2025 LTFM no-plan gap>=0: n", d.height, "max early", d["early"].max(), "exceptions >", CC, ":", int((d["early"] > CC).sum()), "by month n:", d.group_by("month").len().sort("month")["len"].to_list())
print("   early>900:", int((d["early"] > 900).sum()), " >800:", int((d["early"] > 800).sum()))
r = pl.read_parquet("data/interim/local/logic_rows/C_dep2026.parquet").filter((C("apt") == "LTFM") & (~C("plan")) & (C("gap") >= 0))
x = r.with_columns((C("gap") + CC).alias("bound")).filter(C("served") > C("bound"))
x = x.with_columns(C("bound").alias("new"), (C("served") - C("bound")).alias("cut"))
print("2026 rows lowered:", x.height, "total cut s", int(x["cut"].sum()), "sum cut^2", float((x["cut"] ** 2).sum()))
print(x.select(["flight", "sched_d", "sched_hm", "mvt_hm", "gap", "served", "new", "cut", "ades", "stand", "runway"]).sort("cut", descending=True))
# stake under alternatives: truth = bound; truth = gap (copy-like); truth = 2025 median label of no-plan gap<3000 (about 1,000)
for nm, tv in [("truth=bound", x["new"]), ("truth=gap", x["gap"]), ("truth=1000", pl.Series([1000] * x.height))]:
    s = float(((x["served"] - tv) ** 2).sum() - ((x["new"] - tv) ** 2).sum())
    print(f"gain in squared error if {nm}: {s:.3e} = {s / (2 * 344841 * 218.0 ** 1):.3f}")  # placeholder scale, replaced below
tot = 2 * 344841 * 218.0  # scale used by earlier notes: 1 s of RMSE ~ 2 x n x RMSE = 1.5e8
for nm, tv in [("truth=bound", x["new"]), ("truth=gap", x["gap"]), ("truth=1000", pl.Series([1000] * x.height))]:
    s = float(((x["served"] - tv) ** 2).sum() - ((x["new"] - tv) ** 2).sum())
    print(f"  {nm}: gain {s:.3e} sq s = {s / tot:.3f} s of RMSE")
# worst case: truth = served (the row really is at the served value)
s = float(((x["served"] - x["new"]) ** 2).sum()); print(f"  truth=served: loss {s:.3e} sq s = {s / tot:.3f} s")
# write
b = pl.read_parquet("data/interim/agent_bundle_v145_final.parquet")
mp = dict(zip(x["id"].to_list(), x["new"].to_list()))
vals = [int(mp.get(i, v)) for i, v in zip(b[ID].to_list(), b[T].to_list())]
out = b.with_columns(pl.Series(T, vals).cast(b.schema[T]))
out.write_parquet("data/interim/agent_logicC_LTFM_noplan-early-bound_final.parquet")
print("wrote; moved", int((out[T] != b[T]).sum()))
