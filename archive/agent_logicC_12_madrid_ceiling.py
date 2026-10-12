"""logicC 12: Madrid and Barcelona labels never exceed an hour (2025: LEMD max 3,598 in 212,242 rows, LEBL max 3,604 in 179,705 rows; density thins to the edge:
LEMD 4 rows in 3,550-3,600 then 0 in 3,600-4,000; no plan row of either airport has a window edge above 3,600 in 2025 or in the scored file).
Served values above 3,606 are lowered to 3,606. Base v145. Out: data/interim/agent_logicC_LEMD_ceiling3606_final.parquet"""
import polars as pl
C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
CAP = 3606
r = pl.read_parquet("data/interim/local/logic_rows/C_dep2026.parquet").filter(C("apt").is_in(["LEMD", "LEBL"]) & (C("served") > CAP))
print(r.select(["apt", "flight", "sched_d", "sched_hm", "plan", "gap", "gl", "ga", "served"]))
cut = (r["served"] - CAP)
print("rows", r.height, "sum cut^2", float((cut.cast(pl.Float64) ** 2).sum()), "= %.4f s of RMSE (lower bound on the gain)" % (float((cut.cast(pl.Float64) ** 2).sum()) / 1.5e8))
b = pl.read_parquet("data/interim/agent_bundle_v145_final.parquet")
ids = set(r["id"].to_list())
vals = [CAP if i in ids else v for i, v in zip(b[ID].to_list(), b[T].to_list())]
out = b.with_columns(pl.Series(T, vals).cast(b.schema[T]))
out.write_parquet("data/interim/agent_logicC_LEMD_ceiling3606_final.parquet")
print("moved", int((out[T] != b[T]).sum()))
