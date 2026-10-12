import polars as pl
d=pl.read_parquet("data/interim/agent_eb136_tbl.parquet"); print(d.columns)
print(d.select(["cplan"]).head(1))
w=pl.read_parquet("data/interim/agent_waithedge_tbl.parquet")
for c in ["cov","rule","est_kind"]: print(w.group_by(c).len().sort("len",descending=True).head(12))
t=pl.read_parquet("data/interim/local/logic_rows/C_dep2025_typed.parquet", n_rows=3); print(t.columns)
