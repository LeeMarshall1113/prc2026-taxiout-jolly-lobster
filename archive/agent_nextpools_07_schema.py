import polars as pl
for p in ["data/interim/agent_coverage_census_2026.parquet","data/interim/agent_coverage_census_2025.parquet"]:
    d=pl.read_parquet(p); print(p,d.height,d.schema); print(d.head(2))
    for c in d.columns:
        if "cov" in c or "class" in c: print(d.group_by(c).len().sort("len",descending=True).head(10))
