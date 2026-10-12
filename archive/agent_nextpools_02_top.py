import numpy as np, polars as pl
C = pl.col
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
SE=1.5e8
t = pl.read_parquet("data/interim/agent_nextpools_frame.parquet")
tot=t["e2"].sum()
s=t.sort("e2",descending=True)
for k in [20,100,500,2000,10000]:
    print(k, s.head(k)["e2"].sum()/tot)
print(s.head(60).select("apt","day","plan","trk","gs","W","y","served","r","fs","disr").with_columns(pl.col(pl.Float64).round(0)))
# which airport-day carry most SSE
d=t.group_by("apt","day").agg(n=pl.len(), sse=C("e2").sum()/SE, bias=C("r").mean()).sort("sse",descending=True)
print(d.head(40).with_columns(pl.col(pl.Float64).round(2)))
print("top 40 days share", d.head(40)["sse"].sum()/(tot/SE))
