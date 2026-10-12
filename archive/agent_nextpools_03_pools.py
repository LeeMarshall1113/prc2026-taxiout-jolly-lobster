import numpy as np, polars as pl
C = pl.col
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
SE=1.5e8
t = pl.read_parquet("data/interim/agent_nextpools_frame.parquet")
tot=t["e2"].sum()/SE
g = (t.group_by("apt","pl","trk","dl").agg(n=pl.len(), sse=C("e2").sum()/SE, bias=C("r").mean(), tail=(C("e2").filter(C("r").abs()>1800).sum())/SE,
      top20=C("e2").sort(descending=True).head(20).sum()/SE, disrsh=C("e2").filter(C("disr")).sum()/SE, jan=C("e2").filter(C("mon")==1).sum()/SE, jul=C("e2").filter(C("mon")==7).sum()/SE)
     .with_columns((C("sse")/tot).alias("share")).sort("sse",descending=True))
print(g.head(24).with_columns(pl.col(pl.Float64).round(2)))
print("LFPG junk row and 01-23:")
print(t.filter((C("apt")=="LFPG")&(C("day")=="2025-01-23")).group_by("pl","trk").agg(n=pl.len(), sse=C("e2").sum()/SE, big=C("e2").max()/SE))
x=t.filter(~((C("apt")=="LFPG")&(C("day")=="2025-01-23")))
print("excluding LFPG 01-23 total", x["e2"].sum()/SE)
