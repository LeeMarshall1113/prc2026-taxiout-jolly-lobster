"""next-pools step 1: error map of the faithful 2025 held-out chain (agent_waithedge_tbl: served = agent_faithful_lib.fchain), eight frame months.
SSE by airport x family. Row-rule rows (R1,R2,R3a,R3b) excluded. Writes data/interim/agent_nextpools_frame.parquet and prints tables."""
import numpy as np
import polars as pl
C = pl.col
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
SE = 1.5e8
t = pl.read_parquet("data/interim/agent_waithedge_tbl.parquet").with_columns(C("rule").fill_null(""))
print("months", t["mon"].unique().sort().to_list(), t.height)
t = t.filter(~C("rule").is_in(["R1", "R2", "R3a", "R3b"]))
t = t.with_columns((C("y") - C("served")).alias("r"))
t = t.with_columns((C("r") ** 2).alias("e2"))
# disruption day: airport-day RMS residual in top decile within airport (diagnostic)
dd = t.group_by("apt", "day").agg(drms=C("e2").mean().sqrt(), dn=pl.len())
dd = dd.with_columns((C("drms") >= C("drms").quantile(0.9).over("apt")).alias("disr"))
t = t.join(dd.select("apt", "day", "disr"), on=["apt", "day"])
t = t.with_columns(
    pl.when(C("cov") == "matched").then(pl.lit("trk")).when(C("cov") == "points_no_liftoff").then(pl.lit("est")).otherwise(pl.lit("unt")).alias("trk"),
    pl.when(C("gs") > 3600).then(pl.lit("g>1h")).when(C("gs") > 0).then(pl.lit("late")).otherwise(pl.lit("early")).alias("dl"),
    pl.when(C("plan")).then(pl.lit("P")).otherwise(pl.lit("N")).alias("pl"))
t = t.with_columns((C("pl") + "/" + C("trk") + "/" + C("dl") + "/" + pl.when(C("disr")).then(pl.lit("D")).otherwise(pl.lit("-"))).alias("fam"))
t.write_parquet("data/interim/agent_nextpools_frame.parquet")
tot = t["e2"].sum()
print("total SSE/SE", tot / SE, "rows", t.height, "rmse", (tot / t.height) ** .5)
print(t.group_by("apt").agg(n=pl.len(), sse=C("e2").sum() / SE, rmse=C("e2").mean().sqrt()).with_columns((C("sse") / (tot / SE)).alias("sh")).sort("sse", descending=True))
for k in ["pl", "trk", "dl", "disr"]:
    print(t.group_by(k).agg(n=pl.len(), sse=C("e2").sum() / SE).with_columns((C("sse") / (tot / SE)).alias("sh")).sort("sse", descending=True))
g = t.group_by("apt", "fam").agg(n=pl.len(), sse=C("e2").sum() / SE, bias=C("r").mean(), top20=C("e2").sort(descending=True).head(20).sum() / SE).with_columns((C("sse") / (tot / SE)).alias("sh")).sort("sse", descending=True)
print(g.head(50).with_columns(pl.col(pl.Float64).round(2)))
# airport x month cell for top airports
print(t.group_by("apt", "mon").agg(n=pl.len(), sse=C("e2").sum() / SE).with_columns((C("sse") / (tot / SE)).alias("sh")).sort("sse", descending=True).head(25).with_columns(pl.col(pl.Float64).round(3)))
