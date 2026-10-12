"""day-equation 2, step 2: build a candidate file on v156 from one or more joint-solve row tables.
  python agent_dayeq2_02_build.py NAME APT:USE [APT:USE ...]
The airport's rows take the joint values solved on the pre-equation base (the row table's `new`); other rows keep v156. Then, last: the Istanbul
no-plan bound (gs >= 0: value <= gs + 1,200), the plan-clock window (|value - (take-off - LOBT)| <= 3,606), floor 0; the six named flights never change.
Out: data/interim/agent_dayeq2_{NAME}_final.parquet; prints rows moved by airport and the rows moved by more than 1,000 s."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(80)
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
NAME = sys.argv[1]; pieces = [p.split(":") for p in sys.argv[2:]]
b = pl.read_parquet(I + "agent_bundle_v156_final.parquet")
r = (pl.scan_parquet(R + "data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
     .select(C(ID), C("ADEP_mvt").alias("apt"), C("FLIGHT_mvt").alias("fs"), C("FLIGHT_ID_mvt").is_not_null().alias("plan"),
             (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("gl"),
             (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("gs")).collect())
j = b.join(r, on=ID, how="left").with_columns(C(T).cast(pl.Float64).alias("new"))
for piece in pieces:
    apt, use = piece[0], piece[1]; sub = piece[2] if len(piece) > 2 else "all"
    t = pl.read_parquet(I + f"agent_dayeq2_rows_{apt}_{use}.parquet")
    if sub == "planonly":  # only the with-plan candidate rows (rows no earlier piece touched)
        t = t.filter(C("cand") & C("plan"))
    t = t.select(C("id").alias(ID), C("new").alias("nn"), C("pred").alias("base"))
    j = j.join(t, on=ID, how="left")
    nrep = j.filter(C("nn").is_not_null()).height
    j = j.with_columns(pl.when(C("nn").is_not_null() & ~C("fs").is_in(HEDGED)).then(C("nn")).otherwise(C("new")).alias("new")).drop("nn", "base")
    print(f"piece {apt}:{use}: {nrep} rows carry a joint value")
# Istanbul no-plan bound, window, floor, hedged untouched
j = j.with_columns(pl.when((C("apt") == "LTFM") & ~C("plan") & (C("gs") >= 0)).then(pl.min_horizontal(C("new"), C("gs") + 1200)).otherwise(C("new")).alias("new"))
lo = pl.max_horizontal(C("gl") - 3606, pl.lit(0)); hi = C("gl") + 3606
j = j.with_columns(pl.when(C("gl").is_null()).then(C("new")).otherwise(pl.min_horizontal(pl.max_horizontal(C("new"), lo), hi)).alias("new"))
j = j.with_columns(pl.max_horizontal(C("new"), pl.lit(0.0)).round().alias("new"))
j = j.with_columns(pl.when(C("fs").is_in(HEDGED)).then(C(T).cast(pl.Float64)).otherwise(C("new")).alias("new"))
out = b.select(ID).join(j.select(ID, C("new").cast(b.schema[T]).alias(T)), on=ID, how="left"); assert (out[ID] == b[ID]).all() and out[T].null_count() == 0
out.write_parquet(I + f"agent_dayeq2_{NAME}_final.parquet")
d = (out[T].cast(pl.Float64) - b[T].cast(pl.Float64)).to_numpy()
print(f"wrote agent_dayeq2_{NAME}_final.parquet: rows moved {(d != 0).sum()}, rms (moved) {np.sqrt((d[d != 0] ** 2).mean()) if (d != 0).any() else 0:.1f} s, max |dp| {np.abs(d).max():.0f}, sum dp^2/row {(d ** 2).sum() / len(d):.1f}, mean {d.mean():+.3f}")
jj = j.with_columns(pl.Series("d", d))
print(jj.group_by("apt").agg(n=pl.len(), moved=(C("d") != 0).sum(), mean=C("d").mean().round(2), rms=(C("d") ** 2).mean().sqrt().round(1), over1000=(C("d").abs() > 1000).sum()).sort("apt"))
print("rows moved by more than 1,000 s:", int((np.abs(d) > 1000).sum()))
print(jj.filter(C("d").abs() > 1000).select("apt", "fs", "plan", "gs", C(T).alias("v156"), "new", "d").sort(C("d").abs(), descending=True).head(60))
print("hedged flights:", jj.filter(C("fs").is_in(HEDGED)).select("fs", C(T), "new", "d"))
