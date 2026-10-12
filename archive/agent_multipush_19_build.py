"""experiment multipush 19: build data/interim/agent_multipush_final.parquet = v136 with the duplicate-pair rule applied.
Rule: LFPG departure WITH a plan whose partner record (same flight string and take-off date, take-off within 1800 s, group of exactly 2) has NO plan:
new = 0.75 * clip(take-off - schedule - 750, 60, 14400) + 0.25 * v136, only if take-off - schedule - 780 <= 14400. Rows that differ between v132 and v136 are excluded."""
import sys
sys.path.insert(0, ".")
import numpy as np, polars as pl
C = pl.col
I = "data/interim/"
q = pl.read_parquet(I + "agent_multipush_pairs_lfpg26_all.parquet").filter(C("plan"))
v136 = pl.read_parquet(I + "agent_bundle_v136_final.parquet")
v132 = pl.read_parquet(I + "agent_bundle_v132_final.parquet")
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
print(v136.schema[T], v136.schema[ID])
moved = v136.join(v132.select(C(ID), C(T).alias("t132")), on=ID).filter(C(T) != C("t132")).select(C(ID).cast(pl.Float64).alias("id"))
print("rows moved by v133-v136:", moved.height)
q = q.with_columns((C("mvt") - C("sched")).alias("gap")).filter((C("gap") - 780) <= 14400).join(moved, on="id", how="anti")
print("candidate rows", q.height)
cur = v136.select(C(ID).cast(pl.Float64).alias("id"), C(T).alias("v136"))
q = q.join(cur, on="id").with_columns((0.75 * (C("gap") - 750).clip(60, 14400) + 0.25 * C("v136")).round(0).alias("new"))
print(q.select("fl", "day", "gap", "v136", "new"))
upd = q.select(C("id").alias(ID), C("new").cast(v136.schema[T]).alias("new_t"))
out = v136.with_columns(C(ID).cast(pl.Float64).alias("_k")).join(upd.rename({ID: "_k"}), on="_k", how="left").with_columns(pl.coalesce(C("new_t"), C(T)).cast(v136.schema[T]).alias(T)).drop("_k", "new_t")
out = out.select(v136.columns)
out.write_parquet(I + "agent_multipush_final.parquet")
d = (q["new"] - q["v136"]).to_numpy()
print("rows changed", len(d), " sum dp^2", float((d ** 2).sum()), " per scored row", float((d ** 2).sum()) / 344841, " rms dp", float(np.sqrt((d ** 2).mean())))
