"""Three files from the published daily delay identity, applied as deltas (piece minus v150) on a base file:
  A = base + day-level calibration (agent_dayeq_level_final), B = base + Rome no-plan types (agent_dayeq_types_final),
  AB = A with the rows of the types piece set to their B values (the type posterior supersedes the level shift on those rows).
usage: python archive/agent_bundle_dayeq.py <base parquet> <VA> <VB> <VAB>  -> data/interim/agent_bundle_v<V>_final.parquet"""
import sys, polars as pl
C = pl.col; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; D = "data/interim/"
base, VA, VB, VAB = sys.argv[1:5]
b = pl.read_parquet(base); r = pl.read_parquet(D + "agent_bundle_v150_final.parquet").rename({T: "r"})
lv = pl.read_parquet(D + "agent_dayeq_level_final.parquet").rename({T: "lv"}); ty = pl.read_parquet(D + "agent_dayeq_types_final.parquet").rename({T: "ty"})
j = b.join(r, on=ID).join(lv, on=ID).join(ty, on=ID).with_columns((C("lv") - C("r")).alias("dl"), (C("ty") - C("r")).alias("dt"))
j = j.with_columns((C(T) + C("dl")).alias("A"), (C(T) + C("dt")).alias("B")).with_columns(pl.when(C("dt") != 0).then(C("B")).otherwise(C("A")).alias("AB"))
assert (j[ID] == b[ID]).all()
for v, col in ((VA, "A"), (VB, "B"), (VAB, "AB")):
    o = j.select(ID, pl.max_horizontal(C(col), pl.lit(0)).round().cast(b.schema[T]).alias(T)); o.write_parquet(D + f"agent_bundle_v{v}_final.parquet")
    d = o[T] - b[T]; print(f"v{v} ({col}): rows changed {int((d != 0).sum())}, mean {float(d.mean()):+.3f}, max {int(d.max())}, min {int(d.min())}")
print("rows in both pieces:", j.filter((C("dl") != 0) & (C("dt") != 0)).height)
