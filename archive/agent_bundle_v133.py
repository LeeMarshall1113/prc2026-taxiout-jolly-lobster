"""v133 candidate = v132 + Rome no-plan mid-gap mixture at half weight (docs/LAB-NOTES.md, lever A) + callsign-keyed reach of the shipped hidden-plan reading (docs/LAB-NOTES.md). Disjoint row sets."""
import polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
b = pl.read_parquet(I + "agent_bundle_v132_final.parquet"); a = pl.read_parquet(I + "agent_rome2_A_final.parquet"); c = pl.read_parquet(I + "agent_fix2_c_final.parquet")
assert (a[ID] == b[ID]).all() and (c[ID] == b[ID]).all()
ma, mc = (a[T] != b[T]), (c[T] != b[T]); assert int((ma & mc).sum()) == 0, "overlap"
out = b.with_columns(pl.when(ma).then(a[T]).when(mc).then(c[T]).otherwise(b[T]).alias(T))
out.write_parquet(I + "agent_bundle_v133_final.parquet"); print("A rows", int(ma.sum()), "C rows", int(mc.sum()), "total changed", int((out[T] != b[T]).sum()))
