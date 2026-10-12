"""v136 = v135 + lever B (hidden plan with AOBT_3 - LOBT >= 3,606 s: take-off minus AOBT_3; docs/LAB-NOTES.md round 2), built 2026-10-03."""
import polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
b33, b35, bB = (pl.read_parquet(I + f) for f in ("agent_bundle_v133_final.parquet", "agent_bundle_v135_final.parquet", "agent_rome3B_final.parquet"))
assert (b33[ID] == b35[ID]).all() and (b33[ID] == bB[ID]).all()
mB, m35 = bB[T] != b33[T], b35[T] != b33[T]; assert int((mB & m35).sum()) == 0, "overlap"
out = b35.with_columns(pl.when(mB).then(bB[T]).otherwise(b35[T]).alias(T))
out.write_parquet(I + "agent_bundle_v136_final.parquet"); out.write_parquet("submissions/jolly-lobster_v136.parquet")
print("B rows", int(mB.sum()), "changed vs v135", int((out[T] != b35[T]).sum()))
