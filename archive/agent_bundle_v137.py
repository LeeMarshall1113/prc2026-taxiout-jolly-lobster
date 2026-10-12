"""v137 = v136 + LFPG duplicate-pair rule (16 rows; docs/LAB-NOTES.md) + hidden-plan cap by callsign at seven airports (6 rows; docs/LAB-NOTES.md)."""
import polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
b, mp, dr = (pl.read_parquet(I + f) for f in ("agent_bundle_v136_final.parquet", "agent_multipush_final.parquet", "agent_detrules_final.parquet"))
assert (b[ID] == mp[ID]).all() and (b[ID] == dr[ID]).all()
m1, m2 = mp[T] != b[T], dr[T] != b[T]; assert int((m1 & m2).sum()) == 0, "overlap"
out = b.with_columns(pl.when(m1).then(mp[T]).when(m2).then(dr[T]).otherwise(b[T]).alias(T))
out.write_parquet(I + "agent_bundle_v137_final.parquet"); out.write_parquet("submissions/jolly-lobster_v137.parquet")
print("dup rows", int(m1.sum()), "cap rows", int(m2.sum()), "changed", int((out[T] != b[T]).sum()))
