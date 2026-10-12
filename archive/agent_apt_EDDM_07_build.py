"""EDDM experiment 07: candidate = v143 final with the Munich January -38 s shift removed (EDDM, Jan 2026, gap<1800 only rows change).
Evidence: docs/LAB-NOTES.md section 1 (like-for-like instruments A, B: bias +2 s, interval -10..+15)."""
import sys, numpy as np, polars as pl
C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ref = pl.read_parquet("data/interim/agent_bundle_v143_final.parquet")
rk = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt")=="DEP").select(C(ID), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
      (C("MVT_TIME_UTC_mvt")-C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs")).collect())
hit = set(rk.filter((C("apt")=="EDDM")&(C("mo")==1)&(C("gs")<1800))[ID].to_list())
ids = ref[ID].to_list(); old = ref[T].to_numpy().astype(float)
m = np.array([i in hit for i in ids]); new = np.where(m, old + 38.0, old)
out = ref.with_columns(pl.Series(T, np.rint(new)).cast(ref.schema[T]))
assert out[ID].equals(ref[ID]) and (out[T].to_numpy()[~m] == ref[T].to_numpy()[~m]).all()
print("rows moved", int(m.sum()), "mean on them", old[m].mean(), "->", new[m].mean())
out.write_parquet("data/interim/agent_apt_EDDM_final.parquet")
