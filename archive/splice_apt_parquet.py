"""Splice two submissions by airport: TREAT rows at the named airports, CONTROL rows elsewhere (serving twin of splice_npz.py).

    python archive/splice_apt_parquet.py CONTROL.parquet TREAT.parquet LFPG OUT.parquet
"""
import sys

import polars as pl

from prc.config import RAW_DIR

C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
fc, ft, apts, out = sys.argv[1:5]
c, t = pl.read_parquet(fc), pl.read_parquet(ft).select(ID, C(T).alias("_t"))
apt = pl.scan_parquet(RAW_DIR / "ranking.parquet").select(C(ID), C("ADEP_mvt")).collect()
j = c.join(t, on=ID, how="left", maintain_order="left").join(apt, on=ID, how="left", maintain_order="left")
assert j.height == c.height and j["_t"].null_count() == 0
m = C("ADEP_mvt").is_in(apts.split(","))
res = j.with_columns(pl.when(m).then(C("_t")).otherwise(C(T)).cast(c[T].dtype).alias(T)).select(c.columns)
res.write_parquet(out)
d = (res[T].cast(pl.Float64) - c[T].cast(pl.Float64))
print(f"spliced {j.filter(m).height:,} {apts} rows from {ft}; rows changed {(d != 0).sum():,}, mean shift {d.mean():+.3f}s -> {out}")
