"""Compare a rebuilt file with the stored one on the ids they share: max absolute difference per column and rows that differ.

    python final/28_compare_files.py NEW STORED [--cols a,b,c] [--key MVT_ID_mvt]

Float columns count as equal when both are NaN/null or the values match to 1e-9 (relative); other columns by equality.
Prints one line per column that differs and a summary line. Used to test that a final/ script reproduces the shipped
January/July output.
"""
import argparse
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("new")
ap.add_argument("stored")
ap.add_argument("--cols", default=None)
ap.add_argument("--key", default="MVT_ID_mvt")
a = ap.parse_args()
n = pl.read_parquet(a.new).with_columns(pl.col(a.key).cast(pl.Float64)).unique(subset=a.key, keep="first")
s = pl.read_parquet(a.stored).with_columns(pl.col(a.key).cast(pl.Float64)).unique(subset=a.key, keep="first")
cols = a.cols.split(",") if a.cols else [c for c in n.columns if c in s.columns and c != a.key]
j = n.join(s, on=a.key, how="inner", suffix="__s")
print(f"new {n.height:,} rows, stored {s.height:,} rows, shared ids {j.height:,}")
tot_bad = 0
for c in cols:
    x, y = j[c], j[c + "__s"]
    if x.dtype.is_float() or x.dtype.is_integer():
        xx, yy = x.cast(pl.Float64).fill_null(np.nan).to_numpy(), y.cast(pl.Float64).fill_null(np.nan).to_numpy()
        ok = (np.isnan(xx) & np.isnan(yy)) | np.isclose(xx, yy, rtol=1e-9, atol=1e-9, equal_nan=False)
        bad = int((~ok).sum())
        mx = float(np.nanmax(np.abs(xx - yy)[~ok])) if bad else 0.0
    else:
        bad = int(((x != y).fill_null(False) | (x.is_null() != y.is_null())).sum())
        mx = float("nan")
    tot_bad += bad
    if bad:
        print(f"  {c}: {bad:,} rows differ, max abs diff {mx:.6g}")
print(f"columns compared {len(cols)}; rows x columns differing {tot_bad:,}")
