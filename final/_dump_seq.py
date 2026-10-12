"""Take-off-sequence features for the dump rows (stage 20, part d).

Derived from the 2026 half of archive/agent_smallbundle_01_seqfeat.py: the (airport, stand, runway) taxi table T is fitted on
all of 2025, applied to the ranking file's departures, and the 21 sequence columns are built over that file's movements
(level columns demeaned by airport and month). Dump rows only. Changes: ranking file and dump file are arguments.
Run from the repo root.

    python final/_dump_seq.py --ranking data/raw/final_ranking.parquet --rows FILE_rows --out FILE_seq
"""
import argparse
import glob
import sys
import time

sys.path.insert(0, ".")
import polars as pl

import archive.agent_seqpost_feat as F

C = pl.col
ID = "MVT_ID_mvt"
ap = argparse.ArgumentParser()
ap.add_argument("--ranking", default="data/raw/final_ranking.parquet")
ap.add_argument("--rows", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
T0 = time.time()
dep25, arr25 = F.load_movements(sorted(glob.glob("data/raw/training_2025-*.parquet")))
T_all = F.fit_T(dep25)
del arr25
dep26, arr26 = F.load_movements([args.ranking])
f26 = F.build(dep26, arr26, F.apply_T(dep26, T_all), demean=True).with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
d26 = pl.read_parquet(args.rows, columns=[ID]).with_columns(C(ID).cast(pl.Float64))
o26 = d26.join(f26, on=ID, how="left")
o26.write_parquet(args.out)
print(f"deps {dep26.height:,}; dump rows {d26.height:,}; with features {o26['sq_slip'].is_not_null().sum():,} "
      f"({time.time() - T0:.0f}s) -> {args.out}")
