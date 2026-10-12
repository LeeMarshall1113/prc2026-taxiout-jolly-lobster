"""experiment smallbundle 01: take-off-sequence features (archive/agent_seqpost_feat.py, imported) for the re-stack rows.
2025: T fitted on 2025 months other than the row's own month (apply_seq_residual.py's training convention).
2026: T from all of 2025 (apply_seq_residual.py's serve convention). Level columns demeaned by (airport, month) per frame.
Writes data/interim/agent_smallbundle_seq_{2025,2026}.parquet (MVT_ID_mvt Float64 + the 21 SEQ columns), dump rows only.
"""
import glob
import sys
import time

sys.path.insert(0, ".")
import polars as pl

import archive.agent_seqpost_feat as F
from archive.apply_seq_residual import tt_excluding

C = pl.col
ID = "MVT_ID_mvt"
T0 = time.time()
dep25, arr25 = F.load_movements(sorted(glob.glob("data/raw/training_2025-*.parquet")))
f25 = F.build(dep25, arr25, tt_excluding(dep25, dep25, lambda j: {j}), demean=True)
d25 = pl.read_parquet("data/interim/agent_offblock_rows_c238.parquet", columns=[ID]).with_columns(C(ID).cast(pl.Float64))
f25 = f25.with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
o25 = d25.join(f25, on=ID, how="left")
o25.write_parquet("data/interim/agent_smallbundle_seq_2025.parquet")
print(f"2025: deps {dep25.height:,}; dump rows {d25.height:,}; with features {o25['sq_slip'].is_not_null().sum():,} "
      f"({time.time() - T0:.0f}s)")
T_all = F.fit_T(dep25)
del f25, arr25
dep26, arr26 = F.load_movements(["data/raw/ranking.parquet"])
f26 = F.build(dep26, arr26, F.apply_T(dep26, T_all), demean=True).with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
d26 = pl.read_parquet("data/interim/agent_offblock_rows_2026.parquet", columns=[ID]).with_columns(C(ID).cast(pl.Float64))
o26 = d26.join(f26, on=ID, how="left")
o26.write_parquet("data/interim/agent_smallbundle_seq_2026.parquet")
print(f"2026: deps {dep26.height:,}; dump rows {d26.height:,}; with features {o26['sq_slip'].is_not_null().sum():,}")
for nm, o in (("2025", o25), ("2026", o26)):
    print(nm, "non-null share:", " ".join(f"{c}={1 - o[c].null_count() / o.height:.2f}" for c in F.SEQ[:3] + ["sq_arr_same15", "sq_arr_W15"]))
print(f"done {time.time() - T0:.0f}s")
