"""experiment md: take-off-sequence (seq) features for the NEW-day dump rows (docs/LAB-NOTES.md).
= archive/agent_smallbundle_01_seqfeat.py's 2025 half (T fitted on 2025 months other than the row's own month, level columns
demeaned by (airport, month) over the 2025 frame), joined to data/interim/agent_md_rows_new.parquet instead of the c238 rows.
The 2025 frame is all 12 months either way, so these values are on the same footing as agent_smallbundle_seq_2025.parquet."""
import glob
import sys

sys.path.insert(0, ".")
import polars as pl

import archive.agent_seqpost_feat as F
from archive.apply_seq_residual import tt_excluding

C = pl.col
ID = "MVT_ID_mvt"
dep25, arr25 = F.load_movements(sorted(glob.glob("data/raw/training_2025-*.parquet")))
f25 = F.build(dep25, arr25, tt_excluding(dep25, dep25, lambda j: {j}), demean=True)
f25 = f25.with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
# consistency: the rebuilt frame must equal the production file on the c238 rows
old = pl.read_parquet("data/interim/agent_smallbundle_seq_2025.parquet")
chk = old.select(ID).join(f25, on=ID, how="left")
diff = max(float(((chk[c].cast(pl.Float64) - old[c].cast(pl.Float64)).abs()).max() or 0.0) for c in F.SEQ)
print(f"rebuild vs agent_smallbundle_seq_2025 on {old.height:,} c238 rows: max |diff| over SEQ {diff}")
assert diff < 1e-6, "seq rebuild differs from production"
RIN = sys.argv[1] if len(sys.argv) > 1 else "data/interim/agent_md_rows_new.parquet"
ROUT = sys.argv[2] if len(sys.argv) > 2 else "data/interim/agent_md_seq_new.parquet"
dn = pl.read_parquet(RIN, columns=[ID]).with_columns(C(ID).cast(pl.Float64))
o = dn.join(f25, on=ID, how="left")
o.write_parquet(ROUT)
print(f"new rows {dn.height:,}; with features {o['sq_slip'].is_not_null().sum():,} -> {ROUT}")
