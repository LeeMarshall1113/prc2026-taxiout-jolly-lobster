"""experiment restack5: compare two submission-shaped parquets (docs/LAB-NOTES.md reproduction checks).
    python archive/agent_restack5_cmp.py A.parquet B.parquet     exit 0 iff frames are equal after sorting by id
"""
import sys

import polars as pl

a = pl.read_parquet(sys.argv[1]).sort("MVT_ID_mvt")
b = pl.read_parquet(sys.argv[2]).sort("MVT_ID_mvt")
same_ids = a.height == b.height and a["MVT_ID_mvt"].equals(b["MVT_ID_mvt"]) and a.columns == b.columns
nd = int(((a["TAXITIME_SEC_mvt"].cast(pl.Float64) - b["TAXITIME_SEC_mvt"].cast(pl.Float64)).abs() > 0).sum()) if same_ids else -1
eq = same_ids and a.equals(b) and a.schema == b.schema
print(f"CMP {sys.argv[1]} vs {sys.argv[2]}: ids/columns equal {same_ids}; rows differing {nd}; frames equal {eq}")
sys.exit(0 if eq else 1)
