"""Fixes that name specific flights or rows of the January and July 2026 data: one file, one table, off unless asked.

Everything in final/30_general_rules.py and final/32_published_daily.py is defined on input columns and fires on any month.
The items below are not: each names a flight string or a row id that was looked at in the January/July data. They are kept
here, with where each came from, so that a reader can see exactly which scored rows are treated by name. They are applied only
with --with-row-fixes, and every row they touch is listed in the output.

Table columns: key, kind, value, months, shipped, reason, source.
  kind  keep_before   the row keeps the value it had before the published-daily layer (final/32), not the layer's value
        set_value     the row is set to `value` (only items with shipped = no; needs --include-unshipped)
  months  take-off months (UTC) the item applies to under --match dated; empty = any month
  shipped  yes = part of the last uploaded file (submissions/jolly-lobster_v162.parquet); no = prepared, never uploaded

  python final/31_scored_row_fixes.py --before P0.parquet --pred P1.parquet --ranking R.parquet --out Q.parquet --with-row-fixes
      P0 = predictions before final/32, P1 = after final/32.
  --match dated    (default for the February/June build) only in the months listed in the table (1 and 7), the months in which the string was inspected
  --match string   as shipped in the January/July file: matched by flight string alone, in every month
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL, RAW, TARGET_COL  # noqa: E402

C = pl.col

# key, kind, value, months, shipped, reason, source
FIXES = [
    ("WMTT3508", "keep_before", None, [1, 7], True,
     "Rome no-plan row whose record type is a schedule copy or a stamp written a day early; the published daily delay reads zero under both",
     "docs/LAB-NOTES.md; docs/PIPELINE.md section 5"),
    ("SVII3006", "keep_before", None, [1, 7], True, "same: copy or day-early alternative, equation silent", "docs/LAB-NOTES.md"),
    ("ITYTY672", "keep_before", None, [1, 7], True, "same: copy or day-early alternative, equation silent (one of its rows is model-dependent)", "docs/LAB-NOTES.md"),
    ("NOSOS431", "keep_before", None, [1, 7], True, "same: copy or day-early alternative, equation silent (day carries a confounded excess)", "docs/LAB-NOTES.md"),
    ("QFAA7006", "keep_before", None, [1, 7], True, "same: copy or day-early alternative, equation silent", "docs/LAB-NOTES.md"),
    ("TAY9LA", "keep_before", None, [1, 7], True, "same at Rome; the same string at Paris is also left unmoved by the day layers", "docs/LAB-NOTES.md; archive/agent_nextpools_05_build.py"),
]
# One single-row decision that was prepared and never uploaded (the board closed to uploads first). Not applied by default.
UNSHIPPED = [
    ("id:213143319", "set_value", 48776.0, [7], False,
     "Rome no-plan row of 2026-07-05 restored to its scheduled gap on the evidence of the published Rome IFR count and the next day's delay; break-even probability 10%, estimated 1%",
     "docs/LAB-NOTES.md and s10; docs/LAB-NOTES.md (v164 plan)"),
]


def apply_fixes(before: pl.DataFrame, cur: pl.DataFrame, ranking: Path, match: str = "string", include_unshipped: bool = False) -> pl.DataFrame:
    """Apply the table to `cur`; `before` holds the values before final/32. Returns id and answer, rows of `cur` in order."""
    b = before.select(C(ID_COL), C(TARGET_COL).alias("before"))
    meta = (pl.scan_parquet(ranking).filter(C("PHASE_mvt") == "DEP")
            .select(C(ID_COL), C("FLIGHT_mvt").str.strip_chars().alias("fl"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mon")).collect())
    j = cur.join(b, on=ID_COL, how="left").join(meta, on=ID_COL, how="left", maintain_order="left")
    v = j[TARGET_COL].cast(pl.Float64).to_numpy().copy()
    n_touch = 0
    for key, kind, value, months, shipped, reason, src in FIXES + (UNSHIPPED if include_unshipped else []):
        if kind == "keep_before":
            m = (j["fl"] == key).fill_null(False).to_numpy()
            if match == "dated":
                m &= j["mon"].is_in(months).fill_null(False).to_numpy()
            new = j["before"].cast(pl.Float64).to_numpy()
        else:
            m = (j[ID_COL] == float(key.split(":")[1])).to_numpy()
            new = np.full(len(v), value)
        moved = m & (v != new)
        where = sorted(set(zip(j.filter(pl.Series(moved))["apt"].to_list(), j.filter(pl.Series(moved))["mon"].to_list())))
        print(f"  {key:14s} {kind:12s} rows matched {int(m.sum()):3d}, changed {int(moved.sum()):3d}" + "".join(f"  {x} m{y:02d}" for x, y in where))
        v = np.where(m, new, v)
        n_touch += int(moved.sum())
    print(f"row fixes: {n_touch} rows changed")
    return j.select(ID_COL).with_columns(pl.Series(TARGET_COL, np.rint(v)).cast(cur.schema[TARGET_COL]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--with-row-fixes", action="store_true", help="apply the table; without it the output equals --pred")
    ap.add_argument("--include-unshipped", action="store_true", help="also apply the items that were never uploaded")
    ap.add_argument("--match", choices=["string", "dated"], default="dated")
    a = ap.parse_args()
    cur = pl.read_parquet(a.pred)
    if not a.with_row_fixes:
        print("row fixes: OFF, input unchanged")
        out = cur
    else:
        out = apply_fixes(pl.read_parquet(a.before), cur, a.ranking, a.match, a.include_unshipped)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.select(ID_COL, TARGET_COL).write_parquet(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
