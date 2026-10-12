"""Fallback file F0: a complete, valid final submission that needs no new model.

  * January and July rows: the values of an earlier submission, matched by MVT_ID.
  * February and June rows: the mean taxi time of the same airport in the same calendar month of 2025,
    from the 2025 training labels.

The file has the template's row order and answer dtype. No 2026 label is read: the template supplies ids
only, the ranking file supplies airport and take-off month (input columns), and the labels come from 2025.

  python final/00_fallback.py
  python final/00_fallback.py --carry-months 1,7 --fill-months 2,6 --carry-file submissions/jolly-lobster_v162.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from fcommon import ID_COL, OUT, RAW, REPO, TARGET_COL, departure_inputs, parse_months, template_ids, training_label_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--template", type=Path, default=RAW / "final_submitting.parquet")
    ap.add_argument("--carry-file", type=Path, default=REPO / "submissions" / "jolly-lobster_v162.parquet")
    ap.add_argument("--carry-months", default="1,7")
    ap.add_argument("--fill-months", default="2,6")
    ap.add_argument("--out", type=Path, default=OUT / "F0.parquet")
    a = ap.parse_args()
    carry_m, fill_m = parse_months(a.carry_months), parse_months(a.fill_months)

    ids, dtype = template_ids(a.template)
    dep = departure_inputs(a.ranking).select(ID_COL, "ADEP_mvt", "month")
    carry = pl.read_parquet(a.carry_file).select(ID_COL, TARGET_COL).rename({TARGET_COL: "carried"})
    stats = training_label_stats().select("apt", "month", pl.col("mean").alias("fill"))

    frame = (ids.join(dep, on=ID_COL, how="left", maintain_order="left")
                .join(carry, on=ID_COL, how="left", maintain_order="left")
                .join(stats, left_on=["ADEP_mvt", "month"], right_on=["apt", "month"], how="left", maintain_order="left"))
    if frame.height != ids.height:
        raise SystemExit("a join changed the row count")
    month = frame["month"]
    if month.null_count():
        raise SystemExit(f"{month.null_count():,} template ids are not departures of the ranking file")
    unexpected = sorted(set(month.unique().to_list()) - set(carry_m) - set(fill_m))
    if unexpected:
        raise SystemExit(f"months {unexpected} are in the file but neither carried nor filled")

    use_carry = month.is_in(carry_m)
    value = pl.when(pl.col("month").is_in(carry_m)).then(pl.col("carried").cast(pl.Float64)).otherwise(pl.col("fill"))
    out = frame.select(pl.col(ID_COL), value.round(0).cast(dtype).alias(TARGET_COL))
    nulls = out[TARGET_COL].null_count()
    if nulls:
        raise SystemExit(f"{nulls:,} rows have no value (carry file or training mean missing for them)")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(a.out)
    print(f"wrote {a.out}: {out.height:,} rows; carried {int(use_carry.sum()):,}, filled {int((~use_carry).sum()):,}")


if __name__ == "__main__":
    main()
