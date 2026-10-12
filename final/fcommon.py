"""Shared helpers for the final-phase scripts in final/. Nothing here reads a 2026 departure label.

Conventions
  * ID_COL / TARGET_COL are the competition column names.
  * A departure's month is the UTC calendar month of its take-off time (MVT_TIME_UTC_mvt).
  * The template (final_submitting.parquet) fixes row order, ids and the integer type of the answer.
    Only its id column is ever used; its answer column is a placeholder and is never read.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "final"
ID_COL, TARGET_COL = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def parse_months(text: str) -> list[int]:
    """'1,7' -> [1, 7]."""
    return [int(t) for t in text.split(",") if t.strip()]


def departure_inputs(ranking: Path) -> pl.DataFrame:
    """One row per departure with the INPUT columns the file checks need (no label column is selected)."""
    cols = [ID_COL, "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt"]
    return (pl.scan_parquet(ranking).filter(pl.col("PHASE_mvt") == "DEP").select(cols)
            .with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")).collect())


def template_ids(template: Path):
    """Template ids in template order, plus the template's answer dtype (the answer values are not read)."""
    scan = pl.scan_parquet(template)
    dtype = scan.collect_schema()[TARGET_COL]
    return scan.select(ID_COL).collect(), dtype


def training_label_stats(raw_dir: Path = RAW) -> pl.DataFrame:
    """Per airport x calendar month of 2025 (departures): mean, median and count of the training label.

    2025 labels are training data. Months are taken from the take-off time, not the file name.
    """
    files = sorted(raw_dir.glob("training_2025-*.parquet"))
    if not files:
        raise SystemExit(f"no training_2025-*.parquet in {raw_dir}")
    return (pl.scan_parquet([str(f) for f in files])
            .filter(pl.col("PHASE_mvt") == "DEP")
            .select(pl.col("ADEP_mvt").alias("apt"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"),
                    pl.col(TARGET_COL).cast(pl.Float64).alias("y"))
            .drop_nulls()
            .group_by("apt", "month")
            .agg(pl.col("y").mean().alias("mean"), pl.col("y").median().alias("median"), pl.len().alias("n"))
            .sort("apt", "month").collect())
