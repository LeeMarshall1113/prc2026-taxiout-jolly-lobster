"""Write the frame the month-set stages run on: the ranking file with the departure rows of the chosen months only.

Arrival rows of every month are kept (the stages join departures to arrival records up to 8 hours ahead and to the
destination airport; a month's last evening has no arrivals of the next month, as January 31 had none). A departure's month
is the UTC month of its take-off time.

    python final/_mkrank.py --ranking data/raw/final_ranking.parquet --months 2,6 --out data/final/ranking_J5.parquet
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import parse_months  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ranking", required=True)
ap.add_argument("--months", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
months = parse_months(a.months)
d = pl.read_parquet(a.ranking)
keep = (pl.col("PHASE_mvt") != "DEP") | pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months)
o = d.filter(keep)
o.write_parquet(a.out)
dep = o.filter(pl.col("PHASE_mvt") == "DEP")
print(f"{a.out}: {o.height:,} rows, {dep.height:,} departures of months {months}, {o.height - dep.height:,} arrival rows")
