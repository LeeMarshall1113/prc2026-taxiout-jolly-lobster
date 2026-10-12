"""ADS-B ground coverage per airport and month, from the points day files and the movement counts of a ranking file.

Measure: distinct aircraft (ICAO address) with at least one surface point (altitude flag -1) per airport and day, divided by that
day's movements (departures plus arrivals) at the airport. It is a reception measure that needs no matching; the share of
departures that end up with a matched surface track is printed by final/20_adsb_dumps.py once the dumps exist. Reads one
day file at a time with two columns (well under 1 GB). Nothing here reads a label.

    python final/29_adsb_coverage.py [--ranking data/raw/final_ranking.parquet] [--year 2026]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import REPO  # noqa: E402

C = pl.col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default=str(REPO / "data/raw/final_ranking.parquet"))
    ap.add_argument("--year", default="2026")
    args = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(str(REPO / f"data/adsb/points/{args.year}-*.parquet"))):
        day = os.path.basename(f)[:-8]
        p = (pl.scan_parquet(f).filter(C("alt") == -1).group_by("apt")
             .agg(C("icao").n_unique().alias("ground_aircraft")).collect().with_columns(pl.lit(day).alias("day")))
        rows.append(p)
    g = pl.concat(rows)
    mv = (pl.scan_parquet(args.ranking)
          .select("PHASE_mvt", "ADEP_mvt", "ADES_mvt", C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day"))
          .with_columns(pl.when(C("PHASE_mvt") == "DEP").then(C("ADEP_mvt")).otherwise(C("ADES_mvt")).alias("apt"))
          .group_by("apt", "day").agg(pl.len().alias("movements")).collect())
    j = g.join(mv, on=["apt", "day"], how="inner").with_columns(C("day").str.slice(5, 2).alias("month"))
    t = (j.group_by("apt", "month").agg(pl.len().alias("days"), C("ground_aircraft").sum(), C("movements").sum())
         .with_columns((C("ground_aircraft") / C("movements")).round(3).alias("per_movement")).sort("apt", "month"))
    print("ground aircraft per movement (share), by airport and month:")
    print(t.pivot(on="month", index="apt", values="per_movement"))
    print("days counted:")
    print(t.pivot(on="month", index="apt", values="days"))


if __name__ == "__main__":
    main()
