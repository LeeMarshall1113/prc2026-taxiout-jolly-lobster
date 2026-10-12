"""Align each airport-month's mean prediction to EUROCONTROL's published 2026 taxi-out level (run 2026-09-23).

    python archive/apply_ec_align.py --in submissions/jolly-lobster_v100.parquet --out submissions/jolly-lobster_v101.parquet

Source: data/external/eurocontrol/taxi_out_additional_time_2026.csv (EUROCONTROL AIU open data; organiser-cleared).
Per airport a and scored month m (Jan, Jul 2026):
  target T = ec_mean(a,m) + offset(a, xf(a,m))      ec_mean = mean taxi-out of EUROCONTROL's valid flights
  offset   = per-airport linear fit of (true mean y<=7200 - ec_mean) on xf = excluded fraction (TF-VALID)/TF, 2025
             months (de-icing flights are excluded by EUROCONTROL, so the offset grows with the de-icing share)
  rows     = all rows of (a,m) except the K highest predictions; K = EUROCONTROL's excluded count at LIRF/EGLL (where
             it equals the y > 7200 s count in every 2025 month), else our own count of predictions > 7200 s
  shift    = T - mean(prediction on those rows), added to those rows; then clip at 0 and round.
Skipped: LTFM (offset not predictable, LOMO sd 149 s) and LFPG (4/4 validation months worse).
Validation: archive/ec_align_study.py (2025 leave-one-month-out on fold predictions: RMSE improves, 25/36 cells better
before excluding LFPG).
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from prc.config import RAW_DIR

C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
MONSTER_EXCL = {"LIRF", "EGLL"}
SKIP = {"LTFM", "LFPG"}


def ec(year):
    e = pl.read_csv(f"data/external/eurocontrol/taxi_out_additional_time_{year}.csv", infer_schema_length=0)
    return e.filter(C("APT_ICAO").is_in(APTS)).select(
        C("MONTH_NUM").cast(int).alias("mo"), C("APT_ICAO").alias("apt"),
        (C("TF").cast(int) - C("VALID_FL").cast(int)).alias("excl"),
        ((C("TF").cast(int) - C("VALID_FL").cast(int)) / C("TF").cast(int)).alias("xf"),
        ((C("TOTAL_REF_TIME_MIN").cast(float) + C("TOTAL_ADD_TIME_MIN").cast(float)) * 60
         / C("TOTAL_REF_NB_FL").cast(int)).alias("ec_mean"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = (pl.scan_parquet(sorted(str(p) for p in RAW_DIR.glob("training_*.parquet"))).filter(C("PHASE_mvt") == "DEP")
          .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"), C("TAXITIME_SEC_mvt").alias("y"))
          .filter(C("y") <= 7200).group_by("apt", "mo").agg(C("y").mean().alias("tm")).collect())
    gap = tr.join(ec(2025), on=["apt", "mo"]).with_columns((C("tm") - C("ec_mean")).alias("gap"))
    fit = {}
    for apt in APTS:
        g = gap.filter(C("apt") == apt)
        b, c0 = np.polyfit(g["xf"].to_numpy(), g["gap"].to_numpy(), 1)
        fit[apt] = (c0, b)
    e26 = {(r["apt"], r["mo"]): r for r in ec(2026).iter_rows(named=True) if r["mo"] in (1, 7)}

    sub = pl.read_parquet(a.inp)
    rk = (pl.scan_parquet(RAW_DIR / "ranking.parquet").filter(C("PHASE_mvt") == "DEP")
          .select(ID, C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo")).collect())
    d = sub.with_row_index("_o").join(rk, on=ID, how="left").sort("_o")
    assert d.height == sub.height and d["apt"].null_count() == 0
    p = d[T].to_numpy().astype(float)
    out = p.copy()
    total = 0.0
    for apt, mo, idx in d.with_row_index("_i").group_by("apt", "mo").agg(C("_i")).rows():
        if apt in SKIP or (apt, mo) not in e26:
            continue
        r = e26[(apt, mo)]
        target = r["ec_mean"] + fit[apt][0] + fit[apt][1] * r["xf"]
        idx = np.asarray(idx)
        K = r["excl"] if apt in MONSTER_EXCL else int((p[idx] > 7200).sum())
        keep = idx[np.argsort(p[idx], kind="stable")[: max(len(idx) - K, 0)]]
        shift = target - p[keep].mean()
        out[keep] = p[keep] + shift
        total += shift * len(keep)
        print(f"  {apt} {mo}: rows {len(idx):,}  K {K}  target {target:7.1f}  mean {p[keep].mean():7.1f}  shift {shift:+6.1f}")
    out = np.rint(np.clip(out, 0, None))
    sub.with_columns(pl.Series(T, out).cast(sub[T].dtype)).write_parquet(a.out)
    dd = out - p
    print(f"wrote {a.out}: mean shift {dd.mean():+.2f}s  RMS {np.sqrt((dd ** 2).mean()):.2f}s  rows moved {(dd != 0).sum():,}")


if __name__ == "__main__":
    main()
