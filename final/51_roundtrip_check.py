"""Stage 51: round trip of a candidate file against the published daily figures (label-free for 2026).

From the predicted off-block stamp (take-off minus prediction) of each departure, per airport and UTC day of the stamp:
  (a) the number of IFR departures against the published FLT_DEP_IFR_2 (+-2 allowed);
  (b) the delay total sum max(stamp - schedule, 0) in minutes against the published DLY_ALL_PRE_2, as a residual per departure, scored as a
      z value against the same residual measured on the TRUE stamps of the twelve 2025 months of that airport (2025 training labels:
      median and 1.4826 x MAD of the per-departure residual).
Days with a stamp date outside the file's months, the last day of each month (late stamps cut off) and days whose published departure count
differs from ours by more than 15 are skipped. Zurich is scored on IFR flights only, as the daily layer does.

  python final/51_roundtrip_check.py data/final/final_A.parquet [--ranking ...]
With --pre FILE (the file before the daily layer) the delay residual is scored against what a layer capped at lam should leave:
res = offset + (1 - lam) x res_pre, offset and spread per airport-month from the days the layer moved; the FLAG line counts days more than 6 robust sigma off it.
Prints per airport-month: days checked, share of days with the IFR count inside +-2, share with |z| > 6, worst z. Exit status 0 (report only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL, RAW, REPO, TARGET_COL  # noqa: E402

C = pl.col
EURO = REPO / "data" / "external" / "eurocontrol"
IFR_ONLY_DELAY = {"LSZH"}


def published() -> pl.DataFrame:
    parts = []
    for f in sorted(EURO.glob("all_pre_departure_delays_20*.csv")):
        d = pl.read_csv(f, infer_schema_length=0)
        parts.append(d.select(C("APT_ICAO").alias("apt"), C("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"),
                              C("FLT_DEP_1").cast(pl.Float64, strict=False).alias("n_pub"), C("FLT_DEP_IFR_2").cast(pl.Float64, strict=False).alias("ifr_pub"),
                              C("DLY_ALL_PRE_2").cast(pl.Float64, strict=False).alias("dly_pub")))
    return pl.concat(parts)


def day_table(f: pl.DataFrame) -> pl.DataFrame:
    """f: apt, stamp_day, sched_gap_min (stamp - sched in min), ifr (bool). One row per airport-day."""
    return (f.group_by("apt", "day").agg(n=pl.len(), ifr_n=C("ifr").sum(),
                                         dly_all=C("late").sum(), dly_ifr=(C("late") * C("ifr")).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--pre", type=Path, default=None, help="the file before the daily layer: score against the residual a capped layer should leave")
    ap.add_argument("--lam", type=float, default=0.5, help="cap of the daily layer")
    a = ap.parse_args()
    pub = published()

    dep = (pl.scan_parquet(a.ranking).filter(C("PHASE_mvt") == "DEP")
           .select(C(ID_COL).cast(pl.Float64), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   (C("FLIGHT_RULE_mvt") == "I").fill_null(False).alias("ifr")).collect())
    def table(path: Path) -> pl.DataFrame:
        cand = pl.read_parquet(path).select(C(ID_COL).cast(pl.Float64), C(TARGET_COL).cast(pl.Float64).alias("p"))
        f = dep.join(cand, on=ID_COL).with_columns((C("T") - pl.duration(seconds=C("p"))).alias("stamp"))
        f = f.with_columns(C("stamp").dt.date().alias("day"), (((C("stamp") - C("S")).dt.total_seconds() / 60.0).clip(lower_bound=0)).alias("late"),
                           C("T").dt.month().alias("month"))
        last_day = f.group_by("apt").agg(C("day").max().alias("last"))
        t = day_table(f).join(last_day, on="apt").filter(C("day") < C("last"))
        t = t.join(pub, on=["apt", "day"], how="inner").filter((C("n_pub") - C("n")).abs() <= 15)
        t = t.with_columns(pl.when(C("apt").is_in(list(IFR_ONLY_DELAY))).then(C("dly_ifr")).otherwise(C("dly_all")).alias("dly"),
                           pl.when(C("apt").is_in(list(IFR_ONLY_DELAY))).then(C("ifr_n")).otherwise(C("n")).alias("n_dly"))
        return t.with_columns(((C("dly_pub") - C("dly")) / C("n_dly")).alias("res"), (C("ifr_n") - C("ifr_pub")).abs().alias("ifr_diff"))

    t = table(a.file)
    months = sorted(set(t["day"].dt.month().to_list()))
    # 2025 baseline from the true stamps
    tr = (pl.scan_parquet(sorted(str(x) for x in RAW.glob("training_2025-*.parquet"))).filter(C("PHASE_mvt") == "DEP")
          .select(C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"), C("BLOCK_TIME_UTC_mvt").alias("B"),
                  (C("FLIGHT_RULE_mvt") == "I").fill_null(False).alias("ifr")).drop_nulls(["B"]).collect()
          .with_columns(C("B").dt.date().alias("day"), (((C("B") - C("S")).dt.total_seconds() / 60.0).clip(lower_bound=0)).alias("late")))
    tb = day_table(tr).join(pub, on=["apt", "day"], how="inner").filter((C("n_pub") - C("n")).abs() <= 15)
    tb = tb.with_columns(pl.when(C("apt").is_in(list(IFR_ONLY_DELAY))).then(C("dly_ifr")).otherwise(C("dly_all")).alias("dly"),
                         pl.when(C("apt").is_in(list(IFR_ONLY_DELAY))).then(C("ifr_n")).otherwise(C("n")).alias("n_dly"))
    tb = tb.with_columns(((C("dly_pub") - C("dly")) / C("n_dly")).alias("res"))
    base = tb.group_by("apt").agg(med=C("res").median(), sd=(1.4826 * (C("res") - C("res").median()).abs().median()))
    t = t.join(base, on="apt", how="left").with_columns(((C("res") - C("med")) / C("sd").clip(lower_bound=1e-6)).alias("z"), C("day").dt.month().alias("month"))
    if a.pre:
        # The daily layer is capped at lam (0.5): it removes at most lam of the gap between the published total and the implied one, so the residual
        # that should remain is (1 - lam) x the pre-layer residual (both measured from the 2025 identity offset). z_exp is the departure from that.
        tp = table(a.pre).select("apt", "day", C("res").alias("res_pre"), C("dly").alias("dly_pre"))
        t = t.join(tp, on=["apt", "day"], how="left").with_columns(
            ((C("dly") - C("dly_pre")).abs() < 1e-9).alias("inactive"), C("day").dt.month().alias("month"))
        # Expected: res = offset + (1 - lam) x res_pre, with the offset (the own-month bias of the layer) and the spread estimated per airport-month
        # from the days the layer moved; a day is flagged when it lies more than 6 robust sigma (floor 0.05 min per departure) off that line.
        t = t.with_columns((C("res") - (1 - a.lam) * C("res_pre")).alias("dev"))
        ref = (t.filter(~C("inactive")).group_by("apt", "month").agg(off=C("dev").median(), mad=(C("dev") - C("dev").median()).abs().median()))
        t = t.join(ref, on=["apt", "month"], how="left").with_columns(
            ((C("dev") - C("off")) / (1.4826 * C("mad")).clip(lower_bound=0.05)).alias("z_exp"))
        t = t.with_columns(pl.when(C("inactive")).then(None).otherwise(C("z_exp")).alias("z_exp"))

    out = (t.group_by("apt", "month").agg(days=pl.len(), ifr_days=C("ifr_pub").is_not_null().sum(),
                                          ifr_ok=((C("ifr_diff") <= 2) & C("ifr_pub").is_not_null()).sum(),
                                          z_gt6=(C("z").abs() > 6).sum(), worst_z=C("z").abs().max(), res_mean=C("res").mean(),
                                          **({"inactive": C("inactive").sum(), "exp_gt6": (C("z_exp").abs() > 6).sum(), "worst_exp": C("z_exp").abs().max()} if a.pre else {}))
           .with_columns((C("ifr_ok") / C("ifr_days").clip(lower_bound=1)).alias("ifr_share")).sort("apt", "month").with_columns(pl.col(pl.Float64).round(2)))
    print(f"round trip of {a.file.name}: months {months}")
    with pl.Config(tbl_rows=60, tbl_cols=12):
        print(out)
    n_bad = int((out["z_gt6"]).sum())
    print(f"REPORT  {n_bad} airport-day(s) with |z| > 6 on the delay total (raw, informative only: a capped layer leaves residual); count inside +-2 on {int(out['ifr_ok'].sum())} of {int(out['ifr_days'].sum())} IFR-published days")
    if a.pre:
        fl = t.filter(C("z_exp").abs() > 6).sort(C("z_exp").abs(), descending=True)
        print(f"FLAG  {fl.height} airport-day(s) where the residual differs from (1 - {a.lam:g}) x the pre-layer residual by more than 6 sigma "
              f"({int(t['inactive'].sum())} days where the layer moved nothing are not scored: stop days, count mismatch)")
        if fl.height:
            print(fl.select("apt", "day", "res_pre", "res", "z_exp").head(25))


if __name__ == "__main__":
    main()
