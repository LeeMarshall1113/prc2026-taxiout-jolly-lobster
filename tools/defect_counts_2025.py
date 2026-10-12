#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 the authors of prc-taxiout-2026. Free software under the GNU GPL v3 or later; no warranty.
"""Counts behind docs/DATA-DEFECTS.md, re-run from the challenge's 2025 training files.

    python tools/defect_counts_2025.py [--data data/raw] [--section all|headline|copies|dayearly|window|refiled|dups|late|hold|istanbul|heli|default|daily|summary]

Reads data/raw/training_2025-*.parquet (departures only: PHASE_mvt == DEP; 2,085,047 rows at the ten airports) and, for
the section ``daily``, data/external/eurocontrol/all_pre_departure_delays_2025.csv (EUROCONTROL's published daily
airport file). No model, no 2026 data. Standard scientific Python only (polars, numpy); the stand-default section also
imports tools/check_offblock_stamps.py from the same folder. Each section alone peaks at 1.2 to 2.4 GB; all sections in one run peak at about 3 GB, so run them one at a
time (and under a memory cap if the machine is shared).

Definitions used throughout (seconds): gap = take-off minus schedule; y = take-off minus recorded off-block (the label);
gl = take-off minus LOBT (the last planned off-block of the flight plan); ga = take-off minus AOBT_3 (the network manager's
off-block); L = gl - 3,606 (the lower edge of the plan-clock window); "plan row" = LOBT present.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_fmt_str_lengths(40)

COLS = ["MVT_ID_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
        "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "LOBT_flt",
        "EOBT_1_flt", "AOBT_3_flt"]
TZ = {"LIRF": "Europe/Rome", "LSZH": "Europe/Zurich", "EGLL": "Europe/London", "LFPG": "Europe/Paris",
      "EHAM": "Europe/Amsterdam", "LEMD": "Europe/Madrid", "LEBL": "Europe/Madrid", "EDDF": "Europe/Berlin",
      "EDDM": "Europe/Berlin", "LTFM": "Europe/Istanbul"}
W = 3606


def secs(a: str, b: str) -> pl.Expr:
    return (pl.col(a) - pl.col(b)).dt.total_milliseconds() / 1000.0


def load(data: str) -> pl.DataFrame:
    parts = [pl.read_parquet(f, columns=COLS).filter(pl.col("PHASE_mvt") == "DEP")
             for f in sorted(glob.glob(str(Path(data) / "training_2025-*.parquet")))]
    if not parts:
        sys.exit(f"no training_2025-*.parquet under {data}")
    d = pl.concat(parts)
    return d.with_columns(
        y=secs("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"), gap=secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        gl=secs("MVT_TIME_UTC_mvt", "LOBT_flt"), ga=secs("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        sl=secs("SCHED_TIME_UTC_mvt", "LOBT_flt"), plan=pl.col("LOBT_flt").is_not_null(),
        month=pl.col("MVT_TIME_UTC_mvt").dt.month(), apt=pl.col("ADEP_mvt"))


def ldate(d: pl.DataFrame, apt: str) -> pl.DataFrame:
    """Add local calendar dates (airport zone) of take-off, schedule and recorded off-block for one airport."""
    tz = TZ[apt]
    f = lambda c: pl.col(c).dt.replace_time_zone("UTC").dt.convert_time_zone(tz).dt.date()
    return d.filter(pl.col("apt") == apt).with_columns(d_take=f("MVT_TIME_UTC_mvt"), d_sched=f("SCHED_TIME_UTC_mvt"),
                                                         d_block=f("BLOCK_TIME_UTC_mvt"))


def hdr(t: str) -> None:
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


# ----------------------------------------------------------------------------------------------------------------
def headline(d):
    hdr("HEADLINE: how much of the spread of the recorded stamp sits in a few records")
    d = d.with_columns(med=pl.col("y").median().over(["apt", "month"])).with_columns(sq=(pl.col("y") - pl.col("med")) ** 2)
    tot = d["sq"].sum()
    print(f"departures {d.height}; squared deviation of the label from its airport-month median, total {tot:.4e}")
    for name, cond in [("label > 7,200 s", pl.col("y") > 7200), ("label > 3,600 s", pl.col("y") > 3600),
                       ("label > 1,800 s", pl.col("y") > 1800), ("label < 0", pl.col("y") < 0)]:
        x = d.filter(cond)
        print(f"  {name:<18} {x.height:>8d} rows ({100 * x.height / d.height:.3f}%) carry {100 * x['sq'].sum() / tot:.1f}% of it")
    srt = d["sq"].sort(descending=True).to_numpy()
    c = np.cumsum(srt) / tot
    for q in (0.25, 0.5, 0.75):
        print(f"  the largest {int((c < q).sum() + 1):>6d} rows carry {int(q * 100)}%")
    print("  per airport: rows needed for half of the airport's own squared deviation, and the top-50 share")
    for a in sorted(d["apt"].unique().to_list()):
        r = d.filter(pl.col("apt") == a)
        s = r["sq"].sort(descending=True).to_numpy()
        cc = np.cumsum(s) / s.sum()
        print(f"    {a}  {int((cc < 0.5).sum() + 1):>6d} of {r.height:>7d} rows for half; top 50 rows {100 * cc[49]:.1f}%")


def copies(d):
    hdr("SCHEDULE COPIES: label within 90 s of the schedule gap, schedule gap >= 1,800 s")
    inf = d.filter(pl.col("gap") >= 1800)
    inf = inf.with_columns(copy=((pl.col("y") - pl.col("gap")).abs() <= 90))
    print(inf.group_by("apt").agg(informative=pl.len(), copies=pl.col("copy").sum(),
                                    copies_plan=(pl.col("copy") & pl.col("plan")).sum(),
                                    copies_noplan=(pl.col("copy") & ~pl.col("plan")).sum(),
                                    plan_rows=pl.col("plan").sum(), noplan_rows=(~pl.col("plan")).sum()).with_columns(
        rate=pl.col("copies") / pl.col("informative"),
        rate_plan=pl.col("copies_plan") / pl.col("plan_rows"),
        rate_noplan=pl.col("copies_noplan") / pl.col("noplan_rows")).sort("apt"))
    print("\nRome by month (informative rows): no-plan copies / no-plan rows, with-plan copies / with-plan rows")
    r = inf.filter(pl.col("apt") == "LIRF")
    print(r.group_by("month").agg(np_copy=(pl.col("copy") & ~pl.col("plan")).sum(), np_n=(~pl.col("plan")).sum(),
                                   wp_copy=(pl.col("copy") & pl.col("plan")).sum(), wp_n=pl.col("plan").sum()).sort("month"))
    print("\nplan-window test: plan rows whose schedule is more than 3,606 s from the plan's last off-block (a copy would not")
    print("have been joined to the plan): rows and copies, by airport")
    ci = d.filter(pl.col("plan") & (pl.col("sl").abs() > W))
    print(ci.group_by("apt").agg(rows=pl.len(), copies=(((pl.col("y") - pl.col("gap")).abs() <= 90) & (pl.col("gap") >= 1800)).sum()).sort("apt"))
    print(f"  all airports: {ci.height} rows, copies {ci.filter(((pl.col('y') - pl.col('gap')).abs() <= 90) & (pl.col('gap') >= 1800)).height}")
    print("\nBarcelona: where the stamp equals the schedule, minute histogram of (y - gap) for informative plan rows with gap 1,800-3,700 s")
    b = d.filter((pl.col("apt") == "LEBL") & pl.col("plan") & (pl.col("gap") >= 1800) & (pl.col("gap") < 3700))
    off = (b["y"] - b["gap"]).to_numpy()
    print("  offset from the schedule in minutes (stamp minus schedule = -(y - gap)) -> rows:")
    for m in range(-5, 6):
        n = int(((-off >= m * 60 - 30) & (-off < m * 60 + 30)).sum())
        print(f"    {m:+d} min: {n}")
    exact = int((np.abs(off) <= 30).sum())
    print(f"  Barcelona rows within +-30 s of the schedule: {exact} of {len(off)}; mean of the 10 flanking minutes: "
          f"{np.mean([int(((-off >= m * 60 - 30) & (-off < m * 60 + 30)).sum()) for m in list(range(-5, 0)) + list(range(1, 6))]):.0f}")
    print("  the same test at the other airports (rows within +-30 s of the schedule; mean of the flanking +-1..5 minutes)")
    for a in sorted(d["apt"].unique().to_list()):
        x = d.filter((pl.col("apt") == a) & pl.col("plan") & (pl.col("gap") >= 1800) & (pl.col("gap") < 3700))
        o = -(x["y"] - x["gap"]).to_numpy()
        z = int((np.abs(o) <= 30).sum())
        fl = np.mean([int(((o >= m * 60 - 30) & (o < m * 60 + 30)).sum()) for m in list(range(-5, 0)) + list(range(1, 6))])
        print(f"    {a}: {z} vs {fl:.0f} ({z / fl:.1f}x)")
    print("\nBarcelona copies (plan rows, gap 1,800-3,700, |y-gap|<=90): plan never updated (LOBT = schedule)?")
    bb = b.with_columns(copy=((pl.col("y") - pl.col("gap")).abs() <= 90), same=(pl.col("sl").abs() <= 60))
    print(bb.group_by("same").agg(n=pl.len(), copies=pl.col("copy").sum()).with_columns(rate=pl.col("copies") / pl.col("n")))
    print("  by month:")
    print(bb.group_by("month").agg(n=pl.len(), copies=pl.col("copy").sum()).with_columns(rate=pl.col("copies") / pl.col("n")).sort("month"))


def dayearly(d):
    hdr("DAY-EARLY STAMPS: label 82,800-95,000 s")
    de = d.filter((pl.col("y") >= 82800) & (pl.col("y") <= 95000))
    print(f"all airports: {de.height} rows")
    print(de.group_by("apt").agg(n=pl.len(), plan=pl.col("plan").sum()).sort("apt"))
    print(de.select(["apt", "month", "FLIGHT_mvt", "plan", "gap", "y", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]).sort(["apt", "MVT_TIME_UTC_mvt"]))
    r = ldate(d, "LIRF")
    r = r.with_columns(later=(pl.col("d_take") > pl.col("d_sched")), de=((pl.col("y") >= 82800) & (pl.col("y") <= 95000)),
                       copy=((pl.col("y") - pl.col("gap")).abs() <= 90))
    print(f"\nRome departures {r.height}; take-off on a later local day than the schedule {r.filter(pl.col('later')).height}; same local day {r.filter(~pl.col('later')).height}")
    print(f"  day-early on same-day rows: {r.filter(~pl.col('later') & pl.col('de')).height}; on later-day rows: {r.filter(pl.col('later') & pl.col('de')).height}")
    print("  day-early rows: stamp's local date equals the schedule's local date?",
          r.filter(pl.col("de")).select((pl.col("d_block") == pl.col("d_sched")).sum().alias("same_date"), pl.len().alias("n")).row(0))
    print("\nRome no-plan rows with a schedule gap of 6 h or more:")
    x = r.filter(~pl.col("plan") & (pl.col("gap") >= 21600))
    x = x.with_columns(kind=pl.when(pl.col("de")).then(pl.lit("day-early")).when(pl.col("copy")).then(pl.lit("copy")).otherwise(pl.lit("other")))
    print(f"  {x.height} rows; stamp on the schedule's local date: {x.filter(pl.col('d_block') == pl.col('d_sched')).height}")
    print(x.group_by(["later", "kind"]).len().sort(["later", "kind"]))
    print("\nRome day-early rows by schedule gap and plan state:")
    print(r.filter(pl.col("de")).select(["FLIGHT_mvt", "plan", "gap", "later"]).sort("gap"))
    print("\nWhere a real stamp made after local midnight is dated: Rome rows with take-off on a later local day and gap < 6 h")
    z = r.filter(pl.col("later") & (pl.col("gap") < 21600) & ~pl.col("de"))
    print(z.group_by("plan").agg(n=pl.len(), stamp_on_takeoff_date=(pl.col("d_block") == pl.col("d_take")).sum(),
                                  stamp_on_sched_date=(pl.col("d_block") == pl.col("d_sched")).sum()))
    print("\nOther airports: same-day day-early rows")
    for a in ["LSZH", "LFPG", "EGLL", "EHAM", "LEBL", "LEMD", "EDDF", "EDDM", "LTFM"]:
        q = ldate(d, a).filter((pl.col("y") >= 82800) & (pl.col("y") <= 95000))
        if q.height:
            print(a, q.select(["FLIGHT_mvt", "plan", "gap", "y", "d_take", "d_sched", "d_block"]))
    z = ldate(d, "LSZH")
    print("Zurich take-off on a later local day than the schedule:", z.filter(pl.col("d_take") > pl.col("d_sched")).height, "of", z.height)
    print("\nlong labels (> 40,000 s) outside Rome:")
    print(d.filter((pl.col("y") > 40000) & (pl.col("apt") != "LIRF")).select(["apt", "FLIGHT_mvt", "plan", "gap", "y", "month"]))
    print("Rome labels > 40,000 s:", d.filter((pl.col("y") > 40000) & (pl.col("apt") == "LIRF")).height,
          "(no plan", d.filter((pl.col("y") > 40000) & (pl.col("apt") == "LIRF") & ~pl.col("plan")).height, ")")


def window(d):
    hdr("PLAN-CLOCK WINDOW: |y - gl| <= 3,606 on every plan row?")
    p = d.filter(pl.col("plan"))
    p = p.with_columns(dev=(pl.col("y") - pl.col("gl")).abs())
    print(p.group_by("apt").agg(plan_rows=pl.len(), exceptions=(pl.col("dev") > W).sum(), max_dev=pl.col("dev").max(),
                                  p999=pl.col("dev").quantile(0.999)).sort("apt"))
    print(f"all airports: {p.height} plan rows, exceptions {p.filter(pl.col('dev') > W).height}, max |y - gl| {p['dev'].max():.0f}")
    print("\nmonthly maximum |y - gl|:", p.group_by("month").agg(m=pl.col("dev").max()).sort("month")["m"].to_list())
    print("\nRows where the plan clock and the schedule are far apart (|schedule - LOBT| > 3,606 s):",
          p.filter(pl.col("sl").abs() > W).height, "of", p.height)
    print("\nno-plan rows: how many have a stamp that could not have been joined (lower bound on the displaced-stamp population)?")
    print(d.group_by("apt").agg(no_plan=(~pl.col("plan")).sum(), no_plan_rate=(~pl.col("plan")).mean()).sort("apt"))


def refiled(d):
    hdr("STAMPS AT THE PLAN CLOCK ON RE-FILED ROME FLIGHTS (the v148 family and its softer neighbour)")
    for a in ["LIRF", "EGLL", "LFPG", "LTFM", "EDDF", "EDDM", "EHAM", "LSZH", "LEMD", "LEBL"]:
        p = d.filter((pl.col("apt") == a) & pl.col("plan"))
        p = p.with_columns(L=pl.col("gl") - W).with_columns(pos=(pl.col("y") - pl.col("L")) / W)
        f = p.filter((pl.col("L") > 2800) & ((pl.col("L") - pl.col("ga")) > 1800))
        if f.height == 0:
            print(a, "0 rows")
            continue
        print(f"{a}: rows meeting 'lower edge > 2,800 and more than 1,800 s above AOBT_3': {f.height}; label within 60 s of take-off minus LOBT: "
              f"{f.filter((pl.col('y') - pl.col('gl')).abs() <= 60).height}; within 900 s: {f.filter((pl.col('y') - pl.col('gl')).abs() <= 900).height}; "
              f"mean position in the window (0 = late edge, 1 = LOBT) {f['pos'].mean():.2f}; label at the late edge (pos < 0.02): {f.filter(pl.col('pos') < 0.02).height}")
    p = d.filter((pl.col("apt") == "LIRF") & pl.col("plan")).with_columns(L=pl.col("gl") - W, dle=secs("EOBT_1_flt", "LOBT_flt"))
    p = p.with_columns(pos=(pl.col("y") - pl.col("L")) / W)
    fam = p.filter((pl.col("L") > 2800) & ((pl.col("L") - pl.col("ga")) > 1800))
    print(f"\nRome: of the {fam.height} rows of the family, those whose two plan clocks differ by an hour or more "
          f"(EOBT_1 minus LOBT >= 3,600 s): {fam.filter(pl.col('dle') >= 3600).height}")
    print(f"  plan clock equal to the schedule (|schedule - LOBT| <= 60 s) on {fam.filter(pl.col('sl').abs() <= 60).height} of the {fam.height}")
    print("Rome softer neighbour: EOBT_1 minus LOBT >= 3,600 s, AOBT_3 reading below the lower edge, 0 < lower edge <= 2,800 s:")
    s = p.filter((pl.col("dle") >= 3600) & (pl.col("ga") < pl.col("L")) & (pl.col("L") > 0) & (pl.col("L") <= 2800))
    print(f"  {s.height} rows, label within 60 s of take-off minus LOBT on {s.filter((pl.col('y') - pl.col('gl')).abs() <= 60).height}, "
          f"mean position {s['pos'].mean() if s.height else float('nan'):.2f}")
    rest = p.filter((pl.col("dle") < 3600) & (pl.col("ga") < pl.col("L")) & (pl.col("L") > 0) & (pl.col("L") <= 2800))
    print(f"  the same rows without the clock-difference condition: {rest.height + s.height}; of the {rest.height} others the label is "
          f"within 60 s of take-off minus LOBT on {rest.filter((pl.col('y') - pl.col('gl')).abs() <= 60).height}")


def dups(d):
    hdr("DUPLICATE MOVEMENT RECORDS: same airport and flight string, take-off within 1,800 s of another record")
    s = d.select(["apt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "plan", "y", "gap", "MVT_ID_mvt"]).drop_nulls(["FLIGHT_mvt"]).with_columns(
        ts=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s")).sort(["apt", "FLIGHT_mvt", "ts"])
    s = s.with_columns(prev=(pl.col("ts") - pl.col("ts").shift(1)).over(["apt", "FLIGHT_mvt"]),
                       nxt=(pl.col("ts").shift(-1) - pl.col("ts")).over(["apt", "FLIGHT_mvt"]))
    hit = s.filter((pl.col("prev") <= 1800) | (pl.col("nxt") <= 1800))
    print("records in a close pair, by airport:")
    print(hit.group_by("apt").agg(records=pl.len(), plan_records=pl.col("plan").sum()).sort("apt"))
    # pair-level for Paris
    for a in ["LFPG"]:
        x = hit.filter(pl.col("apt") == a)
        x = x.with_columns(grp=((pl.col("prev").is_null()) | (pl.col("prev") > 1800)).cast(pl.Int64).cum_sum())
        g = x.group_by("grp").agg(n=pl.len(), nplan=pl.col("plan").sum(), ts_span=pl.col("ts").max() - pl.col("ts").min(),
                                   y_plan=pl.col("y").filter(pl.col("plan")).max(), y_noplan=pl.col("y").filter(~pl.col("plan")).max(),
                                   gap_plan=pl.col("gap").filter(pl.col("plan")).max())
        print(f"\n{a}: groups {g.height}; sizes", g["n"].value_counts().sort("n").rows())
        pp = g.filter((pl.col("n") == 2) & (pl.col("nplan") == 1))
        print(f"  pairs of exactly two with one plan record and one without: {pp.height}")
        print(f"  plan record's label larger than its partner's by more than 300 s: {pp.filter((pl.col('y_plan') - pl.col('y_noplan')) > 300).height}")
        print(f"  median take-off spread within pair {pp['ts_span'].median():.0f} s (min {pp['ts_span'].min():.0f}, max {pp['ts_span'].max():.0f})")
        print(f"  median plan-record label {pp['y_plan'].median():.0f} s; median no-plan record label {pp['y_noplan'].median():.0f} s")
        print(f"  pairs with plan-record gap of 3,000-50,000 s: {pp.filter((pl.col('gap_plan') >= 3000) & (pl.col('gap_plan') < 50000)).height};"
              f" of those with the plan record larger by 300 s: {pp.filter((pl.col('gap_plan') >= 3000) & (pl.col('gap_plan') < 50000) & ((pl.col('y_plan') - pl.col('y_noplan')) > 300)).height}")
        print(f"  pairs with both records carrying a plan: {g.filter((pl.col('n') == 2) & (pl.col('nplan') == 2)).height}")
        print("  pairs by month:", pp.join(x.group_by("grp").agg(m=pl.col("ts").min()), on="grp").with_columns(
            mo=pl.from_epoch("m", "s").dt.month()).group_by("mo").len().sort("mo")["len"].to_list())


def late(d):
    hdr("LATE STAMPS: label near zero on ordinary departures (label < 60 s, not helicopter / runway H)")
    heli = (pl.col("RUNWAY_mvt") == "H") | pl.col("AIRCRAFT_TYPE_mvt").is_in(
        ["A139", "AS50", "AS32", "EC20", "EC30", "EC35", "EC45", "B429", "B06", "A109", "A119", "S76", "H60"])
    x = d.with_columns(heli=heli.fill_null(False), nz=((pl.col("y") < 60) & ~heli.fill_null(False)),
                       late=(pl.col("plan") & (pl.col("y") < 300) & ((pl.col("ga") - pl.col("y")) >= 300) & ~heli.fill_null(False)),
                       neg=(pl.col("y") <= 0))
    print(x.group_by("apt").agg(rows=pl.len(), near_zero=pl.col("nz").sum(), late_vs_NM=pl.col("late").sum(), le0=pl.col("neg").sum(),
                                  rate_nz=pl.col("nz").mean(), rate_late_plan=pl.col("late").sum() / pl.col("plan").sum()).sort("apt"))
    print("\nmonthly near-zero rate (%) at the three airports where it matters: min / max over the twelve months")
    for a in ["LFPG", "EHAM", "LEBL", "LSZH"]:
        m = x.filter(pl.col("apt") == a).group_by("month").agg(r=100 * pl.col("nz").mean()).sort("month")["r"].to_list()
        print(f"  {a}: min {min(m):.2f}, max {max(m):.2f}   {[round(v, 2) for v in m]}")
    print("\nAmsterdam non-helicopter plan rows with label < 300 s: AOBT_3 minus the stamp, median (minutes) against ordinary rows")
    e = x.filter((pl.col("apt") == "EHAM") & pl.col("plan") & ~pl.col("heli"))
    print(f"  label < 300: {e.filter(pl.col('y') < 300).height} rows ({100 * e.filter(pl.col('y') < 300).height / e.height:.2f}% of plan rows); "
          f"median (ga - y)/60 = {((e.filter(pl.col('y') < 300)['ga'] - e.filter(pl.col('y') < 300)['y']) / 60).median():.1f} min; "
          f"ordinary rows {((e.filter(pl.col('y') >= 300)['ga'] - e.filter(pl.col('y') >= 300)['y']) / 60).median():.1f} min")
    print("\nhour-of-day rate of late0 (plan, y<300, NM later by 300 s) at EHAM, LFPG: max cell over 24 local-ish UTC hours")
    for a in ["EHAM", "LFPG"]:
        h = x.filter((pl.col("apt") == a) & pl.col("plan")).with_columns(h=pl.col("MVT_TIME_UTC_mvt").dt.hour()).group_by("h").agg(r=100 * pl.col("late").mean(), n=pl.len()).sort("h")
        print(f"  {a}:", [round(v, 2) for v in h["r"].to_list()])


def hold(d):
    hdr("HOLD RECORDS: label >= 3,600 s, stamp between 10 min before and 40 min after the schedule, not a copy, not day-shifted")
    x = d.with_columns(stamp=pl.col("gap") - pl.col("y"))
    h = x.filter((pl.col("y") >= 3600) & (pl.col("stamp") >= -600) & (pl.col("stamp") <= 2400) & ((pl.col("y") - pl.col("gap")).abs() > 90)
                 & ~((pl.col("y") >= 82800) & (pl.col("y") <= 95000)))
    print(h.group_by("apt").agg(n=pl.len(), plan=pl.col("plan").sum(), median_y=pl.col("y").median(), median_gap=pl.col("gap").median()).sort("apt"))
    h = h.with_columns(day=pl.col("MVT_TIME_UTC_mvt").dt.date())
    print("\nconcentration on days: share of each airport's hold-like rows on its 5 busiest days (and the number of days with >= 5 rows)")
    for a in sorted(h["apt"].unique().to_list()):
        c = h.filter(pl.col("apt") == a).group_by("day").len().sort("len", descending=True)
        print(f"  {a}: {int(c['len'].sum())} rows on {c.height} days; top-5 days {c['len'].head(5).sum() / c['len'].sum() * 100:.0f}%  {list(zip([str(v) for v in c['day'].head(5).to_list()], c['len'].head(5).to_list()))}; days with >=5 rows: {c.filter(pl.col('len') >= 5).height}")
    print("\nwhy inputs cannot tell: among rows with schedule gap >= 5,400 s and < 50,000 s (take-off long after schedule),")
    print("share that are hold-like, and the share whose stamp is at the real push (|y - ga| <= 900 on plan rows):")
    big = x.filter((pl.col("gap") >= 5400) & (pl.col("gap") < 50000)).with_columns(
        hold=((pl.col("y") >= 3600) & (pl.col("stamp") >= -600) & (pl.col("stamp") <= 2400) & ((pl.col("y") - pl.col("gap")).abs() > 90)),
        real=(pl.col("plan") & ((pl.col("y") - pl.col("ga")).abs() <= 900)))
    print(big.group_by("apt").agg(rows=pl.len(), hold=pl.col("hold").sum(), hold_rate=pl.col("hold").mean(), real_at_NM=pl.col("real").sum()).sort("apt"))
    print("\nplan rows with gap 5,400-50,000 s, by how far the plan clock (LOBT) sits after the schedule: rows and hold-like rate")
    q = big.filter(pl.col("plan") & pl.col("apt").is_in(["EGLL", "LTFM", "LFPG", "LIRF"])).with_columns(
        moved=pl.when(-pl.col("sl") < 900).then(pl.lit("a: plan clock within 15 min of schedule"))
        .when(-pl.col("sl") < 2700).then(pl.lit("b: 15-45 min after schedule")).otherwise(pl.lit("c: 45 min or more after schedule")))
    print(q.group_by(["apt", "moved"]).agg(rows=pl.len(), hold=pl.col("hold").sum(), hold_rate=pl.col("hold").mean()).sort(["apt", "moved"]))


def istanbul(d):
    hdr("ISTANBUL NO-PLAN ROWS: flights to Ercan (LCEN) against the rest")
    t = d.filter(pl.col("apt") == "LTFM")
    n = t.filter(~pl.col("plan")).with_columns(lcen=(pl.col("ADES_mvt") == "LCEN"))
    print(n.group_by("lcen").agg(rows=pl.len(), mean_y=pl.col("y").mean(), over3600=(pl.col("y") > 3600).sum(),
                                   gap_lt_3000=(pl.col("gap") < 3000).mean()))
    print("with plan, to LCEN:", t.filter(pl.col("plan") & (pl.col("ADES_mvt") == "LCEN")).height, "of", t.filter(pl.col("ADES_mvt") == "LCEN").height, "LCEN departures")
    e = n.filter(pl.col("gap") >= 0).with_columns(ex=pl.col("y") - pl.col("gap"))
    print(f"no-plan, take-off at or after schedule: {e.height} rows; max (y - gap) {e['ex'].max():.0f}; above +1,200 s: {e.filter(pl.col('ex') > 1200).height}")
    o = d.filter((pl.col("apt") != "LTFM") & (pl.col("apt") != "LIRF") & ~pl.col("plan") & (pl.col("gap") >= 0)).with_columns(ex=pl.col("y") - pl.col("gap"))
    print(f"outside Istanbul and Rome the same bound: {o.filter(pl.col('ex') > 1200).height} exceptions in {o.height} no-plan rows")
    print("Istanbul no-plan by month (rows, mean label):",
          n.group_by("month").agg(n=pl.len(), m=pl.col("y").mean()).sort("month").rows())


def heli(d):
    hdr("HELICOPTER AND GENERAL-AVIATION MOVEMENTS")
    h = d.filter(pl.col("RUNWAY_mvt") == "H")
    print("runway code H:", h.group_by("apt").agg(n=pl.len(), under120=(pl.col("y") < 120).sum(), mean_y=pl.col("y").mean(),
                                                    med_y=pl.col("y").median(), no_plan=(~pl.col("plan")).sum()).rows())
    print("types on runway H at LSZH:", h["AIRCRAFT_TYPE_mvt"].value_counts().sort("count", descending=True).head(8).rows())
    e = d.filter((pl.col("apt") == "EHAM") & pl.col("STAND_mvt").str.starts_with("HG").fill_null(False))
    print(f"Amsterdam helicopter stands (HG*): {e.height} rows, no plan {e.filter(~pl.col('plan')).height}, median label {e['y'].median():.0f} s, "
          f"mean {e['y'].mean():.0f}, max {e['y'].max():.0f}; types {e['AIRCRAFT_TYPE_mvt'].value_counts().sort('count', descending=True).head(3).rows()}")
    e2 = d.filter((pl.col("apt") == "EHAM") & (pl.col("y") > 7200) & pl.col("STAND_mvt").str.starts_with("HG").fill_null(False))
    print("  of which labels over 7,200 s:", e2.select(["FLIGHT_mvt", "y", "STAND_mvt", "RUNWAY_mvt"]).rows())
    print("\nflight rule V, by airport (rows, mean label, share with label < 120 s):")
    print(d.filter(pl.col("FLIGHT_RULE_mvt") == "V").group_by("apt").agg(n=pl.len(), mean_y=pl.col("y").mean(), lt120=(pl.col("y") < 120).mean()).sort("apt"))


def default(d):
    hdr("NETWORK-MANAGER OFF-BLOCK THAT IS A STAND DEFAULT (tools/check_offblock_stamps.py, over the whole year)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_offblock_stamps as c
    t = d.select(["apt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt", "LOBT_flt"]).rename(
        {"apt": "airport", "FLIGHT_mvt": "flight", "MVT_TIME_UTC_mvt": "takeoff", "AOBT_3_flt": "aobt3", "STAND_mvt": "stand", "RUNWAY_mvt": "runway", "LOBT_flt": "lobt"})
    t = t.with_columns(pl.col("takeoff").dt.replace_time_zone(None).cast(pl.Datetime("us")), pl.col("aobt3").dt.replace_time_zone(None).cast(pl.Datetime("us"))).with_row_index("row")
    f = c.default_stand_flag(t)
    t = t.with_columns(f.alias("dflt"))
    print(t.filter(pl.col("aobt3").is_not_null()).group_by("airport").agg(plan_rows=pl.len(), flagged=pl.col("dflt").sum(), share=pl.col("dflt").mean()).sort("airport"))


def daily(d):
    hdr("PORTAL DAILY ALL-CAUSE PRE-DEPARTURE DELAY VERSUS THE RECORDED STAMPS (2025)")
    f = Path("data/external/eurocontrol/all_pre_departure_delays_2025.csv")
    if not f.exists():
        print("published daily file not found:", f)
        return
    pub = pl.read_csv(f, infer_schema_length=20000).filter(pl.col("APT_ICAO").is_in(list(TZ))).select(
        ["FLT_DATE", "APT_ICAO", "FLT_DEP_1", "DLY_ALL_PRE_2"]).with_columns(pl.col("FLT_DATE").str.to_date())
    x = d.with_columns(delay=((pl.col("gap") - pl.col("y")) / 60.0).clip(lower_bound=0.0))  # (off-block minus schedule) / 60, floored at 0; grouped by the UTC date of the stamp
    res = []
    for variant, flt in [("all rows", pl.lit(True)), ("without VFR", pl.col("FLIGHT_RULE_mvt").fill_null("I") != "V"),
                         ("without VFR and without departures to the same airport", (pl.col("FLIGHT_RULE_mvt").fill_null("I") != "V") & (pl.col("ADES_mvt") != pl.col("ADEP_mvt")))]:
        rows = []
        for a in sorted(TZ):
            q = x.filter((pl.col("apt") == a) & flt).with_columns(
                dd=pl.col("BLOCK_TIME_UTC_mvt").dt.date())
            s = q.group_by("dd").agg(ours=pl.col("delay").sum()).rename({"dd": "FLT_DATE"})
            p = pub.filter(pl.col("APT_ICAO") == a).join(s, on="FLT_DATE", how="left").with_columns(pl.col("ours").fill_null(0.0))
            r = (p["DLY_ALL_PRE_2"] - p["ours"]).abs()
            slope = float(np.polyfit(p["ours"].to_numpy(), p["DLY_ALL_PRE_2"].to_numpy(), 1)[0])
            rows.append((a, p.height, round(float(r.median()), 1), round(float(r.quantile(0.9)), 1), round(float((p["DLY_ALL_PRE_2"] - p["ours"]).mean()), 1), round(slope, 3)))
        print(f"\n{variant}: airport, days, median |published - ours| (min), 90th percentile, mean (published - ours), slope")
        for r in rows:
            print("  ", r)
    print("\nRome: days with long-gap schedule copies (copy: |y - gap| <= 90 s, gap > 7,200 s)")
    r = x.filter(pl.col("apt") == "LIRF").with_columns(cp=((pl.col("y") - pl.col("gap")).abs() <= 90) & (pl.col("gap") > 7200),
                                                         dd=pl.col("BLOCK_TIME_UTC_mvt").dt.date())
    cdays = r.filter(pl.col("cp")).group_by("dd").agg(n=pl.len(), true_delay=(pl.col("gap") / 60.0).sum())
    s = r.group_by("dd").agg(ours=pl.col("delay").sum()).rename({"dd": "FLT_DATE"})
    p = pub.filter(pl.col("APT_ICAO") == "LIRF").join(s, on="FLT_DATE").join(cdays.rename({"dd": "FLT_DATE"}), on="FLT_DATE")
    print(f"  copy days {p.height}, copies {int(p['n'].sum())}, mean true gap of the copies {float(p['true_delay'].sum() / p['n'].sum()):.0f} min")
    print(f"  published minus the stamp-based sum: mean {float((p['DLY_ALL_PRE_2'] - p['ours']).mean()):.1f} min/day;"
          f" published minus the sum if the copies had been real late departures: {float((p['DLY_ALL_PRE_2'] - p['ours'] - p['true_delay']).mean()):.1f} min/day")


def summary(d):
    hdr("SUMMARY TABLE: rows and share of squared deviation by defect (squared deviation of the label from its airport-month median)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_offblock_stamps as c
    t = d.select(["apt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "LOBT_flt", "AOBT_3_flt", "RUNWAY_mvt",
                  "STAND_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "ADES_mvt", "EOBT_1_flt"]).rename(
        {"apt": "airport", "FLIGHT_mvt": "flight", "MVT_TIME_UTC_mvt": "takeoff", "SCHED_TIME_UTC_mvt": "sched", "BLOCK_TIME_UTC_mvt": "offblock",
         "LOBT_flt": "lobt", "AOBT_3_flt": "aobt3", "RUNWAY_mvt": "runway", "STAND_mvt": "stand", "AIRCRAFT_TYPE_mvt": "actype",
         "FLIGHT_RULE_mvt": "rule"})
    for k in ("takeoff", "sched", "offblock", "lobt", "aobt3", "EOBT_1_flt"):
        t = t.with_columns(pl.col(k).dt.replace_time_zone(None).cast(pl.Datetime("us")))
    t = t.with_row_index("row")
    t = c.add_flags(t)
    t = t.with_columns(month=pl.col("takeoff").dt.month(), ap=pl.col("airport"))
    t = t.with_columns(med=pl.col("label").median().over(["airport", "month"])).with_columns(sq=(pl.col("label") - pl.col("med")) ** 2)
    tot_all = t["sq"].sum()
    tot_ap = {r["airport"]: r["sq"] for r in t.group_by("airport").agg(pl.col("sq").sum()).iter_rows(named=True)}
    n_ap = {r["airport"]: r["n"] for r in t.group_by("airport").agg(n=pl.len()).iter_rows(named=True)}
    p = pl.col("lobt").is_not_null()
    L = pl.col("gl") - W
    t = t.with_columns(refiled=(p & (L > 2800) & ((L - (pl.col("takeoff") - pl.col("aobt3")).dt.total_milliseconds() / 1000.0) > 1800) & (pl.col("airport") == "LIRF")),
                       ncen=(~p & (pl.col("airport") == "LTFM") & (pl.col("ADES_mvt") != "LCEN")),
                       hg=((pl.col("airport") == "EHAM") & pl.col("stand").str.starts_with("HG").fill_null(False)),
                       runh=((pl.col("airport") == "LSZH") & (pl.col("runway") == "H")))
    pops = [
        ("schedule copy, Rome, no plan", (pl.col("airport") == "LIRF") & pl.col("schedule_copy") & ~p),
        ("schedule copy, Rome, with plan", (pl.col("airport") == "LIRF") & pl.col("schedule_copy") & p),
        ("schedule copy flag, Barcelona", (pl.col("airport") == "LEBL") & pl.col("schedule_copy")),
        ("day-shifted stamp (all airports)", pl.col("day_shifted")),
        ("stamp at the plan clock, re-filed Rome flights (the family of 38)", pl.col("refiled")),
        ("duplicate movement records, Paris", (pl.col("airport") == "LFPG") & pl.col("duplicate_movement")),
        ("late stamp, label < 60 s, Amsterdam / Zurich / Paris / Barcelona", pl.col("airport").is_in(["EHAM", "LSZH", "LFPG", "LEBL"]) & pl.col("near_zero_label")),
        ("hold-like record, Heathrow / Istanbul / Paris / Rome", pl.col("airport").is_in(["EGLL", "LTFM", "LFPG", "LIRF"]) & pl.col("hold_like")),
        ("no-plan Istanbul flights other than to Ercan", pl.col("ncen")),
        ("helicopter and runway-H movements (Zurich H, Amsterdam HG stands)", pl.col("runh") | pl.col("hg")),
        ("network-manager off-block equal to a stand default (plan rows)", pl.col("nm_default_offblock")),
        ("any record defect (union of the seven record flags)", pl.col("any_record_defect")),
        ("label above 7,200 s (any cause)", pl.col("label") > 7200),
    ]
    print(f"{'population':<72}{'rows':>9}{'% of rows':>10}{'% of its airports sq.dev':>26}{'% of all sq.dev':>16}")
    for name, cond in pops:
        x = t.filter(cond)
        if x.height == 0:
            print(f"{name:<72}{0:>9}")
            continue
        aps = x["airport"].unique().to_list()
        rows_ap = sum(n_ap[a] for a in aps)
        sq_ap = sum(tot_ap[a] for a in aps)
        print(f"{name:<72}{x.height:>9d}{100 * x.height / rows_ap:>10.3f}{100 * x['sq'].sum() / sq_ap:>26.1f}{100 * x['sq'].sum() / tot_all:>16.1f}   [{','.join(sorted(aps))}]")
    # Rome on its own
    r = t.filter(pl.col("airport") == "LIRF")
    print(f"\nRome: schedule copies + day-shifted + refiled family carry {100 * r.filter(pl.col('schedule_copy') | pl.col('day_shifted') | pl.col('refiled'))['sq'].sum() / r['sq'].sum():.1f}% of Rome's squared deviation "
          f"({r.filter(pl.col('schedule_copy') | pl.col('day_shifted') | pl.col('refiled')).height} of {r.height} rows)")


SECTIONS = {"headline": headline, "copies": copies, "dayearly": dayearly, "window": window, "refiled": refiled,
            "dups": dups, "late": late, "hold": hold, "istanbul": istanbul, "heli": heli, "default": default, "daily": daily, "summary": summary}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--section", default="all", choices=["all"] + list(SECTIONS))
    a = ap.parse_args()
    d = load(a.data)
    print(f"departures loaded: {d.height}")
    for k, fn in SECTIONS.items():
        if a.section in ("all", k):
            fn(d)
