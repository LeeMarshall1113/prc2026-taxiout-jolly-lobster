#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# check_offblock_stamps.py: flag departures whose recorded off-block stamp is not a pushback.
# Copyright (C) 2026 the authors of prc-taxiout-2026
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details
# (https://www.gnu.org/licenses/).
"""Flag departures whose recorded off-block stamp is not a pushback, and summarise by airport and month.

    python tools/check_offblock_stamps.py movements.parquet [more.parquet ...]   # prints the summary, writes a flag table
    python tools/check_offblock_stamps.py movements.csv --out flags.csv --summary summary.csv
    python tools/check_offblock_stamps.py --selftest

Self-contained: needs polars and numpy only (no other file of this repository, no model, no external data).
Runs in well under a minute and under 2 GB on one month of the challenge's training files (about 380,000 rows, of
which about 190,000 are departures).

INPUT. One row per movement, with the challenge's column names (a different naming is mapped with
``--map key=COLUMN``, repeatable; keys are in the first column below).

  key       default column          needed for
  --------  ----------------------  ----------------------------------------------------------------------
  airport   ADEP_mvt                REQUIRED  departure airport (summaries are per airport and month)
  flight    FLIGHT_mvt              REQUIRED  flight string (duplicate movements)
  takeoff   MVT_TIME_UTC_mvt        REQUIRED  recorded take-off time
  sched     SCHED_TIME_UTC_mvt      REQUIRED  scheduled off-block time
  offblock  BLOCK_TIME_UTC_mvt      optional  recorded off-block time. Without it the label-based flags stay empty
                                              (label = take-off minus off-block, in seconds)
  lobt      LOBT_flt                optional  last planned off-block time of the flight plan the movement is
                                              attached to (null = no flight plan). Without it every row counts as
                                              "no plan" and the plan-window flags stay empty
  aobt3     AOBT_3_flt              optional  the network manager's own off-block field (stand-default flag)
  runway    RUNWAY_mvt              optional  runway code (helicopter flag: code H or HEL)
  stand     STAND_mvt               optional  stand (stand-default flag)
  actype    AIRCRAFT_TYPE_mvt       optional  ICAO type (helicopter flag)
  rule      FLIGHT_RULE_mvt         optional  flight rule, I or V
  phase     PHASE_mvt               optional  if present, only rows equal to DEP are kept

Times may be timezone-aware or naive and are read as UTC. A parquet or CSV file is read by extension.

FLAGS (one boolean column each; all use seconds). gap = take-off minus schedule; y = take-off minus off-block
(the label); gl = take-off minus lobt.

  Defects of the record (summarised as "record defects"):
  schedule_copy         y present, gap >= --copy-min-gap (1,800 s) and |y - gap| <= 90 s: the off-block equals the
                        schedule. Real at Rome and (rarely) Barcelona; elsewhere a coincidence of a smooth distribution.
  day_shifted           y present, 82,800 <= y <= 95,000: the stamp is dated about one day early.
  duplicate_movement    another movement with the same airport and flight string has a take-off within 1,800 s.
                        Needs no label. Both records of a pair are flagged.
  near_zero_label       y present and y < 60 s, not a helicopter movement: a late stamp, made close to take-off.
  hold_like             y >= 3,600 s, the stamp is 10 min before to 40 min after the schedule, not a schedule copy and
                        not day-shifted: stamp near the schedule, then a wait of hours.
  heli_or_runway_h      runway H / HEL, or a helicopter ICAO type: take-off is the off-block, label near zero.
  label_outside_window  plan present and y present and |y - gl| > 3,606 s: breaks the plan-clock window (the
                        database attaches a movement to its plan only when the off-block is within an hour of the
                        plan's last off-block). Should never fire on the challenge's data; it fires on a join error.

  Input-only conditions (no label needed):
  copy_impossible_plan  plan present and |sched - lobt| > 3,606 s: a copy of the schedule would not have been joined
                        to the plan, so on this row the off-block cannot be a schedule copy.
  nm_default_offblock   aobt3 equals take-off minus a per-stand default, rounded to the minute (see
                        ``default_stand_flag``): the network manager's off-block is a default, not an observation.
  no_plan               lobt is null: flight plan missing.
  vfr                   flight rule V (the portal's delay statistics are easiest to reproduce without these).

SUMMARY. Per airport and month: rows, rate of each flag, and for each flag the share of the airport-month's squared
deviation from its median label that the flagged rows carry (rows with a label only), plus the same for "any record
defect". The deviation is from the airport-month median, not from a model: it measures how much of the spread of the
recorded stamp the flagged rows own, before any modelling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_COLS = {
    "airport": "ADEP_mvt", "flight": "FLIGHT_mvt", "takeoff": "MVT_TIME_UTC_mvt", "sched": "SCHED_TIME_UTC_mvt",
    "offblock": "BLOCK_TIME_UTC_mvt", "lobt": "LOBT_flt", "aobt3": "AOBT_3_flt", "runway": "RUNWAY_mvt",
    "stand": "STAND_mvt", "actype": "AIRCRAFT_TYPE_mvt", "rule": "FLIGHT_RULE_mvt", "phase": "PHASE_mvt",
}
REQUIRED = ("airport", "flight", "takeoff", "sched")

WINDOW_S = 3606          # plan-clock window: observed maximum 3,600 s plus 6 s of slack
COPY_TOL_S = 90          # a copy is within a minute and a half of the schedule
DAY_LO, DAY_HI = 82_800, 95_000
DUP_S = 1_800
NEAR_ZERO_S = 60
HOLD_MIN_Y = 3_600
HOLD_BEFORE_S, HOLD_AFTER_S = 600, 2_400   # stamp from 10 min before to 40 min after the schedule
DEFAULT_BOX_S = 35       # stand-default box: +-35 s around the key's mode, on a 10 s grid
DEFAULT_MIN_ROWS = 100
DEFAULT_MIN_SHARE = 0.30
HELI_RUNWAYS = {"H", "HEL"}
HELI_TYPES = {  # ICAO helicopter types seen at the ten airports and the common others; extend as needed
    "A109", "A119", "A139", "A169", "AS32", "AS35", "AS50", "AS55", "AS65", "B06", "B105", "B407", "B412", "B427",
    "B429", "B505", "EC20", "EC25", "EC30", "EC35", "EC45", "EC55", "EC75", "EN28", "H125", "H135", "H145", "H160",
    "H500", "MD52", "MD90", "R22", "R44", "R66", "S61", "S76", "S92", "AW09", "AW19", "AW39", "AW69", "AW89", "KA32",
}

RECORD_FLAGS = ["schedule_copy", "day_shifted", "duplicate_movement", "near_zero_label", "hold_like",
                "heli_or_runway_h", "label_outside_window"]
CONDITION_FLAGS = ["copy_impossible_plan", "nm_default_offblock", "no_plan", "vfr"]
ALL_FLAGS = RECORD_FLAGS + CONDITION_FLAGS


# ----------------------------------------------------------------------------------------------------------------
# reading and normalising
# ----------------------------------------------------------------------------------------------------------------
def _to_utc_naive(s: pl.Series) -> pl.Series:
    """Datetime (aware or naive) or ISO string -> naive UTC microsecond datetime."""
    if s.dtype == pl.String:
        s = s.str.to_datetime(strict=False, time_zone="UTC")
    if isinstance(s.dtype, pl.Datetime):
        if s.dtype.time_zone is not None:
            s = s.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        return s.cast(pl.Datetime("us"))
    raise TypeError(f"column {s.name!r} is not a time column ({s.dtype})")


def read_table(path: str | Path) -> pl.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pl.read_parquet(p)
    if p.suffix.lower() in (".csv", ".txt"):
        return pl.read_csv(p, infer_schema_length=10_000, try_parse_dates=True)
    raise ValueError(f"unsupported input type {p.suffix!r}: use .parquet or .csv")


def normalise(df: pl.DataFrame, cols: dict[str, str]) -> pl.DataFrame:
    """Keep the mapped columns under neutral names; departures only when a phase column exists."""
    missing = [k for k in REQUIRED if cols[k] not in df.columns]
    if missing:
        raise SystemExit("required column(s) missing: " + ", ".join(f"{k} ({cols[k]})" for k in missing))
    if cols["phase"] in df.columns:
        df = df.filter(pl.col(cols["phase"]) == "DEP")
    out = {}
    for k, c in cols.items():
        if k == "phase" or c not in df.columns:
            continue
        s = df[c]
        out[k] = _to_utc_naive(s) if k in ("takeoff", "sched", "offblock", "lobt", "aobt3") else s
    t = pl.DataFrame(out).with_row_index("row")
    if "MVT_ID_mvt" in df.columns:
        t = t.with_columns(df["MVT_ID_mvt"].alias("movement_id"))
    return t


def _secs(a: str, b: str) -> pl.Expr:
    """Seconds from column b to column a (a - b)."""
    return (pl.col(a) - pl.col(b)).dt.total_milliseconds().cast(pl.Float64) / 1000.0


# ----------------------------------------------------------------------------------------------------------------
# detectors
# ----------------------------------------------------------------------------------------------------------------
def default_stand_flag(t: pl.DataFrame, box_s: int = DEFAULT_BOX_S, min_rows: int = DEFAULT_MIN_ROWS,
                       min_share: float = DEFAULT_MIN_SHARE) -> pl.Series:
    """True where aobt3 looks like take-off minus a per-stand default taxi time rounded to the minute.

    d = take-off minus aobt3 (seconds). For a key (airport, stand, runway), falling back to (airport, stand) and then
    (airport, runway), take the mode of d on a 10 s grid. The key is a *default key* when it has at least ``min_rows``
    rows and at least ``min_share`` of them lie within +-``box_s`` seconds of the mode. A row is flagged when its key
    is a default key and d lies inside the box. The most specific key with at least ``min_rows`` rows decides a row;
    rows whose keys are all smaller are never flagged. Label-free; the modes are learned
    on the table given, so a very small table has few qualifying keys. False positives: a stand with a genuinely
    peaked real taxi time (dense stand peaks at large hubs); the flag measures "looks defaulted", not "is defaulted".
    """
    n = t.height
    if "aobt3" not in t.columns or "stand" not in t.columns:
        return pl.Series("nm_default_offblock", np.zeros(n, dtype=bool))
    keysets = [["airport", "stand", "runway"], ["airport", "stand"], ["airport", "runway"]]
    keysets = [[k for k in ks] for ks in keysets if all(k in t.columns for k in ks)]
    base = t.select(["row", "airport"] + [c for c in ("stand", "runway") if c in t.columns]).with_columns(
        d=((t["takeoff"] - t["aobt3"]).dt.total_milliseconds().cast(pl.Float64) / 1000.0))
    base = base.filter(pl.col("d").is_not_null() & pl.col("stand").is_not_null())
    base = base.with_columns(g=((pl.col("d") / 10.0).round() * 10.0))
    flagged = np.zeros(n, dtype=bool)
    decided = np.zeros(n, dtype=bool)
    for ks in keysets:
        b = base.drop_nulls(ks)
        cnt = b.group_by(ks + ["g"]).len().sort(["len", "g"], descending=[True, False])
        mode = cnt.group_by(ks, maintain_order=True).first().rename({"g": "mode"}).select(ks + ["mode"])
        b = b.join(mode, on=ks, how="left").with_columns(inbox=((pl.col("d") - pl.col("mode")).abs() <= box_s))
        stats = b.group_by(ks).agg(n=pl.len(), share=pl.col("inbox").mean())
        big = stats.filter(pl.col("n") >= min_rows).with_columns(is_default=(pl.col("share") >= min_share)).select(ks + ["is_default"])
        b = b.join(big, on=ks, how="inner")        # rows of keys with enough rows; the others fall through to a coarser key
        rows = b.filter(pl.col("inbox") & pl.col("is_default"))["row"].to_numpy()
        allr = b["row"].to_numpy()
        take = ~decided[rows]
        flagged[rows[take]] = True
        decided[allr] = True                       # a key with enough rows decides its rows, in the box or out of it
    return pl.Series("nm_default_offblock", flagged)


def duplicate_flag(t: pl.DataFrame, within_s: int = DUP_S) -> pl.Series:
    """True where another movement of the same airport and flight string has a take-off within ``within_s`` seconds."""
    n = t.height
    s = (t.select(["row", "airport", "flight", "takeoff"]).drop_nulls(["flight", "takeoff"])
         .with_columns(ts=pl.col("takeoff").dt.epoch("s")).sort(["airport", "flight", "ts"]))
    g = s.with_columns(
        prev=(pl.col("ts") - pl.col("ts").shift(1)).over(["airport", "flight"]),
        nxt=(pl.col("ts").shift(-1) - pl.col("ts")).over(["airport", "flight"]),
    )
    hit = g.filter((pl.col("prev") <= within_s) | (pl.col("nxt") <= within_s))["row"].to_numpy()
    out = np.zeros(n, dtype=bool)
    out[hit] = True
    return pl.Series("duplicate_movement", out)


def add_flags(t: pl.DataFrame, copy_min_gap: int = 1_800) -> pl.DataFrame:
    has_label = "offblock" in t.columns
    has_plan = "lobt" in t.columns
    n = t.height
    cols = {"gap": _secs("takeoff", "sched")}
    if has_label:
        cols["label"] = _secs("takeoff", "offblock")
    if has_plan:
        cols["gl"] = _secs("takeoff", "lobt")
        cols["sl"] = _secs("sched", "lobt")
    t = t.with_columns(**cols)
    F = pl.col
    plan = F("lobt").is_not_null() if has_plan else pl.lit(False)
    none = pl.lit(False)

    e: dict[str, pl.Expr] = {}
    if has_label:
        y, gap = F("label"), F("gap")
        e["schedule_copy"] = (y.is_not_null() & (gap >= copy_min_gap) & ((y - gap).abs() <= COPY_TOL_S)).fill_null(False)
        e["day_shifted"] = (y.is_not_null() & (y >= DAY_LO) & (y <= DAY_HI)).fill_null(False)
        heli = _heli_expr(t)
        e["near_zero_label"] = (y.is_not_null() & (y < NEAR_ZERO_S) & ~heli).fill_null(False)
        stamp_vs_sched = gap - y                                      # off-block minus schedule, seconds
        e["hold_like"] = (y.is_not_null() & (y >= HOLD_MIN_Y) & (stamp_vs_sched >= -HOLD_BEFORE_S)
                          & (stamp_vs_sched <= HOLD_AFTER_S) & ((y - gap).abs() > COPY_TOL_S)
                          & ~((y >= DAY_LO) & (y <= DAY_HI))).fill_null(False)
        e["label_outside_window"] = (plan & y.is_not_null() & ((y - F("gl")).abs() > WINDOW_S)).fill_null(False) if has_plan else none
    else:
        for k in ("schedule_copy", "day_shifted", "near_zero_label", "hold_like", "label_outside_window"):
            e[k] = none
    e["heli_or_runway_h"] = _heli_expr(t)
    e["copy_impossible_plan"] = (plan & (F("sl").abs() > WINDOW_S)).fill_null(False) if has_plan else none
    e["no_plan"] = ~plan if has_plan else pl.lit(True)
    e["vfr"] = (F("rule") == "V").fill_null(False) if "rule" in t.columns else none
    t = t.with_columns(**e)
    t = t.with_columns(duplicate_flag(t), default_stand_flag(t))
    t = t.with_columns(any_record_defect=pl.any_horizontal(*[pl.col(c) for c in RECORD_FLAGS]))
    return t


def _heli_expr(t: pl.DataFrame) -> pl.Expr:
    parts = []
    if "runway" in t.columns:
        parts.append(pl.col("runway").str.to_uppercase().is_in(sorted(HELI_RUNWAYS)).fill_null(False))
    if "actype" in t.columns:
        parts.append(pl.col("actype").is_in(sorted(HELI_TYPES)).fill_null(False))
    return pl.any_horizontal(*parts) if parts else pl.lit(False)


# ----------------------------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------------------------
def summarise(t: pl.DataFrame) -> pl.DataFrame:
    """Long table: airport, month, flag, n_rows, n_flagged, rate, n_labelled, sqdev_share."""
    t = t.with_columns(month=pl.col("takeoff").dt.strftime("%Y-%m"))
    flags = ALL_FLAGS + ["any_record_defect"]
    if "label" in t.columns:
        med = t.group_by(["airport", "month"]).agg(med=pl.col("label").median())
        t = t.join(med, on=["airport", "month"], how="left").with_columns(sq=(pl.col("label") - pl.col("med")) ** 2)
    else:
        t = t.with_columns(sq=pl.lit(None, dtype=pl.Float64))
    parts = []
    base = t.group_by(["airport", "month"]).agg(n_rows=pl.len(), n_lab=pl.col("sq").is_not_null().sum(),
                                                 tot_sq=pl.col("sq").sum())
    for f in flags:
        a = t.group_by(["airport", "month"]).agg(n_flagged=pl.col(f).sum(),
                                                  flag_sq=pl.when(pl.col(f)).then(pl.col("sq")).otherwise(0.0).sum())
        a = base.join(a, on=["airport", "month"]).with_columns(
            flag=pl.lit(f), rate=pl.col("n_flagged") / pl.col("n_rows"),
            sqdev_share=pl.when(pl.col("tot_sq") > 0).then(pl.col("flag_sq") / pl.col("tot_sq")).otherwise(None))
        parts.append(a.select(["airport", "month", "flag", "n_rows", "n_flagged", "rate", "n_lab", "sqdev_share", "flag_sq", "tot_sq"]))
    return pl.concat(parts).sort(["airport", "month", "flag"])


def copy_spike(t: pl.DataFrame) -> str:
    """Evidence that ``schedule_copy`` is real at an airport and not a coincidence of a smooth distribution.

    For plan rows with a schedule gap of 1,800-3,700 s, count the rows whose off-block is within +-30 s of the schedule
    and compare with the mean count in the ten neighbouring minutes (-5..-1 and +1..+5 minutes). A ratio near 1 means the
    flagged rows are what a smooth distribution puts there anyway; a copy habit shows as a ratio well above 1
    (Rome 13.7 and Barcelona 5.8 on the 2025 training year; the other eight airports 0.6 to 1.1).
    """
    if "label" not in t.columns or "lobt" not in t.columns:
        return "schedule-copy evidence: needs offblock and lobt columns"
    x = t.filter(pl.col("lobt").is_not_null() & (pl.col("gap") >= 1800) & (pl.col("gap") < 3700) & pl.col("label").is_not_null())
    lines = ["schedule-copy evidence (plan rows, gap 1,800-3,700 s): rows within +-30 s of the schedule / mean of the 10 flanking minutes (ratio)"]
    for a in sorted(x["airport"].unique().to_list()):
        o = (x.filter(pl.col("airport") == a)["gap"] - x.filter(pl.col("airport") == a)["label"]).to_numpy()  # stamp minus schedule
        z = int((np.abs(o) <= 30).sum())
        fl = float(np.mean([((o >= m * 60 - 30) & (o < m * 60 + 30)).sum() for m in list(range(-5, 0)) + list(range(1, 6))]))
        lines.append(f"  {a}: {z} / {fl:.0f} = {z / fl:.1f}" if fl > 0 else f"  {a}: {z} / 0")
    return "\n".join(lines)


def format_summary(s: pl.DataFrame, max_months: int = 3) -> str:
    """One block per airport; flags down the side; for each month: rows flagged, rate (% of rows), share of squared
    deviation (% of the airport-month's total). With more than ``max_months`` months the months are pooled."""
    flags = ALL_FLAGS + ["any_record_defect"]
    months = sorted(s["month"].unique().to_list())
    pooled = len(months) > max_months
    if pooled:
        s = (s.group_by(["airport", "flag"]).agg(n_rows=pl.col("n_rows").sum(), n_flagged=pl.col("n_flagged").sum(),
                                                  flag_sq=pl.col("flag_sq").sum(), tot_sq=pl.col("tot_sq").sum())
             .with_columns(month=pl.lit("all months"), rate=pl.col("n_flagged") / pl.col("n_rows"),
                           sqdev_share=pl.when(pl.col("tot_sq") > 0).then(pl.col("flag_sq") / pl.col("tot_sq")).otherwise(None)))
        months = ["all months"]
    d = {}
    for r in s.iter_rows(named=True):
        d[(r["airport"], r["month"], r["flag"])] = r
    lines = []
    for a in sorted(s["airport"].unique().to_list()):
        head = "  " + f"{'flag':<22}" + "".join(f"{m:>12}{'rate %':>8}{'sq-dev %':>9}" for m in months)
        lines.append(f"{a}   (rows: " + ", ".join(f"{m} {d[(a, m, flags[0])]['n_rows']}" for m in months) + ")")
        lines.append(head)
        for f in flags:
            cells = []
            for m in months:
                r = d[(a, m, f)]
                sh = "-" if r["sqdev_share"] is None else f"{100 * r['sqdev_share']:.1f}"
                cells.append(f"{r['n_flagged']:>12d}{100 * r['rate']:>8.2f}{sh:>9}")
            lines.append(f"  {f:<22}" + "".join(cells))
        lines.append("")
    lines.append("sq-dev % = share of the airport-month's squared deviation from its median label carried by the flagged rows "
                 "(rows with a label only); the flags overlap, so the columns do not add up.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# selftest
# ----------------------------------------------------------------------------------------------------------------
def _synthetic() -> tuple[pl.DataFrame, dict[str, int]]:
    """A tiny table with one named example of every defect among ordinary departures. Returns (table, flag -> row)."""
    from datetime import datetime, timedelta
    rng = np.random.default_rng(7)
    t0 = datetime(2025, 7, 10, 6, 0, 0)
    rows = []

    def add(ap, fl, sched_off, taxi, plan_off=0, has_plan=True, runway="25", stand="S1", actype="A320", rule="I",
            block_vs_sched=None, aobt_default=None, take_gap=None):
        sched = t0 + timedelta(seconds=sched_off)
        if block_vs_sched is None:
            block_vs_sched = int(rng.integers(-300, 900))
        block = sched + timedelta(seconds=block_vs_sched)
        take = block + timedelta(seconds=taxi) if take_gap is None else sched + timedelta(seconds=take_gap)
        if take_gap is not None:
            block = take - timedelta(seconds=taxi)
        lobt = sched + timedelta(seconds=plan_off) if has_plan else None
        aobt = (take - timedelta(seconds=aobt_default)).replace(second=0, microsecond=0) if (has_plan and aobt_default) else (
            block if has_plan else None)
        rows.append(dict(ADEP_mvt=ap, FLIGHT_mvt=fl, PHASE_mvt="DEP", MVT_TIME_UTC_mvt=take, BLOCK_TIME_UTC_mvt=block,
                         SCHED_TIME_UTC_mvt=sched, RUNWAY_mvt=runway, STAND_mvt=stand, AIRCRAFT_TYPE_mvt=actype,
                         FLIGHT_RULE_mvt=rule, LOBT_flt=lobt, AOBT_3_flt=aobt))
        return len(rows) - 1

    # ordinary departures: plan present, taxi 500-1,200 s, stand S1 with a defaulted network-manager field (default 600 s)
    for i in range(400):
        add("TEST", f"ORD{i:04d}", int(i * 90), int(rng.integers(500, 1200)), plan_off=int(rng.integers(-120, 120)),
            stand="S1", aobt_default=600)
    # stand S2: real network-manager field (no default)
    for i in range(150):
        add("TEST", f"REAL{i:04d}", int(40_000 + i * 90), int(rng.integers(500, 1200)), stand="S2")
    ex = {}
    ex["schedule_copy"] = add("TEST", "COPY1", 80_000, 9_000, block_vs_sched=0, has_plan=False)   # stamp = schedule, left 2.5 h later
    ex["day_shifted"] = add("TEST", "DAYE1", 90_000, 900, take_gap=1_000, has_plan=False)
    rows[ex["day_shifted"]]["BLOCK_TIME_UTC_mvt"] = rows[ex["day_shifted"]]["MVT_TIME_UTC_mvt"] - timedelta(seconds=87_300)
    a = add("TEST", "DUP1", 100_000, 900, has_plan=True)
    b = add("TEST", "DUP1", 100_000, 900, has_plan=False, take_gap=a * 0 + (rows[a]["MVT_TIME_UTC_mvt"] - rows[a]["SCHED_TIME_UTC_mvt"]).total_seconds() + 25)
    ex["duplicate_movement"] = a
    ex["near_zero_label"] = add("TEST", "LATE1", 110_000, 20)
    ex["hold_like"] = add("TEST", "HOLD1", 120_000, 9_000, block_vs_sched=300, has_plan=False)
    ex["heli_or_runway_h"] = add("TEST", "HELI1", 130_000, 5, runway="H", actype="EC35", has_plan=False)
    ex["label_outside_window"] = add("TEST", "WIN1", 140_000, 900, plan_off=9_000)   # plan clock 2.5 h after the stamp
    ex["copy_impossible_plan"] = add("TEST", "CIMP1", 150_000, 900, plan_off=5_400)
    ex["no_plan"] = add("TEST", "NOPL1", 160_000, 900, has_plan=False)
    ex["vfr"] = add("TEST", "VFR1", 170_000, 900, rule="V", has_plan=False)
    ex["nm_default_offblock"] = 0                                   # first ordinary row sits on stand S1
    return pl.DataFrame(rows), ex


def selftest() -> int:
    df, ex = _synthetic()
    t = normalise(df, DEFAULT_COLS)
    t = add_flags(t)
    ok = True
    for f, i in ex.items():
        v = bool(t[f][i])
        print(f"  {f:<22} row {i:>4}  flagged={v}")
        ok &= v
    # the ordinary row 400+ on stand S2 must not carry the defaults, and ordinary rows must not carry record defects
    rec_on_ordinary = int(t.filter(pl.col("row") < 400)["any_record_defect"].sum())
    print(f"  record defects among the 400 ordinary rows: {rec_on_ordinary} (expected 0)")
    real_default = int(t.filter((pl.col("row") >= 400) & (pl.col("row") < 550))["nm_default_offblock"].sum())
    print(f"  stand S2 rows flagged as defaulted: {real_default} (expected 0)")
    ok &= rec_on_ordinary == 0 and real_default == 0
    s = summarise(t)
    ok &= s.height == len(ALL_FLAGS) + 1
    assert ok, "selftest failed"
    print("selftest passed")
    return 0


# ----------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="*", help="one or more parquet or CSV files with one row per movement (several files are stacked; the duplicate check then spans them)")
    ap.add_argument("--out", help="per-flight flag table (.parquet or .csv); default <first input stem>.flags.parquet in the working directory")
    ap.add_argument("--summary", help="write the per-airport-month summary (long format) to this CSV")
    ap.add_argument("--map", action="append", default=[], metavar="KEY=COLUMN",
                    help="column mapping, e.g. --map airport=dep_airport (keys: " + ", ".join(DEFAULT_COLS) + ")")
    ap.add_argument("--copy-min-gap", type=int, default=1_800, help="smallest schedule gap for a schedule copy (s)")
    ap.add_argument("--airports", help="comma-separated airports to keep (default all)")
    ap.add_argument("--selftest", action="store_true", help="build a tiny synthetic table and assert every defect is flagged")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.input:
        ap.error("input file required (or --selftest)")
    cols = dict(DEFAULT_COLS)
    for m in a.map:
        k, _, v = m.partition("=")
        if k not in cols or not v:
            ap.error(f"bad --map {m!r}")
        cols[k] = v
    parts = [normalise(read_table(f), cols) for f in a.input]
    t = parts[0] if len(parts) == 1 else pl.concat(parts, how="diagonal_relaxed").drop("row").with_row_index("row")
    del parts
    if a.airports:
        t = t.filter(pl.col("airport").is_in(a.airports.split(","))).with_columns(row=pl.int_range(pl.len()))
    t = add_flags(t, a.copy_min_gap)
    s = summarise(t)
    print(format_summary(s))
    print("\n" + copy_spike(t))
    out = Path(a.out) if a.out else Path(Path(a.input[0]).stem + ".flags.parquet")
    keep = [c for c in ("movement_id", "airport", "flight", "takeoff", "sched", "offblock", "label", "gap", "gl")
            if c in t.columns] + ALL_FLAGS + ["any_record_defect"]
    o = t.select(keep)
    if out.suffix.lower() == ".csv":
        o.write_csv(out)
    else:
        o.write_parquet(out)
    if a.summary:
        s.write_csv(a.summary)
    print(f"\nwrote {out} ({o.height} rows)" + (f" and {a.summary}" if a.summary else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
