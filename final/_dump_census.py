"""Coverage census of the departures of the chosen months: why a departure is not matched to an ADS-B ground track (stage 20, part f).

Derived from archive/agent_coverage_01_census.py (2026 half only; the 2025 population and its error shares are not rebuilt):
same matcher gates, same classification, one row per departure with its category. Changes: the ranking file, the months and
the output file are arguments. Label-free. Run from the repo root.

    python final/_dump_census.py --ranking data/raw/final_ranking.parquet --months 2,6 --out data/final/J6_census.parquet
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import polars as pl

from archive.agent_noplan_adsb_match import (Gates, add_derived_callsign, flight_callsign_maps, match_departures,
                                              runs_and_tracks)

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--max-days", type=int, default=0, help="debug: truncate each day list")
ap.add_argument("--skip-2026", action="store_true")
ap.add_argument("--ranking", default="data/raw/final_ranking.parquet")
ap.add_argument("--months", default="2,6")
ap.add_argument("--days", default="")
ap.add_argument("--out", default="data/final/J6_census.parquet")
args = ap.parse_args()

# PRODUCTION gates = NP1 (v103, current leading matcher): SHIP's airborne_callsign default (True) + AIRCS
# (air_cs_all on, LIRF skipped) + noplan_callsign on (noplan_callsign_skip defaults to LIRF already).
G = Gates(air_cs_all=True, air_cs_skip=("LIRF",), noplan_callsign=True)
AIRPORTS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ---- flight_callsign_maps, built once exactly as production does (training + ranking plan rows) ----
cs_src = [pl.scan_parquet(src).filter(C("PHASE_mvt") == "DEP")
            .select(C("AOBT_3_flt").is_not_null().alias("plan"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight")).collect()
          for src in ("data/raw/training_*.parquet", args.ranking)]
CS_MAPS = flight_callsign_maps(cs_src)
log("flight_callsign_maps built")


def allruns_and_points_keys(day: str, points_dir: Path):
    """Duplicate of agent_noplan_adsb_match.runs_and_tracks's pre-liftoff-filter grouping, PLUS the raw
    per-airport callsign set (ground or air) -- both diagnostic-only, not fed back into any matching."""
    p = pl.read_parquet(points_dir / f"{day}.parquet").filter(C("icao").is_not_null()).sort(["apt", "icao", "t"])
    pts_cs = (p.filter(C("callsign").is_not_null())
                .select("apt", C("callsign").str.strip_chars().str.to_uppercase().alias("cs"))
                .unique())
    if G.slow_low_ground:
        from archive.agent_noplan_adsb_match import ELEVATION_FT
        elev = C("apt").replace_strict(ELEVATION_FT, default=None, return_dtype=pl.Float64)
        slow_low = ((C("alt") != -1) & ((C("alt").cast(pl.Float64) - elev).abs() <= G.slow_low_band_ft)
                    & (C("gs") <= G.slow_low_gs_kt)).fill_null(False)
        p = p.with_columns(((C("alt") == -1) | slow_low).alias("ground"))
    else:
        p = p.with_columns((C("alt") == -1).alias("ground"))
    p = p.with_columns(
        C("t").shift(1).over(["apt", "icao"]).alias("prev_t"),
        C("ground").shift(1).over(["apt", "icao"]).alias("prev_ground"),
        C("callsign").forward_fill().over(["apt", "icao"]).alias("cs_ff"),
        C("actype").forward_fill().over(["apt", "icao"]).alias("type_ff"),
    )
    gr = p.filter(C("ground")).with_columns(
        (C("prev_t").is_null() | ~C("prev_ground").fill_null(False) | ((C("t") - C("prev_t")) > G.gap_s))
        .alias("run_start"))
    gr = gr.with_columns(C("run_start").cast(pl.Int32).cum_sum().over(["apt", "icao"]).alias("run"))
    allruns = (gr.group_by(["apt", "icao", "run"])
                 .agg(C("cs_ff").drop_nulls().last().alias("callsign"),
                      C("cs_ff").drop_nulls().first().alias("callsign_first"),
                      C("type_ff").drop_nulls().last().alias("adsb_type"))
                 .with_columns(C("callsign").str.strip_chars().str.to_uppercase(),
                               C("callsign_first").str.strip_chars().str.to_uppercase()))
    allruns_keys = (pl.concat([allruns.select("apt", C("callsign").alias("k")),
                               allruns.select("apt", C("callsign_first").alias("k"))], how="vertical_relaxed")
                    .filter(C("k").is_not_null()).unique())
    return allruns_keys, pts_cs


def runs_keys_with_t(runs: pl.DataFrame):
    """(apt, key, liftoff_t) for every key a run in `runs` (post-liftoff-filter, already carries
    callsign_air / cs_all per G) would match on -- ground callsign, airborne callsign, and every cs_all entry."""
    parts = [runs.select("apt", C("callsign").alias("key"), "liftoff_t")]
    if "callsign_air" in runs.columns:
        parts.append(runs.filter(C("callsign_air").is_not_null())
                         .select("apt", C("callsign_air").alias("key"), "liftoff_t"))
    if "cs_all" in runs.columns:
        parts.append(runs.filter(C("cs_all").is_not_null()).explode("cs_all")
                         .select("apt", C("cs_all").alias("key"), "liftoff_t"))
    return pl.concat(parts, how="vertical_relaxed").filter(C("key").is_not_null())


def runs_types_with_t(runs: pl.DataFrame):
    return runs.select("apt", "adsb_type", "liftoff_t").filter(C("adsb_type").is_not_null())


def classify_day(deps_day: pl.DataFrame, day: str, points_dir: Path, match_s_default: float):
    """Returns a DataFrame MVT_ID_mvt, apt, plan, category for this day's rows."""
    runs, touch, tracks = runs_and_tracks(day, G, points_dir=points_dir)
    allruns_keys, pts_cs = allruns_and_points_keys(day, points_dir)
    cand, nc, both = match_departures(deps_day, runs, G)
    matched_ids = both.select("MVT_ID_mvt").unique()
    unmatched = deps_day.join(matched_ids, on="MVT_ID_mvt", how="anti")
    if unmatched.height == 0:
        return both.select("MVT_ID_mvt", "apt", "plan").with_columns(pl.lit("matched").alias("category"))

    rkeys = runs_keys_with_t(runs)
    rtypes = runs_types_with_t(runs)

    def key_cascade(rows: pl.DataFrame, key_col: str, window: float) -> pl.DataFrame:
        """rows must have columns apt, key_col, mvt_t. Returns rows + 'category' for the callsign-keyed path."""
        r = rows.filter(C(key_col).is_not_null())
        none_key = rows.filter(C(key_col).is_null())
        if r.height == 0:
            return none_key.with_columns(pl.lit(None, dtype=pl.Utf8).alias("category")) if none_key.height else r.with_columns(pl.lit(None, dtype=pl.Utf8).alias("category"))
        # in-window candidate?
        cwin = (r.join(rkeys, left_on=["apt", key_col], right_on=["apt", "key"], how="left")
                  .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dm")))
        best = (cwin.group_by("MVT_ID_mvt").agg(C("dm").min().alias("best_dm")))
        r2 = r.join(best, on="MVT_ID_mvt", how="left")
        in_pts = r2.join(pts_cs, left_on=["apt", key_col], right_on=["apt", "cs"], how="semi")
        in_allruns = r2.join(allruns_keys, left_on=["apt", key_col], right_on=["apt", "k"], how="semi")
        # best_dm not null: some run in 'runs' (post lift-off-filter) DOES carry this key, at some time --
        # the row is unmatched only because that candidate sits outside the match window, or lost a
        # uniqueness collision to a competing departure claiming the same run. Combined bucket (see docstring).
        cat = (r2.with_columns(
            pl.when(C("best_dm").is_not_null()).then(pl.lit("key_time_mismatch"))
              .when(C("MVT_ID_mvt").is_in(in_allruns["MVT_ID_mvt"])).then(pl.lit("points_no_liftoff"))
              .when(C("MVT_ID_mvt").is_in(in_pts["MVT_ID_mvt"])).then(pl.lit("points_no_liftoff"))
              .otherwise(pl.lit("no_adsb_points")).alias("category")))
        out = cat.select("MVT_ID_mvt", "apt", "plan", "category")
        if none_key.height:
            out = pl.concat([out, none_key.select("MVT_ID_mvt", "apt", "plan").with_columns(pl.lit(None, dtype=pl.Utf8).alias("category"))], how="vertical_relaxed")
        return out

    up = unmatched.filter(C("plan"))
    un = unmatched.filter(~C("plan"))
    out_parts = []
    if up.height:
        out_parts.append(key_cascade(up.select("MVT_ID_mvt", "apt", "plan", "callsign", "mvt_t"), "callsign", G.match_s))

    if un.height:
        no_key = un.filter(C("cs2").is_null() & C("type").is_null())
        if no_key.height:
            out_parts.append(no_key.select("MVT_ID_mvt", "apt", "plan").with_columns(pl.lit("no_plan_no_key").alias("category")))
        rest = un.join(no_key.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
        has_cs2 = rest.filter(C("cs2").is_not_null())
        no_cs2 = rest.filter(C("cs2").is_null())
        if has_cs2.height:
            cs2_cls = key_cascade(has_cs2.select("MVT_ID_mvt", "apt", "plan", C("cs2").alias("callsign2"), "mvt_t"),
                                   "callsign2", G.match_s)
            if G.noplan_callsign_skip:
                skip_apts = set(G.noplan_callsign_skip)
                cs2_cls = cs2_cls.with_columns(
                    pl.when(C("apt").is_in(list(skip_apts))).then(pl.lit("airport_skip")).otherwise(C("category")).alias("category"))
            out_parts.append(cs2_cls)
        if no_cs2.height:
            typed = no_cs2.filter(C("type").is_not_null())
            if typed.height:
                twin = (typed.join(rtypes, left_on="apt", right_on="apt", how="left")
                             .filter(C("adsb_type") == C("type"))
                             .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dm")))
                best_t = twin.group_by("MVT_ID_mvt").agg(C("dm").min().alias("best_dm"))
                t2 = typed.join(best_t, on="MVT_ID_mvt", how="left")
                # best_dm not null: a run of the SAME type exists (at some time) -- unmatched only because
                # it sits outside noplan_match_s or lost a collision (combined bucket, as above). best_dm
                # null: no run of this type at all -- a genuine type disagreement, not a timing issue.
                cat_t = t2.with_columns(
                    pl.when(C("best_dm").is_not_null()).then(pl.lit("key_time_mismatch"))
                      .otherwise(pl.lit("type_mismatch")).alias("category"))
                out_parts.append(cat_t.select("MVT_ID_mvt", "apt", "plan", "category"))
    result = pl.concat(out_parts, how="vertical_relaxed") if out_parts else unmatched.select("MVT_ID_mvt", "apt", "plan").with_columns(pl.lit("unclassified").alias("category"))
    result = result.with_columns(C("category").fill_null("unclassified"))
    all_out = pl.concat([both.select("MVT_ID_mvt", "apt", "plan").with_columns(pl.lit("matched").alias("category")), result], how="vertical_relaxed")
    return all_out


C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)
if not args.skip_2026:
    have2026 = sorted(p.stem for p in Path("data/adsb/points").glob("2026-*.parquet")
                       if int(p.stem[5:7]) in [int(x) for x in args.months.split(",")])
    if args.days:
        have2026 = [d for d in have2026 if d in args.days.split(',')]
    if args.max_days:
        have2026 = have2026[: args.max_days]
    log(f"2026: {len(have2026)} Jan/Jul days with points")
    dep26 = (pl.scan_parquet(args.ranking).filter(C("PHASE_mvt") == "DEP")
             .select(C("MVT_ID_mvt"), C("ADEP_mvt").alias("apt"), C_DAY.alias("day"),
                     C("AOBT_3_flt").is_not_null().alias("plan"),
                     C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                     C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
                     C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
                     C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"))
             .filter(C("day").is_in(have2026)).collect())
    dep26 = add_derived_callsign(dep26, CS_MAPS)
    log(f"2026 universe: {dep26.height:,} departures")
    all_2026 = []
    for i, day in enumerate(have2026):
        dd = dep26.filter(C("day") == day)
        if dd.height == 0:
            continue
        r = classify_day(dd, day, Path("data/adsb/points"), G.match_s).with_columns(pl.lit(day).alias("day"))
        all_2026.append(r)
        if (i + 1) % 10 == 0:
            log(f"2026 classified {i + 1}/{len(have2026)} days")
    cls26 = pl.concat(all_2026, how="vertical_relaxed")
    cls26.write_parquet(args.out)
    log(f"2026 done: {cls26.height:,} rows classified")
    n26 = cls26.height
