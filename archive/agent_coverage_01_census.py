"""experiment coverage 01: CENSUS of why scored departures are NOT ADS-B matched, tag "coverage" (2026-09-24).

docs/LAB-NOTES.md STEP 1. Matching-only, NO model fit -- reuses prc.adsb_match's shape (via the
existing air_cs_all/noplan_callsign copy, archive/agent_noplan_adsb_match.py) under PRODUCTION gates, i.e. the
same flags data/interim/modal/NP1.npz shipped with: --air-cs-all on --air-cs-skip LIRF --noplan-callsign on
(archive/agent_noplan_stage1_eval.py SHIP + AIRCS + noplan-callsign, the current v103/leading matcher).

For every scored departure NOT in match_departures()'s output, classify WHY, in priority order:
  no_plan_no_key    no-plan row: no derived callsign (cs2) AND no AIRCRAFT_TYPE_mvt -- neither match path can
                     ever fire (class e).
  no_adsb_points    the row's matching key (callsign, or cs2 for no-plan) never appears ANYWHERE in that
                     airport-day's raw ADS-B points at all (ground or air) (class a).
  points_no_liftoff the key appears in the raw points and forms at least one GROUND run that day (pre-filter
                     'allruns'), but no run bearing that key ever passes the lift-off-detection filter into
                     'runs' (class b).
  key_time_mismatch a lift-off run DOES exist bearing that key (in 'runs', after the airborne-callsign /
                     air_cs_all key expansion), but no candidate run sits within the match window (match_s for
                     plan/derived-callsign, noplan_match_s for time+type), or it lost a uniqueness collision to
                     a competing departure (classes c+d, combined and disclosed as such -- distinguishing them
                     needs re-deriving match_departures' own join, not done here).
  type_mismatch     no-plan only, cs2 path exhausted (null or itself unmatched): a lift-off run sits inside the
                     time window but no run's AIRCRAFT_TYPE_mvt agrees (adsb_type != type).
  airport_skip      informational sub-flag: LIRF no-plan rows lose the derived-callsign path to
                     noplan_callsign_skip regardless of the above (they still get a type+time chance).
  matched           in match_departures()'s output (cand/nc); not unmatched, listed for the rollup only.

Populations:
  2025 -- the SAME 238 scored days / 1,352,077 rows as data/interim/modal/NP1.npz (joined by MVT_ID_mvt for
          exact alignment), so category SSE shares are against the model NP1 actually shipped, not a re-derived
          population. plan/callsign/type/flight/mvt_t come from data/raw/training_*.parquet.
  2026 -- data/raw/ranking.parquet departures in Jan+Jul 2026 (the scored months) against data/adsb/points'
          full 2026-01-*/2026-07-* coverage (31+31 days, confirmed present). No y/SSE (unscored); row-count /
          share of departures only, per the task's own framing ("a class big in 2025 but small in 2026 matters
          less").

Read-only on prc/, data/raw, data/adsb/points, data/interim/modal/NP1.npz. Writes
data/interim/agent_coverage_census.parquet only. No model fit, no uploads, no commits, no S3/truthing, no
edits to prc/ or any existing archive/*.py.

    ./.venv/Scripts/python.exe archive/capped_run.py --cap-gb 3 -- \
        ./.venv/Scripts/python.exe archive/agent_coverage_01_census.py [--max-days N] [--skip-2026]
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
ap.add_argument("--out", default="data/interim/agent_coverage_census.parquet")
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
          for src in ("data/raw/training_*.parquet", "data/raw/ranking.parquet")]
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


# ---------------------------------------------------------------------------
# 2025: same 238 days / rows as data/interim/modal/NP1.npz
# ---------------------------------------------------------------------------
z = np.load("data/interim/modal/NP1.npz", allow_pickle=True)
np1_ids_df = pl.DataFrame({"MVT_ID_mvt": z["ids"].astype(np.float64)})
np1_days = sorted(set(z["days"].astype(str).tolist()))
if args.max_days:
    np1_days = np1_days[: args.max_days]
log(f"NP1.npz: {np1_ids_df.height:,} ids across {len(np1_days)} days")

C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)
dep25 = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
         .select(C("MVT_ID_mvt").cast(pl.Float64).alias("MVT_ID_mvt"), C("ADEP_mvt").alias("apt"),
                 C_DAY.alias("day"), C("AOBT_3_flt").is_not_null().alias("plan"),
                 C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                 C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
                 C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
                 C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"))
         .filter(C("day").is_in(np1_days))
         .collect()
         .join(np1_ids_df, on="MVT_ID_mvt", how="semi"))
dep25 = add_derived_callsign(dep25, CS_MAPS)
log(f"2025 universe: {dep25.height:,} of {np1_ids_df.height:,} NP1 ids resolved from training_*.parquet")

all_2025 = []
for i, day in enumerate(np1_days):
    dd = dep25.filter(C("day") == day)
    if dd.height == 0:
        continue
    r = classify_day(dd, day, Path("data/adsb/points"), G.match_s).with_columns(pl.lit(day).alias("day"))
    all_2025.append(r)
    if (i + 1) % 20 == 0:
        log(f"2025 classified {i + 1}/{len(np1_days)} days")
cls25 = pl.concat(all_2025, how="vertical_relaxed")
cls25.write_parquet("data/interim/agent_coverage_census_2025.parquet")
log(f"2025 done: {cls25.height:,} rows classified")

# join SSE from NP1.npz
mid = np.asarray([str(x) for x in z["ids"]])
sq = (z["both_guard"].astype(float) - z["y"].astype(float)) ** 2
sse_lut = dict(zip(mid, sq))
n_total = len(mid)
sse_total = float(sq.sum())
board_rmse = float(np.sqrt(sq.mean()))
c25 = cls25.with_columns(C("MVT_ID_mvt").cast(pl.Utf8).map_elements(lambda s: sse_lut.get(s, float("nan")), return_dtype=pl.Float64).alias("sse"))
rollup25 = (c25.group_by(["apt", "category"]).agg(pl.len().alias("n"), C("sse").sum().alias("sse_sum"))
              .with_columns((C("sse_sum") / sse_total).alias("share_of_total_sse"))
              .sort("sse_sum", descending=True))
rollup25.write_parquet("data/interim/agent_coverage_census_2025_rollup.parquet")
print("\n=== 2025 (238 NP1 days, n={:,}, total SSE {:.3e}, board RMSE {:.3f}) TOP 25 BY SSE ===".format(n_total, sse_total, board_rmse))
print(f"{'apt':<5}{'category':<20}{'n':>10}{'share_sse':>12}")
for r in rollup25.head(25).iter_rows(named=True):
    print(f"{r['apt']:<5}{r['category']:<20}{r['n']:>10,}{r['share_of_total_sse']:>12.2%}")
print("\n=== 2025 ROLLUP BY CATEGORY (all airports) ===")
byc = c25.group_by("category").agg(pl.len().alias("n"), C("sse").sum().alias("sse_sum")).with_columns((C("sse_sum") / sse_total).alias("share")).sort("share", descending=True)
for r in byc.iter_rows(named=True):
    print(f"  {r['category']:<20} n={r['n']:>9,}  share_of_total_SSE={r['share']:>7.2%}")

# ---------------------------------------------------------------------------
# 2026: Jan + Jul ranking days
# ---------------------------------------------------------------------------
if not args.skip_2026:
    have2026 = sorted(p.stem for p in Path("data/adsb/points").glob("2026-*.parquet")
                       if p.stem[5:7] in ("01", "07"))
    if args.max_days:
        have2026 = have2026[: args.max_days]
    log(f"2026: {len(have2026)} Jan/Jul days with points")
    dep26 = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
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
    cls26.write_parquet("data/interim/agent_coverage_census_2026.parquet")
    log(f"2026 done: {cls26.height:,} rows classified")
    n26 = cls26.height
    rollup26 = (cls26.group_by(["apt", "category"]).agg(pl.len().alias("n"))
                  .with_columns((C("n") / n26).alias("share_of_rows")).sort("n", descending=True))
    rollup26.write_parquet("data/interim/agent_coverage_census_2026_rollup.parquet")
    print(f"\n=== 2026 (Jan+Jul, {len(have2026)} days, n={n26:,}) TOP 25 BY ROW COUNT ===")
    print(f"{'apt':<5}{'category':<20}{'n':>10}{'share':>10}")
    for r in rollup26.head(25).iter_rows(named=True):
        print(f"{r['apt']:<5}{r['category']:<20}{r['n']:>10,}{r['share_of_rows']:>10.2%}")
    print("\n=== 2026 ROLLUP BY CATEGORY (all airports) ===")
    byc26 = cls26.group_by("category").agg(pl.len().alias("n")).with_columns((C("n") / n26).alias("share")).sort("share", descending=True)
    for r in byc26.iter_rows(named=True):
        print(f"  {r['category']:<20} n={r['n']:>9,}  share_of_rows={r['share']:>7.2%}")

log("ALL DONE")
