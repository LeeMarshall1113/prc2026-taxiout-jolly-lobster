"""experiment trackfeat COPY of archive/agent_frontier_stage1_eval.py (2026-09-23) adding --stack-track / --floor-stack-track
(on/off, default off): appends scripts/agent_trackfeat_features.TRACKFEAT_COLS -- path length, stop count/total/max, the
initial gate-dwell gap and a late-stop (runway-holding-point) proxy, all computed from the row's own track between its
run start and lift-off -- to the second stage and/or the floor stack, on top of whatever --stack-feats / --floor-stack-feats
already puts there (normally quality_anch, the shipped anchors). With both track flags off this reproduces the frontier
copy exactly. docs/LAB-NOTES.md.

experiment frontier COPY of archive/agent_adsbq_stage1_eval.py (2026-09-23) adding --floor-stack-feats quality_anch (the floor
stack also sees the anchor columns; docs/LAB-NOTES.md). With any other --floor-stack-feats it reproduces the adsbq copy.

experiment adsbq COPY of archive/adsb_stage1_eval.py (2026-09-23) adding --stack-feats quality_anch: the second
stage also sees the extra track anchors of archive/agent_adsbq_anchors.py (the same implementation
archive/agent_adsbq_apply.py serves with). With any other --stack-feats it must reproduce the original.
docs/LAB-NOTES.md.

ADS-B Stage 1 evaluation: does a ground-track off-block estimate beat the model?

    .venv/Scripts/python.exe archive/adsb_stage1_eval.py [--in-sample] [--radius 150]

The go/no-go gate for Stages 2-3 (the 2026 scored months, ~205 GB). Uses the days in
data/adsb/points and OUT-OF-FOLD predictions for them from the configuration v19 ships
(results/preds_n_feat_1_7.parquet, NM neighbour features -- every Stage 1 day is in January
or July 2025). Measuring against an older, weaker control would overstate the gain.

The matching and estimation code lives in prc/adsb_match.py, shared with serving, so the
2026 correction runs exactly what this evaluation validated.

MATCHING. A flight-plan departure with a callsign is matched to an aircraft whose ground run
ends in a LIFT-OFF within 10 minutes of the airport's take-off stamp, same airport, same
callsign. Anchoring at take-off matters: the inbound leg often carries the same callsign
and sits on the same stand.

NO-PLAN departures have no callsign (it comes from the NM record they failed to match), and
for them ADS-B is the ONLY measurement of off-block. They are matched on lift-off within
120 s of the take-off stamp, same airport, same aircraft type; a row with two candidates, or
a run already claimed by a plan departure or by another no-plan row, is dropped. On
2025-01-15 this agreed with the callsign join on 92% of plan rows. The estimator's bias and
error variance for no-plan rows come from PLAN rows at the same airport on other days --
the ground track does not know whether NM matched -- while the model's variance comes from
no-plan rows, where the model is far weaker.

ESTIMATORS of the hidden off-block time, tried in this order:
  stand    last fix within R metres of THIS flight's stand before lift-off -- VALID ONLY if
           the next fix follows within 120 s. On 2025-01-15 the 35 flights whose last stand
           fix was followed by >10 minutes of silence had RMSE 4,547s (a fix hours before
           push-back); the 761 with continuous tracking had RMSE ~160s.
  appear   first fix of the lift-off run, when that fix lies within 400 m of the stand and
           the run is continuous (no silence over 180 s): the transponder came on at
           push-back.
Stand positions come from other days' flights at their TRUE off-block time, and from all
days' arrivals at their on-block time (labelled in 2026 too).

EVERYTHING CALIBRATED IS LEAVE-ONE-DAY-OUT: stand positions, each estimator's per-airport
bias, and the blend weight. --in-sample exists only to debug on a single day.

BLEND. Matched, valid rows get w * (take-off - estimate - bias) + (1 - w) * model, with w the
inverse-variance weight from WINSORISED variances on other days (a MAD weight ignored the
tail that RMSE is made of). A guard variant also refuses corrections that move the
prediction by more than 1,800s. All other departures keep the model.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import polars as pl

from prc.adsb_match import (FLOOR_QUALITY, QUALITY, Gates, add_derived_callsign, departure_floor, dwell_sample, estimate,
                            flight_callsign_maps, match_arrivals, match_departures, runs_and_tracks, stand_map,
                            stand_samples, wvar)
from archive.agent_adsbq_anchors import ANCHOR_COLS, REL_COLS, anchors
from archive.agent_trackfeat_features import TRACKFEAT_COLS, trackfeats

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--in-sample", action="store_true")
ap.add_argument("--radius", type=float, default=150.0)
ap.add_argument("--liftoff-s", type=float, default=600.0,
                help="first airborne fix must follow the last ground fix within this (Heathrow loses "
                     "ground reception before the runway, so the original 180s missed most of its "
                     "departures; 600s validated on the July days, see prc/adsb_match.Gates)")
ap.add_argument("--no-arrivals", action="store_true",
                help="learn stand positions from departures only (the pre-2026-09-16 behaviour)")
ap.add_argument("--preds", default="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet",
                help="out-of-fold predictions to blend against (fold (1,7) covers every Stage 1 day)")
ap.add_argument("--days", default="", help="comma-separated subset of the extracted days (default: all)")
ap.add_argument("--report-days", default="",
                help="also report each blend on these days' rows alone (calibration still uses every day)")
ap.add_argument("--stand-samples", choices=["nearest", "parked", "parked_fill"], default=Gates().stand_samples,
                help="how stand positions are sampled; see prc.adsb_match.Gates")
ap.add_argument("--parked-airports", default=",".join(Gates().parked_airports),
                help="comma-separated airports the --stand-samples mode applies to (default: every airport)")
ap.add_argument("--airborne-callsign", choices=["on", "off"], default="on" if Gates().airborne_callsign else "off",
                help="plan rows may also match a run on its post-lift-off callsign; see prc.adsb_match.Gates")
ap.add_argument("--start-estimator", choices=["on", "off"], default="on" if Gates().start_estimator else "off",
                help="last-resort 'start' estimate: slow first fix of a continuous run at any mapped stand")
ap.add_argument("--slow-low-ground", choices=["on", "off"], default="on" if Gates().slow_low_ground else "off",
                help="treat slow points at numeric altitudes near field elevation as ground")
ap.add_argument("--floor-stack-anch-skip", default="", help="frontier: airports whose rows get NaN anchor columns in the floor stack")
ap.add_argument("--stack-track", choices=["on", "off"], default="off",
                help="trackfeat: append TRACKFEAT_COLS (path length, stop profile, gate-dwell gap, holding-stop proxy; "
                     "archive/agent_trackfeat_features.py) to the main stack, on top of --stack-feats")
ap.add_argument("--floor-stack-track", choices=["on", "off"], default="off",
                help="trackfeat: append TRACKFEAT_COLS to the floor stack, on top of --floor-stack-feats")
ap.add_argument("--floor-stack-track-skip", default="", help="trackfeat: airports whose rows get NaN track columns in the floor stack")
ap.add_argument("--track-cols", default=",".join(TRACKFEAT_COLS),
                help="comma-separated subset of TRACKFEAT_COLS actually appended by --stack-track/--floor-stack-track "
                     "(default: all 8) -- lets a decomposition arm isolate one mechanism (e.g. just tf_path_m,tf_avgspd_kt) "
                     "without recomputing trackfeats() per row")
ap.add_argument("--floor-stack-feats", choices=["basic", "hasest", "quality", "quality2", "quality_st", "quality_ratio", "quality_mo", "quality_anch"], default="basic",
                help="hasest: give the floor stack a flag for whether the row has a stand/appear estimate")
ap.add_argument("--floor-stand-anchor", choices=["on", "off"], default="off",
                help="anchor the taxi floor on the last parked stop AT the flight's stand, not a de-icing pad")
ap.add_argument("--prune-tracks", choices=["on", "off"], default="on",
                help="keep only matched departures' and arrivals' tracks per day (result-identical; memory "
                     "scales with matched aircraft, not every aircraft). off = the original behaviour.")
ap.add_argument("--appear-m", type=float, default=Gates().appear_m,
                help="'appear' needs the run's first fix this close to the stand")
ap.add_argument("--guard-s", type=float, default=1800.0,
                help="the guard variant refuses corrections larger than this")
ap.add_argument("--pre-chain", choices=["off", "clamp", "clamp_box"], default="off",
                help="apply production post-processing to the MODEL prediction before the ADS-B step: "
                     "clamp = the schedule-gap clamp (y <= gap below a 50,000s gap); clamp_box = also the "
                     "LOBT box on plan rows. Production applies both AFTER, so the stack currently reads a "
                     "prediction the chain later overwrites.")
ap.add_argument("--spill-tracks", choices=["on", "off"], default="off",
                help="hold one day's tracks in memory and pickle the rest to scratch (result-identical; the 177-day run drops from ~8.7 GB to ~2 GB)")
ap.add_argument("--blends", default="",
                help="comma-separated subset of plan_noguard,plan_guard,noplan_noguard,noplan_guard,both_noguard,both_guard (default: all six)")
ap.add_argument("--floor-stack-trees", type=int, default=250)
ap.add_argument("--floor-stack-leaves", type=int, default=15)
ap.add_argument("--floor-stack-minchild", type=int, default=100)
ap.add_argument("--stack-trees", type=int, default=300)
ap.add_argument("--stack-leaves", type=int, default=15)
ap.add_argument("--stack-minchild", type=int, default=50)
ap.add_argument("--floor-stack-snapshot", choices=["on", "off"], default="off",
                help="on: train each day's floor stack on predictions before any stack increment (serving-equivalent)")
ap.add_argument("--floor-stack-lower-skip", default="",
                help="airports where the floor stack stays raise-only even with --floor-stack-lower")
ap.add_argument("--floor-stack-lower", type=float, default=0.0,
                help="let the floor stack lower a prediction by at most this many seconds (0 = raise only)")
ap.add_argument("--floor-stack", choices=["on", "off"], default="off",
                help="after the hard floor, add a LightGBM increment trained leave-one-day-out on the rows that "
                     "have a floor (features lb - p, lb, p, airport, hour, NM taxi - p; target y - p), clipped at "
                     "zero so it can only raise a prediction. docs/LAB-NOTES.md.")
ap.add_argument("--tight-anchor-m", type=float, default=Gates().tight_anchor_m,
                help="emit a 'tight' estimator kind anchored on the last fix this close to the stand centroid")
ap.add_argument("--tight-airports", default=",".join(Gates().tight_airports),
                help="airports the tight anchor applies to (default: every airport)")
ap.add_argument("--floor-q", choices=["median", "zero"], default="median",
                help="zero: the hard floor raises to lb itself, leaving the whole increment to the floor stack")
ap.add_argument("--stack-skip-airports", default="",
                help="airports whose plan rows keep the inverse-variance blend instead of the second stage")
ap.add_argument("--stack-feats", choices=["basic", "quality", "quality_lb", "quality_fq", "quality_anch"], default="basic",
                help="quality: add this row's track evidence (distance of the anchor fix from the centroid, its "
                     "speed, fix count and span inside the circle, gap to the next fix, fastest speed inside, time "
                     "since it was last stationary) plus stand and aircraft type to the second stage")
ap.add_argument("--stack", choices=["gbm", "none"], default="none",
                help="plan rows: the LightGBM second stage serving uses (features x - p, NM taxi - p, p, x, "
                     "airport, kind, hour), trained leave-one-day-out, instead of the inverse-variance weight. "
                     "Serving has defaulted to it since v24 and it makes about 99 per cent of the plan-row corrections there, "
                     "so a gate judged without it is judged on a blend that does not ship. docs/LAB-NOTES.md.")
ap.add_argument("--stand-anchor", choices=["last_in_circle", "last_slow"], default=Gates().stand_anchor,
                help="where the stand estimate anchors inside the circle; see docs/LAB-NOTES.md")
ap.add_argument("--stand-anchor-kt", type=float, default=Gates().stand_anchor_kt,
                help="'last_slow' anchors on the last fix at or below this ground speed")
ap.add_argument("--floor-kt", type=float, default=Gates().floor_kt,
                help="a fix at or above this speed means the aircraft has left its stand")
ap.add_argument("--floor-turn-s", type=float, default=Gates().floor_turn_s,
                help="a parked stop this long ends the inbound leg and starts the departure part")
ap.add_argument("--osm-fill", choices=["on", "off"], default="on" if Gates().osm_fill else "off",
                help="fill stands neither the block-time samples nor the dwell fill can map from OpenStreetMap")
ap.add_argument("--osm-airports", default=",".join(Gates().osm_airports),
                help="airports the OSM fill applies to (default: every airport)")
ap.add_argument("--dwell-fill", choices=["on", "off"], default="on" if Gates().dwell_fill else "off",
                help="map stands nobody arrives at from DEPARTURES that were transmitting while parked "
                     "(prc.adsb_match.dwell_sample); label-free, so the held-out day's own departures count")
ap.add_argument("--dwell-airports", default=",".join(Gates().dwell_airports),
                help="airports the dwell fill applies to (default: every airport)")
ap.add_argument("--noplan-callsign", choices=["on", "off"], default="on" if Gates().noplan_callsign else "off",
                help="match no-plan rows on the callsign derived from FLIGHT_mvt before falling back to time + type")
ap.add_argument("--save-blends", default="", help="write each blend's predictions (report-days runs) to this .npz")
ap.add_argument("--floor", choices=["on", "off"], default="off",
                help="also report the taxi-time floor (prc.adsb_match.departure_floor) applied AFTER each blend: "
                     "p = max(p, lb + q), q = the airport's median (y - lb) on other days' rows where the model "
                     "sat below lb, separately for rows with and without a stand/appear estimate")
args = ap.parse_args()
TRACK_COLS = [c for c in TRACKFEAT_COLS if c in set(args.track_cols.split(","))]

G = Gates(liftoff_s=args.liftoff_s, radius_m=args.radius, appear_m=args.appear_m, stand_samples=args.stand_samples,
          parked_airports=tuple(a for a in args.parked_airports.split(",") if a),
          airborne_callsign=args.airborne_callsign == "on", start_estimator=args.start_estimator == "on",
          slow_low_ground=args.slow_low_ground == "on", noplan_callsign=args.noplan_callsign == "on",
          dwell_fill=args.dwell_fill == "on",
          dwell_airports=tuple(a for a in args.dwell_airports.split(",") if a),
          osm_fill=args.osm_fill == "on", osm_airports=tuple(a for a in args.osm_airports.split(",") if a),
          floor_kt=args.floor_kt, floor_turn_s=args.floor_turn_s,
          stand_anchor=args.stand_anchor, stand_anchor_kt=args.stand_anchor_kt,
          tight_anchor_m=args.tight_anchor_m,
          tight_airports=tuple(a for a in args.tight_airports.split(",") if a),
          floor_stand_anchor=args.floor_stand_anchor == "on")
GUARD_S = args.guard_s   # guard variant: refuse corrections larger than this

# 2025 days only: they are the ones with labels and out-of-fold predictions. The 2026 scored
# days live in the same directory since Stage 2 and would only cost time and memory here.
days = sorted(p.stem for p in Path("data/adsb/points").glob("2025-*.parquet"))
if args.days:
    want = set(args.days.split(","))
    days = [d for d in days if d in want]
if not days:
    sys.exit("no extracted days in data/adsb/points")
if len(days) == 1 and not args.in_sample:
    sys.exit("only one day extracted: leave-one-day-out is impossible; use --in-sample to debug")
print(f"days: {', '.join(days)}" + ("   ** IN-SAMPLE: optimistic, debugging only **" if args.in_sample else ""))

C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)
all_dep = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
           .select(C("MVT_ID_mvt"), C("ADEP_mvt").alias("apt"), C_DAY.alias("day"),
                   C("AOBT_3_flt").is_not_null().alias("plan"),
                   C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                   C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                   C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
                   C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
                   C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"),
                   C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int32).alias("hour"),
                   (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("nm_taxi"),
                   C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("block_t"))
           .filter(C("day").is_in(days)).collect())
pred = pl.concat([pl.read_parquet(f).select("MVT_ID_mvt", "y", "bagged") for f in args.preds.split(",")])
pred = pred.unique(subset="MVT_ID_mvt", keep="first")
universe = all_dep.join(pred, on="MVT_ID_mvt", how="inner")
if G.noplan_callsign:
    cs_src = [pl.scan_parquet(src).filter(C("PHASE_mvt") == "DEP")
                .select(C("AOBT_3_flt").is_not_null().alias("plan"),
                        C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                        C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight")).collect()
              for src in ("data/raw/training_*.parquet", "data/raw/ranking.parquet")]
    universe = add_derived_callsign(universe, flight_callsign_maps(cs_src))
    npl = universe.filter(~C("plan"))
    print(f"no-plan callsigns derived from FLIGHT_mvt for {npl['cs2'].is_not_null().mean():.1%} of {npl.height:,} no-plan rows "
          f"(skipping {', '.join(G.noplan_callsign_skip) or 'no airport'})")
print(f"baseline predictions: {args.preds}")
print(f"departures on these days with out-of-fold predictions: {universe.height:,} (of {all_dep.height:,})")

arrivals = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "ARR")
            .select(C("ADES_mvt").alias("apt"), C_DAY.alias("day"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("land_t"),
                    C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("onblock_t"))
            .filter(C("day").is_in(days) & C("callsign").is_not_null()).collect())

NOPLAN_BY_CS, dwell_rows = [], []
matched, arr_matched, tracks_by_day = [], [], {}
if args.spill_tracks == "on":
    import pickle
    import tempfile
    _SPILL = Path(tempfile.mkdtemp(prefix="adsb_tracks_"))
    print(f"spilling tracks to {_SPILL}")
    # Remove the spill dir at exit (2026-09-22: leaked dirs had filled ~170 GB of C:).
    import atexit
    import shutil
    atexit.register(shutil.rmtree, _SPILL, ignore_errors=True)
_CACHE = {}


def get_tracks(day):
    """This day's pruned tracks, from memory or from the spill file (one day is cached)."""
    if args.spill_tracks != "on":
        return tracks_by_day[day]
    if day not in _CACHE:
        _CACHE.clear()
        with open(tracks_by_day[day], "rb") as fh:
            _CACHE[day] = pickle.load(fh)
    return _CACHE[day]
print(f"  {'day':<12}" + "".join(f"{a:>6}" for a in sorted(universe["apt"].unique().to_list())) + "   matched / plan departures")
for day in days:
    runs, touch, tracks = runs_and_tracks(day, G)
    arr_matched.append(match_arrivals(arrivals.filter(C("day") == day), touch, G))
    deps_day = universe.filter(C("day") == day)
    cand, nc, both = match_departures(deps_day, runs, G)
    matched.append(both)
    if G.dwell_fill:
        for r in both.filter(C("stand").is_not_null()).iter_rows(named=True):
            pos = dwell_sample(r, tracks, G)
            if pos:
                dwell_rows.append(("dwell", day, r["apt"], r["stand"], pos[0], pos[1], "dwell"))
    # KEEP ONLY THE AIRCRAFT READ LATER. Every later use of tracks_by_day -- stand samples for
    # matched plan departures and matched arrivals, then estimate() and departure_floor() for
    # matched departures -- is a single tracks.get((apt, icao)) of that row's OWN aircraft; nothing
    # iterates over the rest. So pruning to this day's matched departures and arrivals is
    # result-identical by construction, and was also checked bit-for-bit on 30 days
    # (--prune-tracks on vs off). It is what lets the ~181-day evaluation fit under 12 GB:
    # holding every aircraft's fixes for every day grew ~linearly to an estimated ~13 GB.
    # archive/adsb_apply.py has pruned the same way since it was written.
    if args.prune_tracks == "on":
        keep = set(zip(both["apt"].to_list(), both["icao"].to_list()))
        if arr_matched[-1].height:
            keep |= set(zip(arr_matched[-1]["apt"].to_list(), arr_matched[-1]["icao"].to_list()))
        kept = {k: v for k, v in tracks.items() if k in keep}
    else:
        kept = tracks
    if args.spill_tracks == "on":
        _p = _SPILL / f"{day}.pkl"
        with open(_p, "wb") as fh:
            pickle.dump(kept, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tracks_by_day[day] = _p
        del kept
    else:
        tracks_by_day[day] = kept
    del runs, touch, tracks
    d = deps_day.filter(C("plan") & C("callsign").is_not_null())
    dn = deps_day.filter(~C("plan") & C("type").is_not_null())
    if G.noplan_callsign and nc.height:
        NOPLAN_BY_CS.append(nc.filter(C("cs2").is_not_null() & (C("dmatch") > G.noplan_match_s)).height)
    cov = {a: cand.filter(C("apt") == a).height / max(d.filter(C("apt") == a).height, 1)
           for a in sorted(universe["apt"].unique().to_list())}
    print(f"  {day:<12}" + "".join(f"{v:>6.0%}" for v in cov.values())
          + f"   {cand.height:,} / {d.height:,}   no-plan {nc.height:,} / {dn.height:,}")
m = pl.concat(matched)
if NOPLAN_BY_CS:
    print(f"no-plan matches beyond the {G.noplan_match_s:.0f}s time window (so from the derived callsign): {sum(NOPLAN_BY_CS):,}")

# Stand positions. A parked aircraft blocks in and out at the same spot, so a sample is
# either a departure's fix at its TRUE off-block ("dep") or an arrival's fix at its on-block
# ("arr"). At serve time in 2026 arrival on-block times are labelled and departure off-block
# times are not -- so for an evaluation day, same-day arrivals are fair and same-day
# departures are not. That is exactly the leave-one-day-out rule applied below.
samples = []
for _day in days:  # by day, so the spill loads each file once
    _tr = get_tracks(_day)
    for r in m.filter(C("plan") & (C("day") == _day)).iter_rows(named=True):  # a no-plan BLOCK stamp may be stale
        for pos in stand_samples(_tr, r["apt"], r["icao"], r["block_t"], G, arrival=False):
            samples.append(("dep", r["day"], r["apt"], r["stand"], pos[0], pos[1], pos[2]))
n_dep = len(samples)
if not args.no_arrivals:
    _am = pl.concat(arr_matched)
    for _day in days:  # by day, so the spill loads each file once
        _tr = get_tracks(_day)
        for r in _am.filter(C("day") == _day).iter_rows(named=True):
            for pos in stand_samples(_tr, r["apt"], r["icao"], r["onblock_t"], G, arrival=True):
                samples.append(("arr", r["day"], r["apt"], r["stand"], pos[0], pos[1], pos[2]))
blk = pl.DataFrame(samples + dwell_rows, schema=["src", "day", "apt", "stand", "lat", "lon", "quality"], orient="row")
if G.dwell_fill:
    print(f"departure dwell samples (label-free): {len(dwell_rows):,}")
print(f"\nstand-position samples: {n_dep:,} departures at true off-block, {len(samples) - n_dep:,} arrivals at on-block"
      + ("  (arrivals disabled)" if args.no_arrivals else "")
      + (f"; parked {blk.filter(C('quality') == 'parked').height:,}" if G.stand_samples != "nearest" else ""))

rows = []
for day in days:
    src = blk if args.in_sample else blk.filter((C("src") != "dep") | (C("day") != day))
    # A stand position is used only if its samples agree: median distance to the centroid
    # within the spread gate. Reused or ambiguous stand labels spread out and are dropped.
    cmap = stand_map(src, G)
    for r in m.filter(C("day") == day).iter_rows(named=True):
        kind, taxi, qual = estimate(r, cmap, get_tracks(day), G, with_quality=True)
        anc = (anchors(r, cmap, get_tracks(day), G, taxi) if "quality_anch" in (args.stack_feats, args.floor_stack_feats)
               else dict.fromkeys(ANCHOR_COLS, float("nan")))
        tf = (trackfeats(r, get_tracks(day), G) if args.stack_track == "on" or args.floor_stack_track == "on"
              else dict.fromkeys(TRACKFEAT_COLS, float("nan")))
        lb, fq = (departure_floor(r, get_tracks(day), G, cmap, with_quality=True) if args.floor == "on"
                  else (None, {k: float("nan") for k in FLOOR_QUALITY}))
        rows.append((r["MVT_ID_mvt"], day, r["apt"], bool(r["plan"]), kind, taxi, r["mvt_t"] - r["block_t"], float(r["bagged"]), lb,
                     int(r["hour"]) if r.get("hour") is not None else 0, float(r["nm_taxi"]) if r.get("nm_taxi") is not None else float("nan"),
                     r["stand"], r["type"], *[qual[q] for q in QUALITY], *[fq[q] for q in FLOOR_QUALITY],
                     *[anc[k] for k in ANCHOR_COLS], *[tf[k] for k in TRACKFEAT_COLS]))
est = pl.DataFrame(rows, schema=["MVT_ID_mvt", "day", "apt", "plan", "kind", "taxi", "y_true", "model", "lb", "hour", "nm_taxi",
                                 "stand", "type", *QUALITY, *FLOOR_QUALITY, *ANCHOR_COLS, *TRACKFEAT_COLS],
                   orient="row", infer_schema_length=None)


def mad(x):
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


for plan_rows in (True, False):
    print(f"\nESTIMATOR ERROR vs TRUTH, {'PLAN' if plan_rows else 'NO-PLAN'} departures, after each airport's median "
          f"bias (+ = says a longer taxi)")
    print(f"  {'airport':<8}{'matched':>8}{'stand':>7}{'appear':>7}{'none':>6}{'stand RMSE':>11}{'stand MAD':>10}"
          f"{'appear RMSE':>12}{'model RMSE*':>12}")
    ep = est.filter(C("plan") == plan_rows)
    for apt in sorted(ep["apt"].unique().to_list()):
        e = ep.filter(C("apt") == apt)
        cells = []
        for kind in ("stand", "appear"):
            s = e.filter(C("kind") == kind)
            err = (s["taxi"] - s["y_true"]).to_numpy()
            cells.append(err - np.median(err) if len(err) else err)
        val = e.filter(C("kind").is_not_null())
        rm_model = float(np.sqrt(((np.clip(val["model"].to_numpy(), 0, None) - val["y_true"].to_numpy()) ** 2).mean())) if val.height else float("nan")
        print(f"  {apt:<8}{e.height:>8,}{len(cells[0]):>7,}{len(cells[1]):>7,}{e.filter(C('kind').is_null()).height:>6,}"
              f"{(np.sqrt((cells[0] ** 2).mean()) if len(cells[0]) else float('nan')):>10.0f}s{mad(cells[0]):>9.0f}s"
              f"{(np.sqrt((cells[1] ** 2).mean()) if len(cells[1]) else float('nan')):>11.0f}s{rm_model:>11.0f}s")
    print("  *model RMSE on the same rows that received a valid estimate")

if G.start_estimator:
    ss = est.filter(C("plan") & (C("kind") == "start"))
    print("\n'start' estimates on plan rows (RMSE after airport median bias / MAD / model RMSE on the same rows): " + "; ".join(
        f"{a} n={s.height} {np.sqrt(((e - np.median(e)) ** 2).mean()):.0f}s / {mad(e):.0f}s / "
        f"{np.sqrt(((np.clip(s['model'].to_numpy(), 0, None) - s['y_true'].to_numpy()) ** 2).mean()):.0f}s"
        for a in sorted(ss['apt'].unique().to_list()) for s in [ss.filter(C('apt') == a)]
        for e in [(s['taxi'] - s['y_true']).to_numpy()]))

OUT = {}
y = universe["y"].to_numpy().astype(float)
p0 = np.clip(universe["bagged"].to_numpy().astype(float), 0, None)
if args.pre_chain != "off":
    _raw = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
            .select(C("MVT_ID_mvt"),
                    (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("_gs"),
                    (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("_gl"),
                    C("AOBT_3_flt").is_not_null().alias("_plan")).collect())
    _j = universe.select("MVT_ID_mvt").join(_raw, on="MVT_ID_mvt", how="left")
    _gs = _j["_gs"].fill_null(np.nan).to_numpy().astype(float)
    _gl = _j["_gl"].fill_null(np.nan).to_numpy().astype(float)
    _pl = _j["_plan"].fill_null(False).to_numpy()
    _n0 = int((np.isfinite(_gs) & (_gs >= 5000) & (_gs < 50000) & (p0 > _gs)).sum())
    p0 = np.where(np.isfinite(_gs) & (_gs >= 5000) & (_gs < 50000) & (p0 > _gs), _gs, p0)
    _n1 = 0
    if args.pre_chain == "clamp_box":
        _box = _pl & np.isfinite(_gl)
        _lo, _hi = _gl - 3606.0, _gl + 3606.0
        _n1 = int((_box & ((p0 < _lo) | (p0 > _hi))).sum())
        p0 = np.where(_box, np.clip(p0, np.where(np.isfinite(_lo), _lo, -np.inf), np.where(np.isfinite(_hi), _hi, np.inf)), p0)
    p0 = np.clip(p0, 0, None)
    print(f"--pre-chain {args.pre_chain}: gap clamp moved {_n0:,} model predictions, LOBT box {_n1:,}")
idx = {k: i for i, k in enumerate(universe["MVT_ID_mvt"].to_list())}
apts, dys = universe["apt"].to_numpy(), universe["day"].to_numpy()


def model_var(tr_np, apt):
    """Model error variance on NO-PLAN rows: this airport if it has 30 rows, else all airports."""
    s = tr_np.filter(C("apt") == apt)
    if s.height < 30:
        s = tr_np
    if s.height < 30:
        return None
    return wvar(np.clip(s["model"].to_numpy(), 0, None) - s["y_true"].to_numpy())


FLOOR_MIN_ROWS = 20
fl = est.filter(C("lb").is_not_null()).with_columns(C("kind").is_not_null().alias("has_est"))


FLOOR_APTS = sorted(fl["apt"].unique().to_list()) if fl.height else []


FLOOR_STANDS = {k: i for i, k in enumerate(sorted({(a, s) for a, s in zip(fl["apt"].to_list(), fl["stand"].to_list()) if s}))} if fl.height else {}
FLOOR_TYPES = {k: i for i, k in enumerate(sorted({t for t in fl["type"].to_list() if t}))} if fl.height else {}
FLOOR_CATS = ([3, 7 + len(FLOOR_QUALITY), 8 + len(FLOOR_QUALITY)] if args.floor_stack_feats == "quality_st"
              else [3, 7 + len(FLOOR_QUALITY)] if args.floor_stack_feats == "quality_mo" else [3])


def _floor_x(frame, p_now):
    """Features for the floor stack, in the order the model is trained on."""
    lb = frame["lb"].to_numpy().astype(float)
    nm = frame["nm_taxi"].to_numpy().astype(float) if "nm_taxi" in frame.columns else np.full(len(lb), np.nan)
    apt = np.array([FLOOR_APTS.index(a) for a in frame["apt"].to_list()], dtype=float)
    cols = [lb - p_now, lb, p_now, apt, frame["hour"].to_numpy().astype(float), nm - p_now]
    if args.floor_stack_feats in ("hasest", "quality", "quality2", "quality_st", "quality_ratio", "quality_mo", "quality_anch"):
        cols.append(frame["has_est"].cast(pl.Int8).to_numpy().astype(float))
    if args.floor_stack_feats in ("quality", "quality2", "quality_st", "quality_ratio", "quality_mo", "quality_anch"):
        cols += [frame[q].to_numpy().astype(float) for q in FLOOR_QUALITY]
    if args.floor_stack_feats == "quality_anch":
        # frontier (docs/LAB-NOTES.md section 3, A10): the floor stack also sees the stand-crossing anchors, in
        # the same form the main stack gets them (scripts/agent_adsbq_anchors.stack_columns: REL_COLS relative to p)
        anc_cols = [frame[k].cast(pl.Float64).fill_null(np.nan).to_numpy() - p_now if k in REL_COLS
                    else frame[k].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in ANCHOR_COLS]
        if args.floor_stack_anch_skip:
            # --floor-stack-anch-skip: these airports' rows carry NaN anchors in the floor stack (training and scoring)
            sk = np.isin(np.array(frame["apt"].to_list()), args.floor_stack_anch_skip.split(","))
            anc_cols = [np.where(sk, np.nan, c) for c in anc_cols]
        cols += anc_cols
    if args.floor_stack_feats == "quality_mo":
        # the scored set is January and July; reception and de-icing differ by season, and the stack
        # is trained across Jan-Mar and Jul-Sep with no way to tell them apart
        cols.append(np.array([int(d[5:7]) for d in frame["day"].to_list()], float))
    if args.floor_stack_feats == "quality_ratio":
        # how TIGHT the floor is, which a tree cannot form from lb and p by splitting on each
        cols.append(lb / np.maximum(p_now, 1.0))
        cols.append(lb / np.maximum(nm, 1.0))
        cols.append(np.maximum(p_now, 1.0) / np.maximum(nm, 1.0))
    if args.floor_stack_feats == "quality_st":
        cols.append(np.array([FLOOR_STANDS.get((a, s), -1) for a, s in zip(frame["apt"].to_list(), frame["stand"].to_list())], float))
        cols.append(np.array([FLOOR_TYPES.get(t, -1) for t in frame["type"].to_list()], float))
    if args.floor_stack_feats == "quality2":
        cols.append(frame["taxi"].cast(pl.Float64).fill_null(np.nan).to_numpy() - p_now)
        cols += [frame[q].cast(pl.Float64).fill_null(np.nan).to_numpy() for q in QUALITY]
    if args.floor_stack_track == "on":
        # trackfeat: path length, stop profile, gate-dwell gap, holding-stop proxy -- appended last, so
        # nothing above (including FLOOR_CATS' categorical indices) shifts
        tcols = [frame[k].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in TRACK_COLS]
        if args.floor_stack_track_skip:
            sk = np.isin(np.array(frame["apt"].to_list()), args.floor_stack_track_skip.split(","))
            tcols = [np.where(sk, np.nan, c) for c in tcols]
        cols += tcols
    return np.column_stack(cols)


def apply_floor(p_in):
    """Raise each matched row's prediction to lb + q, q calibrated leave-one-day-out."""
    out = p_in.copy()
    for day in days:
        other = fl.filter(C("day") != day)
        below = other.filter(C("model").clip(lower_bound=0) < C("lb"))
        qs = {} if args.floor_q == "zero" else {
            (a, h): float(np.median((g["y_true"] - g["lb"]).to_numpy())) if g.height >= FLOOR_MIN_ROWS else 0.0
            for (a, h), g in below.partition_by(["apt", "has_est"], as_dict=True).items()}
        for r in fl.filter(C("day") == day).iter_rows(named=True):
            i = idx.get(r["MVT_ID_mvt"])
            if i is None:
                continue
            floor = r["lb"] + qs.get((r["apt"], r["has_est"]), 0.0)
            if out[i] < floor:
                out[i] = floor
    if args.floor_stack == "on":
        import lightgbm as lgb
        # snapshot on: every day's stack trains on predictions AFTER the hard floor and BEFORE any stack
        # increment, as serving does (adsb_apply trains once on calibration rows). off (the original):
        # training rows of days already processed carry their own increments, so each day's corrections
        # leak into later days' targets.
        p_snap = out.copy() if args.floor_stack_snapshot == "on" else None
        for day in days:
            tr = fl.filter(C("day") != day)
            te = fl.filter(C("day") == day)
            ti = np.array([idx.get(m, -1) for m in tr["MVT_ID_mvt"].to_list()])
            ei = np.array([idx.get(m, -1) for m in te["MVT_ID_mvt"].to_list()])
            ok_t, ok_e = ti >= 0, ei >= 0
            if ok_t.sum() < 500 or not ok_e.any():
                continue
            p_tr = (p_snap if p_snap is not None else out)[ti[ok_t]]
            xt = _floor_x(tr.filter(pl.Series(ok_t)), p_tr)
            yt = np.clip(tr.filter(pl.Series(ok_t))["y_true"].to_numpy().astype(float) - p_tr, -3600, 3600)
            m = lgb.LGBMRegressor(n_estimators=args.floor_stack_trees, max_depth=-1 if args.floor_stack_leaves > 15 else 4,
                                  num_leaves=args.floor_stack_leaves, learning_rate=0.05,
                                  min_child_samples=args.floor_stack_minchild, verbose=-1).fit(xt, yt, categorical_feature=FLOOR_CATS)
            p_te = out[ei[ok_e]]
            add = np.clip(m.predict(_floor_x(te.filter(pl.Series(ok_e)), p_te)), -args.floor_stack_lower, None)
            if args.floor_stack_lower > 0 and args.floor_stack_lower_skip:
                skip = np.isin(np.array(te.filter(pl.Series(ok_e))["apt"].to_list()), args.floor_stack_lower_skip.split(","))
                add = np.where(skip, np.maximum(add, 0.0), add)
            if args.floor_stack_lower > 0:
                # lowering never goes under lb itself: the aircraft was taxiing fast at MVT - lb, so y >= lb
                out[ei[ok_e]] = np.maximum(p_te + add, te.filter(pl.Series(ok_e))["lb"].to_numpy().astype(float))
            else:
                out[ei[ok_e]] = p_te + add
    return out


if args.floor == "on":
    print()
    print(f"FLOOR ON: every blend below has the taxi floor applied after it; floors on {fl.height:,} matched rows "
          f"({fl.filter(C('has_est')).height:,} with a stand/appear estimate)")
    fa = apply_floor(p0)
    ea = (y - fa) ** 2
    print(f"  floor alone on the model: all {math.sqrt(((y - p0) ** 2).mean()):.2f} -> {math.sqrt(ea.mean()):.2f}; rows raised {int((fa > p0).sum()):,}")

STACK_APTS = sorted(est.filter(C("kind").is_not_null())["apt"].unique().to_list())
STACK_KINDS = ("stand", "appear", "start", "tight")
# stand and aircraft type as categoricals: a code review measured per-(airport, stand) bias at
# 120s against per-airport 130s, with type adding separately
_seen = est.filter(C("kind").is_not_null())
STACK_STANDS = {k: i for i, k in enumerate(sorted({(a, s) for a, s in zip(_seen["apt"].to_list(), _seen["stand"].to_list()) if s}))}
STACK_TYPES = {k: i for i, k in enumerate(sorted({t for t in _seen["type"].to_list() if t}))}
STACK_CATS = ([4, 5] if args.stack_feats == "basic"
              else [4, 5, 7 + len(QUALITY), 8 + len(QUALITY)] if args.stack_feats in ("quality", "quality_anch")
              else [4, 5, 7 + len(QUALITY) + len(FLOOR_QUALITY), 8 + len(QUALITY) + len(FLOOR_QUALITY)]
              if args.stack_feats == "quality_fq"
              else [4, 5, 9 + len(QUALITY), 10 + len(QUALITY)])


def _stack_x(frame, bias_of, p_now):
    """Second-stage features, the same list scripts/adsb_apply.stack_features builds."""
    taxi = frame["taxi"].to_numpy().astype(float)
    bias = np.array([bias_of.get((a, k), np.nan) for a, k in zip(frame["apt"].to_list(), frame["kind"].to_list())])
    x = np.clip(taxi - bias, 0, None)
    nm = frame["nm_taxi"].to_numpy().astype(float)
    apt = np.array([STACK_APTS.index(a) for a in frame["apt"].to_list()], dtype=float)
    kind = np.array([STACK_KINDS.index(k) for k in frame["kind"].to_list()], dtype=float)
    cols = [x - p_now, nm - p_now, p_now, x, apt, kind, frame["hour"].to_numpy().astype(float)]
    if args.stack_feats in ("quality", "quality_lb", "quality_fq", "quality_anch"):
        cols += [frame[q].to_numpy().astype(float) for q in QUALITY]
        if args.stack_feats == "quality_lb":
            lbv = frame["lb"].to_numpy().astype(float)
            cols += [lbv, lbv - p_now]
        if args.stack_feats == "quality_fq":
            cols += [frame[q].cast(pl.Float64).fill_null(np.nan).to_numpy() for q in FLOOR_QUALITY]
        cols += [np.array([STACK_STANDS.get((a, st), -1) for a, st in zip(frame["apt"].to_list(), frame["stand"].to_list())], float),
                 np.array([STACK_TYPES.get(ty, -1) for ty in frame["type"].to_list()], float)]
        if args.stack_feats == "quality_anch":
            # after stand and type, so STACK_CATS is unchanged; ci_R and appear_raw enter as value - p, like x - p
            cols += [frame[k].cast(pl.Float64).fill_null(np.nan).to_numpy() - p_now if k in REL_COLS
                     else frame[k].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in ANCHOR_COLS]
    if args.stack_track == "on":
        # trackfeat: appended last (after anchors if any), so STACK_CATS is unchanged. These are all
        # durations/distances, not estimated off-block times, so they enter raw (not value - p)
        cols += [frame[k].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in TRACK_COLS]
    return np.column_stack(cols), np.isfinite(x) & np.isfinite(nm)


def stack_corrections(day, train, test_plan, p_base):
    """LightGBM corrections for one day's plan rows, trained on the other days."""
    import lightgbm as lgb

    bias_of = {}
    for (apt, kind), tr in train.filter(C("plan") & C("kind").is_not_null()).partition_by(["apt", "kind"], as_dict=True).items():
        if tr.height >= 30:
            bias_of[(apt, kind)] = float(np.median((tr["taxi"] - tr["y_true"]).to_numpy()))
    tr_rows = train.filter(C("plan") & C("kind").is_not_null())
    if not tr_rows.height or not test_plan.height:
        return {}
    ti = np.array([idx.get(m, -1) for m in tr_rows["MVT_ID_mvt"].to_list()])
    ok = ti >= 0
    if ok.sum() < 200:
        return {}
    tr_rows = tr_rows.filter(pl.Series(ok))
    p_tr = p_base[ti[ok]]
    Xt, good = _stack_x(tr_rows, bias_of, p_tr)
    yt = np.clip(tr_rows["y_true"].to_numpy().astype(float) - p_tr, -3600, 3600)
    if good.sum() < 200:
        return {}
    model = lgb.LGBMRegressor(n_estimators=args.stack_trees, max_depth=-1 if args.stack_leaves > 15 else 4,
                              num_leaves=args.stack_leaves, learning_rate=0.05,
                              min_child_samples=args.stack_minchild, verbose=-1).fit(Xt[good], yt[good], categorical_feature=STACK_CATS)
    ei = np.array([idx.get(m, -1) for m in test_plan["MVT_ID_mvt"].to_list()])
    oke = ei >= 0
    if not oke.any():
        return {}
    te = test_plan.filter(pl.Series(oke))
    Xe, goode = _stack_x(te, bias_of, p_base[ei[oke]])
    out = {}
    if goode.any():
        corr = np.clip(model.predict(Xe[goode]), -GUARD_S, GUARD_S)
        for mid, c in zip(np.asarray(te["MVT_ID_mvt"].to_list())[goode], corr):
            out[mid] = float(c)
    return out


_WANT = {b for b in args.blends.split(",") if b}
for scope, guard in (("plan", False), ("plan", True), ("noplan", False), ("noplan", True), ("both", False), ("both", True)):
    if _WANT and f"{scope}_{'guard' if guard else 'noguard'}" not in _WANT:
        continue
    p1, used = p0.copy(), 0
    for day in days:
        train = est if args.in_sample else est.filter(C("day") != day)
        tr_np = train.filter(~C("plan") & C("kind").is_not_null())
        test = est.filter((C("day") == day) & C("kind").is_not_null())
        if scope != "both":
            test = test.filter(C("plan") == (scope == "plan"))
        if not test.height:
            continue
        stacked = {}
        if args.stack == "gbm" and scope in ("plan", "both"):
            tp = test.filter(C("plan"))
            skip = [a for a in args.stack_skip_airports.split(",") if a]
            if skip:
                tp = tp.filter(~C("apt").is_in(skip))
            stacked = stack_corrections(day, train, tp, p0)
            for mid, c in stacked.items():
                i = idx.get(mid)
                if i is not None:
                    p1[i] = max(p0[i] + c, 0.0)
                    used += 1
        for (apt, kind, is_plan), te in test.partition_by(["apt", "kind", "plan"], as_dict=True).items():
            if is_plan and stacked:
                te = te.filter(~C("MVT_ID_mvt").is_in(list(stacked)))
                if not te.height:
                    continue
            # the estimator's own bias and variance always come from plan rows: the ground
            # track does not know whether NM matched the flight
            tr = train.filter((C("apt") == apt) & (C("kind") == kind) & C("plan"))
            if tr.height < 30:
                continue
            err = (tr["taxi"] - tr["y_true"]).to_numpy()
            bias = float(np.median(err))
            v_a = wvar(err - bias)
            v_m = (wvar(np.clip(tr["model"].to_numpy(), 0, None) - tr["y_true"].to_numpy()) if is_plan
                   else model_var(tr_np, apt))
            if v_m is None:
                continue
            w = v_m / (v_m + v_a) if (v_m + v_a) > 0 else 0.0
            for r in te.iter_rows(named=True):
                i = idx.get(r["MVT_ID_mvt"])
                if i is None:
                    continue
                x = max(r["taxi"] - bias, 0.0)
                new = w * x + (1 - w) * p0[i]
                if guard and abs(new - p0[i]) > GUARD_S:
                    continue
                p1[i] = new
                used += 1
    if args.floor == "on":
        p1 = apply_floor(p1)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    core = y <= 10000
    label = f"{scope.upper()} rows, " + (f"with the {GUARD_S}s guard" if guard else "no guard")
    print(f"\nBLEND, {label} (leave-one-day-out{' -- NO, in-sample' if args.in_sample else ''}): corrected {used:,} of {len(y):,}")
    print(f"  all departures {math.sqrt(e0.mean()):.2f} -> {math.sqrt(e1.mean()):.2f} "
          f"({math.sqrt(e1.mean()) - math.sqrt(e0.mean()):+.2f}s)   core "
          f"{math.sqrt(e1[core].mean()) - math.sqrt(e0[core].mean()):+.2f}s")
    print("  by airport: " + "  ".join(
        f"{a} {math.sqrt(e1[apts == a].mean()) - math.sqrt(e0[apts == a].mean()):+.1f}" for a in sorted(set(apts))))
    if len(days) > 1:
        print("  by day:     " + "  ".join(
            f"{d[5:]} {math.sqrt(e1[dys == d].mean()) - math.sqrt(e0[dys == d].mean()):+.1f}" for d in days))
    saved = (e0.sum() - e1.sum()) / len(y)
    print(f"  board projection at 2025 coverage: {math.sqrt(max(271.969 ** 2 - saved, 0)) - 271.969:+.2f}s "
          f"(squared error saved per departure {saved:,.0f} s^2)")
    if args.report_days:
        rep = np.isin(dys, args.report_days.split(","))
        rc = rep & core
        print(f"  report days only ({int(rep.sum()):,} rows): all {math.sqrt(e0[rep].mean()):.2f} -> {math.sqrt(e1[rep].mean()):.2f} "
              f"({math.sqrt(e1[rep].mean()) - math.sqrt(e0[rep].mean()):+.2f}s)   core "
              f"{math.sqrt(e1[rc].mean()) - math.sqrt(e0[rc].mean()):+.2f}s")
        OUT[(scope, guard)] = p1.copy()


if args.save_blends:
    np.savez(args.save_blends, ids=np.asarray(universe["MVT_ID_mvt"].to_list()), y=y, p0=p0, days=dys, apts=apts,
             **{f"{sc}_{'guard' if gd else 'noguard'}": v for (sc, gd), v in OUT.items()})
    print(f"saved blends to {args.save_blends}")
