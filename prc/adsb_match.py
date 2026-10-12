"""ADS-B ground tracks -> off-block estimates. Shared by the Stage 1 evaluation and by serving.

One implementation, so the 2026 correction runs exactly the code the leave-one-day-out
evaluation validated (a flag and its implementation living in different files is the bug
this project has shipped eight times). archive/adsb_stage1_eval.py is the evaluation; see
its docstring for why each gate exists and what it cost to learn.

Points come from prc.adsb (one parquet per day, ten airport boxes, ground or below 3,000 ft).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

C = pl.col
POINTS_DIR = Path("data/adsb/points")


@dataclass(frozen=True)
class Gates:
    gap_s: float = 20 * 60          # a ground run is broken by a silence longer than this
    # the first airborne fix must follow the run's last ground fix within this. 600s, not the
    # original 180s: Heathrow loses ground reception before the runway. Found on the January
    # Stage 1 days, validated on July alone under a rule fixed beforehand (guarded plan-row
    # blend -2.72s / core -3.20s against -2.63s / -3.10s at 180s).
    liftoff_s: float = 600.0
    match_s: float = 10 * 60        # plan rows: |lift-off - take-off stamp| within this, same callsign
    noplan_match_s: float = 120.0   # no-plan rows: lift-off this close, and the aircraft type must agree
    lookback_s: float = 3 * 3600    # how far before lift-off to look for the aircraft at its stand
    stand_next_s: float = 120.0     # a stand fix counts only if tracking continues within this
    # 600 m, not the original 400: on the 30 days with serving's second stage in the evaluation it
    # reads -0.334s all / -0.395s core on top of the quality stack, no airport worse.
    appear_m: float = 600.0         # 'appear' needs the run's first fix this close to the stand
    run_gap_s: float = 180.0        # 'appear' needs a run with no silence longer than this
    stand_spread_m: float = 60.0    # use a stand position only if its samples sit this close to their median
    dwell_window_s: float = 600.0   # a slow fix inside the stand circle must occur this soon before leaving it
    dwell_gs_kt: float = 3.0        # 'slow' = parked or pushing back
    radius_m: float = 150.0         # stand circle
    fix_tol_s: float = 120.0        # a stand-position sample needs a fix this close to the block time
    max_taxi_s: float = 14400.0     # estimates outside [0, this] are discarded
    # Stand samples. 'nearest' = the fix nearest the block time (the original). 'parked' = the
    # median position of fixes at <= parked_gs_kt within parked_window_s AFTER an arrival's
    # on-block (BEFORE a departure's off-block), falling back to 'nearest' where there are too
    # few; the map is then built from parked samples first. At Barcelona and Madrid the nearest
    # fix to a labelled on-block is still moving at a median 11-12 kt, so only 26-28% of stands
    # passed the spread gate; parked samples sit within a median 2 m.
    stand_samples: str = "parked_fill"
    # Airports the stand_samples mode applies to; empty = every airport. Everywhere else stays
    # on 'nearest'. Barcelona alone: on the 10 Stage 1 days parked_fill everywhere cost Schiphol
    # 1.5s; at Barcelona only it gives -3.89s all / -5.06s core against nearest's -3.65s /
    # -4.74s, identical elsewhere, no day worse. The airport was chosen after seeing per-airport
    # results.
    # EDDF, LEMD and LSZH added 2026-09-17 on the 30 days: all -6.99s against -6.93s, each of the
    # three better on its own rows (LEMD -5.4 against -4.6), no airport worse.
    parked_airports: tuple = ("LEBL", "EDDF", "LEMD", "LSZH")
    parked_window_s: float = 900.0
    parked_gs_kt: float = 1.0
    parked_min_fixes: int = 3
    # Plan rows may also match a run on the callsign its aircraft transmits once AIRBORNE (first
    # callsign within airborne_cs_window_s after lift-off). At Frankfurt in 2026 many ground runs
    # still carry the inbound flight's callsign, or none, until after take-off: on two days 348
    # plan departures were lost that way (TWB403 on the ground, TWB404 in the air).
    # Validated on the 10 Stage 1 days: -3.90s all / -5.07s core against -3.89s / -5.06s, no airport
    # worse by more than 0.1s (stale ground callsigns were rarer in 2025).
    airborne_callsign: bool = True
    airborne_cs_window_s: float = 900.0
    # 'start' estimator, tried last: the first fix of a continuous lift-off run, when that fix
    # is slow (<= start_gs_kt) and lies within radius_m of ANY mapped stand at the airport. It
    # needs no position for the flight's own stand label, which Frankfurt, Madrid and Zurich
    # often lack (unmapped, or no fix near it), and many transponders come on at push-back.
    start_estimator: bool = False
    start_gs_kt: float = 5.0
    # Treat a point as ground when it reports a numeric (pressure) altitude within
    # slow_low_band_ft of field elevation at <= slow_low_gs_kt. Some transponders never set the
    # on-ground flag: at Frankfurt two 2026 days hold 3,299 such slow points at 100-400 ft
    # (elevation 364 ft), which break ground runs and hide aircraft at their stands.
    # Validated on the 10 Stage 1 days: -3.91s all / -5.09s core against -3.90s / -5.07s, no airport
    # worse (unflagged slow points were rare in 2025).
    slow_low_ground: bool = True
    slow_low_band_ft: float = 800.0
    slow_low_gs_kt: float = 40.0
    # Where the 'stand' estimate anchors inside the circle. 'last_in_circle' is the fix the
    # aircraft was last seen at before crossing 150 m; 'last_slow' is the last fix at or below
    # stand_anchor_kt, i.e. the last moment it was not moving, which is what an airport records as
    # off-block.
    stand_anchor: str = "last_in_circle"
    stand_anchor_kt: float = 1.0
    # A TIGHT anchor, emitted as its own estimator kind: the last fix within tight_m of the stand
    # centroid, with tracking continuing within stand_next_s. The 150 m crossing sits a median
    # +288s after the recorded off-block; the 20 m one sits +29s, and its error about the median
    # is smaller at the airports whose stand map is built from clean samples (EDDM 72s against
    # 114s, LSZH 83s against 149s) and larger where the map is not (EHAM, LEBL). Own kind, so it
    # gets its own bias and variance rather than diluting the stand estimate's.
    tight_anchor_m: float = 0.0        # 0 disables it
    tight_airports: tuple = ()         # empty = every airport
    # Taxi-time floor (departure_floor): an aircraft moving at >= floor_kt toward its lift-off
    # has left its stand, so y >= MVT - t(that fix). The departure part of a run starts after
    # its last parked stop (consecutive fixes at <= floor_park_kt spanning >= floor_turn_s).
    floor_kt: float = 10.0
    floor_park_kt: float = 1.0
    floor_turn_s: float = 600.0
    # Anchor the floor on the last parked stop AT THIS FLIGHT'S STAND rather than the last parked stop
    # of any kind (a de-icing pad or a long runway queue). Falls back to the old rule. Default off.
    floor_stand_anchor: bool = False
    # Stand positions from DEPARTURES that were transmitting while parked (dwell_sample), used
    # only for stands the block-time samples cannot map. Label-free, so 2026 departures map 2026
    # stands -- the point of it: Madrid and Frankfurt lose most of their matched departures to
    # stands no tracked arrival ever reached. Everywhere at once fails its rule (30 days: all
    # -6.96s against -6.93s, and Schiphol 0.9s worse); at EDDF and LEMD, the two airports whose
    # block-time map leaves most matched departures unmapped, -6.98s with every other airport
    # identical. Those two were chosen after seeing per-airport results, as LEBL was for parked.
    # Stand positions from OpenStreetMap (aeroway=parking_position / gate, tagged with the stand
    # name in `ref`), used only for stands the block-time samples and the dwell fill cannot map.
    # Fetched by archive/fetch_osm_stands.py; ODbL, attribution "(c) OpenStreetMap contributors".
    # Round 4 re-judged this on the corrected harness (the earlier -0.133s reading was taken
    # without serving's second stage). Everywhere at once now reads -0.367s and passes five of the
    # six checks, failing only LIRF at +0.131s against a +0.10s bar while LSZH runs -3.65s and
    # EDDM -1.41s. The round-2 rule pre-declared a per-airport version in exactly that case:
    # dropping Rome takes LIRF to +0.019s and the whole thing to -0.415s, every check passing.
    # Rome is already excluded from the second stage, so this is the same call made twice.
    # v31 adds LIRF, which v30 excluded. That exclusion followed the 2025 evaluation (+0.131s at
    # LIRF), but Rome had 0.0 per cent surface coverage on all 30 of those days, so an OSM position
    # there had no ground fixes to attach to and the harness could only measure noise. Validated
    # instead against 2026 labelled ARRIVALS: stands that ONLY OSM maps give
    # median -46s / IQR 101s against -30s / 79s for measured stands, and on 2026 DEPARTURES the
    # stand map goes 39 -> 187 and estimates 83 -> 150. The new rows are the same population as the
    # old ones (median taxi 371s vs 380s, taxi - model -499s vs -494s) and their offset matches the
    # fitted LIRF appear bias of -503s almost exactly.
    # ROME IS BACK OUT, 2026-09-18, ON THE BOARD ITSELF. v31 enabled it and scored 249.0678 against
    # v30's 249.0634 -- flat, marginally worse, agreeing with the 2025 harness's 0.048s. The
    # validation above was all correct and all beside the point: the positions ARE accurate and the
    # coverage gain IS real (743 -> 1,196 Rome rows corrected), but at Rome the appear estimator's
    # RMSE is 445s against the model's 367s. The estimator is WORSE THAN THE MODEL there, so more
    # coverage of it buys nothing at any weight. Coverage was never Rome's binding constraint.
    osm_fill: bool = True
    osm_airports: tuple = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM")
    dwell_fill: bool = True
    dwell_airports: tuple = ("EDDF", "LEMD")   # empty = every airport
    dwell_park_kt: float = 1.0
    dwell_spread_m: float = 30.0
    dwell_min_s: float = 900.0
    dwell_min_fixes: int = 3
    # No-plan rows carry no NM callsign, so they are matched on lift-off time + aircraft type
    # and any row with a competing candidate is dropped: 2,568 of 5,290 matched in 2026. With
    # noplan_callsign they are first matched on the callsign DERIVED from FLIGHT_mvt
    # (flight_callsign_maps), the way plan rows are. Measured on the 30 days: 55 more no-plan
    # corrections for -0.03s, against a -0.10s rule. Off.
    noplan_callsign: bool = False
    # Airports whose no-plan rows keep the old matching. LIRF: half its no-plan BLOCK stamps are
    # copies of the schedule, so the truth there is not the taxi ADS-B measures, and 2025 has no
    # Rome ground coverage to test it on -- the Rome no-plan gate owns those rows.
    noplan_callsign_skip: tuple = ("LIRF",)
    # A LightGBM increment on top of the hard floor, trained on the calibration rows that have a
    # floor (lb - p, lb, p, airport, hour, NM taxi - p -> y - p) and CLIPPED AT ZERO: it may raise
    # a prediction, never lower one. The hard floor is a step that does nothing where the
    # prediction sits just above it, though the same evidence applies. 30 days, confirmed under a
    # rule fixed beforehand: all -0.98s, core -1.16s, 30 of 30 days better, no airport worse,
    # -0.70s after deleting the 50 rows it helps most.
    floor_stack: bool = True


# field elevations (ft), for slow_low_ground
ELEVATION_FT = {"EDDF": 364, "EDDM": 1487, "EGLL": 83, "EHAM": -11, "LEBL": 12, "LEMD": 2001, "LFPG": 392,
                "LIRF": 13, "LSZH": 1416, "LTFM": 325}


def metres(lat, lon, lat0, lon0):
    dy = (lat - lat0) * 110_540.0
    dx = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    return np.sqrt(dx * dx + dy * dy)


def runs_and_tracks(day: str, g: Gates, points_dir: Path = POINTS_DIR):
    """Lift-off runs, touchdowns, and every fix per aircraft for one day."""
    p = pl.read_parquet(points_dir / f"{day}.parquet").filter(C("icao").is_not_null()).sort(["apt", "icao", "t"])
    if g.slow_low_ground:
        elev = C("apt").replace_strict(ELEVATION_FT, default=None, return_dtype=pl.Float64)
        slow_low = ((C("alt") != -1) & ((C("alt").cast(pl.Float64) - elev).abs() <= g.slow_low_band_ft)
                    & (C("gs") <= g.slow_low_gs_kt)).fill_null(False)
        p = p.with_columns(((C("alt") == -1) | slow_low).alias("ground"))
    else:
        p = p.with_columns((C("alt") == -1).alias("ground"))
    p = p.with_columns(
        C("t").shift(1).over(["apt", "icao"]).alias("prev_t"),
        C("ground").shift(1).over(["apt", "icao"]).alias("prev_ground"),
        C("t").shift(-1).over(["apt", "icao"]).alias("next_t"),
        C("ground").shift(-1).over(["apt", "icao"]).alias("next_ground"),
        C("callsign").forward_fill().over(["apt", "icao"]).alias("cs_ff"),
        C("actype").forward_fill().over(["apt", "icao"]).alias("type_ff"),
    )
    gr = p.filter(C("ground")).with_columns(
        (C("prev_t").is_null() | ~C("prev_ground").fill_null(False) | ((C("t") - C("prev_t")) > g.gap_s))
        .alias("run_start"))
    gr = gr.with_columns(C("run_start").cast(pl.Int32).cum_sum().over(["apt", "icao"]).alias("run"),
                         (C("t") - C("prev_t")).alias("dt_prev"))
    allruns = (gr.group_by(["apt", "icao", "run"])
                 .agg(C("t").min().alias("start_t"), C("t").max().alias("liftoff_t"),
                      C("lat").sort_by("t").first().alias("start_lat"), C("lon").sort_by("t").first().alias("start_lon"),
                      C("gs").sort_by("t").first().alias("start_gs"),
                      C("dt_prev").filter(~C("run_start")).max().alias("max_gap"),
                      C("cs_ff").drop_nulls().last().alias("callsign"),
                      C("cs_ff").drop_nulls().first().alias("callsign_first"),
                      C("type_ff").drop_nulls().last().alias("adsb_type"),
                      C("prev_t").first().alias("before_t"), C("prev_ground").first().alias("before_ground"),
                      C("next_t").last().alias("after_t"), C("next_ground").last().alias("after_ground"))
                 .with_columns(C("callsign").str.strip_chars().str.to_uppercase(),
                               C("callsign_first").str.strip_chars().str.to_uppercase()))
    runs = allruns.filter((C("after_ground") == False) & ((C("after_t") - C("liftoff_t")) <= g.liftoff_s))  # noqa: E712
    if g.airborne_callsign:
        air = (p.filter(~C("ground") & C("callsign").is_not_null())
                .select("apt", "icao", C("t").alias("t_air"),
                        C("callsign").str.strip_chars().str.to_uppercase().alias("callsign_air"))
                .sort("t_air"))
        runs = (runs.sort("liftoff_t")
                    .join_asof(air, left_on="liftoff_t", right_on="t_air", by=["apt", "icao"], strategy="forward",
                               tolerance=g.airborne_cs_window_s)
                    .drop("t_air"))
    # touchdowns: a ground run whose previous fix was airborne moments before -- an arrival
    touch = (allruns.filter((C("before_ground") == False) & ((C("start_t") - C("before_t")) <= g.liftoff_s))  # noqa: E712
                    .select("apt", "icao", C("start_t").alias("touchdown_t"), C("callsign_first").alias("callsign")))
    # every fix (ground AND low airborne) per aircraft, so 'next fix after the stand fix'
    # sees the whole track, not only ground fixes
    tracks = {}
    for (apt, icao), s in p.select("apt", "icao", "t", "lat", "lon", "gs").partition_by(["apt", "icao"], as_dict=True).items():
        tracks[(apt, icao)] = (s["t"].to_numpy(), s["lat"].to_numpy().astype(float), s["lon"].to_numpy().astype(float),
                               s["gs"].fill_null(0.0).to_numpy().astype(float))
    return runs, touch, tracks


def match_arrivals(arrivals_day: pl.DataFrame, touch: pl.DataFrame, g: Gates) -> pl.DataFrame:
    """Arrivals (apt, callsign, land_t, ...) to touchdowns by callsign and time."""
    return (arrivals_day.join(touch, on=["apt", "callsign"], how="inner")
                        .with_columns((C("touchdown_t") - C("land_t")).abs().alias("dm"))
                        .filter(C("dm") <= g.match_s).sort("dm")
                        .unique(subset=["apt", "callsign", "land_t"], keep="first"))


FLIGHT_RE = r"^([A-Z0-9]{2}[A-Z]?)(\d+)[A-Z]?$"


def flight_callsign_maps(frames, min_share: float = 0.8):
    """Learn FLIGHT_mvt -> transmitted callsign from PLAN rows. Label-free.

    Plan rows carry both the airport's flight designator (FLIGHT_mvt, 'LH400') and the NM
    callsign the aircraft transmits ('DLH400', but also 'DLH8UL' -- Lufthansa gives many
    flights an alphanumeric callsign). Two maps, each kept only where one answer holds at
    least min_share of a key's rows:
        exact   'LH400'    -> 'DLH400'     (89% of plan rows; reproduces the callsign 99.7%)
        prefix  'LH'       -> 'DLH'        (a fallback: prefix + digits is right 51% of the
                                            time, so it only ever matches when a run with
                                            exactly that callsign is actually there)
    """
    plan = (pl.concat([f.select("plan", "flight", "callsign") for f in frames], how="vertical_relaxed")
              .filter(C("plan") & C("flight").is_not_null() & C("callsign").is_not_null()))
    em = (plan.group_by("flight", "callsign").len()
              .with_columns((C("len") / C("len").sum().over("flight")).alias("share"))
              .sort("len", descending=True).unique(subset="flight", keep="first").filter(C("share") >= min_share))
    pm = (plan.with_columns(C("flight").str.extract(FLIGHT_RE, 1).alias("fp"), C("callsign").str.slice(0, 3).alias("cp"))
              .filter(C("fp").is_not_null()).group_by("fp", "cp").len()
              .with_columns((C("len") / C("len").sum().over("fp")).alias("share"))
              .sort("len", descending=True).unique(subset="fp", keep="first").filter(C("share") >= min_share))
    return (em.select("flight", C("callsign").alias("cs_exact")), pm.select("fp", "cp"))


def add_derived_callsign(frame: pl.DataFrame, maps) -> pl.DataFrame:
    """Add 'cs2': the callsign FLIGHT_mvt implies, exact map first, prefix + digits second."""
    exact, prefix = maps
    return (frame.join(exact, on="flight", how="left")
                 .with_columns(C("flight").str.extract(FLIGHT_RE, 1).alias("_fp"),
                               C("flight").str.extract(FLIGHT_RE, 2).alias("_fn"))
                 .join(prefix, left_on="_fp", right_on="fp", how="left")
                 .with_columns(pl.coalesce(C("cs_exact"),
                                           pl.when(C("cp").is_not_null() & C("_fn").is_not_null())
                                             .then(C("cp") + C("_fn").str.strip_chars_start("0"))
                                             .otherwise(None)).alias("cs2"))
                 .drop("cs_exact", "_fp", "_fn", "cp"))


def match_departures(deps_day: pl.DataFrame, runs: pl.DataFrame, g: Gates):
    """Departures (MVT_ID_mvt, apt, plan, callsign, type, mvt_t, ...) to lift-off runs.

    Plan rows by callsign within match_s; no-plan rows by time within noplan_match_s and
    aircraft type, unique both ways, never a run a plan row already claimed.
    Returns (plan matches, no-plan matches, both stacked).
    """
    d = deps_day.filter(C("plan") & C("callsign").is_not_null())
    if g.airborne_callsign and "callsign_air" in runs.columns:
        # every run keyed by its ground callsign, and again by its airborne callsign when that differs
        long = pl.concat([
            runs.with_columns(C("callsign").alias("cs_key")),
            runs.filter(C("callsign_air").is_not_null() & (C("callsign_air") != C("callsign").fill_null("")))
                .with_columns(C("callsign_air").alias("cs_key")),
        ], how="vertical_relaxed").drop("callsign")
        cand = (d.with_columns(C("callsign").alias("cs_key")).join(long, on=["apt", "cs_key"], how="inner").drop("cs_key")
                 .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dmatch"))
                 .filter(C("dmatch") <= g.match_s).sort("dmatch")
                 .unique(subset="MVT_ID_mvt", keep="first")
                 .unique(subset=["apt", "icao", "liftoff_t"], keep="first"))
    else:
        cand = (d.join(runs, on=["apt", "callsign"], how="inner")
                 .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dmatch"))
                 .filter(C("dmatch") <= g.match_s).sort("dmatch").unique(subset="MVT_ID_mvt", keep="first"))
    free = runs.join(cand.select("apt", "icao", "liftoff_t"), on=["apt", "icao", "liftoff_t"], how="anti")
    # No-plan rows first by the callsign DERIVED from FLIGHT_mvt (flight_callsign_maps): they
    # carry no NM callsign, and time + type alone drops every row that has a competing candidate.
    ncs = None
    if g.noplan_callsign and "cs2" in deps_day.columns:
        dcs = deps_day.filter(~C("plan") & C("cs2").is_not_null())
        if "callsign_air" in free.columns:
            keys = pl.concat([
                free.with_columns(C("callsign").alias("cs_key")),
                free.filter(C("callsign_air").is_not_null() & (C("callsign_air") != C("callsign").fill_null("")))
                    .with_columns(C("callsign_air").alias("cs_key")),
            ], how="vertical_relaxed")
        else:
            keys = free.with_columns(C("callsign").alias("cs_key"))
        ncs = (dcs.drop("callsign").with_columns(C("cs2").alias("cs_key"))
                  .join(keys, on=["apt", "cs_key"], how="inner").drop("cs_key")
                  .filter((C("liftoff_t") - C("mvt_t")).abs() <= g.match_s))
        once = ncs.group_by("MVT_ID_mvt").len().filter(C("len") == 1).select("MVT_ID_mvt")
        once_run = ncs.group_by(["apt", "icao", "liftoff_t"]).len().filter(C("len") == 1).select("apt", "icao", "liftoff_t")
        ncs = (ncs.join(once, on="MVT_ID_mvt", how="semi").join(once_run, on=["apt", "icao", "liftoff_t"], how="semi")
                  .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dmatch")))
        if g.noplan_callsign_skip:
            ncs = ncs.filter(~C("apt").is_in(list(g.noplan_callsign_skip)))
        free = free.join(ncs.select("apt", "icao", "liftoff_t"), on=["apt", "icao", "liftoff_t"], how="anti")
        deps_day = deps_day.join(ncs.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
    dn = deps_day.filter(~C("plan") & C("type").is_not_null())
    nc = (dn.drop("callsign").join(free, on="apt", how="inner")
            .filter(((C("liftoff_t") - C("mvt_t")).abs() <= g.noplan_match_s) & (C("adsb_type") == C("type"))))
    once_row = nc.group_by("MVT_ID_mvt").len().filter(C("len") == 1).select("MVT_ID_mvt")
    once_run = nc.group_by(["apt", "icao", "liftoff_t"]).len().filter(C("len") == 1).select("apt", "icao", "liftoff_t")
    nc = (nc.join(once_row, on="MVT_ID_mvt", how="semi").join(once_run, on=["apt", "icao", "liftoff_t"], how="semi")
            .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dmatch")))
    if ncs is not None and ncs.height:
        nc = pl.concat([ncs.select(nc.columns), nc], how="vertical_relaxed")
    both = pl.concat([cand, nc.select(cand.columns)], how="vertical_relaxed")
    return cand, nc, both


def fix_near(tracks: dict, apt, icao, when: float, g: Gates):
    tr = tracks.get((apt, icao))
    if tr is None:
        return None
    t, la, lo, _ = tr
    k = int(np.argmin(np.abs(t - when)))
    return (la[k], lo[k]) if abs(t[k] - when) <= g.fix_tol_s else None


def dwell_sample(r: dict, tracks: dict, g: Gates):
    """Where a matched DEPARTURE stood before it taxied, with no block time involved.

    Stand positions come from block times today, which only arrivals carry at serve time, so a
    stand nobody arrives at through a tracked aircraft has no position and every departure off it
    is lost (Madrid 19,394 matched departures, Frankfurt 7,008). A departure that was transmitting
    while parked says where its own stand is: take the longest run of consecutive fixes at
    <= dwell_park_kt within dwell_spread_m of its first fix, inside the lookback window before
    lift-off, and require it to last dwell_min_s. Returns (lat, lon) or None.
    """
    tr = tracks.get((r["apt"], r["icao"]))
    if tr is None:
        return None
    t, la, lo, gs = tr
    idx = np.flatnonzero((t >= r["liftoff_t"] - g.lookback_s) & (t <= r["liftoff_t"]))
    if len(idx) < g.dwell_min_fixes:
        return None
    best, cur = None, None
    for i in idx:
        if gs[i] <= g.dwell_park_kt and (cur is None or metres(la[i], lo[i], la[cur[0]], lo[cur[0]]) <= g.dwell_spread_m):
            cur = [i, i] if cur is None else [cur[0], i]
        elif gs[i] <= g.dwell_park_kt:
            cur = [i, i]
        else:
            cur = None
        if cur is not None and (best is None or (t[cur[1]] - t[cur[0]]) > (t[best[1]] - t[best[0]])):
            best = list(cur)
    if best is None or (t[best[1]] - t[best[0]]) < g.dwell_min_s:
        return None
    sel = np.arange(best[0], best[1] + 1)
    return float(np.median(la[sel])), float(np.median(lo[sel]))


def stand_samples(tracks: dict, apt, icao, when: float, g: Gates, arrival: bool) -> list:
    """Stand-position samples [(lat, lon, quality), ...] at a block time.

    'nearest'      the fix nearest the block time, if within fix_tol_s
    'parked'       the median of slow fixes just after an arrival's on-block / just before a
                   departure's off-block, else the nearest fix
    'parked_fill'  both, as separate samples; stand_map uses the parked ones only for stands
                   the nearest ones cannot map
    """
    tr = tracks.get((apt, icao))
    if tr is None:
        return []
    t, la, lo, gs = tr
    mode = g.stand_samples if (not g.parked_airports or apt in g.parked_airports) else "nearest"
    out = []
    parked = None
    if mode in ("parked", "parked_fill"):
        if arrival:
            w = (t >= when) & (t <= when + g.parked_window_s)
        else:
            w = (t <= when) & (t >= when - g.parked_window_s)
        idx = np.flatnonzero(w & (gs <= g.parked_gs_kt))
        if len(idx) >= g.parked_min_fixes:
            parked = (float(np.median(la[idx])), float(np.median(lo[idx])), "parked")
    k = int(np.argmin(np.abs(t - when)))
    nearest = (la[k], lo[k], "nearest") if abs(t[k] - when) <= g.fix_tol_s else None
    if mode == "parked":
        out = [parked] if parked else ([nearest] if nearest else [])
    elif mode == "parked_fill":
        out = [s for s in (nearest, parked) if s]
    else:
        out = [nearest] if nearest else []
    return out


def _centroids(samples: pl.DataFrame, g: Gates) -> dict:
    cent = (samples.group_by(["apt", "stand"])
                   .agg(C("lat").median().alias("clat"), C("lon").median().alias("clon"), pl.len().alias("k"),
                        C("lat"), C("lon"))
                   .filter(C("k") >= 2))
    cmap = {}
    for a, s, clat, clon, k, lats, lons in cent.select("apt", "stand", "clat", "clon", "k", "lat", "lon").iter_rows():
        spread = float(np.median(metres(np.asarray(lats, float), np.asarray(lons, float), clat, clon)))
        if spread <= g.stand_spread_m:
            cmap[(a, s)] = (clat, clon)
    return cmap


def stand_map(samples: pl.DataFrame, g: Gates) -> dict:
    """(apt, stand) -> (lat, lon) from samples (apt, stand, lat, lon[, quality]) that agree within the spread.

    'parked' mode maps stands from parked samples first and falls back to all samples;
    'parked_fill' maps stands from nearest-fix samples first and uses parked samples only for
    stands those cannot map. Without a quality column every sample counts alike.
    """
    dwell = samples.filter(C("quality") == "dwell") if "quality" in samples.columns else samples.head(0)
    block = samples.filter(C("quality") != "dwell") if "quality" in samples.columns else samples
    if "quality" not in samples.columns or g.stand_samples == "nearest":
        cmap = _centroids(block, g)
    elif g.stand_samples == "parked_fill":
        cmap = _centroids(block.filter(C("quality") == "nearest"), g)
        for key, pos in _centroids(block.filter(C("quality") == "parked"), g).items():
            cmap.setdefault(key, pos)
    else:
        cmap = _centroids(block.filter(C("quality") == "parked"), g)
        for key, pos in _centroids(block, g).items():
            cmap.setdefault(key, pos)
    # Departure dwell positions map only what block times could not (dwell_fill).
    if g.dwell_fill and dwell.height:
        if g.dwell_airports:
            dwell = dwell.filter(C("apt").is_in(list(g.dwell_airports)))
        for key, pos in _centroids(dwell, g).items():
            cmap.setdefault(key, pos)
    # OpenStreetMap maps only what neither of those could.
    if g.osm_fill:
        for key, pos in osm_stand_map(g).items():
            cmap.setdefault(key, pos)
    return cmap


OSM_STANDS = Path("data/external/osm/stands.parquet")
_OSM_CACHE: dict = {}


def osm_stand_map(g: Gates) -> dict:
    """(apt, stand) -> (lat, lon) from OpenStreetMap, median position per stand name."""
    if not OSM_STANDS.exists():
        return {}
    if "map" not in _OSM_CACHE:
        f = (pl.read_parquet(OSM_STANDS)
             .with_columns(C("ref").str.strip_chars().str.to_uppercase().alias("ref"))
             .filter(C("ref").is_not_null() & (C("ref") != ""))
             .group_by("apt", "ref").agg(C("lat").median(), C("lon").median()))
        _OSM_CACHE["map"] = {(a, r): (la, lo) for a, r, la, lo in f.iter_rows()}
    m = _OSM_CACHE["map"]
    if g.osm_airports:
        keep = set(g.osm_airports)
        return {k: v for k, v in m.items() if k[0] in keep}
    return m


_STAND_CACHE: list = [None, {}]  # [the stand map object the arrays belong to, {apt: (lats, lons)}]


def _airport_stands(cmap: dict, apt):
    """Arrays of every mapped stand position at an airport, for the stand map passed in.

    Cached for the most recent map only, checked by identity rather than id(): the evaluation
    rebuilds the map for every held-out day, and a freed dict's id can be reused by the next.
    """
    if _STAND_CACHE[0] is not cmap:
        _STAND_CACHE[0], _STAND_CACHE[1] = cmap, {}
    arrays = _STAND_CACHE[1]
    if apt not in arrays:
        pos = [v for (a, _), v in cmap.items() if a == apt]
        arrays[apt] = (np.array([p[0] for p in pos], float), np.array([p[1] for p in pos], float))
    return arrays[apt]


# What the track says about how much to trust THIS row's estimate. A code review measured that
# a day-out model on these seven cuts the post-bias residual from 130s to 118s -- 17.9% of its
# variance -- while neither the inverse-variance blend (one weight per airport and kind) nor the
# second stage can see any of it. The hard gates stand_next_s and dwell_window_s approximate two
# of them with a yes/no.
FLOOR_QUALITY = ("fq_has_stop", "fq_nstops", "fq_accel_s", "fq_first_gs", "fq_run_to_fast_s", "fq_stand_m")
QUALITY = ("q_dist_m", "q_gs_kt", "q_fixes", "q_span_s", "q_next_gap_s", "q_max_gs_kt", "q_since_slow_s")


def estimate(r: dict, cmap: dict, tracks: dict, g: Gates, with_quality: bool = False):
    """(kind, taxi[, quality]) for one matched departure; kind is 'stand', 'tight', 'appear',
    'start' or None. quality is a dict over QUALITY describing this row's evidence."""
    kind, est = None, None
    qual = dict.fromkeys(QUALITY, float("nan"))
    c = cmap.get((r["apt"], r["stand"]))
    tr = tracks.get((r["apt"], r["icao"]))
    if c is not None and tr is not None:
        t, la, lo, gs = tr
        win = np.flatnonzero((t <= r["liftoff_t"]) & (t >= r["liftoff_t"] - g.lookback_s))
        if len(win):
            near = win[metres(la[win], lo[win], c[0], c[1]) <= g.radius_m]
            if len(near):
                j = near.max()
                # Dwell: the aircraft was slow inside the circle shortly before leaving it.
                # A taxi-by past a mapped stand is continuous but never slow there.
                dwell = near[(t[near] >= t[j] - g.dwell_window_s) & (gs[near] <= g.dwell_gs_kt)]
                if len(dwell) and j + 1 < len(t) and t[j + 1] - t[j] <= g.stand_next_s:
                    anchor = j
                    if g.stand_anchor == "last_slow":
                        stopped = near[gs[near] <= g.stand_anchor_kt]
                        if len(stopped):
                            anchor = int(stopped.max())
                    kind, est = "stand", float(t[anchor])
                    if with_quality:
                        d_near = metres(la[near], lo[near], c[0], c[1])
                        slow_in = near[gs[near] <= 1.0]
                        qual.update(
                            q_dist_m=float(metres(la[anchor], lo[anchor], c[0], c[1])),
                            q_gs_kt=float(gs[anchor]),
                            q_fixes=float(len(near)),
                            q_span_s=float(t[near.max()] - t[near.min()]),
                            q_next_gap_s=float(t[j + 1] - t[j]) if j + 1 < len(t) else float("nan"),
                            q_max_gs_kt=float(gs[near].max()),
                            q_since_slow_s=float(t[anchor] - t[slow_in.max()]) if len(slow_in) else float("nan"),
                        )
                        del d_near
                    if g.tight_anchor_m > 0 and (not g.tight_airports or r["apt"] in g.tight_airports):
                        tight = near[metres(la[near], lo[near], c[0], c[1]) <= g.tight_anchor_m]
                        if len(tight):
                            k2 = int(tight.max())
                            if k2 + 1 < len(t) and t[k2 + 1] - t[k2] <= g.stand_next_s:
                                kind, est = "tight", float(t[k2])
                                if with_quality:
                                    qual.update(q_dist_m=float(metres(la[k2], lo[k2], c[0], c[1])),
                                                q_gs_kt=float(gs[k2]),
                                                q_next_gap_s=float(t[k2 + 1] - t[k2]))
        if kind is None and r["max_gap"] is not None and r["max_gap"] <= g.run_gap_s:
            d_start = float(metres(np.array([r["start_lat"]]), np.array([r["start_lon"]]), c[0], c[1])[0])
            if d_start <= g.appear_m:
                kind, est = "appear", float(r["start_t"])
                if with_quality:
                    qual.update(q_dist_m=d_start, q_gs_kt=float(r["start_gs"]) if r.get("start_gs") is not None else float("nan"),
                                q_next_gap_s=float(r["max_gap"]))
    if (kind is None and g.start_estimator and r.get("start_gs") is not None and r["max_gap"] is not None
            and r["max_gap"] <= g.run_gap_s and r["start_gs"] <= g.start_gs_kt):
        slat, slon = _airport_stands(cmap, r["apt"])
        if len(slat) and float(metres(slat, slon, r["start_lat"], r["start_lon"]).min()) <= g.radius_m:
            kind, est = "start", float(r["start_t"])
    taxi = None if est is None else r["mvt_t"] - est
    if taxi is not None and not (0 <= taxi <= g.max_taxi_s):
        kind, taxi = None, None
    return (kind, taxi, qual) if with_quality else (kind, taxi)


def departure_floor(r: dict, tracks: dict, g: Gates, cmap: dict | None = None,
                    with_quality: bool = False):
    """Floor on the taxi time of one matched departure: MVT - t(first fix at >= floor_kt of the
    departure part of its lift-off run), or None.

    An aircraft moving at taxi speed toward lift-off has already left its stand, so its off-block
    came first and y = MVT - BLOCK >= MVT - t(fix) -- whether or not it was ever seen at the stand.
    A run holds the inbound taxi too when the aircraft never fell silent for gap_s: the departure
    part starts after the run's last parked stop, and a run that began at a touchdown with no
    such stop cannot be separated, so it gets no floor.
    """
    tr = tracks.get((r["apt"], r["icao"]))
    if tr is None:
        return (None, {k: float("nan") for k in FLOOR_QUALITY}) if with_quality else None
    t, lat, lon, gs = tr
    idx = np.flatnonzero((t >= r["start_t"]) & (t <= r["liftoff_t"]))
    if not len(idx):
        return (None, {k: float("nan") for k in FLOOR_QUALITY}) if with_quality else None
    slow = gs[idx] <= g.floor_park_kt
    stops, i, n = [], 0, len(idx)
    while i < n:
        if slow[i]:
            j = i
            while j + 1 < n and slow[j + 1]:
                j += 1
            if t[idx[j]] - t[idx[i]] >= g.floor_turn_s:
                stops.append((i, j))
            i = j + 1
        else:
            i += 1
    begin = stops[-1][1] if stops else None          # the last qualifying stop: the original rule
    if g.floor_stand_anchor and stops and cmap is not None:
        centre = cmap.get((r["apt"], r.get("stand")))
        if centre is not None:
            at_stand = [s for s in stops
                        if metres(float(np.median(lat[idx[s[0]:s[1] + 1]])), float(np.median(lon[idx[s[0]:s[1] + 1]])),
                                  centre[0], centre[1]) <= g.radius_m]
            if at_stand:
                begin = at_stand[-1][1]              # the last stop AT the stand, before any de-icing pad
    from_stop = begin is not None
    if begin is None:
        touch_run = (r.get("before_ground") is False and r.get("before_t") is not None
                     and r["start_t"] - r["before_t"] <= g.liftoff_s)
        if touch_run:
            return (None, {k: float("nan") for k in FLOOR_QUALITY}) if with_quality else None
        begin = 0
    part = idx[begin:]
    moving = part[gs[part] >= g.floor_kt]
    if not len(moving):
        return (None, {k: float("nan") for k in FLOOR_QUALITY}) if with_quality else None
    lb = r["mvt_t"] - float(t[moving[0]])
    lb = lb if 0 <= lb <= g.max_taxi_s else None
    if not with_quality:
        return lb
    fast_t = float(t[moving[0]])
    fq = {"fq_has_stop": 1.0 if from_stop else 0.0, "fq_nstops": float(len(stops)),
          "fq_accel_s": fast_t - float(t[idx[begin]]), "fq_first_gs": float(gs[idx[0]]),
          "fq_run_to_fast_s": fast_t - float(t[idx[0]]), "fq_stand_m": float("nan")}
    if from_stop and cmap is not None:
        centre = cmap.get((r["apt"], r.get("stand")))
        anchor_stop = next((s for s in stops if s[1] == begin), None)
        if centre is not None and anchor_stop is not None:
            a, b = anchor_stop
            fq["fq_stand_m"] = float(metres(float(np.median(lat[idx[a:b + 1]])), float(np.median(lon[idx[a:b + 1]])),
                                            centre[0], centre[1]))
    return lb, fq


def wvar(x) -> float:
    """Variance after winsorising at the 1st and 99th percentiles."""
    if len(x) < 5:
        return float("nan")
    lo, hi = np.percentile(x, [1, 99])
    return float(np.var(np.clip(x, lo, hi)))
