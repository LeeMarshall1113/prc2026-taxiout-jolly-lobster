# STAGECAP SERVING COPY of archive/agent_merged_apply_taxinm.py (2026-09-25, docs/LAB-NOTES.md): adds ONLY
# --stack-trees/--stack-leaves/--stack-minchild/--stack-lr and --floor-stack-trees/--floor-stack-leaves/--floor-stack-minchild/
# --floor-stack-lr for the main and floor stack LightGBMs (defaults = the hard-coded 300/15/50/0.05 and 250/15/100/0.05; depth
# rule as archive/agent_taxinm_stage1_eval.py: leaves > 15 -> max_depth -1, else 4). At defaults this is the original.
#
# TAXINM SERVING COPY of agent_merged_apply_lfpg.py (2026-09-23, LEVER 1 GENERALIZED, lead
# instruction, URGENT -- ships as v105): adds --floor-stack-taxinm-apts <comma list> (default empty), the
# SAME taxi-phase NM-callsign mechanism as --floor-stack-lfpgnm (kept, unmodified, still present in this
# file) generalized to any of the 9 covered airports via archive/agent_taxinm_features.py -- the serving
# port of archive/agent_taxinm_stage1_eval.py, which Modal judged: unspliced TX-vs-NP1 -1.102s (fails only
# EHAM +0.485); per-airport splices PASS at EDDF, EGLL, LEMD, LSZH (+ LFPG, relaxed clause); the COMBINED
# splice over S={EDDF,EGLL,LEMD,LSZH,LFPG} PASSES every clause (-0.909 all rows, minus-top-50 -0.794,
# Jan-Mar -0.772, Jul-Sep -0.968, 227/238 days better). Ship path: build v105 with ALL 9 airports enabled
# (matching the judged configuration's shared floor-stack pool exactly), then splice ONLY S's rows into
# v103's raw output (data/interim/v103_raw.parquet) with archive/splice_apt_parquet.py -- see
# archive/build_v105_taxinm.sh. Do NOT edit agent_merged_apply_lfpg.py -- v104 ships from it.
# With --floor-stack-taxinm-apts empty AND --floor-stack-lfpgnm off, this file must reproduce
# archive/agent_merged_apply_np.py exactly (verified below). docs/LAB-NOTES.md.
#
# LFPG SERVING COPY of archive/agent_merged_apply_np.py (2026-09-23, LEVER 1, team instruction): adds
# --floor-stack-lfpgnm on|off (default off) -- LFPG NM-callsign taxi-phase matching with NO lift-off requirement
# (archive/agent_lfpgnewstand_features.py, the mechanism archive/agent_lfpgnewstand_stage1_eval.py validated: Modal
# splice PASS, -0.147 all rows, LFPG -1.09, minus-top-50 -0.056, Jul-Sep -0.158, Jan-Mar +0.000 -- the SPLICE result,
# not the unspliced arm, which FAILS on other airports through the shared floor-stack training pool; this is why v104
# ships as a splice, LFPG rows only, same as this note's design). With --floor-stack-lfpgnm off this file must
# reproduce agent_merged_apply_np.py exactly (verified: --selftest, flag off vs the unmodified _np.py file, below).
# Do NOT edit archive/agent_merged_apply_np.py -- v103 is building with it. docs/LAB-NOTES.md.
#
# NP1 SERVING COPY of archive/agent_merged_apply.py (2026-09-23): matching from archive/agent_noplan_adsb_match.py
# (air_cs_all + no-plan derived-callsign matching) and a --noplan-callsign flag. Gated on Modal 2026-09-23: NP1 vs AIRCS
# 238 days, all four clauses PASS (-0.102 s, no airport worse, Jan-Mar -0.102, Jul-Sep -0.120). With --noplan-callsign
# off this reproduces agent_merged_apply.py (same matching code for plan rows).
"""experiment MERGED serving script (2026-09-23) = archive/agent_frontier_apply.py + the postchain airborne-callsign matching
taken from archive/agent_postchain_adsb_match_aircs.py (Gates.air_cs_all / air_cs_skip; gated 177 days, -0.306 s, LIRF
excluded). --air-cs-all on|off (default off) and --air-cs-skip (default LIRF). All new flags default OFF: with
--air-cs-all off this script is agent_frontier_apply.py; with it on and the frontier knobs off it is
archive/agent_postchain_adsb_apply_aircs.py (quality) plus quality_anch support. docs/LAB-NOTES.md.

experiment frontier COPY of archive/agent_adsbq_apply.py (2026-09-23) adding three knobs, all OFF by default (with none of
them it must reproduce agent_adsbq_apply.py): --kind-off APT:KIND[,..] (treat that airport's estimator kind as absent,
e.g. LIRF:appear), --calib-weight-recent W (calibration rows in --recent-months get weight W in both LightGBM stacks) and
--air-cs-all on (plan rows match on any airborne callsign; the postchain experiment's EDDF fix). docs/LAB-NOTES.md.

experiment adsbq COPY of archive/adsb_apply.py (2026-09-23) adding --stack-feats quality_anch: the second stage also
sees the extra track anchors of archive/agent_adsbq_anchors.py (interpolated tight-circle crossings, run start,
visit length, time-from-last-fix). Everything else is byte-for-byte the live script's behaviour; with
--stack-feats quality this copy must reproduce adsb_apply.py exactly. docs/LAB-NOTES.md.

Blend ADS-B off-block estimates into a submission. Same code path the Stage 1 evaluation validated.

    .venv/Scripts/python.exe archive/adsb_apply.py --in submissions/X.parquet --out submissions/Y.parquet \
        [--scope both|plan] [--no-guard] [--serve-days 2026-01-01,...] [--calib-days 2025-01-06,...]

CALIBRATION (2025 Stage 1 days, all of them, leave-one-day-out stand maps exactly as in
archive/adsb_stage1_eval.py): per (airport, estimator) the median bias and winsorised error
variance of the estimate on PLAN rows, and the model's winsorised error variance -- on plan
rows from the fold predictions (--calib-preds), on no-plan rows pooled as in the evaluation.

SERVING (days in --serve-days, default every 2026 day extracted): departures come from
ranking.parquet (the target is withheld there; nothing here reads it), matched by callsign
(plan) or lift-off time + aircraft type (no-plan). Stand positions come from 2026 arrivals at
their labelled on-block time plus every 2025 Stage 1 sample. A matched row with a valid
estimate becomes w * max(taxi - bias, 0) + (1 - w) * prediction, w = v_model / (v_model +
v_estimate); with the guard (default, the variant the evaluation favoured) a correction that
would move the prediction by more than 1,800s is refused.

--selftest DAY runs the serving path on one 2025 Stage 1 day instead -- that day's departures
from the training file with their labels unused, calibration and stand maps from the OTHER
days only, and the fold prediction as the "submission" -- and prints the RMSE change. It
should land close to, not exactly on, the evaluation's leave-one-day-out figure for that day:
the evaluation calibrates on other days whose stand maps may use the held-out day's
departures, which serving never can, and it blends unrounded predictions.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import polars as pl

from archive.agent_noplan_adsb_match import (FLOOR_QUALITY, POINTS_DIR, QUALITY, Gates, add_derived_callsign, departure_floor, dwell_sample, estimate, flight_callsign_maps, match_arrivals,
                            match_departures, runs_and_tracks, stand_map, stand_samples, wvar)
from archive.agent_adsbq_anchors import ANCHOR_COLS, anchors, stack_columns
from archive.agent_lfpgnewstand_features import LFPGNM_COLS, lfpg_nm_match
from archive.agent_taxinm_features import TAXINM_COLS, taxinm_match

C = pl.col
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
GUARD_S = 1800.0
C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp")
ap.add_argument("--out")
ap.add_argument("--scope", choices=["both", "plan"], default="both")
ap.add_argument("--no-guard", action="store_true")
ap.add_argument("--serve-days", default="")
ap.add_argument("--calib-days", default="")
ap.add_argument("--calib-preds", default="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet")
ap.add_argument("--selftest", default="", help="a 2025 Stage 1 day to run the serving path on, with labels held out")
ap.add_argument("--stack", choices=["gbm", "none"], default="gbm",
                help="plan rows: a LightGBM second stage (features x - p, NM taxi - p, p, x, airport, kind, hour; "
                     "trained on the calibration rows) instead of the fixed inverse-variance weight. Stage 1 "
                     "leave-one-day-out against the guarded blend: -0.18s all / -0.24s core, 8 of 10 days better. "
                     "No-plan rows keep the inverse-variance blend.")
ap.add_argument("--stack-feats", choices=["basic", "lb", "quality", "quality_fq", "quality_anch"], default="quality",
                help="quality: give the second stage this row's track evidence (prc.adsb_match.QUALITY) plus stand "
                     "and aircraft type. 30 days, leave-one-day-out, against the same run without them: -1.04s all "
                     "/ -1.23s core, and -1.35s combined with the tight anchor. docs/LAB-NOTES.md.")
ap.add_argument("--stack-skip-airports", default="LIRF",
                help="airports whose plan rows keep the inverse-variance blend instead of the second stage. LIRF: "
                     "Rome is half the remaining error and the quality stack costs it 0.25s while gaining "
                     "everywhere else.")
ap.add_argument("--floor", choices=["on", "off"], default="on",
                help="after every correction, raise each matched departure to its taxi floor lb + q "
                     "(prc.adsb_match.departure_floor; q = the airport's median y - lb on calibration rows where "
                     "the model sat below lb, separately with / without a stand or appear estimate). docs/LAB-NOTES.md: "
                     "confirmed on the 20 Stage 1b days under a rule fixed beforehand, -2.07s all / -2.19s core on top "
                     "of the guarded blend, 20 of 20 days better, no airport worse.")
ap.add_argument("--osm-fill", choices=["on", "off"], default="on" if Gates().osm_fill else "off",
                help="fill stands the block-time samples and the dwell fill cannot map from OpenStreetMap. Round 7, "
                     "30 days, leave-one-day-out: -0.415s all, every check passing, with Rome excluded (Gates."
                     "osm_airports) because everywhere-at-once costs LIRF 0.13s against a 0.10s bar.")
ap.add_argument("--spill-tracks", choices=["on", "off"], default="off",
                help="hold one calibration day's tracks in memory and pickle the rest to scratch; the 177-day "
                     "calibration otherwise keeps every day at once (~8 GB). Result-identical, and it is what "
                     "lets a build share a busy machine.")
ap.add_argument("--floor-q", choices=["median", "zero"], default="median",
                help="zero: the hard floor raises to lb itself and the floor stack owns the whole offset "
                     "(30-day screen -0.274s against the shipped per-airport median offset)")
ap.add_argument("--floor-stack-lower", type=float, default=0.0,
                help="let the floor stack lower a prediction by at most this many seconds, never under lb (0 = raise only)")
ap.add_argument("--floor-stack-lower-skip", default="",
                help="airports where the floor stack stays raise-only even with --floor-stack-lower")
ap.add_argument("--floor-stack-feats", choices=["basic", "hasest", "quality"], default="hasest",
                help="hasest: give the floor stack a flag for whether the row also has a stand or appear estimate. "
                     "Round 7, 30 days, leave-one-day-out: -0.164s all, confirming round 4's -0.162s.")
ap.add_argument("--kind-off", default="", help="APT:KIND[,APT:KIND] estimator kinds treated as absent (frontier K8)")
ap.add_argument("--calib-weight-recent", type=float, default=1.0,
                help="weight of calibration rows in --recent-months in the main and floor stacks (frontier K9)")
ap.add_argument("--recent-months", default="10,11")
ap.add_argument("--floor-stack-anchors", choices=["on", "off"], default="off",
                help="the floor stack also sees the anchor columns (agent_adsbq_anchors.stack_columns relative to the "
                     "floor stack's p), appended after FLOOR_QUALITY; frontier section 3 A10")
ap.add_argument("--floor-stack-anch-skip", default="",
                help="airports whose rows carry NaN anchor columns in the floor stack (calibration and serving), e.g. LIRF")
ap.add_argument("--air-cs-all", choices=["on", "off"], default="off",
                help="plan rows may match a run on ANY callsign its aircraft transmits airborne within the window "
                     "(postchain section 3, 177-day gate PASS -0.306 s); implemented in agent_postchain_adsb_match_aircs")
ap.add_argument("--air-cs-skip", default="LIRF", help="airports that keep the shipped callsign keys with --air-cs-all on")
ap.add_argument("--noplan-callsign", choices=["on", "off"], default="off",
                help="no-plan rows first match on the callsign derived from FLIGHT_mvt, keyed on every airborne callsign (IDEA-NP1)")
ap.add_argument("--floor-stack-lfpgnm", choices=["on", "off"], default="off",
                help="LEVER 1: LFPG NM-callsign taxi-phase matching, NO lift-off requirement "
                     "(archive/agent_lfpgnewstand_features.py). Appends LFPGNM_COLS to the floor stack (LFPG rows "
                     "already matched get them as extra features) AND adds brand-new floor-stack rows for LFPG "
                     "departures production's lift-off-gated matcher misses entirely but this NM matcher finds ground "
                     "evidence for. The raw NM floor is NEVER a hard floor (it is unreliable enough to make LFPG "
                     "dramatically worse if used that way, measured in the stage-1 harness) -- it enters only as a "
                     "learned floor-stack feature; see LFPGNM_SYNTH below for the mechanics.")
ap.add_argument("--stack-trees", type=int, default=300)
ap.add_argument("--stack-leaves", type=int, default=15)
ap.add_argument("--stack-minchild", type=int, default=50)
ap.add_argument("--stack-lr", type=float, default=0.05)
ap.add_argument("--floor-stack-trees", type=int, default=250)
ap.add_argument("--floor-stack-leaves", type=int, default=15)
ap.add_argument("--floor-stack-minchild", type=int, default=100)
ap.add_argument("--floor-stack-lr", type=float, default=0.05)
ap.add_argument("--floor-stack-taxinm-apts", default="",
                help="LEVER 1 GENERALIZED: comma list of airports where taxi-phase NM-callsign matching with NO "
                     "lift-off requirement (archive/agent_taxinm_features.py) is enabled -- the same mechanism as "
                     "--floor-stack-lfpgnm, generalized to any airport, independent of it (both may be set; they "
                     "use separate columns/synthetic-row tracking). Default empty = no effect.")
args = ap.parse_args()
KIND_OFF = {tuple(s.split(":")) for s in args.kind_off.split(",") if s}
FS_ANCH_SKIP = {a for a in args.floor_stack_anch_skip.split(",") if a}
RECENT_MONTHS = {m.zfill(2) for m in args.recent_months.split(",") if m}


def _match(deps_day, runs, day):
    """match_departures from agent_postchain_adsb_match_aircs: with G.air_cs_all the runs from runs_and_tracks carry every
    airborne callsign (cs_all) and plan rows match on them (air_cs_skip airports excepted); off = prc.adsb_match exactly."""
    return match_departures(deps_day, runs, G)


def _kind_filter(apt_, kind_, taxi_, qual_):
    """--kind-off: the row keeps no estimate of that kind (its quality columns go NaN, as for a row with no kind)."""
    if kind_ is not None and (apt_, kind_) in KIND_OFF:
        return None, None, dict.fromkeys(qual_, float("nan"))
    return kind_, taxi_, qual_
G = Gates(osm_fill=args.osm_fill == "on", air_cs_all=args.air_cs_all == "on",
          air_cs_skip=tuple(a for a in args.air_cs_skip.split(",") if a), noplan_callsign=args.noplan_callsign == "on")
FLOOR_MIN_ROWS = 20
KINDS = ("stand", "appear", "start", "tight")  # must match scripts/adsb_stage1_eval.STACK_KINDS

on_disk = sorted(p.stem for p in POINTS_DIR.glob("*.parquet"))
calib_days = [d for d in on_disk if d.startswith("2025-")]
if args.calib_days:
    calib_days = [d for d in calib_days if d in set(args.calib_days.split(","))]
if args.selftest:
    if args.selftest not in on_disk or not args.selftest.startswith("2025-"):
        sys.exit(f"--selftest day {args.selftest} is not an extracted 2025 day")
    calib_days = [d for d in calib_days if d != args.selftest]
    serve_days = [args.selftest]
else:
    if not (args.inp and args.out):
        sys.exit("--in and --out are required unless --selftest")
    serve_days = [d for d in on_disk if d.startswith("2026-")]
    if args.serve_days:
        serve_days = [d for d in serve_days if d in set(args.serve_days.split(","))]
if len(calib_days) < 2 or not serve_days:
    sys.exit(f"need >= 2 calibration days and >= 1 serving day; have {len(calib_days)} and {len(serve_days)}")
print(f"calibration days ({len(calib_days)}): {', '.join(calib_days)}")
print(f"serving days ({len(serve_days)}): {serve_days[0]} .. {serve_days[-1]}")


def departures(src, days):
    out = (pl.scan_parquet(src).filter(C("PHASE_mvt") == "DEP")
            .select(C(ID), C("ADEP_mvt").alias("apt"), C_DAY.alias("day"), C("AOBT_3_flt").is_not_null().alias("plan"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                    C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"),
                    C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int32).alias("hour"),
                    (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("nm_taxi"),
                    C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"))
            .filter(C("day").is_in(days)).collect())
    if G.noplan_callsign:
        out = add_derived_callsign(out, CS_MAPS)
    return out


def arrivals(src, days):
    return (pl.scan_parquet(src).filter(C("PHASE_mvt") == "ARR")
            .select(C("ADES_mvt").alias("apt"), C_DAY.alias("day"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("land_t"),
                    C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("onblock_t"))
            .filter(C("day").is_in(days) & C("callsign").is_not_null()).collect())


def lfpgnm_for_days(deps, days, src_days_source):
    """LEVER 1: NM-callsign taxi-phase match (no lift-off requirement) for every LFPG row in `deps`
    on `days`, reading points directly (independent of runs_and_tracks/prune, same design as
    archive/agent_lfpgnewstand_stage1_eval.py). Returns MVT_ID_mvt + LFPGNM_COLS + lfnm_floor
    (NaN/absent where no NM match or no ground fix). `src_days_source` is only for the log line."""
    parts = []
    for day in days:
        dl = deps.filter((C("apt") == "LFPG") & (C("day") == day)).select("MVT_ID_mvt", "callsign", "mvt_t")
        if dl.height == 0:
            continue
        pts = (pl.scan_parquet(f"data/adsb/points/{day}.parquet")
               .filter(C("apt") == "LFPG").select("icao", "callsign", "t", "alt", "gs").collect())
        parts.append(lfpg_nm_match(dl, pts))
    out = (pl.concat(parts) if parts
           else pl.DataFrame(schema={"MVT_ID_mvt": pl.Int64, **{c: pl.Float64 for c in LFPGNM_COLS}, "lfnm_floor": pl.Float64}))
    print(f"lfpgnewstand ({src_days_source}): {out.filter(pl.col('lfnm_floor').is_not_null() & pl.col('lfnm_floor').is_finite()).height:,} "
          f"LFPG NM floor matches of {deps.filter(C('apt') == 'LFPG').height:,} LFPG departures on {len(days)} days")
    return out


def taxinm_for_days(deps, days, apts, src_days_source):
    """LEVER 1 GENERALIZED: taxinm_for_days is lfpgnm_for_days generalized to a list of airports.
    Returns MVT_ID_mvt + TAXINM_COLS + txnm_floor for every row at any airport in `apts`."""
    parts = []
    for apt in apts:
        for day in days:
            dl = deps.filter((C("apt") == apt) & (C("day") == day)).select("MVT_ID_mvt", "callsign", "mvt_t")
            if dl.height == 0:
                continue
            pts = (pl.scan_parquet(f"data/adsb/points/{day}.parquet")
                   .filter(C("apt") == apt).select("icao", "callsign", "t", "alt", "gs").collect())
            parts.append(taxinm_match(dl, pts, apt))
    out = (pl.concat(parts) if parts
           else pl.DataFrame(schema={"MVT_ID_mvt": pl.Int64, **{c: pl.Float64 for c in TAXINM_COLS}, "txnm_floor": pl.Float64}))
    n_ok = out.filter(pl.col("txnm_floor").is_not_null() & pl.col("txnm_floor").is_finite()).height
    print(f"taxinm ({src_days_source}): {n_ok:,} NM floor matches of {deps.filter(C('apt').is_in(apts)).height:,} "
          f"departures at {','.join(apts)} on {len(days)} days")
    return out


TRAIN = "data/raw/training_*.parquet"
CS_MAPS = None
if G.noplan_callsign:
    CS_MAPS = flight_callsign_maps([pl.scan_parquet(p).filter(C("PHASE_mvt") == "DEP")
                                    .select(C("AOBT_3_flt").is_not_null().alias("plan"),
                                            C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                                            C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight")).collect()
                                    for p in (TRAIN, "data/raw/ranking.parquet")])
    print("no-plan derived callsigns ON (IDEA-NP1)")

# ---- calibration on 2025 days: matches, stand samples, leave-one-day-out estimates ----
cal_dep = departures(TRAIN, calib_days).join(
    pl.scan_parquet(TRAIN).filter(C("PHASE_mvt") == "DEP")
      .select(C(ID), C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("block_t")).collect(), on=ID)
cal_preds = pl.concat([pl.read_parquet(f).select(ID, "y", "bagged") for f in args.calib_preds.split(",")]).unique(subset=ID, keep="first")
cal_dep = cal_dep.join(cal_preds, on=ID, how="inner")
cal_arr = arrivals(TRAIN, calib_days)
tracks_cal, cal_matched, samples = {}, [], []
if args.spill_tracks == "on":
    import pickle
    import tempfile
    _SPILL = Path(tempfile.mkdtemp(prefix="adsb_apply_tracks_"))
    print(f"spilling calibration tracks to {_SPILL}")
    # Remove the ~4.4 GB spill dir at exit (2026-09-22: 38 leaked dirs had filled ~170 GB of C:).
    import atexit
    import shutil
    atexit.register(shutil.rmtree, _SPILL, ignore_errors=True)
_CACHE = {}


def cal_tracks(day):
    """One calibration day's pruned tracks, from memory or from its spill file."""
    if args.spill_tracks != "on":
        return tracks_cal[day]
    if day not in _CACHE:
        _CACHE.clear()
        with open(tracks_cal[day], "rb") as fh:
            _CACHE[day] = pickle.load(fh)
    return _CACHE[day]
for day in calib_days:
    runs, touch, tracks = runs_and_tracks(day, G)
    _, _, both = _match(cal_dep.filter(C("day") == day), runs, day)
    cal_matched.append(both)
    for r in both.filter(C("plan")).iter_rows(named=True):
        for pos in stand_samples(tracks, r["apt"], r["icao"], r["block_t"], G, arrival=False):
            samples.append(("dep", day, r["apt"], r["stand"], pos[0], pos[1], pos[2]))
    for r in match_arrivals(cal_arr.filter(C("day") == day), touch, G).iter_rows(named=True):
        for pos in stand_samples(tracks, r["apt"], r["icao"], r["onblock_t"], G, arrival=True):
            samples.append(("arr", day, r["apt"], r["stand"], pos[0], pos[1], pos[2]))
    if G.dwell_fill:
        for r in both.filter(C("stand").is_not_null()).iter_rows(named=True):
            pos = dwell_sample(r, tracks, G)
            if pos:
                samples.append(("dwell", day, r["apt"], r["stand"], pos[0], pos[1], "dwell"))
    # only matched departures' aircraft are read again (estimates, floors): 30 days of every
    # aircraft's fixes would not fit a small slot
    keep = set(zip(both["apt"].to_list(), both["icao"].to_list()))
    kept = {k: v for k, v in tracks.items() if k in keep}
    if args.spill_tracks == "on":
        _p = _SPILL / f"{day}.pkl"
        with open(_p, "wb") as fh:
            pickle.dump(kept, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tracks_cal[day] = _p
        del kept
    else:
        tracks_cal[day] = kept
    del runs, touch, tracks
blk = pl.DataFrame(samples, schema=["src", "day", "apt", "stand", "lat", "lon", "quality"], orient="row")
cm = pl.concat(cal_matched)
rows, floor_rows, FQ_BY_ID, FANC_BY_ID = [], [], {}, {}
for day in calib_days:
    cmap = stand_map(blk.filter((C("src") != "dep") | (C("day") != day)), G)
    for r in cm.filter(C("day") == day).iter_rows(named=True):
        kind, taxi, qual = _kind_filter(r["apt"], *estimate(r, cmap, cal_tracks(day), G, with_quality=True))
        if args.floor == "on":
            lb, fq = departure_floor(r, cal_tracks(day), G, cmap, with_quality=True)
            FQ_BY_ID[r[ID]] = fq
            if lb is not None and args.floor_stack_anchors == "on" and r["apt"] not in FS_ANCH_SKIP:
                # stored compactly (one float64 array per row, ANCHOR_COLS order): ~700k calibration floor rows at 264 days
                _a = anchors(r, cmap, cal_tracks(day), G, taxi)
                FANC_BY_ID[r[ID]] = np.array([_a[k] for k in ANCHOR_COLS], np.float64)
            if lb is not None:
                floor_rows.append((r[ID], r["apt"], kind is not None, lb, r["mvt_t"] - r["block_t"], float(r["bagged"]),
                                   int(r["hour"]), float(r["nm_taxi"]) if r["nm_taxi"] is not None else float("nan"),
                                   *[fq[q] for q in FLOOR_QUALITY]))
        if kind is not None:
            anc = anchors(r, cmap, cal_tracks(day), G, taxi) if args.stack_feats == "quality_anch" else dict.fromkeys(ANCHOR_COLS, float("nan"))
            rows.append((r[ID], r["apt"], bool(r["plan"]), kind, taxi, r["mvt_t"] - r["block_t"], float(r["bagged"]),
                         r["hour"], r["nm_taxi"], r["stand"], r["type"], *[qual[q] for q in QUALITY], *[anc[k] for k in ANCHOR_COLS]))
cal = pl.DataFrame(rows, schema=[ID, "apt", "plan", "kind", "taxi", "y_true", "model", "hour", "nm_taxi",
                                 "stand", "type", *QUALITY, *ANCHOR_COLS], orient="row", infer_schema_length=None)
tr_np = cal.filter(~C("plan"))
params = {}
for (apt, kind), tr in cal.filter(C("plan")).partition_by(["apt", "kind"], as_dict=True).items():
    if tr.height < 30:
        continue
    err = (tr["taxi"] - tr["y_true"]).to_numpy()
    bias = float(np.median(err))
    v_a = wvar(err - bias)
    v_m_plan = wvar(np.clip(tr["model"].to_numpy(), 0, None) - tr["y_true"].to_numpy())
    s = tr_np.filter(C("apt") == apt)
    if s.height < 30:
        s = tr_np
    v_m_np = wvar(np.clip(s["model"].to_numpy(), 0, None) - s["y_true"].to_numpy()) if s.height >= 30 else None
    params[(apt, kind)] = (bias, v_a, v_m_plan, v_m_np, tr.height)
print("calibration: " + "; ".join(f"{a}/{k} n={n} bias {b:+.0f}s w_plan {vm / (vm + va):.2f}"
                                  for (a, k), (b, va, vm, _, n) in sorted(params.items())))

floor_q = {}
LFPGNM_SYNTH_IDS = set()
TAXINM_APTS = [a for a in args.floor_stack_taxinm_apts.split(",") if a]
TAXINM_SYNTH_IDS = set()
if args.floor == "on":
    fr = pl.DataFrame(floor_rows, schema=[ID, "apt", "has_est", "lb", "y_true", "model", "hour", "nm_taxi", *FLOOR_QUALITY],
                      orient="row", infer_schema_length=None)
    if args.floor_stack_lfpgnm == "on":
        # LEVER 1: (1) append LFPGNM_COLS to every calibration floor row (NaN off LFPG, by construction of
        # lfpgnm_for_days); (2) add brand-new floor rows for LFPG calibration departures never matched by
        # production at all but NM-matched here -- most of the mechanism's value, per
        # docs/LAB-NOTES.md. lb is the raw NM-callsign floor: unreliable as a hard floor
        # (see the p_cal fix below), used only as a floor-stack FEATURE.
        lfnm_cal = lfpgnm_for_days(cal_dep, calib_days, "calibration")
        fr = fr.join(lfnm_cal.select(["MVT_ID_mvt", *LFPGNM_COLS]), on=ID, how="left")
        matched_lfpg_ids = fr.filter(C("apt") == "LFPG").select(ID)
        new_pop = (cal_dep.filter(C("apt") == "LFPG").join(matched_lfpg_ids, on=ID, how="anti")
                   .join(lfnm_cal.filter(C("lfnm_floor").is_not_null() & C("lfnm_floor").is_finite())
                         .select(["MVT_ID_mvt", *LFPGNM_COLS, "lfnm_floor"]), on=ID, how="inner"))
        if new_pop.height:
            new_fr = new_pop.with_columns(
                pl.lit(False).alias("has_est"), C("lfnm_floor").alias("lb"),
                (C("mvt_t") - C("block_t")).alias("y_true"), C("bagged").cast(pl.Float64).alias("model"),
                C("hour").cast(pl.Int64).alias("hour"), C("nm_taxi").cast(pl.Float64).alias("nm_taxi"),
                *[pl.lit(None).cast(pl.Float64).alias(q) for q in FLOOR_QUALITY],
            ).select(fr.columns)
            fr = pl.concat([fr, new_fr], how="vertical_relaxed")
            LFPGNM_SYNTH_IDS = set(new_fr[ID].to_list())
        print(f"lfpgnewstand: {len(LFPGNM_SYNTH_IDS):,} new LFPG calibration floor rows added "
              f"(of {matched_lfpg_ids.height + len(LFPGNM_SYNTH_IDS):,} LFPG calibration departures with a floor)")
    if TAXINM_APTS:
        # LEVER 1 GENERALIZED: same design as --floor-stack-lfpgnm above, any airport, separate
        # columns/tracking so the two mechanisms never collide even if both flags are set.
        txnm_cal = taxinm_for_days(cal_dep, calib_days, TAXINM_APTS, "calibration")
        fr = fr.join(txnm_cal.select(["MVT_ID_mvt", *TAXINM_COLS]), on=ID, how="left")
        matched_tx_ids = fr.filter(C("apt").is_in(TAXINM_APTS)).select(ID)
        new_pop_tx = (cal_dep.filter(C("apt").is_in(TAXINM_APTS)).join(matched_tx_ids, on=ID, how="anti")
                      .join(txnm_cal.filter(C("txnm_floor").is_not_null() & C("txnm_floor").is_finite())
                            .select(["MVT_ID_mvt", *TAXINM_COLS, "txnm_floor"]), on=ID, how="inner"))
        if new_pop_tx.height:
            # every column fr might have (FLOOR_QUALITY always; LFPGNM_COLS only if --floor-stack-lfpgnm
            # is also on) -- extras .select(fr.columns) does not need are simply dropped, harmlessly
            new_fr_tx = new_pop_tx.with_columns(
                pl.lit(False).alias("has_est"), C("txnm_floor").alias("lb"),
                (C("mvt_t") - C("block_t")).alias("y_true"), C("bagged").cast(pl.Float64).alias("model"),
                C("hour").cast(pl.Int64).alias("hour"), C("nm_taxi").cast(pl.Float64).alias("nm_taxi"),
                *[pl.lit(None).cast(pl.Float64).alias(q) for q in FLOOR_QUALITY],
                *[pl.lit(None).cast(pl.Float64).alias(c) for c in LFPGNM_COLS],
            ).select(fr.columns)
            fr = pl.concat([fr, new_fr_tx], how="vertical_relaxed")
            TAXINM_SYNTH_IDS = set(new_fr_tx[ID].to_list())
        print(f"taxinm: {len(TAXINM_SYNTH_IDS):,} new calibration floor rows added at {','.join(TAXINM_APTS)} "
              f"(of {matched_tx_ids.height + len(TAXINM_SYNTH_IDS):,} departures there with a floor)")
    SYNTH_IDS = LFPGNM_SYNTH_IDS | TAXINM_SYNTH_IDS
    below = fr.filter(C("model").clip(lower_bound=0) < C("lb"))
    if SYNTH_IDS:
        # never let the noisy synthetic floor calibrate the real floor's q offset
        below = below.filter(~C(ID).is_in(list(SYNTH_IDS)))
    if args.floor_q == "median":
        for (a, h), g in below.partition_by(["apt", "has_est"], as_dict=True).items():
            floor_q[(a, h)] = float(np.median((g["y_true"] - g["lb"]).to_numpy())) if g.height >= FLOOR_MIN_ROWS else 0.0
    print(f"taxi floor: {fr.height:,} calibration rows with a floor, {below.height:,} with the model below it; offsets "
          + "; ".join(f"{a}/{'est' if h else 'none'} {q:+.0f}s" for (a, h), q in sorted(floor_q.items())))

APTS = sorted({a for a, _ in params})


STACK_STANDS = {k: i for i, k in enumerate(sorted({(a, st) for a, st in zip(cal["apt"].to_list(), cal["stand"].to_list()) if st}))}
STACK_TYPES = {k: i for i, k in enumerate(sorted({t for t in cal["type"].to_list() if t}))}
# the categorical positions, which must match between training and serving
STACK_CATS = ([4, 5, 7 + len(QUALITY), 8 + len(QUALITY)] if args.stack_feats in ("quality", "quality_anch")
              else [4, 5, 7 + len(QUALITY) + len(FLOOR_QUALITY), 8 + len(QUALITY) + len(FLOOR_QUALITY)]
              if args.stack_feats == "quality_fq" else [4, 5])


def stack_features(apt, kind, taxi, p, nm, hour, lb=None, qual=None, stand=None, actype=None, fq=None, anc=None):
    """Second-stage features, identical for calibration and serving rows; None if not usable."""
    if (apt, kind) not in params or nm is None or not np.isfinite(nm):
        return None
    x = max(taxi - params[(apt, kind)][0], 0.0)
    f = [x - p, nm - p, p, x, APTS.index(apt), KINDS.index(kind), hour]
    if args.stack_feats == "lb":
        v = float(lb) if lb is not None and np.isfinite(float(lb)) else float("nan")
        f += [v, v - p]
    elif args.stack_feats in ("quality", "quality_fq", "quality_anch"):
        q = qual or {}
        f += [float(q.get(k, float("nan"))) for k in QUALITY]
        if args.stack_feats == "quality_fq":
            fqd = fq or {}
            f += [float(fqd.get(k, float("nan"))) for k in FLOOR_QUALITY]
        f += [float(STACK_STANDS.get((apt, stand), -1)), float(STACK_TYPES.get(actype, -1))]
        if args.stack_feats == "quality_anch":
            f += stack_columns(anc or dict.fromkeys(ANCHOR_COLS, float("nan")), p)
    return f


DAY_BY_ID = dict(zip(cal_dep[ID].to_list(), cal_dep["day"].to_list()))


def _row_weight(mid) -> float:
    """--calib-weight-recent: weight of one calibration row (1.0 unless its day is in --recent-months)."""
    d = DAY_BY_ID.get(mid)
    return args.calib_weight_recent if d is not None and d[5:7] in RECENT_MONTHS else 1.0


STACK_SKIP = {a for a in args.stack_skip_airports.split(",") if a}
LB_BY_ID = {r[0]: r[3] for r in floor_rows} if args.floor == "on" else {}
stacker = None
if args.stack == "gbm":
    import lightgbm as lgb
    Xs, ys, ws = [], [], []
    for r in cal.filter(C("plan")).iter_rows(named=True):
        f = stack_features(r["apt"], r["kind"], r["taxi"], max(r["model"], 0.0), r["nm_taxi"], r["hour"],
                           LB_BY_ID.get(r[ID]), {k: r[k] for k in QUALITY}, r["stand"], r["type"], FQ_BY_ID.get(r[ID]),
                           {k: r[k] for k in ANCHOR_COLS})
        if f is not None:
            Xs.append(f)
            ys.append(float(np.clip(r["y_true"] - max(r["model"], 0.0), -3600, 3600)))
            ws.append(_row_weight(r[ID]))
    stacker = lgb.LGBMRegressor(n_estimators=args.stack_trees, max_depth=-1 if args.stack_leaves > 15 else 4,
                                num_leaves=args.stack_leaves, learning_rate=args.stack_lr, min_child_samples=args.stack_minchild,
                                verbose=-1).fit(np.asarray(Xs, float), np.asarray(ys), categorical_feature=STACK_CATS,
                                                sample_weight=np.asarray(ws, float) if args.calib_weight_recent != 1.0 else None)
    if args.calib_weight_recent != 1.0:
        print(f"calibration weight {args.calib_weight_recent} on months {sorted(RECENT_MONTHS)}: "
              f"{sum(1 for w in ws if w != 1.0):,} of {len(ws):,} main-stack rows")
    print(f"second stage: LightGBM trained on {len(ys):,} calibration plan rows, features {args.stack_feats}"
          + (f", skipping {','.join(sorted(STACK_SKIP))}" if STACK_SKIP else ""))

def blended_calibration() -> dict:
    """What the serving path would predict for each calibration row: the same blend, the same
    second stage. The floor stack has to be trained on THAT, because that is what it is applied
    to at serve time -- training it on the raw model instead made it fire on 40% of the 2026 rows
    against 20% in the evaluation."""
    out = {}
    for r in cal.iter_rows(named=True):
        p0 = max(float(r["model"]), 0.0)
        key = (r["apt"], r["kind"])
        if key not in params:
            continue
        if stacker is not None and r["plan"]:
            f = stack_features(r["apt"], r["kind"], r["taxi"], p0, r["nm_taxi"], r["hour"], LB_BY_ID.get(r[ID]),
                               {k: r[k] for k in QUALITY}, r["stand"], r["type"], None, {k: r[k] for k in ANCHOR_COLS})
            if f is not None:
                out[r[ID]] = max(p0 + float(np.clip(stacker.predict(np.asarray([f], float))[0], -GUARD_S, GUARD_S)), 0.0)
                continue
        bias, v_a, v_m_plan, v_m_np, _ = params[key]
        v_m = v_m_plan if r["plan"] else v_m_np
        if v_m is None:
            continue
        w = v_m / (v_m + v_a) if (v_m + v_a) > 0 else 0.0
        new = w * max(r["taxi"] - bias, 0.0) + (1 - w) * p0
        out[r[ID]] = new if abs(new - p0) <= GUARD_S else p0
    return out


FLOOR_APTS, floor_stacker = [], None
if args.floor == "on" and G.floor_stack and (floor_q or args.floor_q == "zero"):
    import lightgbm as lgb
    FLOOR_APTS = sorted(fr["apt"].unique().to_list())
    q_arr = np.array([floor_q.get((a, h), 0.0) for a, h in zip(fr["apt"].to_list(), fr["has_est"].to_list())])
    blend_cal = blended_calibration()
    base = np.array([blend_cal.get(m, mo) for m, mo in zip(fr[ID].to_list(), fr["model"].to_list())], dtype=float)
    # LEVER 1: the raw NM-callsign floor (LFPGNM_SYNTH_IDS rows) must NEVER hard-raise the floor
    # stack's training target -- it is unreliable enough (see the harness's own smoke-tested +692s
    # regression when it was tried) to only ever act as a learned feature. -inf makes the
    # maximum() a no-op for those rows.
    lb_plus_q = fr["lb"].to_numpy() + q_arr
    if SYNTH_IDS:
        _synth_mask = np.array([m in SYNTH_IDS for m in fr[ID].to_list()])
        lb_plus_q = np.where(_synth_mask, -np.inf, lb_plus_q)
    p_cal = np.maximum(np.clip(base, 0, None), lb_plus_q)
    lb_c, nm_c = fr["lb"].to_numpy().astype(float), fr["nm_taxi"].to_numpy().astype(float)
    # The 7th column is the round-7 'hasest' gate: whether this row also has a stand/appear
    # estimate. A floor on a row the estimators could not see is a different animal from a floor
    # that merely confirms one, and the stack could not tell them apart. It must stay in lockstep
    # with the serving-side feature vector below.
    Xf = np.column_stack([lb_c - p_cal, lb_c, p_cal,
                          np.array([FLOOR_APTS.index(a) for a in fr["apt"].to_list()], dtype=float),
                          fr["hour"].to_numpy().astype(float), nm_c - p_cal]
                         + ([np.array(fr["has_est"].to_list(), dtype=float)]
                            if args.floor_stack_feats in ("hasest", "quality") else [])
                         + ([fr[q].to_numpy().astype(float) for q in FLOOR_QUALITY]
                            if args.floor_stack_feats == "quality" else []))
    if args.floor_stack_anchors == "on":
        assert args.floor_stack_feats == "quality", "--floor-stack-anchors needs --floor-stack-feats quality"
        _nan_anc = dict.fromkeys(ANCHOR_COLS, float("nan"))
        _as_dict = lambda v: dict(zip(ANCHOR_COLS, v.tolist())) if v is not None else _nan_anc  # noqa: E731
        Xf = np.column_stack([Xf, np.array([stack_columns(_as_dict(FANC_BY_ID.get(m)), pc)
                                            for m, pc in zip(fr[ID].to_list(), p_cal)], dtype=float)])
        print(f"floor stack: + {len(ANCHOR_COLS)} anchor columns; ci_50 available on "
              f"{np.isfinite(Xf[:, 6 + 1 + len(FLOOR_QUALITY) + 2]).mean():.1%} of calibration floor rows")
    if args.floor_stack_lfpgnm == "on":
        # LEVER 1: appended LAST, so nothing above (including categorical_feature=[3]) shifts.
        Xf = np.column_stack([Xf] + [fr[k].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in LFPGNM_COLS])
        print(f"floor stack: + {len(LFPGNM_COLS)} lfpgnewstand columns (LEVER 1); "
              f"{len(LFPGNM_SYNTH_IDS):,} synthetic (production-unmatched) LFPG calibration rows")
    if TAXINM_APTS:
        # LEVER 1 GENERALIZED: appended LAST (after LFPGNM_COLS if present), so nothing above shifts.
        Xf = np.column_stack([Xf] + [fr[k].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in TAXINM_COLS])
        print(f"floor stack: + {len(TAXINM_COLS)} taxinm columns (LEVER 1 GENERALIZED); "
              f"{len(TAXINM_SYNTH_IDS):,} synthetic (NP1-unmatched) calibration rows at {','.join(TAXINM_APTS)}")
    yf = np.clip(fr["y_true"].to_numpy().astype(float) - p_cal, -3600, 3600)
    floor_stacker = lgb.LGBMRegressor(n_estimators=args.floor_stack_trees,
                                      max_depth=-1 if args.floor_stack_leaves > 15 else 4,
                                      num_leaves=args.floor_stack_leaves, learning_rate=args.floor_stack_lr,
                                      min_child_samples=args.floor_stack_minchild, verbose=-1).fit(
        Xf, yf, categorical_feature=[3],
        sample_weight=(np.array([_row_weight(m) for m in fr[ID].to_list()], float)
                       if args.calib_weight_recent != 1.0 else None))
    if args.floor_stack_feats == "quality":
        print("floor stack calibration fq means: " + ", ".join(
            f"{q} {np.nanmean(fr[q].to_numpy().astype(float)):.2f}" for q in FLOOR_QUALITY))
    print(f"floor stack: LightGBM trained on {len(yf):,} calibration rows with a floor "
          f"({sum(1 for m in fr[ID].to_list() if m in blend_cal):,} of them on their blended prediction)")

# ---- serving ----
if args.selftest:
    srv_dep = departures(TRAIN, serve_days)
    srv_arr = arrivals(TRAIN, serve_days)
    sub = (cal_preds.select(ID, C("bagged").clip(0, None).round(0).cast(pl.Int32).alias(TARGET))
           .join(srv_dep.select(ID), on=ID, how="semi"))
else:
    RANK = "data/raw/ranking.parquet"
    srv_dep = departures(RANK, serve_days)
    srv_arr = arrivals(RANK, serve_days)
    sub = pl.read_parquet(args.inp)

pred = dict(zip(sub[ID].to_list(), sub[TARGET].to_numpy().astype(float)))
# Two passes, one day in memory at a time: 62 days of tracks at once would not fit a small
# slot. Pass 1 collects this period's arrival stand samples; pass 2 matches and estimates.
srv_samples = []
for day in serve_days:
    runs, touch, tracks = runs_and_tracks(day, G)
    for r in match_arrivals(srv_arr.filter(C("day") == day), touch, G).iter_rows(named=True):
        for pos in stand_samples(tracks, r["apt"], r["icao"], r["onblock_t"], G, arrival=True):
            srv_samples.append(("arr", day, r["apt"], r["stand"], pos[0], pos[1], pos[2]))
    if G.dwell_fill:
        # the scored days' own departures map their own stands: no label is involved
        _, _, dm = _match(srv_dep.filter(C("day") == day), runs, day)
        for r in dm.filter(C("stand").is_not_null()).iter_rows(named=True):
            pos = dwell_sample(r, tracks, G)
            if pos:
                srv_samples.append(("dwell", day, r["apt"], r["stand"], pos[0], pos[1], "dwell"))
        del dm
    del runs, touch, tracks
# stand map: this period's arrivals plus every 2025 sample (never a serving day's departure off-block)
smap_src = pl.concat([blk, pl.DataFrame(srv_samples, schema=blk.schema, orient="row")]) if srv_samples else blk
cmap = stand_map(smap_src, G)
new_pred, moved, refused, n_matched, n_stacked, by = dict(pred), 0, 0, 0, 0, {}
n_floor, n_raised, n_stacked_floor = 0, 0, 0
FQ_SERVE, STACK_MOVES = [], []   # diagnostics only: serving fq rows, floor-stack moves
SRV_FANC = {}                    # --floor-stack-anchors: this day's serving anchors per floor row
for day in serve_days:
    runs, _, tracks = runs_and_tracks(day, G)
    _, _, sm = _match(srv_dep.filter(C("day") == day), runs, day)
    n_matched += sm.height
    # LEVER 1: this day's LFPG NM-callsign taxi-phase evidence, independent of sm/runs (same design
    # as archive/agent_lfpgnewstand_stage1_eval.py -- the whole point is to reach rows runs_and_tracks's
    # liftoff gate excludes). LFNM_DAY: mid -> LFPGNM_COLS values (for rows already matched too, as
    # extra floor-stack features). LFNM_FLOOR_DAY: mid -> raw NM floor, LFPG rows only, finite only.
    LFNM_DAY, LFNM_FLOOR_DAY = {}, {}
    if args.floor_stack_lfpgnm == "on":
        day_lfpg_deps = srv_dep.filter((C("apt") == "LFPG") & (C("day") == day)).select("MVT_ID_mvt", "callsign", "mvt_t")
        if day_lfpg_deps.height:
            pts_day = (pl.scan_parquet(f"data/adsb/points/{day}.parquet")
                       .filter(C("apt") == "LFPG").select("icao", "callsign", "t", "alt", "gs").collect())
            lfnm_day = lfpg_nm_match(day_lfpg_deps, pts_day)
            for row in lfnm_day.iter_rows(named=True):
                LFNM_DAY[row["MVT_ID_mvt"]] = [row[c] for c in LFPGNM_COLS]
                fl_ = row["lfnm_floor"]
                if fl_ is not None and np.isfinite(fl_):
                    LFNM_FLOOR_DAY[row["MVT_ID_mvt"]] = fl_
    # LEVER 1 GENERALIZED: same design, any airport in TAXINM_APTS. TXNM_APT_DAY: mid -> apt (needed
    # since, unlike LFNM, this spans multiple airports).
    TXNM_DAY, TXNM_FLOOR_DAY, TXNM_APT_DAY = {}, {}, {}
    if TAXINM_APTS:
        for _apt in TAXINM_APTS:
            day_tx_deps = srv_dep.filter((C("apt") == _apt) & (C("day") == day)).select("MVT_ID_mvt", "callsign", "mvt_t")
            if day_tx_deps.height == 0:
                continue
            pts_tx = (pl.scan_parquet(f"data/adsb/points/{day}.parquet")
                      .filter(C("apt") == _apt).select("icao", "callsign", "t", "alt", "gs").collect())
            txnm_day = taxinm_match(day_tx_deps, pts_tx, _apt)
            for row in txnm_day.iter_rows(named=True):
                TXNM_DAY[row["MVT_ID_mvt"]] = [row[c] for c in TAXINM_COLS]
                TXNM_APT_DAY[row["MVT_ID_mvt"]] = _apt
                fl_ = row["txnm_floor"]
                if fl_ is not None and np.isfinite(fl_):
                    TXNM_FLOOR_DAY[row["MVT_ID_mvt"]] = fl_
    stack_ids, stack_X, floors = [], [], []
    for r in sm.iter_rows(named=True):
        if args.scope == "plan" and not r["plan"]:
            continue
        p0 = pred.get(r[ID])
        kind, taxi, qual = _kind_filter(r["apt"], *estimate(r, cmap, tracks, G, with_quality=True))
        lb, fq = None, None
        if args.floor == "on" and p0 is not None:
            lb, fq = departure_floor(r, tracks, G, cmap, with_quality=True)
            if lb is not None:
                floors.append((r[ID], lb + floor_q.get((r["apt"], kind is not None), 0.0), r["apt"], r["hour"],
                               r["nm_taxi"], lb, kind is not None, fq, False))
                if args.floor_stack_anchors == "on" and r["apt"] not in FS_ANCH_SKIP:
                    SRV_FANC[r[ID]] = anchors(r, cmap, tracks, G, taxi)
        if kind is None or p0 is None or (r["apt"], kind) not in params:
            continue
        key = (r["apt"], "plan" if r["plan"] else "no-plan")
        if stacker is not None and r["plan"] and r["apt"] not in STACK_SKIP:
            anc = anchors(r, cmap, tracks, G, taxi) if args.stack_feats == "quality_anch" else None
            f = stack_features(r["apt"], kind, taxi, p0, r["nm_taxi"], r["hour"], lb, qual, r["stand"], r["type"], fq, anc)
            if f is not None:
                stack_ids.append((r[ID], key))
                stack_X.append(f)
                continue
        bias, v_a, v_m_plan, v_m_np, _ = params[(r["apt"], kind)]
        v_m = v_m_plan if r["plan"] else v_m_np
        if v_m is None:
            continue
        w = v_m / (v_m + v_a) if (v_m + v_a) > 0 else 0.0
        new = w * max(taxi - bias, 0.0) + (1 - w) * p0
        if not args.no_guard and abs(new - p0) > GUARD_S:
            refused += 1
            continue
        new_pred[r[ID]] = new
        moved += 1
        by[key] = by.get(key, 0) + 1
    if stack_X:
        corr = np.clip(stacker.predict(np.asarray(stack_X, float)), -GUARD_S, GUARD_S)
        for (mid, key), c in zip(stack_ids, corr):
            new_pred[mid] = max(pred[mid] + float(c), 0.0)
            moved += 1
            n_stacked += 1
            by[key] = by.get(key, 0) + 1
    # LEVER 1: brand-new floor-stack entries for LFPG departures never matched by production's
    # lift-off-gated matcher (sm) at all, but NM-matched here -- most of the mechanism's value.
    # is_synth=True: never hard-raises (see below), floor-stack FEATURE only.
    n_synth_today = 0
    if args.floor_stack_lfpgnm == "on" and LFNM_FLOOR_DAY:
        matched_ids_today = set(sm[ID].to_list())
        _hour_nm = {row["MVT_ID_mvt"]: (row["hour"], row["nm_taxi"])
                    for row in srv_dep.filter((C("apt") == "LFPG") & (C("day") == day))
                                     .select("MVT_ID_mvt", "hour", "nm_taxi").iter_rows(named=True)}
        for mid, lbv in LFNM_FLOOR_DAY.items():
            if mid in matched_ids_today or mid not in new_pred:
                continue
            hour_v, nm_v = _hour_nm.get(mid, (0, None))
            floors.append((mid, lbv, "LFPG", hour_v, nm_v, lbv, False, None, True))
            n_synth_today += 1
    n_synth_today_tx = 0
    if TAXINM_APTS and TXNM_FLOOR_DAY:
        matched_ids_today_tx = set(sm[ID].to_list())
        _hour_nm_tx = {row["MVT_ID_mvt"]: (row["hour"], row["nm_taxi"])
                       for row in srv_dep.filter(C("apt").is_in(TAXINM_APTS) & (C("day") == day))
                                        .select("MVT_ID_mvt", "hour", "nm_taxi").iter_rows(named=True)}
        for mid, lbv in TXNM_FLOOR_DAY.items():
            if mid in matched_ids_today_tx or mid not in new_pred:
                continue
            hour_v, nm_v = _hour_nm_tx.get(mid, (0, None))
            floors.append((mid, lbv, TXNM_APT_DAY[mid], hour_v, nm_v, lbv, False, None, True))
            n_synth_today_tx += 1
    if n_synth_today or n_synth_today_tx:
        n_matched += n_synth_today + n_synth_today_tx
    # the floor comes after every correction, as evaluated
    fx, fid, fapt, flb = [], [], [], []
    for mid, floor, apt_, hour_, nm_, lb_, hasest_, fq_, is_synth in floors:
        n_floor += 1
        if is_synth:
            # LEVER 1: the raw NM-callsign floor must never hard-raise a prediction -- unreliable
            # enough (smoke-tested: +692s on LFPG when tried) to only ever act as a floor-stack
            # feature, matching archive/agent_lfpgnewstand_stage1_eval.py's apply_floor fix.
            pass
        elif new_pred[mid] < floor:
            new_pred[mid] = floor
            n_raised += 1
        if floor_stacker is not None and apt_ in FLOOR_APTS:
            pn = new_pred[mid]
            nmv = float(nm_) if nm_ is not None and np.isfinite(float(nm_)) else float("nan")
            row = [lb_ - pn, lb_, pn, float(FLOOR_APTS.index(apt_)), float(hour_), nmv - pn]
            if args.floor_stack_feats in ("hasest", "quality"):
                row.append(float(hasest_))
            if args.floor_stack_feats == "quality":
                row += [float((fq_ or {}).get(q, float("nan"))) for q in FLOOR_QUALITY]
            if args.floor_stack_anchors == "on":
                row += [float(v) for v in stack_columns(SRV_FANC.pop(mid, dict.fromkeys(ANCHOR_COLS, float("nan"))), pn)]
            if args.floor_stack_lfpgnm == "on":
                # appended LAST, so nothing above (including categorical_feature=[3]) shifts
                row += [float(v) if v is not None else float("nan") for v in LFNM_DAY.get(mid, [None] * len(LFPGNM_COLS))]
            if TAXINM_APTS:
                # LEVER 1 GENERALIZED: appended LAST (after LFPGNM_COLS if present)
                row += [float(v) if v is not None else float("nan") for v in TXNM_DAY.get(mid, [None] * len(TAXINM_COLS))]
            fx.append(row)
            fid.append(mid)
            fapt.append(apt_)
            flb.append(float(lb_) if not is_synth else float("-inf"))  # LEVER 1: never clamp a synthetic row to its own noisy lb
            if args.floor_stack_feats == "quality":
                FQ_SERVE.append([float((fq_ or {}).get(q, float("nan"))) for q in FLOOR_QUALITY])
    if fx:
        add = np.clip(floor_stacker.predict(np.asarray(fx, float)), -args.floor_stack_lower, None)
        lower_skip = {a for a in args.floor_stack_lower_skip.split(",") if a}
        for mid, a, apt_, lbv in zip(fid, add, fapt, flb):
            if args.floor_stack_lower > 0:
                if apt_ in lower_skip:
                    a = max(a, 0.0)
                if a != 0:
                    before = new_pred[mid]
                    new_pred[mid] = max(new_pred[mid] + float(a), lbv)
                    STACK_MOVES.append(new_pred[mid] - before)
                    n_stacked_floor += 1
            elif a > 0:
                new_pred[mid] += float(a)
                STACK_MOVES.append(float(a))
                n_stacked_floor += 1
    if n_synth_today:
        print(f"  lfpgnewstand {day}: {n_synth_today:,} new LFPG floor-stack rows (production-unmatched, NM-matched)")
    if n_synth_today_tx:
        print(f"  taxinm {day}: {n_synth_today_tx:,} new floor-stack rows at {','.join(TAXINM_APTS)} (NP1-unmatched, NM-matched)")
    del runs, tracks
print(f"serving: {n_matched:,} departures matched; corrected {moved:,} ({n_stacked:,} by the second stage); "
      f"refused by the guard {refused:,}")
print("  by airport: " + ", ".join(f"{a} {p} {n}" for (a, p), n in sorted(by.items())))
if args.floor == "on":
    if FQ_SERVE:
        fqa = np.asarray(FQ_SERVE, float)
        print("floor stack serving fq means:     " + ", ".join(f"{q} {np.nanmean(fqa[:, i]):.2f}" for i, q in enumerate(FLOOR_QUALITY)))
    if STACK_MOVES:
        mv = np.asarray(STACK_MOVES)
        print(f"floor stack moves: {int((mv > 0).sum()):,} raised (mean {mv[mv > 0].mean() if (mv > 0).any() else 0:+.0f}s), "
              f"{int((mv < 0).sum()):,} lowered (mean {mv[mv < 0].mean() if (mv < 0).any() else 0:+.0f}s), net mean {mv.mean():+.1f}s")
    print(f"taxi floor: {n_floor:,} matched departures have a floor; raised {n_raised:,}"
          + (f"; floor stack raised {n_stacked_floor:,} further" if floor_stacker is not None else ""))

out = sub.with_columns(pl.Series(TARGET, np.rint(np.clip([new_pred[i] for i in sub[ID].to_list()], 0, None)).astype(np.int64))
                       .cast(sub[TARGET].dtype))
if args.selftest:
    truth = (pl.scan_parquet(TRAIN).filter(C("PHASE_mvt") == "DEP")
             .select(C(ID), (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y")).collect())
    j = sub.rename({TARGET: "p0"}).join(out.rename({TARGET: "p1"}), on=ID).join(truth, on=ID).join(srv_dep.select(ID, "apt"), on=ID)
    y, p0, p1 = j["y"].to_numpy(), j["p0"].to_numpy().astype(float), j["p1"].to_numpy().astype(float)
    print(f"SELFTEST {args.selftest}: {j.height:,} departures, RMSE {np.sqrt(((y - p0) ** 2).mean()):.2f} -> "
          f"{np.sqrt(((y - p1) ** 2).mean()):.2f} ({np.sqrt(((y - p1) ** 2).mean()) - np.sqrt(((y - p0) ** 2).mean()):+.2f}s)")
    apts_j = j["apt"].to_numpy()
    print("  by airport (RMSE delta, rows changed): " + "  ".join(
        f"{a} {np.sqrt(((y[m] - p1[m]) ** 2).mean()) - np.sqrt(((y[m] - p0[m]) ** 2).mean()):+.2f}s/{int((p1[m] != p0[m]).sum())}"
        for a in sorted(set(apts_j)) for m in [apts_j == a]))
else:
    out.write_parquet(args.out)
    print(f"wrote {args.out}")
