"""agent_postchain COPY of archive/adsb_apply.py (2026-09-23) importing archive/agent_postchain_adsb_match_aircs.py and
adding --air-cs-all (default off = identical to the shipped path). Original docstring follows.

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

from archive.agent_postchain_adsb_match_aircs import (FLOOR_QUALITY, POINTS_DIR, QUALITY, Gates, departure_floor, dwell_sample, estimate, match_arrivals,
                            match_departures, runs_and_tracks, stand_map, stand_samples, wvar)

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
ap.add_argument("--stack-feats", choices=["basic", "lb", "quality", "quality_fq"], default="quality",
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
ap.add_argument("--air-cs-all", choices=["on", "off"], default="off",
                help="key lift-off runs on every airborne callsign in the window (agent_postchain; default off = shipped)")
ap.add_argument("--air-cs-skip", default="LIRF", help="airports that keep the shipped keys when air-cs-all is on")
args = ap.parse_args()
G = Gates(osm_fill=args.osm_fill == "on", air_cs_all=args.air_cs_all == "on",
          air_cs_skip=tuple(x for x in args.air_cs_skip.split(",") if x))
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
    return (pl.scan_parquet(src).filter(C("PHASE_mvt") == "DEP")
            .select(C(ID), C("ADEP_mvt").alias("apt"), C_DAY.alias("day"), C("AOBT_3_flt").is_not_null().alias("plan"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                    C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"),
                    C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int32).alias("hour"),
                    (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("nm_taxi"))
            .filter(C("day").is_in(days)).collect())


def arrivals(src, days):
    return (pl.scan_parquet(src).filter(C("PHASE_mvt") == "ARR")
            .select(C("ADES_mvt").alias("apt"), C_DAY.alias("day"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("land_t"),
                    C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("onblock_t"))
            .filter(C("day").is_in(days) & C("callsign").is_not_null()).collect())


TRAIN = "data/raw/training_*.parquet"

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
    _, _, both = match_departures(cal_dep.filter(C("day") == day), runs, G)
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
rows, floor_rows, FQ_BY_ID = [], [], {}
for day in calib_days:
    cmap = stand_map(blk.filter((C("src") != "dep") | (C("day") != day)), G)
    for r in cm.filter(C("day") == day).iter_rows(named=True):
        kind, taxi, qual = estimate(r, cmap, cal_tracks(day), G, with_quality=True)
        if args.floor == "on":
            lb, fq = departure_floor(r, cal_tracks(day), G, cmap, with_quality=True)
            FQ_BY_ID[r[ID]] = fq
            if lb is not None:
                floor_rows.append((r[ID], r["apt"], kind is not None, lb, r["mvt_t"] - r["block_t"], float(r["bagged"]),
                                   int(r["hour"]), float(r["nm_taxi"]) if r["nm_taxi"] is not None else float("nan"),
                                   *[fq[q] for q in FLOOR_QUALITY]))
        if kind is not None:
            rows.append((r[ID], r["apt"], bool(r["plan"]), kind, taxi, r["mvt_t"] - r["block_t"], float(r["bagged"]),
                         r["hour"], r["nm_taxi"], r["stand"], r["type"], *[qual[q] for q in QUALITY]))
cal = pl.DataFrame(rows, schema=[ID, "apt", "plan", "kind", "taxi", "y_true", "model", "hour", "nm_taxi",
                                 "stand", "type", *QUALITY], orient="row", infer_schema_length=None)
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
if args.floor == "on":
    fr = pl.DataFrame(floor_rows, schema=[ID, "apt", "has_est", "lb", "y_true", "model", "hour", "nm_taxi", *FLOOR_QUALITY],
                      orient="row", infer_schema_length=None)
    below = fr.filter(C("model").clip(lower_bound=0) < C("lb"))
    if args.floor_q == "median":
        for (a, h), g in below.partition_by(["apt", "has_est"], as_dict=True).items():
            floor_q[(a, h)] = float(np.median((g["y_true"] - g["lb"]).to_numpy())) if g.height >= FLOOR_MIN_ROWS else 0.0
    print(f"taxi floor: {fr.height:,} calibration rows with a floor, {below.height:,} with the model below it; offsets "
          + "; ".join(f"{a}/{'est' if h else 'none'} {q:+.0f}s" for (a, h), q in sorted(floor_q.items())))

APTS = sorted({a for a, _ in params})


STACK_STANDS = {k: i for i, k in enumerate(sorted({(a, st) for a, st in zip(cal["apt"].to_list(), cal["stand"].to_list()) if st}))}
STACK_TYPES = {k: i for i, k in enumerate(sorted({t for t in cal["type"].to_list() if t}))}
# the categorical positions, which must match between training and serving
STACK_CATS = ([4, 5, 7 + len(QUALITY), 8 + len(QUALITY)] if args.stack_feats == "quality"
              else [4, 5, 7 + len(QUALITY) + len(FLOOR_QUALITY), 8 + len(QUALITY) + len(FLOOR_QUALITY)]
              if args.stack_feats == "quality_fq" else [4, 5])


def stack_features(apt, kind, taxi, p, nm, hour, lb=None, qual=None, stand=None, actype=None, fq=None):
    """Second-stage features, identical for calibration and serving rows; None if not usable."""
    if (apt, kind) not in params or nm is None or not np.isfinite(nm):
        return None
    x = max(taxi - params[(apt, kind)][0], 0.0)
    f = [x - p, nm - p, p, x, APTS.index(apt), KINDS.index(kind), hour]
    if args.stack_feats == "lb":
        v = float(lb) if lb is not None and np.isfinite(float(lb)) else float("nan")
        f += [v, v - p]
    elif args.stack_feats in ("quality", "quality_fq"):
        q = qual or {}
        f += [float(q.get(k, float("nan"))) for k in QUALITY]
        if args.stack_feats == "quality_fq":
            fqd = fq or {}
            f += [float(fqd.get(k, float("nan"))) for k in FLOOR_QUALITY]
        f += [float(STACK_STANDS.get((apt, stand), -1)), float(STACK_TYPES.get(actype, -1))]
    return f


STACK_SKIP = {a for a in args.stack_skip_airports.split(",") if a}
LB_BY_ID = {r[0]: r[3] for r in floor_rows} if args.floor == "on" else {}
stacker = None
if args.stack == "gbm":
    import lightgbm as lgb
    Xs, ys = [], []
    for r in cal.filter(C("plan")).iter_rows(named=True):
        f = stack_features(r["apt"], r["kind"], r["taxi"], max(r["model"], 0.0), r["nm_taxi"], r["hour"],
                           LB_BY_ID.get(r[ID]), {k: r[k] for k in QUALITY}, r["stand"], r["type"], FQ_BY_ID.get(r[ID]))
        if f is not None:
            Xs.append(f)
            ys.append(float(np.clip(r["y_true"] - max(r["model"], 0.0), -3600, 3600)))
    stacker = lgb.LGBMRegressor(n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.05, min_child_samples=50,
                                verbose=-1).fit(np.asarray(Xs, float), np.asarray(ys), categorical_feature=STACK_CATS)
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
                               {k: r[k] for k in QUALITY}, r["stand"], r["type"])
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
    p_cal = np.maximum(np.clip(base, 0, None), fr["lb"].to_numpy() + q_arr)
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
    yf = np.clip(fr["y_true"].to_numpy().astype(float) - p_cal, -3600, 3600)
    floor_stacker = lgb.LGBMRegressor(n_estimators=250, max_depth=4, num_leaves=15, learning_rate=0.05,
                                      min_child_samples=100, verbose=-1).fit(Xf, yf, categorical_feature=[3])
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
        _, _, dm = match_departures(srv_dep.filter(C("day") == day), runs, G)
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
for day in serve_days:
    runs, _, tracks = runs_and_tracks(day, G)
    _, _, sm = match_departures(srv_dep.filter(C("day") == day), runs, G)
    n_matched += sm.height
    stack_ids, stack_X, floors = [], [], []
    for r in sm.iter_rows(named=True):
        if args.scope == "plan" and not r["plan"]:
            continue
        p0 = pred.get(r[ID])
        kind, taxi, qual = estimate(r, cmap, tracks, G, with_quality=True)
        lb, fq = None, None
        if args.floor == "on" and p0 is not None:
            lb, fq = departure_floor(r, tracks, G, cmap, with_quality=True)
            if lb is not None:
                floors.append((r[ID], lb + floor_q.get((r["apt"], kind is not None), 0.0), r["apt"], r["hour"],
                               r["nm_taxi"], lb, kind is not None, fq))
        if kind is None or p0 is None or (r["apt"], kind) not in params:
            continue
        key = (r["apt"], "plan" if r["plan"] else "no-plan")
        if stacker is not None and r["plan"] and r["apt"] not in STACK_SKIP:
            f = stack_features(r["apt"], kind, taxi, p0, r["nm_taxi"], r["hour"], lb, qual, r["stand"], r["type"], fq)
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
    # the floor comes after every correction, as evaluated
    fx, fid, fapt, flb = [], [], [], []
    for mid, floor, apt_, hour_, nm_, lb_, hasest_, fq_ in floors:
        n_floor += 1
        if new_pred[mid] < floor:
            new_pred[mid] = floor
            n_raised += 1
        if floor_stacker is not None and apt_ in FLOOR_APTS:
            pn = new_pred[mid]
            nmv = float(nm_) if nm_ is not None and np.isfinite(float(nm_)) else float("nan")
            row = [lb_ - pn, lb_, pn, float(FLOOR_APTS.index(apt_)), float(hour_), nmv - pn]
            if args.floor_stack_feats in ("hasest", "quality"):
                row.append(float(hasest_))
            if args.floor_stack_feats == "quality":
                row += [float(fq_[q]) for q in FLOOR_QUALITY]
            fx.append(row)
            fid.append(mid)
            fapt.append(apt_)
            flb.append(float(lb_))
            if args.floor_stack_feats == "quality":
                FQ_SERVE.append([float(fq_[q]) for q in FLOOR_QUALITY])
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
    j = sub.rename({TARGET: "p0"}).join(out.rename({TARGET: "p1"}), on=ID).join(truth, on=ID)
    y, p0, p1 = j["y"].to_numpy(), j["p0"].to_numpy().astype(float), j["p1"].to_numpy().astype(float)
    print(f"SELFTEST {args.selftest}: {j.height:,} departures, RMSE {np.sqrt(((y - p0) ** 2).mean()):.2f} -> "
          f"{np.sqrt(((y - p1) ** 2).mean()):.2f} ({np.sqrt(((y - p1) ** 2).mean()) - np.sqrt(((y - p0) ** 2).mean()):+.2f}s)")
else:
    out.write_parquet(args.out)
    print(f"wrote {args.out}")
