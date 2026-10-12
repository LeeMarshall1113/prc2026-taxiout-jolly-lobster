"""experiment adsbq 01: per-row dump of the ADS-B off-block estimate and its raw track evidence.

    .venv/Scripts/python.exe archive/agent_adsbq_01_dump.py --stage samples   # stand-map samples, cached
    .venv/Scripts/python.exe archive/agent_adsbq_01_dump.py --stage rows      # per matched departure

Read-only on prc/ and data/. Uses prc.adsb_match exactly as production does (Gates() defaults = what
adsb_apply serves), stand maps leave-one-day-out exactly as the Stage 1 evaluation builds them
(a day's own departure off-block samples never map that day; its arrivals may).

Besides the production estimate (kind, taxi, QUALITY) and floor (lb, FLOOR_QUALITY), each row
carries alternative anchors from the same track, for stand-mapped rows:
  cr_R   MVT - t(last fix within R m of the centroid in the 3 h lookback)       R in RADII
  ci_R   same, linearly interpolated to the R-m crossing using the next fix
  gp_R   gap from that fix to the next fix (s)
  sl_R   1 if a fix at <= 3 kt sits inside R within 600 s before it (production's dwell test)
and the run start (appear_raw = MVT - start_t, start_d = its distance from the centroid).
Output: data/interim/agent_adsbq_rows.parquet (+ _samples.parquet cache).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import polars as pl

from prc.adsb_match import (FLOOR_QUALITY, QUALITY, Gates, departure_floor, dwell_sample, estimate, match_arrivals,
                            match_departures, metres, runs_and_tracks, stand_map, stand_samples)

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["samples", "rows"], required=True)
ap.add_argument("--map-step", type=int, default=2, help="every Nth 2025 day in months 1-3,7-9 feeds the stand map")
ap.add_argument("--eval-step", type=int, default=4, help="every Nth such day is evaluated (a subset of the map days)")
ap.add_argument("--max-eval", type=int, default=0)
args = ap.parse_args()

G = Gates()
RADII = (20, 30, 50, 75, 100, 150, 200, 300, 450, 600)
OUT_S = Path("data/interim/agent_adsbq_samples.parquet")
OUT_R = Path("data/interim/agent_adsbq_rows.parquet")
MONTHS = ("01", "02", "03", "07", "08", "09")
alldays = sorted(p.stem for p in Path("data/adsb/points").glob("2025-*.parquet") if p.stem[5:7] in MONTHS)
map_days = alldays[::args.map_step]
eval_days = map_days[::max(args.eval_step // args.map_step, 1)]
if args.max_eval:
    eval_days = eval_days[:args.max_eval]
print(f"map days {len(map_days)}, eval days {len(eval_days)}", flush=True)

C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)
TRAIN = "data/raw/training_*.parquet"


def deps(days):
    return (pl.scan_parquet(TRAIN).filter(C("PHASE_mvt") == "DEP")
            .select(C("MVT_ID_mvt"), C("ADEP_mvt").alias("apt"), C_DAY.alias("day"),
                    C("AOBT_3_flt").is_not_null().alias("plan"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                    C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"),
                    C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int32).alias("hour"),
                    (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("nm_taxi"),
                    C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("block_t"))
            .filter(C("day").is_in(days)).collect())


def arrs(days):
    return (pl.scan_parquet(TRAIN).filter(C("PHASE_mvt") == "ARR")
            .select(C("ADES_mvt").alias("apt"), C_DAY.alias("day"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("land_t"),
                    C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("onblock_t"))
            .filter(C("day").is_in(days) & C("callsign").is_not_null()).collect())


if args.stage == "samples":
    t0 = time.time()
    dep_all, arr_all = deps(map_days), arrs(map_days)
    samples = []
    for i, day in enumerate(map_days):
        runs, touch, tracks = runs_and_tracks(day, G)
        _, _, both = match_departures(dep_all.filter(C("day") == day), runs, G)
        for r in both.filter(C("plan")).iter_rows(named=True):
            for pos in stand_samples(tracks, r["apt"], r["icao"], r["block_t"], G, arrival=False):
                samples.append(("dep", day, r["apt"], r["stand"], pos[0], pos[1], pos[2]))
        for r in match_arrivals(arr_all.filter(C("day") == day), touch, G).iter_rows(named=True):
            for pos in stand_samples(tracks, r["apt"], r["icao"], r["onblock_t"], G, arrival=True):
                samples.append(("arr", day, r["apt"], r["stand"], pos[0], pos[1], pos[2]))
        if G.dwell_fill:
            for r in both.filter(C("stand").is_not_null()).iter_rows(named=True):
                pos = dwell_sample(r, tracks, G)
                if pos:
                    samples.append(("dwell", day, r["apt"], r["stand"], pos[0], pos[1], "dwell"))
        del runs, touch, tracks
        if i % 10 == 0:
            print(f"  {day} samples {len(samples):,}  {time.time() - t0:.0f}s", flush=True)
    pl.DataFrame(samples, schema=["src", "day", "apt", "stand", "lat", "lon", "quality"], orient="row").write_parquet(OUT_S)
    print(f"wrote {OUT_S} ({len(samples):,}) in {time.time() - t0:.0f}s")
    sys.exit(0)

# ---- rows ----
t0 = time.time()
blk = pl.read_parquet(OUT_S)
preds = pl.concat([pl.read_parquet(f).select("MVT_ID_mvt", "y", "bagged") for f in
                   ("results/preds_n_feat_1_7.parquet", "results/preds_n_feat_2_8.parquet",
                    "results/preds_n3_bulk_3_9.parquet")]).unique(subset="MVT_ID_mvt", keep="first")
universe = deps(eval_days).join(preds, on="MVT_ID_mvt", how="inner")
NAN = float("nan")
out = []
for i, day in enumerate(eval_days):
    runs, touch, tracks = runs_and_tracks(day, G)
    _, _, both = match_departures(universe.filter(C("day") == day), runs, G)
    cmap = stand_map(blk.filter((C("src") != "dep") | (C("day") != day)), G)
    for r in both.iter_rows(named=True):
        kind, taxi, qual = estimate(r, cmap, tracks, G, with_quality=True)
        lb, fq = departure_floor(r, tracks, G, cmap, with_quality=True)
        rec = {"MVT_ID_mvt": r["MVT_ID_mvt"], "day": day, "apt": r["apt"], "plan": bool(r["plan"]),
               "stand": r["stand"], "type": r["type"], "hour": int(r["hour"]),
               "nm_taxi": float(r["nm_taxi"]) if r["nm_taxi"] is not None else NAN,
               "y_true": r["mvt_t"] - r["block_t"], "model": float(r["bagged"]),
               "kind": kind, "taxi": taxi if taxi is not None else NAN, "lb": lb if lb is not None else NAN,
               "appear_raw": r["mvt_t"] - r["start_t"], "start_gs": r["start_gs"] if r["start_gs"] is not None else NAN,
               "max_gap": r["max_gap"] if r["max_gap"] is not None else NAN,
               "touch_run": bool(r.get("before_ground") is False and r.get("before_t") is not None
                                 and r["start_t"] - r["before_t"] <= G.liftoff_s),
               "run_len": r["liftoff_t"] - r["start_t"], "dmatch": r["dmatch"]}
        rec.update({q: qual[q] for q in QUALITY})
        rec.update({q: fq[q] for q in FLOOR_QUALITY})
        c = cmap.get((r["apt"], r["stand"]))
        tr = tracks.get((r["apt"], r["icao"]))
        rec["has_c"] = c is not None
        rec["start_d"] = NAN
        for R in RADII:
            rec[f"cr_{R}"] = rec[f"ci_{R}"] = rec[f"gp_{R}"] = NAN
            rec[f"sl_{R}"] = 0
        rec["dens"] = NAN
        rec["t_in150"] = NAN
        if c is not None:
            rec["start_d"] = float(metres(np.array([r["start_lat"]]), np.array([r["start_lon"]]), c[0], c[1])[0])
        if c is not None and tr is not None:
            t, la, lo, gs = tr
            win = np.flatnonzero((t <= r["liftoff_t"]) & (t >= r["liftoff_t"] - G.lookback_s))
            if len(win):
                d = metres(la[win], lo[win], c[0], c[1])
                for R in RADII:
                    ins = np.flatnonzero(d <= R)
                    if not len(ins):
                        continue
                    k = ins.max()                      # position within win
                    j = win[k]
                    rec[f"cr_{R}"] = r["mvt_t"] - t[j]
                    if j + 1 < len(t):
                        rec[f"gp_{R}"] = t[j + 1] - t[j]
                        dn = float(metres(np.array([la[j + 1]]), np.array([lo[j + 1]]), c[0], c[1])[0])
                        if dn > d[k] and t[j + 1] - t[j] <= G.stand_next_s:
                            f = (R - d[k]) / (dn - d[k])
                            rec[f"ci_{R}"] = r["mvt_t"] - (t[j] + min(max(f, 0.0), 1.0) * (t[j + 1] - t[j]))
                    near = win[ins]
                    rec[f"sl_{R}"] = int(bool(((t[near] >= t[j] - G.dwell_window_s) & (gs[near] <= G.dwell_gs_kt)).any()))
                    if R == 150:
                        rec["t_in150"] = t[j] - t[near.min()]
                        w2 = np.flatnonzero((t >= t[j] - 300) & (t <= t[j] + 300))
                        rec["dens"] = float(len(w2))
        out.append(rec)
    del runs, touch, tracks
    print(f"  {day} rows {len(out):,}  {time.time() - t0:.0f}s", flush=True)
df = pl.DataFrame(out, infer_schema_length=None)
df.write_parquet(OUT_R)
print(f"wrote {OUT_R} ({df.height:,} rows) in {time.time() - t0:.0f}s")
