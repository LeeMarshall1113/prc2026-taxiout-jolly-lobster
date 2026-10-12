"""experiment monster2 01: EVERY push-back event of the matched airframe in its turnaround (docs/LAB-NOTES.md).

    python archive/agent_monster2_01_events.py --mode 2025 --days D1,D2 --out data/interim/agent_monster2_ev_<tag>.parquet
    python archive/agent_monster2_01_events.py --mode 2026 --days ...   (label-free)

Matching = the offblock dump's (production NP1: air_cs_all, LIRF air-cs skip, noplan_callsign; + taxinm for rows NP1
misses), EXCEPT --lirf-np on lifts noplan_callsign_skip=("LIRF",) so Rome no-plan rows can match by derived callsign
(non-production; those rows are flagged lirf_np and reported separately).

Push-back reader = the v2 reader's reverse test (archive/agent_offblock_v2.py): consecutive GROUND fixes i -> i+1,
displacement >= 3 m, heading finite, gs >= 0.5 kt at either end, bearing of the displacement > 120 deg from the
reported heading. Added guards (declared before any label was read): dt <= 90 s between the fixes (no displacement
across a reception hole) and gs <= 8 kt at both ends (a push is slow). Reverse steps closer than 180 s are one EVENT;
an event counts if it has >= 2 steps or >= 15 m of reverse travel.

Turnaround window: (last AIRBORNE fix of this airframe before lift-off - 300 s, lift-off - 60 s], looking back at most
16 h (previous day's points file loaded). The last airborne fix before the departure is the inbound landing (or the
previous rotation's), so a push of the airframe's EARLIER departure is outside the window.
Per row: all events (start time, steps, reverse metres, lat/lon), n_push, first/last push, the longest slow run and
the longest silence between the first push and the final push, the airframe's take-off vs MVT (matching check),
and where the airframe was at the labelled off-block (2025 only; reported, never used by a rule).
"""
import argparse
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_noplan_adsb_match import (ELEVATION_FT, Gates, add_derived_callsign, flight_callsign_maps,
                                             match_departures, runs_and_tracks)

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["2025", "2026"], default="2025")
ap.add_argument("--days", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--apts", default="EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH,LTFM")
ap.add_argument("--lirf-np", choices=["on", "off"], default="on")
ap.add_argument("--only-ids", default=None, help="optional parquet with MVT_ID_mvt: keep only these rows in the output")
args = ap.parse_args()
DAYS = args.days.split(",")
APTS = args.apts.split(",")
G = Gates(air_cs_all=True, air_cs_skip=("LIRF",), noplan_callsign=True,
          noplan_callsign_skip=() if args.lirf_np == "on" else ("LIRF",))
T0 = time.time()
LOOKBACK = 16 * 3600
NAN = float("nan")


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)
src = "data/raw/training_*.parquet" if args.mode == "2025" else "data/raw/ranking.parquet"
cols = [C("MVT_ID_mvt"), C("ADEP_mvt").alias("apt"), C_DAY.alias("day"),
        C("AOBT_3_flt").is_not_null().alias("plan"),
        C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
        C("STAND_mvt").cast(pl.Utf8).alias("stand"),
        C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
        C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
        C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t")]
if args.mode == "2025":
    cols.append(C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("block_t"))
dep = (pl.scan_parquet(src).filter(C("PHASE_mvt") == "DEP").select(cols)
       .filter(C("day").is_in(DAYS) & C("apt").is_in(APTS)).collect())
if args.mode == "2026":
    dep = dep.with_columns(pl.lit(None).cast(pl.Float64).alias("block_t"))
cs_src = [pl.scan_parquet(s).filter(C("PHASE_mvt") == "DEP")
          .select(C("AOBT_3_flt").is_not_null().alias("plan"),
                  C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                  C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight")).collect()
          for s in ("data/raw/training_*.parquet", "data/raw/ranking.parquet")]
dep = add_derived_callsign(dep, flight_callsign_maps(cs_src))
del cs_src
assert dep["MVT_ID_mvt"].n_unique() == dep.height
ONLY = None
if args.only_ids:
    ONLY = set(pl.read_parquet(args.only_ids)["MVT_ID_mvt"].to_list())
log(f"{args.mode}: {dep.height:,} departures on {len(DAYS)} days (lirf-np {args.lirf_np})")


def taxinm_icao(deps, day):
    """NM-callsign -> icao for departures NP1 missed (the offblock dump's rule)."""
    if deps.height == 0:
        return {}
    p = (pl.scan_parquet(f"data/adsb/points/{day}.parquet").filter(C("apt").is_in(APTS) & C("callsign").is_not_null())
         .select("apt", "icao", C("callsign").str.strip_chars().str.to_uppercase().alias("callsign"), "t").collect())
    cs = (p.join(deps.select("MVT_ID_mvt", "apt", "callsign", "mvt_t"), on=["apt", "callsign"], how="inner")
          .filter((C("t") >= C("mvt_t") - 5400) & (C("t") <= C("mvt_t") + 900))
          .group_by(["MVT_ID_mvt", "icao"]).len().sort("len", descending=True).unique(subset="MVT_ID_mvt", keep="first"))
    return dict(zip(cs["MVT_ID_mvt"].to_list(), cs["icao"].to_list()))


def full_tracks(day, keys):
    """(apt, icao) -> every fix (ground flag, heading) from the day's and the previous day's points files."""
    icaos = list({k[1] for k in keys})
    prev = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    parts = []
    for d in (prev, day):
        f = Path(f"data/adsb/points/{d}.parquet")
        if f.exists():
            parts.append(pl.scan_parquet(f).filter(C("icao").is_in(icaos) & C("apt").is_in(APTS))
                         .select("apt", "icao", "t", "lat", "lon", "alt", "gs", "track").collect())
    p = pl.concat(parts).unique(subset=["apt", "icao", "t"]).sort(["apt", "icao", "t"])
    elev = C("apt").replace_strict(ELEVATION_FT, default=None, return_dtype=pl.Float64)
    p = p.with_columns(((C("alt") == -1) | (((C("alt").cast(pl.Float64) - elev).abs() <= G.slow_low_band_ft)
                                            & (C("gs") <= G.slow_low_gs_kt))).fill_null(False).alias("ground"))
    out = {}
    for (apt, icao), s in p.partition_by(["apt", "icao"], as_dict=True).items():
        if (apt, icao) in keys:
            out[(apt, icao)] = prep(s["t"].to_numpy(), s["lat"].to_numpy().astype(float), s["lon"].to_numpy().astype(float),
                                    s["gs"].fill_null(0.0).to_numpy().astype(float),
                                    s["track"].fill_null(float("nan")).to_numpy().astype(float), s["ground"].to_numpy())
    return out


def prep(t, la, lo, gs, trk, gr):
    """per-airframe arrays + reverse steps + events, computed once per (apt, icao) per day."""
    n = len(t)
    ev = []
    if n >= 2:
        dt = np.diff(t)
        dy = (la[1:] - la[:-1]) * 110540.0
        dx = (lo[1:] - lo[:-1]) * 111320.0 * np.cos(np.radians(la[:-1]))
        d = np.sqrt(dx * dx + dy * dy)
        brg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        ang = np.abs((brg - trk[:-1] + 180.0) % 360.0 - 180.0)
        ok = (gr[:-1] & gr[1:] & (dt <= 90) & (d >= 3) & np.isfinite(trk[:-1])
              & ((gs[:-1] >= 0.5) | (gs[1:] >= 0.5)) & (np.maximum(gs[:-1], gs[1:]) <= 8))
        rv = np.flatnonzero(ok & (ang > 120))
        if len(rv):
            grp = [[rv[0]]]
            for k in rv[1:]:
                if t[k] - t[grp[-1][-1]] <= 180:
                    grp[-1].append(k)
                else:
                    grp.append([k])
            for g_ in grp:
                dist = float(d[g_].sum())
                if len(g_) >= 2 or dist >= 15:
                    ev.append((float(t[g_[0]]), float(t[g_[-1] + 1]), len(g_), dist, float(la[g_[0]]), float(lo[g_[0]])))
    return dict(t=t, la=la, lo=lo, gs=gs, gr=gr, ev=ev)


def mdist(a, b, c, d):
    return float(math.hypot((a - c) * 110540.0, (b - d) * 111320.0 * math.cos(math.radians(a))))


def row_features(tr, mvt, lift, block):
    t, gs, gr, la, lo = tr["t"], tr["gs"], tr["gr"], tr["la"], tr["lo"]
    t_ref = lift if np.isfinite(lift) else mvt
    t_cut = t_ref - 300
    air = np.flatnonzero((~gr) & (t < t_cut) & (t > mvt - LOOKBACK))
    t_air = float(t[air[-1]]) if len(air) else NAN
    w0 = (t_air + 300) if np.isfinite(t_air) else mvt - LOOKBACK   # landing roll + taxi-in reverses are rare anyway
    w1 = t_ref - 60
    evs = [e for e in tr["ev"] if w0 < e[0] <= w1]
    o = {"t_air_prev": t_air, "w0": w0, "n_push": len(evs)}
    gw = np.flatnonzero(gr & (t > w0) & (t <= t_ref))
    o["n_gfix"] = len(gw)
    o["t_gfirst"] = float(t[gw[0]]) if len(gw) else NAN
    # take-off of this airframe near MVT (matching check): first airborne fix after MVT - 900
    to = np.flatnonzero((~gr) & (t >= mvt - 900) & (t <= mvt + 1800))
    o["d_to"] = float(t[to[0]] - mvt) if len(to) else NAN
    o["ev_t0"] = [e[0] for e in evs]
    o["ev_t1"] = [e[1] for e in evs]
    o["ev_n"] = [e[2] for e in evs]
    o["ev_m"] = [e[3] for e in evs]
    if evs:
        f, l = evs[0], evs[-1]
        o["t_first"], o["t_last"] = f[0], l[0]
        o["first_m"], o["last_m"] = f[3], l[3]
        o["d_first_last_m"] = mdist(f[4], f[5], l[4], l[5])
        # between the first push's end and the final push's start (or the take-off if only one push)
        a, b = f[1], (l[0] if len(evs) > 1 else t_ref)
        k = np.flatnonzero(gr & (t >= a) & (t <= b))
        o["max_sil"] = float(np.diff(t[k]).max()) if len(k) >= 2 else (b - a)
        best, bi = 0.0, None
        i = 0
        while i < len(k):
            if gs[k[i]] <= 1.0:
                j = i
                while j + 1 < len(k) and gs[k[j + 1]] <= 1.0 and t[k[j + 1]] - t[k[j]] <= 600:
                    j += 1
                if t[k[j]] - t[k[i]] > best:
                    best, bi = float(t[k[j]] - t[k[i]]), (k[i], k[j])
                i = j + 1
            else:
                i += 1
        o["max_stop"] = best
        o["stop_d_first_m"] = mdist(float(np.median(la[bi[0]:bi[1] + 1])), float(np.median(lo[bi[0]:bi[1] + 1])),
                                    f[4], f[5]) if bi else NAN
        # fast movement (taxi >= 10 kt) between the first push and the final push: did it go somewhere?
        o["taxi_between"] = float((gs[k] >= 10).sum()) if len(k) else 0.0
    else:
        for c in ("t_first", "t_last", "first_m", "last_m", "d_first_last_m", "max_sil", "max_stop", "stop_d_first_m",
                  "taxi_between"):
            o[c] = NAN
    if np.isfinite(block):
        # where was the airframe at the labelled off-block?  (reported only)
        near = np.flatnonzero(gr & (np.abs(t - block) <= 120))
        o["blk_nfix"] = len(near)
        o["blk_gs_max"] = float(gs[near].max()) if len(near) else NAN
        o["blk_air_before"] = bool(np.isfinite(t_air) and block < t_air)
    else:
        o["blk_nfix"], o["blk_gs_max"], o["blk_air_before"] = 0, NAN, False
    return o


rows = []
for di, day in enumerate(DAYS):
    dd = dep.filter(C("day") == day)
    if dd.height == 0:
        continue
    runs, touch, tracks = runs_and_tracks(day, G)
    del touch, tracks
    cand, nc, both = match_departures(dd, runs, G)
    del runs
    both = both.unique(subset="MVT_ID_mvt", keep="first")
    un = dd.join(both.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").filter(C("callsign").is_not_null())
    tx = taxinm_icao(un, day)
    ur = un.filter(C("MVT_ID_mvt").is_in(list(tx.keys())))
    todo = [(r["MVT_ID_mvt"], r["apt"], r["icao"], bool(r["plan"]), r["mvt_t"], r["liftoff_t"], r["block_t"], "np1")
            for r in both.select("MVT_ID_mvt", "apt", "icao", "plan", "mvt_t", "liftoff_t", "block_t").iter_rows(named=True)]
    todo += [(r["MVT_ID_mvt"], r["apt"], tx[r["MVT_ID_mvt"]], bool(r["plan"]), r["mvt_t"], NAN, r["block_t"], "taxinm")
             for r in ur.select("MVT_ID_mvt", "apt", "plan", "mvt_t", "block_t").iter_rows(named=True)]
    if ONLY is not None:
        todo = [x for x in todo if x[0] in ONLY]
    keys = {(x[1], x[2]) for x in todo}
    ft = full_tracks(day, keys) if keys else {}
    for mid, apt, icao, plan, mvt, lift, block, srcn in todo:
        tr = ft.get((apt, icao))
        if tr is None:
            continue
        o = row_features(tr, mvt, lift if lift is not None else NAN, block if block is not None else NAN)
        rows.append({"MVT_ID_mvt": mid, "day": day, "apt": apt, "icao": icao, "plan": plan, "src": srcn,
                     "lirf_np": apt == "LIRF" and not plan, "mvt_t": mvt,
                     "liftoff_t": lift if lift is not None else NAN, "block_t": block, **o})
    del ft
    if (di + 1) % 5 == 0 or di + 1 == len(DAYS):
        log(f"{day}: {dd.height:,} deps, NP1 {both.height:,}, taxinm {ur.height:,}, rows so far {len(rows):,} ({di + 1}/{len(DAYS)})")

out = pl.DataFrame(rows, infer_schema_length=None)
out.write_parquet(args.out)
log(f"wrote {out.height:,} rows -> {args.out}")
