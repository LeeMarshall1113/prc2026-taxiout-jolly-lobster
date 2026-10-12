"""experiment restack3 01: COPY of archive/agent_offblock_01_dump.py (unchanged matching, stand map and v2 columns) that ALSO
writes production's anchor columns (scripts/agent_adsbq_anchors.anchors, 14 ANCHOR_COLS) and track-shape columns
(scripts/agent_trackfeat_features.trackfeats, 8 TRACKFEAT_COLS) on NP1 rows (taxinm rows: NaN, as production).
For variant B TRK of docs/LAB-NOTES.md. Original docstring follows.

experiment offblock 01: per-row dump of every off-block estimator on matched departures (docs/LAB-NOTES.md).

    python archive/agent_offblock_01_dump.py --mode 2025 --days D1,D2,... --out data/interim/agent_offblock_rows_<tag>.parquet
    python archive/agent_offblock_01_dump.py --mode 2026 --days ...   (label-free: no block time read for departures)

Matching = production NP1 (air_cs_all, LIRF skip, noplan_callsign) + taxi-phase NM matching (taxinm, no lift-off
gate) for departures NP1 misses, at the 9 covered airports. Stand map = production stand_map() over the saved sample
pools (data/interim/agent_adsbq_samples.parquet 89 Jan-Sep days + agent_frontier_samples_on.parquet 31 Oct-Nov days),
leave-one-day-out for departure samples, OSM fill as production. In 2026 mode the pool is all 120 days (no 2026
departure labels exist or are read).

Per row, every estimate is an OFF-BLOCK TIME (epoch s), NaN if unavailable:
  prod_kind / prod_t   production estimate() cascade (stand, appear; tight/start off as shipped)
  e_stand              last in-150 m fix before leaving (only when the stand gate passes; = prod when kind=stand)
  e_appear             run start when continuous and within 600 m of the stand (production's appear, independently)
  e_start              first ground fix of the departure segment (no stand needed)
  e_ci20               interpolated crossing of the 20 m circle round the stand centroid (ungated)
  lb_t                 production departure_floor anchor (first >=10 kt fix of the departure part)
v2 candidates (raw points incl. the reported `track` = heading on the surface, see _00_probe):
  P = the parked dwell (consecutive fixes <= 1 kt spanning >= 30 s): the last one >= 600 s if any (the turnaround),
      else the first slow run of the segment if it starts within 120 s of the segment start (transponder on while
      parked); must end before the first >= 10 kt fix
  e_dend               last fix of P (last seen stationary)
  e_dnext              first fix after P
  e_mv5/10/20          interpolated time the aircraft first gets 5/10/20 m from P's centroid after P
  e_rev                first fix after P start whose displacement to the next fix (>= 3 m) points > 120 deg away
                       from its reported heading (reverse = push-back), within 900 s after P
  e_hdg                first fix after P whose heading differs from P's median heading by > 20 deg
  e_gs1                first fix after P with gs >= 1 kt
  e_rev0               no P: first fix of the segment, when it is ALREADY reversing (push in progress when seen)
Quality: P_span, P_n, P_gap (e_dnext - e_dend), P_dstand (P centroid to mapped stand), seg_n, seg_rate (fixes/min in
the 600 s after P end or segment start), push (reverse seen), first_gs, dist of first fix to stand.
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_adsbq_anchors import ANCHOR_COLS, anchors
from archive.agent_trackfeat_features import TRACKFEAT_COLS, trackfeats
from archive.agent_noplan_adsb_match import (ELEVATION_FT, Gates, add_derived_callsign, departure_floor, estimate,
                                             flight_callsign_maps, match_departures, metres, runs_and_tracks, stand_map)

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["2025", "2026"], default="2025")
ap.add_argument("--days", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--apts", default="EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH")
args = ap.parse_args()
DAYS = args.days.split(",")
APTS = args.apts.split(",")
G = Gates(air_cs_all=True, air_cs_skip=("LIRF",), noplan_callsign=True)
T0 = time.time()


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
        C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"),
        C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int32).alias("hour")]
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
log(f"{args.mode}: {dep.height:,} departures on {len(DAYS)} days")

pool = pl.concat([pl.read_parquet("data/interim/agent_adsbq_samples.parquet"),
                  pl.read_parquet("data/interim/agent_frontier_samples_on.parquet")], how="vertical_relaxed")
log(f"stand-sample pool {pool.height:,} rows, {pool['day'].n_unique()} days")

NAN = float("nan")


def ang_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def slow_runs(gs, t, thr=1.0):
    """maximal runs of consecutive fixes with gs <= thr: list of (i, j, span_s)."""
    out, i, n = [], 0, len(gs)
    while i < n:
        if gs[i] <= thr:
            j = i
            while j + 1 < n and gs[j + 1] <= thr:
                j += 1
            out.append((i, j, t[j] - t[i]))
            i = j + 1
        else:
            i += 1
    return out


def cross_time(t, la, lo, i0, clat, clon, R):
    """first time after index i0 the track is > R m from (clat, clon), interpolated."""
    for k in range(i0 + 1, len(t)):
        d = metres(la[k], lo[k], clat, clon)
        if d > R:
            dp = metres(la[k - 1], lo[k - 1], clat, clon)
            f = 0.0 if d <= dp else min(max((R - dp) / (d - dp), 0.0), 1.0)
            return t[k - 1] + f * (t[k] - t[k - 1])
    return NAN


V2 = ("e_dend", "e_dnext", "e_mv5", "e_mv10", "e_mv20", "e_rev", "e_hdg", "e_gs1", "e_rev0",
      "P_span", "P_n", "P_gap", "P_dstand", "P_long", "seg_n", "seg_rate", "push", "first_gs", "first_dstand",
      "e_start", "seg_start_gap", "e_revany", "e_revany_prev", "revany_gap", "e_pushend", "rev_dist", "rev_gs",
      "rev_dstand", "e_rev0x")


def v2(t, la, lo, gs, trk, t_begin, t_end, stand_c):
    out = dict.fromkeys(V2, NAN)
    k = np.flatnonzero((t >= t_begin) & (t <= t_end))
    if len(k) < 1:
        return out
    t, la, lo, gs, trk = t[k], la[k], lo[k], gs[k], trk[k]
    brk = np.flatnonzero(np.diff(t) > G.gap_s)
    if len(brk):
        s = brk[-1] + 1
        out["seg_start_gap"] = float(t[s] - t[s - 1])
        t, la, lo, gs, trk = t[s:], la[s:], lo[s:], gs[s:], trk[s:]
    n = len(t)
    out["seg_n"] = float(n)
    out["e_start"] = float(t[0])
    out["first_gs"] = float(gs[0])
    if stand_c is not None:
        out["first_dstand"] = float(metres(la[0], lo[0], stand_c[0], stand_c[1]))
    fast = np.flatnonzero(gs >= G.floor_kt)
    # first reverse movement anywhere in the departure part (after the last >= 600 s stop starts, before the
    # first >= 10 kt fix after it): the push-back, whether or not a parked dwell was seen first
    runs0 = slow_runs(gs, t)
    longs0 = [r for r in runs0 if r[2] >= G.floor_turn_s]
    a0 = longs0[-1][0] if longs0 else 0
    ff = fast[fast > a0]
    b0 = ff[0] if len(ff) else n - 1
    revs = []
    for k2 in range(a0, min(b0, n - 1)):
        d = metres(la[k2 + 1], lo[k2 + 1], la[k2], lo[k2])
        if d >= 3 and np.isfinite(trk[k2]) and (gs[k2] >= 0.5 or gs[k2 + 1] >= 0.5):
            brg = (math.degrees(math.atan2((lo[k2 + 1] - lo[k2]) * 111320.0 * math.cos(math.radians(la[k2])),
                                           (la[k2 + 1] - la[k2]) * 110540.0)) + 360) % 360
            revs.append((k2, ang_diff(brg, trk[k2]) > 120))
    first_rev = next((k2 for k2, rv in revs if rv), None)
    if first_rev is not None:
        out["e_revany"] = float(t[first_rev])
        out["e_revany_prev"] = float(t[first_rev - 1]) if first_rev > a0 else NAN
        out["revany_gap"] = float(t[first_rev] - t[first_rev - 1]) if first_rev > a0 else NAN
        last_rev = first_rev
        for k2, rv in revs:
            if k2 > first_rev:
                if rv:
                    last_rev = k2
                else:
                    break
        out["e_pushend"] = float(t[last_rev + 1])
        out["rev_dist"] = float(metres(la[last_rev + 1], lo[last_rev + 1], la[first_rev], lo[first_rev]))
        out["rev_gs"] = float(gs[first_rev]) if gs[first_rev] >= 0.5 else float(gs[first_rev + 1])
        if stand_c is not None:
            out["rev_dstand"] = float(metres(la[first_rev], lo[first_rev], stand_c[0], stand_c[1]))
            if first_rev == 0 and out["rev_gs"] > 0.5:
                out["e_rev0x"] = float(t[0]) - out["rev_dstand"] / (out["rev_gs"] * 0.5144)
    runs = runs0
    longs = [r for r in runs if r[2] >= G.floor_turn_s]
    m_lim = n
    P = None
    if longs:
        P = longs[-1]
        if stand_c is not None:  # prefer the last long stop AT the stand over a de-icing pad / runway hold
            at = [r for r in longs if metres(float(np.median(la[r[0]:r[1] + 1])), float(np.median(lo[r[0]:r[1] + 1])),
                                             stand_c[0], stand_c[1]) <= 150]
            if at:
                P = at[-1]
        out["P_long"] = 1.0
    elif runs and runs[0][2] >= 30 and t[runs[0][0]] - t[0] <= 120:
        P = runs[0]
        out["P_long"] = 0.0
    if P is not None:
        after_fast = fast[fast > P[1]]
        if not len(after_fast) and len(fast) and fast.max() < P[0]:
            P = None  # the dwell is after the last fast fix: an arrival, not this departure
    if P is None:
        # push already in progress when first seen?
        if n >= 2:
            dd = metres(la[1], lo[1], la[0], lo[0])
            if dd >= 3 and np.isfinite(trk[0]):
                brg = (math.degrees(math.atan2((lo[1] - lo[0]) * 111320.0 * math.cos(math.radians(la[0])),
                                               (la[1] - la[0]) * 110540.0)) + 360) % 360
                if ang_diff(brg, trk[0]) > 120:
                    out["e_rev0"] = float(t[0])
                    out["push"] = 1.0
        w = np.flatnonzero((t >= t[0]) & (t <= t[0] + 600))
        out["seg_rate"] = float(len(w)) / 10.0
        return out
    i, j, span = P
    out["P_span"], out["P_n"] = float(span), float(j - i + 1)
    clat, clon = float(np.median(la[i:j + 1])), float(np.median(lo[i:j + 1]))
    if stand_c is not None:
        out["P_dstand"] = float(metres(clat, clon, stand_c[0], stand_c[1]))
    out["e_dend"] = float(t[j])
    if j + 1 < n:
        out["e_dnext"] = float(t[j + 1])
        out["P_gap"] = float(t[j + 1] - t[j])
    w = np.flatnonzero((t >= t[j]) & (t <= t[j] + 600))
    out["seg_rate"] = float(len(w)) / 10.0
    for R in (5, 10, 20):
        out[f"e_mv{R}"] = float(cross_time(t, la, lo, j, clat, clon, R))
    hdg0 = np.nanmedian(trk[i:j + 1]) if np.isfinite(trk[i:j + 1]).any() else NAN
    lim = t[j] + 900
    for k2 in range(j, n - 1):
        if t[k2] > lim:
            break
        d = metres(la[k2 + 1], lo[k2 + 1], la[k2], lo[k2])
        if d >= 3 and np.isfinite(trk[k2]) and (gs[k2] >= 0.5 or gs[k2 + 1] >= 0.5):
            brg = (math.degrees(math.atan2((lo[k2 + 1] - lo[k2]) * 111320.0 * math.cos(math.radians(la[k2])),
                                           (la[k2 + 1] - la[k2]) * 110540.0)) + 360) % 360
            if ang_diff(brg, trk[k2]) > 120:
                out["e_rev"] = float(t[k2])
                out["push"] = 1.0
                break
    if not np.isfinite(out["push"]):
        out["push"] = 0.0
    for k2 in range(j + 1, n):
        if t[k2] > lim:
            break
        if np.isfinite(hdg0) and np.isfinite(trk[k2]) and ang_diff(trk[k2], hdg0) > 20:
            out["e_hdg"] = float(t[k2])
            break
    for k2 in range(j + 1, n):
        if gs[k2] >= 1.0:
            out["e_gs1"] = float(t[k2])
            break
    return out


def raw_tracks(day, keys):
    """(apt, icao) -> ground-fix arrays with heading, for the aircraft in `keys`."""
    icaos = list({k[1] for k in keys})
    p = (pl.scan_parquet(f"data/adsb/points/{day}.parquet").filter(C("icao").is_in(icaos) & C("apt").is_in(APTS))
         .select("apt", "icao", "t", "lat", "lon", "alt", "gs", "track").collect().sort(["apt", "icao", "t"]))
    elev = C("apt").replace_strict(ELEVATION_FT, default=None, return_dtype=pl.Float64)
    p = p.filter(((C("alt") == -1) | (((C("alt").cast(pl.Float64) - elev).abs() <= G.slow_low_band_ft)
                                      & (C("gs") <= G.slow_low_gs_kt))).fill_null(False))
    out = {}
    for (apt, icao), s in p.partition_by(["apt", "icao"], as_dict=True).items():
        if (apt, icao) in keys:
            out[(apt, icao)] = (s["t"].to_numpy(), s["lat"].to_numpy().astype(float), s["lon"].to_numpy().astype(float),
                                s["gs"].fill_null(0.0).to_numpy().astype(float),
                                s["track"].fill_null(float("nan")).to_numpy().astype(float))
    return out


def taxinm_icao(deps, day):
    """NM-callsign -> icao for departures NP1 missed (agent_taxinm_features' rule: most fixes in [mvt-5400, mvt+900])."""
    if deps.height == 0:
        return {}
    p = (pl.scan_parquet(f"data/adsb/points/{day}.parquet").filter(C("apt").is_in(APTS) & C("callsign").is_not_null())
         .select("apt", "icao", C("callsign").str.strip_chars().str.to_uppercase().alias("callsign"), "t").collect())
    cs = (p.join(deps.select("MVT_ID_mvt", "apt", "callsign", "mvt_t"), on=["apt", "callsign"], how="inner")
          .filter((C("t") >= C("mvt_t") - 5400) & (C("t") <= C("mvt_t") + 900))
          .group_by(["MVT_ID_mvt", "icao"]).len().sort("len", descending=True).unique(subset="MVT_ID_mvt", keep="first"))
    return dict(zip(cs["MVT_ID_mvt"].to_list(), cs["icao"].to_list()))


rows = []
for di, day in enumerate(DAYS):
    dd = dep.filter(C("day") == day)
    if dd.height == 0:
        continue
    runs, touch, tracks = runs_and_tracks(day, G)
    cand, nc, both = match_departures(dd, runs, G)
    del runs, touch
    both = both.unique(subset="MVT_ID_mvt", keep="first")
    un = dd.join(both.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").filter(C("callsign").is_not_null())
    tx = taxinm_icao(un, day)
    keys = set(zip(both["apt"].to_list(), both["icao"].to_list()))
    ur = un.filter(C("MVT_ID_mvt").is_in(list(tx.keys())))
    keys |= {(a, tx[m]) for a, m in zip(ur["apt"].to_list(), ur["MVT_ID_mvt"].to_list())}
    rt = raw_tracks(day, keys)
    tracks = {k: v for k, v in tracks.items() if k in keys}
    src_pool = pool.filter((C("src") != "dep") | (C("day") != day)) if args.mode == "2025" else pool
    cmap = stand_map(src_pool, G)
    for r in both.iter_rows(named=True):
        kind, taxi = estimate(r, cmap, tracks, G)
        c = cmap.get((r["apt"], r["stand"]))
        # independent stand / appear / ci20
        e_stand = r["mvt_t"] - taxi if kind == "stand" else NAN
        e_appear = NAN
        if c is not None and r["max_gap"] is not None and r["max_gap"] <= G.run_gap_s:
            if float(metres(r["start_lat"], r["start_lon"], c[0], c[1])) <= G.appear_m:
                e_appear = float(r["start_t"])
        e_ci20 = NAN
        tr = tracks.get((r["apt"], r["icao"]))
        if c is not None and tr is not None:
            t, la, lo, gs = tr
            win = np.flatnonzero((t <= r["liftoff_t"]) & (t >= r["liftoff_t"] - G.lookback_s))
            if len(win):
                d = metres(la[win], lo[win], c[0], c[1])
                ins = np.flatnonzero(d <= 20)
                if len(ins):
                    kk = ins.max()
                    jj = win[kk]
                    if jj + 1 < len(t) and t[jj + 1] - t[jj] <= G.stand_next_s:
                        dn = float(metres(la[jj + 1], lo[jj + 1], c[0], c[1]))
                        if dn > d[kk]:
                            f = min(max((20 - d[kk]) / (dn - d[kk]), 0.0), 1.0)
                            e_ci20 = t[jj] + f * (t[jj + 1] - t[jj])
        lb = departure_floor(r, tracks, G)
        rw = rt.get((r["apt"], r["icao"]))
        feats = v2(*rw, r["liftoff_t"] - G.lookback_s, r["liftoff_t"], c) if rw is not None else dict.fromkeys(V2, NAN)
        rows.append({"MVT_ID_mvt": r["MVT_ID_mvt"], "day": day, "apt": r["apt"], "plan": bool(r["plan"]), "src": "np1",
                     "stand": r["stand"], "type": r["type"], "hour": r["hour"], "mvt_t": r["mvt_t"], "block_t": r["block_t"],
                     "liftoff_t": r["liftoff_t"], "has_c": c is not None, "prod_kind": kind,
                     "prod_t": (r["mvt_t"] - taxi) if taxi is not None else NAN, "e_stand": e_stand, "e_appear": e_appear,
                     "e_ci20": e_ci20, "lb_t": (r["mvt_t"] - lb) if lb is not None else NAN, **feats,
                     **anchors(r, cmap, tracks, G, taxi), **trackfeats(r, tracks, G)})
    for r in ur.iter_rows(named=True):
        ic = tx[r["MVT_ID_mvt"]]
        c = cmap.get((r["apt"], r["stand"]))
        rw = rt.get((r["apt"], ic))
        feats = v2(*rw, r["mvt_t"] - 5400, r["mvt_t"] + 60, c) if rw is not None else dict.fromkeys(V2, NAN)
        rows.append({"MVT_ID_mvt": r["MVT_ID_mvt"], "day": day, "apt": r["apt"], "plan": bool(r["plan"]), "src": "taxinm",
                     "stand": r["stand"], "type": r["type"], "hour": r["hour"], "mvt_t": r["mvt_t"], "block_t": r["block_t"],
                     "liftoff_t": NAN, "has_c": c is not None, "prod_kind": None, "prod_t": NAN, "e_stand": NAN,
                     "e_appear": NAN, "e_ci20": NAN, "lb_t": NAN, **feats,
                     **dict.fromkeys(ANCHOR_COLS, NAN), **dict.fromkeys(TRACKFEAT_COLS, NAN)})
    del tracks, rt
    log(f"{day}: {dd.height:,} deps, NP1 matched {both.height:,}, taxinm {ur.height:,}, cmap {len(cmap):,}  ({di + 1}/{len(DAYS)})")

out = pl.DataFrame(rows, infer_schema_length=None)
out.write_parquet(args.out)
log(f"wrote {out.height:,} rows -> {args.out}")
