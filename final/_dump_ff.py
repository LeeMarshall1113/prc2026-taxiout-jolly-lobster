"""First-fix geometry dump for the back-extrapolation columns (stage 20, part b).

Derived from archive/agent_backx_01_dump.py: same matching and stand map, one row per matched departure with the first
ground fix of the departure segment and its position relative to the mapped stand. Changes: the ranking file is an
argument and mode 2026 is the default. Label-free. Run from the repo root.
"""
import argparse
import math
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_noplan_adsb_match import (ELEVATION_FT, Gates, add_derived_callsign, flight_callsign_maps,
                                             match_departures, metres, runs_and_tracks, stand_map)

C = pl.col
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["2025", "2026"], default="2026")
ap.add_argument("--ranking", default="data/raw/final_ranking.parquet")
ap.add_argument("--days", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--apts", default="EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH")
args = ap.parse_args()
DAYS = args.days.split(",")
APTS = args.apts.split(",")
G = Gates(air_cs_all=True, air_cs_skip=("LIRF",), noplan_callsign=True)
T0 = time.time()
NAN = float("nan")


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)
src = "data/raw/training_*.parquet" if args.mode == "2025" else args.ranking
cols = [C("MVT_ID_mvt"), C("ADEP_mvt").alias("apt"), C_DAY.alias("day"),
        C("AOBT_3_flt").is_not_null().alias("plan"),
        C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
        C("STAND_mvt").cast(pl.Utf8).alias("stand"),
        C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
        C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
        C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"),
        C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int32).alias("hour")]
dep = (pl.scan_parquet(src).filter(C("PHASE_mvt") == "DEP").select(cols)
       .filter(C("day").is_in(DAYS) & C("apt").is_in(APTS)).collect())
dep = dep.with_columns(pl.lit(None).cast(pl.Float64).alias("block_t"))  # label-free in both modes
cs_src = [pl.scan_parquet(s).filter(C("PHASE_mvt") == "DEP")
          .select(C("AOBT_3_flt").is_not_null().alias("plan"),
                  C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                  C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight")).collect()
          for s in ("data/raw/training_*.parquet", args.ranking)]
dep = add_derived_callsign(dep, flight_callsign_maps(cs_src))
del cs_src
assert dep["MVT_ID_mvt"].n_unique() == dep.height
log(f"{args.mode}: {dep.height:,} departures on {len(DAYS)} days")

pool = pl.concat([pl.read_parquet("data/interim/agent_adsbq_samples.parquet"),
                  pl.read_parquet("data/interim/agent_frontier_samples_on.parquet")], how="vertical_relaxed")
log(f"stand-sample pool {pool.height:,} rows, {pool['day'].n_unique()} days")

FF = ("e_start", "first_gs", "first_dstand", "ff_lat", "ff_lon", "ff_trk", "st_lat", "st_lon", "brg_st",
      "mv_brg", "mv_dt", "mv_spd", "rel_mv", "n120")


def ang_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def bearing(la0, lo0, la1, lo1):
    return (math.degrees(math.atan2((lo1 - lo0) * 111320.0 * math.cos(math.radians(la0)),
                                    (la1 - la0) * 110540.0)) + 360) % 360


def ff(t, la, lo, gs, trk, t_begin, t_end, stand_c):
    """first fix of the departure segment: the offblock dump's v2() window + last-segment rule, verbatim."""
    out = dict.fromkeys(FF, NAN)
    k = np.flatnonzero((t >= t_begin) & (t <= t_end))
    if len(k) < 1:
        return out
    t, la, lo, gs, trk = t[k], la[k], lo[k], gs[k], trk[k]
    brk = np.flatnonzero(np.diff(t) > G.gap_s)
    if len(brk):
        s = brk[-1] + 1
        t, la, lo, gs, trk = t[s:], la[s:], lo[s:], gs[s:], trk[s:]
    n = len(t)
    out["e_start"] = float(t[0])
    out["first_gs"] = float(gs[0])
    out["ff_lat"], out["ff_lon"] = float(la[0]), float(lo[0])
    out["ff_trk"] = float(trk[0])
    out["n120"] = float(int((t <= t[0] + 120).sum()))
    for k2 in range(1, n):
        if t[k2] - t[0] > 180:
            break
        d = float(metres(la[k2], lo[k2], la[0], lo[0]))
        if d >= 15:
            out["mv_brg"] = bearing(la[0], lo[0], la[k2], lo[k2])
            out["mv_dt"] = float(t[k2] - t[0])
            out["mv_spd"] = d / max(float(t[k2] - t[0]), 1.0)
            break
    if stand_c is not None:
        out["st_lat"], out["st_lon"] = float(stand_c[0]), float(stand_c[1])
        out["first_dstand"] = float(metres(la[0], lo[0], stand_c[0], stand_c[1]))
        if out["first_dstand"] >= 5:
            out["brg_st"] = bearing(stand_c[0], stand_c[1], la[0], lo[0])
            if np.isfinite(out["mv_brg"]):
                out["rel_mv"] = ang_diff(out["mv_brg"], out["brg_st"])
    return out


def raw_tracks(day, keys):
    """(apt, icao) -> ground-fix arrays with heading, for the aircraft in `keys` (offblock dump, verbatim)."""
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
    """NM-callsign -> icao for departures NP1 missed (offblock dump, verbatim)."""
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
    del runs, touch, tracks
    both = both.unique(subset="MVT_ID_mvt", keep="first")
    un = dd.join(both.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti").filter(C("callsign").is_not_null())
    tx = taxinm_icao(un, day)
    keys = set(zip(both["apt"].to_list(), both["icao"].to_list()))
    ur = un.filter(C("MVT_ID_mvt").is_in(list(tx.keys())))
    keys |= {(a, tx[m]) for a, m in zip(ur["apt"].to_list(), ur["MVT_ID_mvt"].to_list())}
    rt = raw_tracks(day, keys)
    src_pool = pool.filter((C("src") != "dep") | (C("day") != day)) if args.mode == "2025" else pool
    cmap = stand_map(src_pool, G)
    for r in both.iter_rows(named=True):
        c = cmap.get((r["apt"], r["stand"]))
        rw = rt.get((r["apt"], r["icao"]))
        f = ff(*rw, r["liftoff_t"] - G.lookback_s, r["liftoff_t"], c) if rw is not None else dict.fromkeys(FF, NAN)
        rows.append({"MVT_ID_mvt": r["MVT_ID_mvt"], "day": day, "apt": r["apt"], "src": "np1", **f})
    for r in ur.iter_rows(named=True):
        c = cmap.get((r["apt"], r["stand"]))
        rw = rt.get((r["apt"], tx[r["MVT_ID_mvt"]]))
        f = ff(*rw, r["mvt_t"] - 5400, r["mvt_t"] + 60, c) if rw is not None else dict.fromkeys(FF, NAN)
        rows.append({"MVT_ID_mvt": r["MVT_ID_mvt"], "day": day, "apt": r["apt"], "src": "taxinm", **f})
    del rt
    log(f"{day}: {dd.height:,} deps, NP1 {both.height:,}, taxinm {ur.height:,}  ({di + 1}/{len(DAYS)})")

out = pl.DataFrame(rows, infer_schema_length=None)
out.write_parquet(args.out)
log(f"wrote {out.height:,} rows -> {args.out}")
