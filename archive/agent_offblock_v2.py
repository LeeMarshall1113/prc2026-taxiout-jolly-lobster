"""experiment offblock: the v2 off-block reader, ONE implementation shared by the dump (archive/agent_offblock_01_dump.py,
whose inline copy this is, lines 94-285 verbatim apart from the APTS/G parameters) and the stage-1 evaluator copy
(archive/agent_offblock_stage1_eval.py). docs/LAB-NOTES.md.

v2(t, lat, lon, gs, heading, t_begin, t_end, stand_c) -> dict over V2: first reverse movement (push-back: the
displacement between fixes points > 120 deg away from the reported heading), back-extrapolation of a push already in
progress when first seen, parked-dwell end / first 5-20 m from it, and their quality columns. Label-free.
"""
import math

import numpy as np
import polars as pl

from archive.agent_noplan_adsb_match import ELEVATION_FT, Gates, metres

C = pl.col
G = Gates()   # only gap_s, floor_kt, floor_turn_s, slow_low_band_ft, slow_low_gs_kt are read: production defaults
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
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




# columns the stacks see: estimated off-block times enter as (MVT - e) - p, like x - p; the rest raw
V2_REL = ("e_revany", "e_rev0x", "e_pushend", "e_mv10", "e_dend")
V2_RAW = ("push", "rev_gs", "rev_dist", "rev_dstand", "revany_gap", "first_gs", "first_dstand", "P_span", "P_gap",
          "P_dstand", "seg_start_gap")
V2_STACK_COLS = V2_REL + V2_RAW
