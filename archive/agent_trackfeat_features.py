"""experiment trackfeat: NEW per-flight features computed from the ADS-B ground track, distinct from
archive/agent_adsbq_anchors.py's off-block-time anchors. ONE implementation, imported by both the
stage-1 evaluator (archive/agent_trackfeat_stage1_eval.py) and any future serving copy.

Label-free: everything comes from the row's own ground track (tracks[(apt, icao)]) and its matched
run (start_t = run start, liftoff_t = lift-off). Plan: docs/LAB-NOTES.md
("trackfeat" tag). Anchors measure WHEN the aircraft crossed a stand-centred circle; these measure
the SHAPE of the taxi-out itself -- how far it rolled, how often and how long it stopped, and
whether there is a late stop (runway holding point) separate from the initial gate dwell.

TRACKFEAT_COLS, in order:
  tf_path_m       path length (metres) along the track from start_t to liftoff_t (candidate: "path
                   length along the track")
  tf_nstops       number of stops (>= STOP_MIN_S seconds at <= park_kt) between start_t and liftoff_t
  tf_stop_s       total stopped time across those stops
  tf_maxstop_s    the longest single stop
  tf_movegap_s    the gate-dwell time still inside the observed run: the duration of the FIRST stop,
                   only if it begins within START_NEAR_S of start_t (candidate: "the gap between the
                   last parked fix and the first moving fix"). 0 if the run starts already moving.
  tf_holdrun_s    duration of the LAST stop, when it is a DIFFERENT stop from the first one -- a late
                   stop after real taxiing has happened, i.e. a runway-holding-point proxy (candidate:
                   "time at the runway holding point"). NaN if there is no such second stop.
  tf_fmhold_s     time from the end of the initial dwell (first real movement) to the start of that
                   late stop (candidate: "time from first movement to the runway entry", proxied by
                   the holding stop since runways are not geofenced here). NaN with tf_holdrun_s.
  tf_avgspd_kt    path_m / (liftoff_t - start_t), in knots -- a cheap summary of how congested the
                   taxi-out was users of tf_path_m and tf_stop_s.

NOT implemented here (candidates sized and set aside, see docs/LAB-NOTES.md): towing
(no heading in the extracted points, so a reverse segment cannot be told from a slow forward one)
and departure-queue position from other aircraft's tracks (needs a per-airport, per-runway
cross-aircraft join -- a different cost class from the other four, which are all this-row-only).
"""
import numpy as np

STOP_MIN_S = 20.0       # a slow segment must last this long to count as a "stop" (noise floor)
START_NEAR_S = 90.0     # the first stop counts as the initial gate dwell only if it starts this
                         # close to start_t (otherwise the run was already rolling when reception began)
PARK_KT = 1.0            # <= this ground speed is "stopped" (same threshold departure_floor uses
                         # by default for g.floor_park_kt; kept as its own constant so a change to
                         # Gates does not silently change this module)

TRACKFEAT_COLS = ("tf_path_m", "tf_nstops", "tf_stop_s", "tf_maxstop_s", "tf_movegap_s",
                   "tf_holdrun_s", "tf_fmhold_s", "tf_avgspd_kt")


def _stops(t, gs, park_kt=PARK_KT, min_s=STOP_MIN_S):
    """[(t_begin, t_end, duration), ...] for consecutive-fix slow segments, in order."""
    out = []
    slow = gs <= park_kt
    i, n = 0, len(t)
    while i < n:
        if slow[i]:
            j = i
            while j + 1 < n and slow[j + 1]:
                j += 1
            dur = float(t[j] - t[i])
            if dur >= min_s:
                out.append((float(t[i]), float(t[j]), dur))
            i = j + 1
        else:
            i += 1
    return out


def trackfeats(r: dict, tracks: dict, g) -> dict:
    """TRACKFEAT_COLS for one matched departure; NaN/0 where not available."""
    nan = float("nan")
    out = dict.fromkeys(TRACKFEAT_COLS, nan)
    tr = tracks.get((r["apt"], r["icao"]))
    if tr is None or r.get("start_t") is None or r.get("liftoff_t") is None:
        return out
    t, la, lo, gs = tr
    idx = np.flatnonzero((t >= r["start_t"]) & (t <= r["liftoff_t"]))
    if len(idx) < 2:
        return out
    dur = float(t[idx[-1]] - t[idx[0]])
    # path length: sum of consecutive flat-earth distances between fixes in the run window (same
    # approximation metres() uses, applied leg-by-leg instead of against one fixed centroid)
    seglat0, seglon0 = la[idx[:-1]], lo[idx[:-1]]
    seglat1, seglon1 = la[idx[1:]], lo[idx[1:]]
    # metres() takes (lat, lon, lat0, lon0) with lat0/lon0 scalars normally, but it is plain numpy
    # arithmetic so paired arrays work too (vectorised leg-by-leg distance)
    dy = (seglat1 - seglat0) * 110_540.0
    dx = (seglon1 - seglon0) * 111_320.0 * np.cos(np.radians(seglat0))
    path_m = float(np.sqrt(dx * dx + dy * dy).sum())
    out["tf_path_m"] = path_m
    out["tf_avgspd_kt"] = (path_m / 1852.0) / max(dur / 3600.0, 1e-6) if dur > 0 else nan

    stops = _stops(t[idx], gs[idx])
    out["tf_nstops"] = float(len(stops))
    if stops:
        durs = [s[2] for s in stops]
        out["tf_stop_s"] = float(sum(durs))
        out["tf_maxstop_s"] = float(max(durs))
        first = stops[0]
        t0 = float(t[idx[0]])
        if first[0] - t0 <= START_NEAR_S:
            out["tf_movegap_s"] = first[2]
            fm_t = first[1]   # first real movement = end of the initial dwell
            rest = stops[1:]
        else:
            out["tf_movegap_s"] = 0.0
            fm_t = t0
            rest = stops
        if rest:
            hold = rest[-1]
            out["tf_holdrun_s"] = hold[2]
            out["tf_fmhold_s"] = float(hold[0] - fm_t)
    else:
        out["tf_stop_s"] = 0.0
        out["tf_maxstop_s"] = 0.0
        out["tf_movegap_s"] = 0.0
    return out
