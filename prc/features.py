"""Feature construction.

Built against the real schema (see docs/LAB-NOTES.md), not the
website's description of it.

The one structural fact that shapes everything here: a missing flight-plan row
matters enormously, but **only at one airport**. `LIRF + no flight plan` is 1,488
rows, 0.071% of the data, holding 37.4% of the target's total variance with a
conditional mean of 6,531s. Away from Rome a missing flight plan is
unremarkable — 1,019s against a global 991s. So the signal is the *interaction*,
carried by ``airport_plan``, not ``has_flight_plan`` on its own; the flag alone
averages Rome together with 21,000 harmless rows.

It is left as a general per-airport crossing rather than a hard-coded LIRF flag:
it is the same information without hand-picking an airport, the other nine get
to contribute, and it does not silently break if the fault moves or a second
airport develops one. The missingness of every ``_flt`` column is preserved
rather than imputed.

Nothing here uses ``BLOCK_TIME_UTC_mvt`` or ``TAXITIME_SEC_mvt``: both are
blanked for departures in the ranking file. ``MVT_TIME_UTC_mvt`` (wheels-up) is
given for every row and is fair game.
"""

from __future__ import annotations

import numpy as np
import polars as pl

CATEGORICAL = [
    "ADEP_mvt",
    "ADES_mvt",
    "RUNWAY_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
    "AIRCRAFT_OPERATOR_flt",
    "MARKET_SEGMENT_flt",
    "WK_TBL_CAT_flt",
    "FLIGHT_TYPE_flt",
    "FLIGHT_RULE_mvt",
    "callsign_op",
    "airport_plan",
    "ref_level",
]

# Explicit differences between Network Manager clocks, added 2026-09-16.
#
# A tree can only threshold each column separately, so it cannot form the
# difference of two timestamps even when both gaps are present -- the argument
# that already justified lobt_minus_aobt, dep_delay (AOBT_3 - EOBT_1) and
# sched_vs_eobt (SCHED - EOBT_1), which exist and are NOT repeated here. A public write-up by another team reports a gain from this
# family on a model without our linear gap baseline, which reaches much of it already. Measured here against a control holding the clock
# features that already ship: a residual model with these added beats the
# control by -0.33s and -0.31s on the two folds. Small, consistent, unconfirmed.
#
# Pure per-row arithmetic: no aggregation over the frame, so none of the
# January-and-July-in-one-frame bugs. All null when the flight plan is missing.
# Appended LAST in NUMERIC so no existing column index moves, and dropped by
# default in final.DROP until a fold test passes; bulk model only (NOPLAN_DROP).
CLOCK_FEATURES = (
    "clk_lobt_minus_sched",    # LOBT - SCHED: the last planned off-block against the timetable
    "clk_eobt_minus_lobt",     # EOBT_1 - LOBT: how far the plan moved after the first estimate
    "clk_iobt_minus_lobt",     # IOBT - LOBT: initial against last
    "clk_aobt_minus_sched",    # AOBT_3 - SCHED: NM's actual off-block against the timetable
    "clk_arvt3_minus_mvt",     # ARVT_3 - MVT: NM's destination arrival against our take-off
    "clk_arvt3_minus_arvt1",   # ARVT_3 - ARVT_1: drift in NM's arrival estimate
    "clk_arvt3_minus_aobt",    # ARVT_3 - AOBT_3: NM's gate-to-gate for this flight
    "clk_sec_aobt",            # seconds within the minute: a minute-resolution clock sits on :00
    "clk_sec_mvt",
    "clk_sec_sched",
)

_LATE_LO_S, _LATE_HI_S = -3600.0, 10800.0

# The seven the screen validated (family LOFO -1.341s demeaned, all five folds negative,
# -1.231s after deleting the top 100 rows). Appended at the END of NUMERIC so no existing
# column index moves.
LATE_NEIGHBOUR_FEATURES = (
    "late_apt_past15_mean", "late_apt_fol15_mean", "late_rwy_past15_mean",
    "late_rwy_fol15_mean", "late_apt_past60_mean", "late_rwy_past60_mean",
    "late_apt_past60_own_minus",
)

NUMERIC = [
    "hour",
    "minute_of_day",
    "dow",
    "month",
    "doy",
    "is_weekend",
    "has_flight_plan",
    "gap_aobt",
    "gap_eobt",
    "gap_sched",
    "gap_lobt",
    "lobt_minus_aobt",
    "dep_delay",
    "sched_vs_eobt",
    "mvt_sec_00",
    "arr_sub_day",
    "arr_sub_stand",
    "turn_age",
    "turn_taxi_in",
    "turn_type_match",
    "turn_op_match",
    "turn_in_delay",
    "turn_sched_slack",
    "turn_rwy_match",
    "dep_30min",
    "arr_30min",
    "dep_60min",
    "arr_60min",
    "dep_rwy_30min",
    "dep_rwy_headway",
    "arr_taxi_60min",
    "arr_taxi_180min",
    "arr_taxi_trend",
    "arr_slow_60min",
    "arr_slow_180min",
    "sched_demand_30min",
    "rwy_dep_active",
    "rwy_dep_share",
    "rwy_arr_active",
    "rwy_mixed_mode",
    "wake_sep_req",
    "wake_prev_heavy",
    "wake_heavy_60min",
    "cfg_shift",
    "cfg_new",
    "prev_stand_gap",
    "prev_stand_headway",
    "prev_rwy_gap",
    "stand_runway_pair_n",
    "ref_taxi_s",
    *__import__("prc.weather", fromlist=["FEATURES"]).FEATURES,
    # NM neighbour and following context -- _nm_neighbours. Appended LAST so
    # every existing feature keeps its column index.
    "nm_apt_past15_mean",
    "nm_apt_past15_n",
    "nm_apt_fol15_mean",
    "nm_apt_fol15_n",
    "nm_apt_past60_mean",
    "nm_apt_past60_n",
    "nm_apt_fol60_mean",
    "nm_apt_fol60_n",
    "nm_rwy_past15_mean",
    "nm_rwy_past15_n",
    "nm_rwy_fol15_mean",
    "nm_rwy_fol15_n",
    "nm_rwy_past60_mean",
    "nm_rwy_past60_n",
    "nm_rwy_fol60_mean",
    "nm_rwy_fol60_n",
    "nm_rwy_past60_own_minus",
    "nm_rwy_fol60_own_minus",
    "nm_rwy_next_wait",
    "nm_rwy_next_proxy",
] + list(CLOCK_FEATURES) + list(LATE_NEIGHBOUR_FEATURES)

FEATURES = CATEGORICAL + NUMERIC
TARGET = "TAXITIME_SEC_mvt"

# Fitted on training rows by prc.reference and attached afterwards, so build()
# cannot produce them and must not try to select them.
REFERENCE = ["ref_taxi_s", "ref_level"]


def _window_counts(times: np.ndarray, airports: np.ndarray, half_window_s: int) -> np.ndarray:
    """For each row, how many movements share its airport within +/- the window.

    Counted over whichever movements are passed in, so callers control whether
    that means departures, arrivals or both. Uses searchsorted per airport, which
    is exact and cheap; a rolling join would need the frame sorted globally.
    """
    out = np.zeros(len(times), dtype=np.int32)
    order = np.argsort(airports, kind="stable")
    sorted_airports = airports[order]
    bounds = np.searchsorted(sorted_airports, np.unique(sorted_airports), side="left")
    bounds = np.append(bounds, len(sorted_airports))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        idx = order[start:stop]
        t = times[idx]
        t_order = np.argsort(t, kind="stable")
        t_sorted = t[t_order]
        lo = np.searchsorted(t_sorted, t_sorted - half_window_s, side="left")
        hi = np.searchsorted(t_sorted, t_sorted + half_window_s, side="right")
        counts = (hi - lo - 1).astype(np.int32)  # exclude self
        out[idx[t_order]] = counts
    return out


_MAX_HEADWAY_S = 6 * 3600


def _prev_gap(times: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Seconds since the previous movement sharing the same key. -1 if first."""
    out = np.full(len(times), -1.0)
    order = np.lexsort((times, keys))
    k, t = keys[order], times[order]
    same = np.empty(len(order), dtype=bool)
    same[0] = False
    same[1:] = k[1:] == k[:-1]
    gaps = np.empty(len(order))
    gaps[0] = -1.0
    gaps[1:] = np.where(same[1:], t[1:] - t[:-1], -1.0)
    out[order] = gaps
    return out


def _trailing_count(query_times, query_keys, src_times, src_keys, window_s):
    """How MANY source rows fall in the window before each query time.

    _trailing_mean answers a different question and returns the mean, so using
    it with a constant value of 1 yields 1.0 wherever the window is non-empty
    and NaN otherwise -- an occupancy flag, not a count. Anything wanting a
    share of traffic needs this instead.
    """
    out = np.zeros(len(query_times))
    for key in np.unique(query_keys):
        q = np.flatnonzero(query_keys == key)
        srcs = np.flatnonzero(src_keys == key)
        if not len(q) or not len(srcs):
            continue
        st = np.sort(src_times[srcs])
        qt = query_times[q]
        out[q] = (np.searchsorted(st, qt, side="right")
                  - np.searchsorted(st, qt - window_s, side="left"))
    return out


def _trailing_mean(
    query_times: np.ndarray,
    query_keys: np.ndarray,
    src_times: np.ndarray,
    src_keys: np.ndarray,
    src_values: np.ndarray,
    window_s: int,
) -> np.ndarray:
    """Mean of src_values over the ``window_s`` seconds before each query time.

    Strictly backward-looking, so it stays honest: only movements that had
    already happened can inform a prediction. NaN where the window is empty.
    """
    out = np.full(len(query_times), np.nan)
    for key in np.unique(query_keys):
        q = np.flatnonzero(query_keys == key)
        s = np.flatnonzero(src_keys == key)
        if not len(q) or not len(s):
            continue
        order = np.argsort(src_times[s], kind="stable")
        st, sv = src_times[s][order], src_values[s][order]
        csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(sv))])
        ccnt = np.concatenate([[0.0], np.cumsum(~np.isnan(sv))])
        qt = query_times[q]
        hi = np.searchsorted(st, qt, side="right")
        lo = np.searchsorted(st, qt - window_s, side="left")
        n = ccnt[hi] - ccnt[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[q] = np.where(n > 0, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)
    return out


def _wave2(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Congestion the earlier features missed: runway-level, and ground state.

    ``arr_taxi_60min`` is the mean taxi-in of arrivals that landed at this
    airport in the previous hour. It is the closest thing available to a direct
    reading of how congested the surface actually is right now, and it is honest
    at prediction time: ranking.parquet blanks the departure taxi times but
    leaves the whole arrival side intact.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(), frame["ADES_mvt"].to_numpy())
    runway = frame["RUNWAY_mvt"].to_numpy().astype(str)
    apt_rwy = np.char.add(np.char.add(airport.astype(str), "/"), runway)

    dep = phase == "DEP"
    arr = phase == "ARR"
    taxi = frame["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()

    out = {}
    # Same-runway departure pressure, and headway to the previous departure.
    rwy_counts = np.zeros(len(epoch), dtype=np.int32)
    if dep.any():
        rwy_counts[dep] = _window_counts(epoch[dep], apt_rwy[dep], 900)
    out["dep_rwy_30min"] = rwy_counts

    # Anything beyond a few hours is not a headway, it is "nothing recent", and
    # the distinction matters at prediction time: the ranking file holds January
    # and July in one frame, so the first July departure on a runway would
    # otherwise measure back five months. 85 ranking rows exceeded seven days
    # against 10 in a training month. NaN says "no recent departure" in a way
    # that means the same thing in both frames.
    headway = np.full(len(epoch), np.nan)
    if dep.any():
        gaps = _prev_gap(epoch[dep].astype(float), apt_rwy[dep])
        gaps[(gaps < 0) | (gaps > _MAX_HEADWAY_S)] = np.nan
        headway[dep] = gaps
    out["dep_rwy_headway"] = headway

    # Ground state, read off the arrivals that have already landed.
    out["arr_taxi_60min"] = _trailing_mean(
        epoch.astype(float), airport, epoch[arr].astype(float), airport[arr], taxi[arr], 3600
    )

    # The TAIL of that distribution, and its persistence, because the mean is a
    # poor disruption detector. At EHAM in January 2026 -- inside the scored set
    # -- the mean arrival taxi-in rose 593s to 660s, eleven percent, while the
    # number of arrivals taking over an hour went from 2 to 72. The centre moved
    # a tenth; the tail moved thirty-six fold. On 2026-01-05 the daily mean hit
    # 2,050s against a January 2025 daily median of 644s, z = 11.9, where the
    # worst day in all of January 2025 reached z = 0.7.
    #
    # Rates rather than counts, so the value means the same thing in an hour
    # with forty arrivals and an hour with four.
    slow30 = (taxi[arr] > 1800).astype(float)
    slow60 = (taxi[arr] > 3600).astype(float)
    out["arr_slow_60min"] = _trailing_mean(
        epoch.astype(float), airport, epoch[arr].astype(float), airport[arr], slow30, 3600
    )
    out["arr_slow_180min"] = _trailing_mean(
        epoch.astype(float), airport, epoch[arr].astype(float), airport[arr], slow60, 10800
    )
    out["arr_taxi_180min"] = _trailing_mean(
        epoch.astype(float), airport, epoch[arr].astype(float), airport[arr], taxi[arr], 10800
    )
    # Getting worse or recovering: the last hour against the last three.
    out["arr_taxi_trend"] = out["arr_taxi_60min"] - out["arr_taxi_180min"]

    # Planned demand, from scheduled times rather than achieved ones.
    sched = frame["SCHED_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    sched_counts = np.zeros(len(epoch), dtype=np.int32)
    if dep.any():
        sched_counts[dep] = _window_counts(sched[dep], airport[dep], 900)
    out["sched_demand_30min"] = sched_counts
    return out


def _sequence_proxy(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """What the PREVIOUS departure from this stand, and this runway, experienced.

    A flight's own (MVT - AOBT_3) gap is its best single predictor at r=0.536,
    but it is null for the rows that hurt us most. The neighbouring flights'
    gaps are not: they come from other rows, and every input is present in
    ranking.parquet -- MVT_TIME 0% null, AOBT_3 1.53%, STAND and RUNWAY ~0%.

    Measured: the previous same-stand departure's gap correlates 0.27 with this
    row's target overall and 0.38 on flight-plan-present non-LIRF rows, against
    0.11 for the shipped arrival-taxi feature. Inside the no-flight-plan group,
    which has no timestamp features at all because every _flt column is null on
    the row itself, it still carries 0.15 to 0.18.

    Strictly backward-looking: only departures that have already taken off.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(), frame["ADES_mvt"].to_numpy())
    gap = (frame["MVT_TIME_UTC_mvt"] - frame["AOBT_3_flt"]).dt.total_seconds().to_numpy().astype(float)

    n = len(epoch)
    out = {k: np.full(n, np.nan) for k in
           ("prev_stand_gap", "prev_stand_headway", "prev_rwy_gap")}
    dep = phase == "DEP"

    for field, gap_key, head_key in (
        ("STAND_mvt", "prev_stand_gap", "prev_stand_headway"),
        ("RUNWAY_mvt", "prev_rwy_gap", None),
    ):
        key = np.char.add(np.char.add(airport.astype(str), "|"),
                          frame[field].to_numpy().astype(str))
        for k in np.unique(key[dep]):
            i = np.flatnonzero(dep & (key == k))
            if len(i) < 2:
                continue
            o = i[np.argsort(epoch[i], kind="stable")]
            # shift by one: each row sees only its predecessor
            head = (epoch[o[1:]] - epoch[o[:-1]]).astype(float)
            # ...but only if the predecessor is recent enough to mean anything.
            # ranking.parquet holds January and July of 2026 in one frame, so
            # July's first departure from a stand otherwise inherits January's,
            # five months back, and a headway no training month can produce.
            # _wave2 caps for exactly this reason; this must too.
            stale = head > _MAX_HEADWAY_S
            prev = gap[o[:-1]].copy()
            prev[stale] = np.nan
            head[stale] = np.nan
            out[gap_key][o[1:]] = prev
            if head_key:
                out[head_key][o[1:]] = head
    return out


# ICAO wake-turbulence departure separation, in seconds, indexed
# [leader][follower]. A Medium rotating behind a Heavy waits roughly twice as
# long as one behind another Medium, and that is a property of the PAIR, which
# no count of recent departures can represent. J is super-heavy (A388, A225),
# H heavy, M medium, L light.
# Beyond half an hour the aircraft ahead imposes no separation at all.
_MAX_WAKE_GAP_S = 1800

_WAKE_SEP = {
    "J": {"J": 90.0, "H": 120.0, "M": 180.0, "L": 180.0},
    "H": {"J": 60.0, "H": 90.0, "M": 120.0, "L": 180.0},
    "M": {"J": 60.0, "H": 60.0, "M": 60.0, "L": 120.0},
    "L": {"J": 60.0, "H": 60.0, "M": 60.0, "L": 60.0},
}


def _prev_value(times, keys, values):
    """The value carried by the previous movement sharing the same key.

    _prev_gap gives the TIME since that movement; this gives what it was. Same
    lexsort, same semantics, None where there is no predecessor.
    """
    out = np.full(len(times), None, dtype=object)
    order = np.lexsort((times, keys))
    k = keys[order]
    v = values[order]
    same = np.empty(len(order), dtype=bool)
    same[0] = False
    same[1:] = k[1:] == k[:-1]
    prev = np.full(len(order), None, dtype=object)
    prev[1:] = np.where(same[1:], v[:-1], None)
    out[order] = prev
    return out


def _wake(frame: pl.DataFrame) -> dict:
    """Separation the aircraft ahead of this one imposes, and the heavy mix.

    Three things no existing feature can see:

      wake_sep_req      the ICAO separation implied by the previous departure on
                        this runway and this aircraft's own category
      wake_prev_heavy   whether that predecessor was heavy or super
      wake_heavy_60min  the fraction of departures on this runway in the last
                        hour that were heavy or super

    WK_TBL_CAT_flt is 99.2% populated and splits M 77%, H 21%, J 0.7%, L 0.4%.
    All three are strictly backward-looking from wheels-up and built from
    columns that survive into ranking.parquet in full.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(),
                       frame["ADES_mvt"].to_numpy())
    runway = frame["RUNWAY_mvt"].to_numpy().astype(str)
    apt_rwy = np.char.add(np.char.add(airport.astype(str), "/"), runway)
    cat = frame["WK_TBL_CAT_flt"].fill_null("?").to_numpy().astype(str)
    dep = phase == "DEP"

    n = len(epoch)
    out = {"wake_sep_req": np.full(n, np.nan),
           "wake_prev_heavy": np.full(n, np.nan),
           "wake_heavy_60min": np.full(n, np.nan)}
    if not dep.any():
        return out

    prev = _prev_value(epoch[dep].astype(float), apt_rwy[dep], cat[dep])
    own = cat[dep]
    # Wake separation only binds between departures close together in time, and
    # an unbounded backward look is also the frame-composition hazard that
    # _MAX_TURN_S exists for: ranking.parquet holds January AND July in one
    # frame, so July's first departure on a runway would otherwise take its
    # predecessor from January. Measured at 0.01% of scored rows before this cap.
    since = _prev_gap(epoch[dep].astype(float), apt_rwy[dep])
    stale = (since < 0) | (since > _MAX_WAKE_GAP_S)
    sep = np.full(dep.sum(), np.nan)
    ph = np.full(dep.sum(), np.nan)
    for i, (lead, foll) in enumerate(zip(prev, own)):
        if (lead is None or stale[i] or lead not in _WAKE_SEP
                or foll not in _WAKE_SEP["M"]):
            continue
        sep[i] = _WAKE_SEP[lead][foll]
        ph[i] = 1.0 if lead in ("H", "J") else 0.0
    out["wake_sep_req"][dep] = sep
    out["wake_prev_heavy"][dep] = ph

    heavy = np.isin(cat[dep], ("H", "J")).astype(float)
    out["wake_heavy_60min"] = _trailing_mean(
        epoch.astype(float), apt_rwy, epoch[dep].astype(float), apt_rwy[dep],
        heavy, 3600)
    return out


def _config_change(frame: pl.DataFrame) -> dict:
    """Has this airport just switched its runway configuration?

    The second axis the queueing experiment flagged and nobody built. rwy_dep_active
    and rwy_dep_share describe the configuration an aircraft sits inside; they
    say nothing about whether it just CHANGED. A wind shift that reverses the
    landing direction rewrites every taxi route on the field at once, and the
    departures caught mid-transition queue differently from those either side of
    it. The existing features are blind to that because a rolling window looks
    identical before and after a switch once the window has refilled.

      cfg_shift   how much this runway's share of departures moved between the
                  previous half hour and the half hour before it
      cfg_new     this runway carried no departures in the preceding hour, so it
                  has just opened

    Both read only RUNWAY_mvt and movement times, which are fully populated for
    departures and arrivals alike in ranking.parquet.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(),
                       frame["ADES_mvt"].to_numpy()).astype(str)
    runway = frame["RUNWAY_mvt"].to_numpy().astype(str)
    apt_rwy = np.char.add(np.char.add(airport, "/"), runway)
    dep = phase == "DEP"

    n = len(epoch)
    out = {"cfg_shift": np.full(n, np.nan), "cfg_new": np.full(n, np.nan)}
    if not dep.any():
        return out

    t = epoch.astype(float)
    td = epoch[dep].astype(float)
    W = 1800.0

    def share(at):
        """This runway's share of the airport's departures in the W seconds
        before `at`. Zero where the airport had no departures at all."""
        rwy = _trailing_count(at, apt_rwy, td, apt_rwy[dep], W)
        tot = _trailing_count(at, airport, td, airport[dep], W)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, rwy / np.maximum(tot, 1), np.nan)

    now, before = share(t), share(t - W)
    out["cfg_shift"] = np.nan_to_num(now, nan=0.0) - np.nan_to_num(before, nan=0.0)
    # Just opened: carried departures in the last half hour and none in the one
    # before it.
    out["cfg_new"] = ((np.nan_to_num(now, nan=0.0) > 0)
                      & (np.nan_to_num(before, nan=0.0) == 0)).astype(float)
    return out


def _runway_config(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Which runways the airport is actually working, around each departure.

    An airport's configuration -- how many runways are open for departures, how
    the traffic is split across them, and whether departures and arrivals share
    a runway -- changes with wind and time of day and rewrites taxi routing for
    everyone on the field. We know each flight's own assigned runway but have
    had no view of the configuration it sits inside.

    Unlike the rejected reference-taxi feature, this is time-varying and
    airport-wide rather than a static function of a row's own (stand, runway),
    so CatBoost cannot already be recovering it from the raw categoricals.

    Reconstructed from movements we already hold, on both sides: RUNWAY_mvt is
    100% populated on arrival rows in ranking.parquet as well as in training.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    # ARR rows are keyed on ADES: only 16% of them have ADEP among the ten.
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(), frame["ADES_mvt"].to_numpy())
    runway = frame["RUNWAY_mvt"].to_numpy().astype(str)

    n = len(epoch)
    out = {k: np.full(n, np.nan) for k in
           ("rwy_dep_active", "rwy_dep_share", "rwy_arr_active", "rwy_mixed_mode")}
    half = 1800  # +/- 30 minutes

    for apt in np.unique(airport):
        at_apt = airport == apt
        rwys = np.unique(runway[at_apt])
        index = {r: i for i, r in enumerate(rwys)}

        for want, prefix in (("DEP", "dep"), ("ARR", "arr")):
            side = at_apt & (phase == want)
            if not side.any():
                continue
            order = np.argsort(epoch[side], kind="stable")
            times = epoch[side][order]
            codes = np.array([index[r] for r in runway[side][order]])
            # cumulative count per runway, so any window is one subtraction
            onehot = np.zeros((len(times) + 1, len(rwys)), dtype=np.int32)
            onehot[np.arange(1, len(times) + 1), codes] = 1
            cum = np.cumsum(onehot, axis=0)

            q = np.flatnonzero(at_apt & (phase == "DEP"))
            qt = epoch[q]
            lo = np.searchsorted(times, qt - half, side="left")
            hi = np.searchsorted(times, qt + half, side="right")
            counts = cum[hi] - cum[lo]                      # (len(q), len(rwys))
            active = (counts > 0).sum(axis=1)

            if want == "DEP":
                total = counts.sum(axis=1)
                own = np.array([index[r] for r in runway[q]])
                mine = counts[np.arange(len(q)), own]
                out["rwy_dep_active"][q] = active
                with np.errstate(invalid="ignore", divide="ignore"):
                    out["rwy_dep_share"][q] = np.where(total > 0, mine / np.maximum(total, 1), np.nan)
            else:
                out["rwy_arr_active"][q] = active
                own = np.array([index[r] for r in runway[q]])
                # Is this departure's runway also taking arrivals? Mixed mode
                # costs departures time that segregated operation does not.
                out["rwy_mixed_mode"][q] = (counts[np.arange(len(q)), own] > 0).astype(float)
    return out


def _congestion(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Departure and arrival pressure around each movement's wheels-up time.

    Both are knowable at prediction time: the ranking file keeps MVT_TIME_UTC_mvt
    for departures and the whole arrival side untouched.
    """
    epoch = frame["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    phase = frame["PHASE_mvt"].to_numpy()
    # A movement's own airport: ADEP for departures, ADES for arrivals.
    airport = np.where(phase == "DEP", frame["ADEP_mvt"].to_numpy(), frame["ADES_mvt"].to_numpy())

    out = {}
    for half, label in ((900, "30min"), (1800, "60min")):
        for want, name in (("DEP", "dep"), ("ARR", "arr")):
            mask = phase == want
            counts = np.zeros(len(epoch), dtype=np.int32)
            if mask.any():
                sub = _window_counts(epoch[mask], airport[mask], half)
                # Every row gets the count of same-airport movements of this
                # phase; rows of the other phase are matched by interval search
                # against the same arrays.
                full = np.zeros(len(epoch), dtype=np.int32)
                full[mask] = sub
                if (~mask).any():
                    for apt in np.unique(airport):
                        src = mask & (airport == apt)
                        dst = (~mask) & (airport == apt)
                        if not src.any() or not dst.any():
                            continue
                        t_src = np.sort(epoch[src])
                        t_dst = epoch[dst]
                        full[dst] = (
                            np.searchsorted(t_src, t_dst + half, side="right")
                            - np.searchsorted(t_src, t_dst - half, side="left")
                        ).astype(np.int32)
                counts = full
            out[f"{name}_{label}"] = counts
    return out


def _arrival_substitution_tables(frame: pl.DataFrame):
    """How often the airport's ARRIVALS that day had a substituted block time.

    The competition withholds BLOCK_TIME for departures -- it is the target's
    denominator -- but leaves the arrival side fully populated, in training AND
    in ranking.parquet. So the same failure we are trying to detect on a
    departure is directly OBSERVABLE on the arrivals beside it: when the
    airport's off-block sensing is down, arrivals get SCHED written into BLOCK
    too, and we can see it.

    LIRF arrivals run 12.1% substituted in 2025 and 11.7% in the scored 2026
    file, so the signal is stable across the year gap.

    This is the one input to the substitution classifier that is not a function
    of gap_sched. A classifier built on gap_sched alone only ties a five-number
    table built on gap_sched, which is what the first run measured.

    Scoped by calendar DATE, so a training month and the two-month ranking frame
    give the same answer for the same day -- no frame-composition hazard.
    """
    # NOTE: must be called on the FULL frame, before build() drops arrivals.
    arr = frame.filter(pl.col("PHASE_mvt") == "ARR").select(
        pl.col("ADES_mvt").alias("_apt"),
        pl.col("STAND_mvt").alias("_stand"),
        pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_date"),
        (
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds().abs() <= 60
        ).cast(pl.Float64).alias("_sub"),
    )
    if arr.height == 0:
        return None, None

    day = arr.group_by("_apt", "_date").agg(
        pl.col("_sub").mean().alias("arr_sub_day"), pl.len().alias("_dn")
    )
    # Per stand as well, shrunk toward the day rate: a stand may see only a
    # handful of arrivals in a day, and an unshrunk 0/1 would be noise.
    PRIOR = 8.0
    stand = arr.group_by("_apt", "_stand", "_date").agg(
        pl.col("_sub").mean().alias("_sr"), pl.len().alias("_sn")
    ).join(day, on=["_apt", "_date"], how="left").with_columns(
        ((pl.col("_sr") * pl.col("_sn") + pl.col("arr_sub_day") * PRIOR)
         / (pl.col("_sn") + PRIOR)).alias("arr_sub_stand")
    ).select("_apt", "_stand", "_date", "arr_sub_stand")

    return day.drop("_dn"), stand


def _join_arrival_substitution(frame: pl.DataFrame, day, stand) -> pl.DataFrame:
    """Attach the arrival tables to the (departures-only) feature frame."""
    if day is None:
        return frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("arr_sub_day"),
            pl.lit(None, dtype=pl.Float64).alias("arr_sub_stand"),
        )
    return (
        frame.with_columns(
            pl.col("ADEP_mvt").alias("_apt"),
            pl.col("STAND_mvt").alias("_stand"),
            pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_date"),
        )
        .join(day, on=["_apt", "_date"], how="left")
        .join(stand, on=["_apt", "_stand", "_date"], how="left")
        .drop("_apt", "_stand", "_date")
    )


# A turnaround longer than this is not a turnaround. The cap matters for the
# same reason _wave2 and _sequence_proxy cap: ranking.parquet holds January AND
# July 2026 in one frame, so an uncapped backward search hands July's first
# departure from a stand an arrival from January.
_MAX_TURN_S = 24 * 3600
# An inbound gate delay beyond twelve hours, or a taxi-in beyond four, is a
# broken timestamp rather than an operational fact.
_MAX_IN_DELAY_S = 12 * 3600
_MAX_TAXI_IN_S = 4 * 3600


def _turnaround(frame: pl.DataFrame) -> pl.DataFrame:
    """What the aircraft now leaving this stand did when it ARRIVED at it.

    The competition blanks BLOCK_TIME and TAXITIME for departures only, so an
    arrival's on-block time and its taxi-in survive into ranking.parquet in full.
    For a departure that is a turnaround, the arrival that occupied the stand is
    the same airframe, and its taxi-in measured the same apron on the same day.

    Four columns, all strictly backward-looking from the departure's wheels-up:

        turn_age        seconds since that arrival reached the stand
        turn_taxi_in    its taxi-in, in seconds
        turn_type_match its aircraft type equals this departure's
        turn_op_match   its callsign operator equals this departure's

    The two match flags are what separate a real turnaround from an unrelated
    aircraft that happened to use the stand earlier; without them `turn_age` is
    just stand-occupancy noise.

    a public repository by another participant (see the README credits) use this family.

    Must be called on the FULL frame, before build() drops arrivals.
    """
    arr = (
        frame.filter(pl.col("PHASE_mvt") == "ARR")
        .select(
            pl.col("ADES_mvt").alias("_apt"),
            pl.col("STAND_mvt").alias("_stand"),
            pl.col("BLOCK_TIME_UTC_mvt").alias("_on_block"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("turn_taxi_in"),
            pl.col("AIRCRAFT_TYPE_mvt").alias("_a_type"),
            pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{2,3})", 1).alias("_a_op"),
            pl.col("RUNWAY_mvt").alias("_a_rwy"),
            (pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt"))
            .dt.total_seconds().cast(pl.Float64).alias("turn_in_delay"),
        )
        .drop_nulls("_on_block")
        .sort("_on_block")
    )
    if arr.height == 0:
        return None
    return arr


def _join_turnaround(frame: pl.DataFrame, arr) -> pl.DataFrame:
    empty = {
        "turn_age": pl.Float64, "turn_taxi_in": pl.Float64,
        "turn_type_match": pl.Int8, "turn_op_match": pl.Int8,
        "turn_in_delay": pl.Float64, "turn_sched_slack": pl.Float64,
        "turn_rwy_match": pl.Int8,
    }
    if arr is None:
        return frame.with_columns([pl.lit(None, dtype=t).alias(c)
                                   for c, t in empty.items()])
    out = (
        frame.with_columns(
            pl.col("ADEP_mvt").alias("_apt"),
            pl.col("STAND_mvt").alias("_stand"),
            pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{2,3})", 1).alias("_d_op"),
        )
        .sort("MVT_TIME_UTC_mvt")
        .join_asof(arr, left_on="MVT_TIME_UTC_mvt", right_on="_on_block",
                   by=["_apt", "_stand"], strategy="backward")
        .with_columns(
            (pl.col("MVT_TIME_UTC_mvt") - pl.col("_on_block"))
            .dt.total_seconds().cast(pl.Float64).alias("turn_age")
        )
    )
    stale = pl.col("turn_age") > _MAX_TURN_S
    return (
        out.with_columns(
            pl.when(stale).then(None).otherwise(pl.col("turn_age")).alias("turn_age"),
            # Also capped: an eleven-hour "taxi-in" is a timestamp artefact,
            # and the serve frame's maximum was three times any training
            # month's. Never shipped, so this costs nothing to fix now.
            pl.when(stale | (pl.col("turn_taxi_in") > _MAX_TAXI_IN_S))
              .then(None).otherwise(pl.col("turn_taxi_in")).alias("turn_taxi_in"),
            pl.when(stale).then(None)
              .otherwise((pl.col("_a_type") == pl.col("AIRCRAFT_TYPE_mvt")).cast(pl.Int8))
              .alias("turn_type_match"),
            pl.when(stale).then(None)
              .otherwise((pl.col("_a_op") == pl.col("_d_op")).cast(pl.Int8))
              .alias("turn_op_match"),
            # How late the inbound reached its gate. A delay arriving is the
            # single most common reason a departure pushes back late, and it
            # is measurable here because an arrival's on-block time survives
            # into ranking.parquet in full while the departure's does not.
            # Capped for the same reason dep_rwy_headway is capped. Uncapped,
            # the inbound "delay" reached 337,737s in a training month and
            # 434,936s at serve -- three-and-a-half and five DAYS. Those are
            # corrupt schedule timestamps, not delays, and letting the serve
            # frame reach further than any training month can is precisely how
            # _sequence_proxy shipped 203-day headways on a few thousand scored rows.
            # Null past the cap, because "no usable inbound" means the same
            # thing in both frames while a clipped number does not.
            pl.when(stale | (pl.col("turn_in_delay").abs() > _MAX_IN_DELAY_S))
              .then(None).otherwise(pl.col("turn_in_delay"))
              .alias("turn_in_delay"),
            # Scheduled ground time: how long the schedule allowed between this
            # aircraft reaching the stand and its own scheduled off-block. A
            # tight turn behaves differently from an aircraft based overnight.
            pl.when(stale).then(None)
              .otherwise(
                  pl.when((pl.col("SCHED_TIME_UTC_mvt") - pl.col("_on_block"))
                          .dt.total_seconds().abs() > _MAX_TURN_S)
                  .then(None)
                  .otherwise((pl.col("SCHED_TIME_UTC_mvt") - pl.col("_on_block"))
                             .dt.total_seconds().cast(pl.Float64)))
              .alias("turn_sched_slack"),
            # Landed on the runway it is about to depart from: the surface
            # crossing it will or will not have to make.
            pl.when(stale).then(None)
              .otherwise((pl.col("_a_rwy") == pl.col("RUNWAY_mvt")).cast(pl.Int8))
              .alias("turn_rwy_match"),
        )
        .drop("_apt", "_stand", "_on_block", "_a_type", "_a_op", "_d_op", "_a_rwy")
    )


_NM_PROXY_MAX_S = 7200
_NM_NEXT_MAX_S = 7200
_NM_WINDOWS_S = (900, 3600)

NM_NEIGHBOUR_FEATURES = (
    [f"nm_{g}_{side}{w // 60}_{stat}" for g in ("apt", "rwy") for w in _NM_WINDOWS_S
     for side in ("past", "fol") for stat in ("mean", "n")]
    + ["nm_rwy_past60_own_minus", "nm_rwy_fol60_own_minus", "nm_rwy_next_wait", "nm_rwy_next_proxy"]
)


def _nm_neighbours(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Other departures' NM taxi proxy, in the minutes before AND after this one.

    The proxy is MVT - AOBT_3 on OTHER departures -- never this row's label,
    which is MVT - BLOCK, and never anyone's BLOCK. It is kept only on 0..7,200s.
    For each departure and group (airport; airport + runway) this gives the mean
    and count over the 15 and 60 minutes strictly before its movement time and
    strictly after it, this row's own proxy minus the 60-minute runway means, and
    the next later departure on the same runway: its wait and its proxy.

    Looking forward is deliberate. The task is retrospective reconstruction:
    ranking.parquet carries whole days, departure movement times and NM clocks
    for every flight, and the organiser blanks only departure block times.
    a public repository by another participant (GPL-3.0) uses this family; the
    idea is theirs, the code is not. Our own residuals showed little LINEAR
    signal from it (docs/LAB-NOTES.md), so it
    ships only through a fold test.

    Pools stay inside one UTC month and the next-departure wait is capped at two
    hours. ranking.parquet holds January and July in one frame, and every
    uncapped search in this file has handed a July row a January neighbour.
    The row itself and same-second ties are excluded from every window.

    Verified 2026-09-15 against brute force on 300 random March 2025 departures:
    0 mismatches across 17 checks each.
    """
    n = frame.height
    dep = frame["PHASE_mvt"].to_numpy() == "DEP"
    mvt = frame["MVT_TIME_UTC_mvt"]
    epoch = mvt.dt.epoch("s").to_numpy().astype(np.int64)
    proxy = (mvt - frame["AOBT_3_flt"]).dt.total_seconds().to_numpy().astype(float)
    valid = dep & np.isfinite(proxy) & (proxy >= 0) & (proxy <= _NM_PROXY_MAX_S)
    month = mvt.dt.strftime("%Y-%m").to_numpy().astype(str)
    apt = np.char.add(np.char.add(frame["ADEP_mvt"].to_numpy().astype(str), "|"), month)
    rwy = np.char.add(np.char.add(apt, "|"), frame["RUNWAY_mvt"].to_numpy().astype(str))

    out = {name: np.full(n, np.nan) for name in NM_NEIGHBOUR_FEATURES}
    for gname, key in (("apt", apt), ("rwy", rwy)):
        for k in np.unique(key[dep]):
            q = np.flatnonzero(dep & (key == k))
            s = np.flatnonzero(valid & (key == k))
            qt = epoch[q]
            so = s[np.argsort(epoch[s], kind="stable")]
            st = epoch[so]
            cs = np.concatenate([[0.0], np.cumsum(proxy[so])])
            for w in _NM_WINDOWS_S:
                for side in ("past", "fol"):
                    if side == "past":
                        lo, hi = np.searchsorted(st, qt - w, "left"), np.searchsorted(st, qt, "left")
                    else:
                        lo, hi = np.searchsorted(st, qt, "right"), np.searchsorted(st, qt + w, "right")
                    cnt = (hi - lo).astype(float)
                    out[f"nm_{gname}_{side}{w // 60}_n"][q] = cnt
                    with np.errstate(invalid="ignore", divide="ignore"):
                        out[f"nm_{gname}_{side}{w // 60}_mean"][q] = np.where(
                            cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1.0), np.nan)
            if gname == "rwy":
                own = np.where(valid[q], proxy[q], np.nan)
                out["nm_rwy_past60_own_minus"][q] = own - out["nm_rwy_past60_mean"][q]
                out["nm_rwy_fol60_own_minus"][q] = own - out["nm_rwy_fol60_mean"][q]
                j = np.searchsorted(st, qt, "right")
                has = j < len(st)
                wait = np.full(len(q), np.nan)
                nxt = np.full(len(q), np.nan)
                wait[has] = (st[j[has]] - qt[has]).astype(float)
                nxt[has] = proxy[so[j[has]]]
                far = ~(wait <= _NM_NEXT_MAX_S)
                wait[far] = np.nan
                nxt[far] = np.nan
                out["nm_rwy_next_wait"][q] = wait
                out["nm_rwy_next_proxy"][q] = nxt
    return out


def _late_neighbours(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Other departures' off-block LATENESS, in the minutes before and after this one.

    The shipped ``_nm_neighbours`` aggregates MVT - AOBT_3, a TAXI proxy. This is the same
    windowing arithmetic on a different quantity: dep_delay = AOBT_3 - EOBT_1, how late other
    aircraft pushed back. Pushback punctuality reflects turnaround, stand and ground-handling
    stress; surface taxi reflects the movement area. They are different signals and the model
    had only the second.

    Never reads this row's label (MVT - BLOCK) or anyone's BLOCK. Unlike the taxi proxy,
    lateness is SIGNED and so is not clipped at zero; it is bounded two-sided to keep a handful
    of multi-day plan artefacts out of a mean.

    Each column is demeaned within (airport, month). That is not cosmetic: LFPG's median
    neighbour lateness is 574s in 2025 and 231s in 2026, and the demeaned form is immune to that
    drift and also scored better (-1.341s vs -1.284s pooled).

    Screened 2026-09-20 with every shipped nm_* column, the congestion block and weather as
    controls: best single column 9.5x its placebo, family LOFO -1.341s with all five folds
    negative and -1.231s after deleting its hundred best rows, monotone across all ten deciles,
    and negative in BOTH scored months separately. Strongest at the four airports where AOBT_3
    is NM's planned value, and strongest of all at LTFM, which ADS-B cannot see -- so it does
    not overlap the ADS-B stack.
    """
    n = frame.height
    dep = frame["PHASE_mvt"].to_numpy() == "DEP"
    mvt = frame["MVT_TIME_UTC_mvt"]
    epoch = mvt.dt.epoch("s").to_numpy().astype(np.int64)
    late = (frame["AOBT_3_flt"] - frame["EOBT_1_flt"]).dt.total_seconds().to_numpy().astype(float)
    valid = dep & np.isfinite(late) & (late >= _LATE_LO_S) & (late <= _LATE_HI_S)
    proxy = np.where(valid, late, 0.0)
    month = mvt.dt.strftime("%Y-%m").to_numpy().astype(str)
    apt = np.char.add(np.char.add(frame["ADEP_mvt"].to_numpy().astype(str), "|"), month)
    rwy = np.char.add(np.char.add(apt, "|"), frame["RUNWAY_mvt"].to_numpy().astype(str))

    raw = {}
    for gname, key in (("apt", apt), ("rwy", rwy)):
        for w in _NM_WINDOWS_S:
            for side in ("past", "fol"):
                raw[f"late_{gname}_{side}{w // 60}_mean"] = np.full(n, np.nan)
    own_minus = np.full(n, np.nan)
    for gname, key in (("apt", apt), ("rwy", rwy)):
        for k in np.unique(key[dep]):
            q = np.flatnonzero(dep & (key == k))
            s_idx = np.flatnonzero(valid & (key == k))
            qt = epoch[q]
            so = s_idx[np.argsort(epoch[s_idx], kind="stable")]
            st = epoch[so]
            cs = np.concatenate([[0.0], np.cumsum(proxy[so])])
            for w in _NM_WINDOWS_S:
                for side in ("past", "fol"):
                    if side == "past":
                        lo, hi = np.searchsorted(st, qt - w, "left"), np.searchsorted(st, qt, "left")
                    else:
                        lo, hi = np.searchsorted(st, qt, "right"), np.searchsorted(st, qt + w, "right")
                    cnt = (hi - lo).astype(float)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        raw[f"late_{gname}_{side}{w // 60}_mean"][q] = np.where(
                            cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1.0), np.nan)
            if gname == "apt":
                own_minus[q] = np.where(valid[q], late[q], np.nan) - raw["late_apt_past60_mean"][q]
    raw["late_apt_past60_own_minus"] = own_minus

    # demean within (airport, month) -- immunises against the 2025->2026 level drift
    out = {}
    for name in LATE_NEIGHBOUR_FEATURES:
        v = raw[name].copy()
        for k in np.unique(apt[dep]):
            q = np.flatnonzero(dep & (apt == k))
            m = np.nanmean(v[q]) if np.isfinite(v[q]).any() else np.nan
            if np.isfinite(m):
                v[q] = v[q] - m
        out[name] = v
    return out


def build(frame: pl.DataFrame, with_target: bool = True) -> pl.DataFrame:
    """Turn a raw movements frame into the model matrix (departures only).

    ``frame`` must contain both phases: arrivals are dropped from the output but
    are needed first to compute arrival pressure.
    """
    arr_day, arr_stand = _arrival_substitution_tables(frame)
    turn_arr = _turnaround(frame)
    congestion = _congestion(frame)
    congestion.update(_wave2(frame))
    congestion.update(_runway_config(frame))
    congestion.update(_wake(frame))
    congestion.update(_config_change(frame))
    congestion.update(_sequence_proxy(frame))
    congestion.update(_nm_neighbours(frame))
    congestion.update(_late_neighbours(frame))
    frame = frame.with_columns(
        [pl.Series(name, values) for name, values in congestion.items()]
    )

    mvt = pl.col("MVT_TIME_UTC_mvt")
    frame = frame.filter(pl.col("PHASE_mvt") == "DEP").with_columns(
        mvt.dt.hour().alias("hour"),
        # .dt.hour() is Int8 and polars keeps that dtype through the multiply,
        # so hour*60 overflows above 127 and wraps: 21:10 became -10, not 1270.
        # 99.7% of rows carried a wrong value; every model so far trained on it.
        (mvt.dt.hour().cast(pl.Int32) * 60 + mvt.dt.minute()).alias("minute_of_day"),
        mvt.dt.weekday().alias("dow"),
        mvt.dt.month().alias("month"),
        mvt.dt.ordinal_day().alias("doy"),
        (mvt.dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        pl.col("AOBT_3_flt").is_not_null().cast(pl.Int8).alias("has_flight_plan"),
        # Operator identity, recovered from the callsign.
        #
        # AIRCRAFT_OPERATOR_flt is 100% null for the no-flight-plan rows -- the
        # group that carries most of our squared error has no operator column at
        # all. But FLIGHT_mvt sits on the movement side and is 99.99% populated
        # in training and 99.98% in ranking.parquet, and its leading letters are
        # the operator: matched against AIRCRAFT_OPERATOR_flt on the 2,046,656
        # rows where both exist, the prefix agrees with the modal operator
        # 99.48% of the time. 99.88% of ranking prefixes were seen in training.
        #
        # Within the no-plan rows this is worth eta^2 0.069 on the residual
        # after removing airport x aircraft type, so it is not a restatement of
        # either. Handed to CatBoost raw, as a categorical: a hand-rolled target
        # encoding is the mistake ref_taxi_s made twice.
        pl.col("FLIGHT_mvt").str.extract(r"^([A-Z]{2,3})", 1).alias("callsign_op"),
        (mvt - pl.col("AOBT_3_flt")).dt.total_seconds().alias("gap_aobt"),
        (mvt - pl.col("EOBT_1_flt")).dt.total_seconds().alias("gap_eobt"),
        (mvt - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap_sched"),
        (mvt - pl.col("LOBT_flt")).dt.total_seconds().alias("gap_lobt"),
        # How far the LAST-CALCULATED off-block sits from the ACTUAL one.
        #
        # Both are Network Manager fields and both survive into ranking.parquet
        # (1.53% null, the same rows). Where they disagree by more than an hour,
        # the airport's own BLOCK_TIME tracks LOBT rather than AOBT -- so the
        # target follows gap_lobt, not gap_aobt. On the 630 "neither" rows of
        # fold (1,7) the medians are: target 4,202s, gap_lobt 4,381s,
        # gap_aobt 1,384s, and the correlation with the target is 0.618 for
        # gap_lobt against -0.061 for gap_aobt.
        #
        # A tree cannot form this difference itself -- it can only threshold each
        # column separately -- so handing it over explicitly is not redundant
        # with having both gaps present.
        ((mvt - pl.col("LOBT_flt")) - (mvt - pl.col("AOBT_3_flt")))
            .dt.total_seconds().alias("lobt_minus_aobt"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("dep_delay"),
        (pl.col("SCHED_TIME_UTC_mvt") - pl.col("EOBT_1_flt")).dt.total_seconds().alias("sched_vs_eobt"),
        # A sensor reading has uniform seconds; a hand-entered or schedule-copied
        # one piles up on :00. BLOCK_TIME would say more here but is withheld at
        # serve time, so only the movement side is usable.
        (mvt.dt.second() == 0).cast(pl.Int8).alias("mvt_sec_00"),
        # CLOCK_FEATURES -- see its comment at the top of this module.
        (pl.col("LOBT_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("clk_lobt_minus_sched"),
        (pl.col("EOBT_1_flt") - pl.col("LOBT_flt")).dt.total_seconds().alias("clk_eobt_minus_lobt"),
        (pl.col("IOBT_flt") - pl.col("LOBT_flt")).dt.total_seconds().alias("clk_iobt_minus_lobt"),
        (pl.col("AOBT_3_flt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("clk_aobt_minus_sched"),
        (pl.col("ARVT_3_flt") - mvt).dt.total_seconds().alias("clk_arvt3_minus_mvt"),
        (pl.col("ARVT_3_flt") - pl.col("ARVT_1_flt")).dt.total_seconds().alias("clk_arvt3_minus_arvt1"),
        (pl.col("ARVT_3_flt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("clk_arvt3_minus_aobt"),
        pl.col("AOBT_3_flt").dt.second().cast(pl.Float64).alias("clk_sec_aobt"),
        mvt.dt.second().cast(pl.Float64).alias("clk_sec_mvt"),
        pl.col("SCHED_TIME_UTC_mvt").dt.second().cast(pl.Float64).alias("clk_sec_sched"),
    )

    from . import weather

    frame = weather.attach(frame)

    # Per-airport crossing of the flight-plan flag. See the module docstring:
    # a missing flight plan is only dangerous at some airports, so the model
    # needs the pair, not the flag.
    frame = frame.with_columns(
        (pl.col("ADEP_mvt") + pl.lit("|") + pl.col("has_flight_plan").cast(pl.Utf8)).alias(
            "airport_plan"
        )
    )

    # How busy this stand/runway combination is -- a crude proxy for apron
    # layout and taxi distance, which we have no geometry for.
    #
    # Expressed as movements per operating day, NOT a raw count. build() is
    # called per monthly file in training but on the whole ranking file at
    # prediction time, and the ranking file holds two months, so a raw count
    # would arrive about twice as large at serve time than at fit time. Dividing
    # by the number of distinct dates present makes the two agree.
    # Grouped by calendar month as well as by key. Training calls build() once
    # per monthly file, but the real serve-time call passes the whole of
    # ranking.parquet, which is January AND July 2026 -- 62 days in one frame.
    # Dividing by 62 while training divided by 31 halves every key that only
    # appears in one of the two months and biases the rest; measured, only 3.3%
    # of served rows came within 1% of what a single-month build gives, and
    # 13,446 were served exactly half. Grouping by month makes the two agree.
    frame = frame.with_columns(
        pl.col("MVT_TIME_UTC_mvt").dt.truncate("1mo").alias("_ym")
    )
    days = frame.group_by("_ym").agg(
        pl.col("MVT_TIME_UTC_mvt").dt.date().n_unique().alias("_days")
    )
    pair = frame.group_by("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "_ym").agg(
        pl.len().alias("_n")
    ).join(days, on="_ym", how="left").with_columns(
        (pl.col("_n") / pl.max_horizontal(pl.col("_days"), pl.lit(1)))
        .alias("stand_runway_pair_n")
    ).select("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "_ym", "stand_runway_pair_n")
    frame = frame.join(
        pair, on=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "_ym"], how="left"
    ).drop("_ym")


    frame = _join_arrival_substitution(frame, arr_day, arr_stand)
    frame = _join_turnaround(frame, turn_arr)

    built = [f for f in FEATURES if f not in REFERENCE]
    keep = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", *built]
    keep += [TARGET] if with_target else []
    frame = frame.select(list(dict.fromkeys(keep)))
    cats = [c for c in CATEGORICAL if c not in REFERENCE]
    return frame.with_columns([pl.col(c).cast(pl.Utf8).fill_null("__NA__") for c in cats])
