"""Candidate constructions for the high-volume feature audit (2026-09-21). A LIBRARY: no I/O of its own.

Every candidate is a cell of the (quantity x aggregation) grid that archive/agent_audit_00_enumerate.py
found EMPTY in prc/features.py. The list below is the pre-declared candidate set: it was written
before any candidate was screened, and archive/agent_audit_02_screen.py screens every one of them
against one bar and reports every result, pass or fail.

CONSTRUCTION RULES, all inherited from the traps this project has already paid for:

  * Pools are MONTH-SCOPED (key includes the UTC year-month of the row's own MVT), exactly as
    _nm_neighbours is. build() runs per month in training and once over January AND July 2026 at
    serve time; a month-scoped pool gives the same value either way. archive/agent_audit_04_serve.py
    proves it on ranking.parquet rather than asserting it.
  * Nothing reads BLOCK_TIME or TAXITIME of a DEPARTURE (withheld at serve time). Arrival BLOCK and
    TAXITIME are labelled in ranking.parquet and are fair game, as the shipped arr_taxi_60min is.
  * The row's own value never enters its own aggregate. Past windows take side='left' at the anchor,
    forward windows side='right', so the row and any same-second tie are excluded -- the shipped
    convention. Inclusive cohort windows subtract the row's own contribution explicitly.
  * Every proxy is bounded so a handful of multi-day artefacts cannot move a mean.
  * The value handed to the screen is DEMEANED within (airport, month), as the shipped late_* family
    is -- immune to a 2025 -> 2026 level shift, and the airport-month level is exactly what the
    placebo preserves, so the screen reads only within-month variation.

Window arithmetic is vectorised over ALL keys at once: rows are sorted on key*BIG + (t - T0), so a
window search can never cross from one key into the next (BIG exceeds any time span + window).
"""
from __future__ import annotations

import numpy as np
import polars as pl

C = pl.col
TEN = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
BIG = np.int64(100_000_000)          # 3.2 years of seconds: no window can cross a key boundary
T0 = np.int64(1_717_200_000)         # 2024-06-01, so t - T0 is positive for 2025 and 2026 alike

LATE_LO, LATE_HI = -3600.0, 10800.0  # the shipped lateness bound
NM_LO, NM_HI = 0.0, 7200.0           # the shipped taxi-proxy bound
DEL_LIM = 10800.0                    # symmetric bound for schedule/plan deviations
EXC_LIM = 7200.0                     # airborne-excess bound
TURN_MAX = 24 * 3600                 # the shipped turnaround cap

# ---------------------------------------------------------------------------------------------
# THE PRE-DECLARED CANDIDATE LIST. family, name, one-line mechanism. Written before screening.
# ---------------------------------------------------------------------------------------------
CANDIDATES = [
    # A. take-off punctuality of neighbouring departures, MVT - SCHED, plan AND no-plan rows
    ("A", "gs_apt_past15", "how late other departures are leaving vs timetable, last 15 min"),
    ("A", "gs_apt_fol15", "same, next 15 min"),
    ("A", "gs_apt_past60", "same, last 60 min"),
    ("A", "gs_apt_fol60", "same, next 60 min"),
    ("A", "gs_rwy_past15", "same runway, last 15 min"),
    ("A", "gs_rwy_fol15", "same runway, next 15 min"),
    ("A", "gs_cohort60", "MVT - SCHED of flights SCHEDULED within +/-30 min of this one"),
    # B. plan-deviation clocks of neighbouring departures
    ("B", "sve_apt_past60", "SCHED - EOBT_1 of neighbours: announced pre-departure slip"),
    ("B", "sve_apt_fol60", "same, next 60 min"),
    ("B", "aol_apt_past15", "AOBT_3 - LOBT of neighbours: lateness vs the LAST plan"),
    ("B", "aol_apt_fol15", "same, next 15 min"),
    ("B", "aol_apt_past60", "same, last 60 min"),
    ("B", "aol_rwy_past15", "same runway, last 15 min"),
    ("B", "los_apt_past60", "LOBT - SCHED of neighbours: last plan vs timetable"),
    ("B", "eol_apt_past60", "EOBT_1 - LOBT of neighbours: how far plans moved"),
    ("B", "adrift_apt_past60", "ARVT_3 - ARVT_1 of neighbouring departures: arrival-estimate drift"),
    ("B", "planupd_apt_past60", "share of neighbours whose plan moved (EOBT_1 != LOBT)"),
    # C. off-block lateness AOBT_3 - EOBT_1 in scopes / statistics the late_* family lacks
    ("C", "late_op_past60", "same OPERATOR's departures, last 60 min: its ground handling now"),
    ("C", "late_op_fol60", "same operator, next 60 min"),
    ("C", "late_op_both120", "same operator, +/-120 min"),
    ("C", "late_ades_both120", "same DESTINATION, +/-120 min: an ATFM flow held on the ground"),
    ("C", "late_prevstand", "previous departure from the same stand (<=6h): its lateness"),
    ("C", "late_rwy_next", "next departure on the same runway (<=2h): its lateness"),
    ("C", "late_apt_share900", "share of departures in the last 60 min pushing back >15 min late"),
    ("C", "late_apt_sd60", "dispersion of neighbours' lateness, last 60 min"),
    ("C", "late_cohort60", "lateness of flights SCHEDULED within +/-30 min of this one"),
    # D. taxi proxy MVT - AOBT_3 in scopes / statistics the nm_* family lacks
    ("D", "nm_op_past60", "same operator's taxi proxy, last 60 min"),
    ("D", "nm_op_fol60", "same operator, next 60 min"),
    ("D", "nm_ades_both120", "same destination, +/-120 min"),
    ("D", "nm_apt_share1800_p60", "share of departures in the last 60 min taxiing > 30 min"),
    ("D", "nm_apt_share1800_f60", "same, next 60 min"),
    ("D", "nm_apt_sd60", "dispersion of neighbours' taxi proxy, last 60 min"),
    # E. the ARRIVAL side at this airport, clocks nothing aggregates
    ("E", "arrdel_apt_past60", "arrivals' on-block delay BLOCK - SCHED, last 60 min"),
    ("E", "arrland_apt_past60", "arrivals' landing delay MVT - SCHED, last 60 min"),
    ("E", "arrest_apt_past60", "arrivals' landing vs first NM estimate MVT - ARVT_1"),
    ("E", "arrexc_apt_past60", "arrivals' AIRBORNE EXCESS (MVT-AOBT_3)-(ARVT_1-EOBT_1): holding"),
    ("E", "arrexc_apt_both30", "same, +/-30 min"),
    ("E", "arrexc_apt_fol60", "same, next 60 min"),
    ("E", "arrorig_apt_past60", "arrivals' origin lateness AOBT_3 - EOBT_1: inbound punctuality"),
    ("E", "arrclk_apt_past60", "arrivals' MVT - ARVT_3: airport vs NM landing clock"),
    # F. the arrival side, scoped
    ("F", "arrtaxi_rwy_past60", "taxi-in of arrivals on THIS runway (mixed-mode), last 60 min"),
    ("F", "arrexc_rwy_past60", "airborne excess of arrivals on this runway, last 60 min"),
    ("F", "arrtaxi_op_past60", "taxi-in of the same operator's arrivals (its terminal), last 60 min"),
    ("F", "arrdel_op_both120", "on-block delay of the same operator's arrivals, +/-120 min"),
    # G. the linked inbound (same stand, type AND operator match, <=24h, same month)
    ("G", "turn_obt_violation", "min(AOBT_3 - inbound on-block, 0): NM off-block before arrival"),
    ("G", "turn_in_origin_late", "inbound's own origin lateness AOBT_3 - EOBT_1"),
    ("G", "turn_in_excess", "inbound's airborne excess"),
    # H. occupancy anchored at WHEELS-UP (never at this row's off-block: that family is dead)
    ("H", "backlog_sched", "departures scheduled to have left but still on the ground at wheels-up"),
    ("H", "surface_now", "departures off-block and not yet airborne at wheels-up"),
    ("H", "surface_15ago", "same, 15 minutes before wheels-up (own interval excluded)"),
    # I. clock-regime shares among neighbours
    ("I", "mvtsec00_apt_p60", "share of departures whose MVT stamp is on :00 (manual entry)"),
    ("I", "arrsec00_apt_p60", "share of arrivals whose BLOCK stamp is on :00"),
    ("I", "latezero_apt_p60", "share of departures with AOBT_3 == EOBT_1 (plan written as actual)"),
    ("I", "aobteqlobt_apt_p60", "share of departures with AOBT_3 == LOBT"),
    # J. destination side
    ("J", "dest_arrexc_eta", "airborne excess of arrivals at ADES (one of the ten) around our ETA"),
]

# killed at enumeration (archive/agent_audit_00_enumerate.py section 2): the quantity is constant
DEGENERATE = [
    ("I", "aobtsec00_apt_p60", "AOBT_3 is on a whole minute at 99-100% of rows at 9 of 10 airports"),
    ("I", "iobtnelobt_apt_p60", "IOBT == LOBT on 98-100% of rows at every airport"),
]

# ---------------------------------------------------------------------------------------------
# ROUND 2 -- designed AFTER round 1's results were known (1 marginal pass of 55), pre-registered
# before any round-2 candidate was screened. O1/O2 are siblings of round 1's only pass
# (late_rwy_next) and are NOT independent evidence; every other family is a new mechanism.
# ---------------------------------------------------------------------------------------------
CANDIDATES_R2 = [
    # K. copy-regime / broken-stamp detectors on the ARRIVAL side, in a window rather than a day
    ("K", "arrsub6_apt_both60", "share of arrivals +/-60 min with |BLOCK-SCHED| <= 6s: copying NOW"),
    ("K", "arrsub6_apt_both180", "same, +/-3h"),
    ("K", "arrsub60_apt_both60", "same at the 60s definition arr_sub_day uses, +/-60 min"),
    ("K", "arrneg_apt_both120", "share of arrivals +/-2h with on-block at or before landing"),
    ("K", "turn_in_copy6", "latest arrival at this stand (<=24h, no match needed) was a copy"),
    ("K", "arrsub6_op_both180", "share of the same operator's arrivals +/-3h that are copies"),
    # L. network scope -- all ten airports at once
    ("L", "late_net_past60", "mean lateness over all ten airports, last 60 min"),
    ("L", "late_opnet_both60", "same operator's lateness at all ten airports, +/-60 min"),
    ("L", "late_adesnet_eta60", "lateness of flights from all ten to the same ADES, ETA +/-60 min"),
    ("L", "arrexc_net_past60", "arrival airborne excess over all ten airports, last 60 min"),
    ("L", "nm_net_past60", "taxi proxy over all ten airports, last 60 min"),
    # M. pier scope (first character of STAND: letter piers, or the terminal digit)
    ("M", "nm_pier_past60", "taxi proxy of departures from the same pier, last 60 min"),
    ("M", "late_pier_past60", "lateness of departures from the same pier, last 60 min"),
    ("M", "arrtaxi_pier_past60", "taxi-in of arrivals to the same pier, last 60 min"),
    # N. queue anchored at the SCHEDULED time (never at this row's off-block)
    ("N", "backlog_at_sched", "departures due at or before my SCHED still on the ground at my SCHED"),
    ("N", "arrbank_at_sched", "arrivals blocking in within +/-15 min of my SCHED"),
    # O. grid completion -- SIBLINGS of round 1's pass, flagged, not independent
    ("O", "late_rwy_prev", "previous departure on the same runway (<=2h): its lateness"),
    ("O", "gs_rwy_next", "next departure on the same runway (<=2h): its MVT - SCHED"),
    # P. local civil time (month is dropped in production, so DST blurs a UTC hour)
    ("P", "localhour_cellres", "out-of-fold mean residual of (airport, LOCAL hour, weekend)"),
    ("P", "utchour_cellres", "the same by UTC hour -- the reference the ledger found dead"),
    # Q / X. runway and schedule-bank pressure at a point
    ("Q", "arr_rwy_cnt_both10", "arrivals landing on this departure's runway +/-10 min of wheels-up"),
    ("X", "sched_same_minute", "other departures scheduled at exactly the same minute"),
]
# ---------------------------------------------------------------------------------------------
# ROUND 3 -- designed after rounds 1-2 (1 marginal pass of 77), pre-registered before screening.
# Target: constructions the shipped model structurally CANNOT form from what it has.
# V1 is a sibling of round 1's sub-floor arrival-holding readings and is flagged as such.
# ---------------------------------------------------------------------------------------------
CANDIDATES_R3 = [
    # S. surface congestion ADJUSTED FOR THE NEIGHBOURS' STAND/RUNWAY MIX -- the model sees this
    #    row's stand and runway, never its neighbours', so a raw neighbour mean confounds
    #    "the surface is slow" with "the last hour happened to use far stands"
    ("S", "arrtaxi_exc_apt_past60", "arrivals' taxi-in minus their (stand,rwy) month mean, last 60 min"),
    ("S", "arrtaxi_exc_apt_both30", "same, +/-30 min"),
    ("S", "nm_exc_apt_past60", "departures' taxi proxy minus their (stand,rwy) month mean, last 60 min"),
    ("S", "nm_exc_rwy_past15", "same, this runway, last 15 min"),
    ("S", "nm_exc_apt_fol60", "same, airport, next 60 min"),
    # T. slow time scales: works, closures, layout changes lasting days
    ("T", "nm_rwy_lag24h", "taxi proxy on this runway at the same time YESTERDAY (+/-30 min)"),
    ("T", "nm_pair7d_dev", "this (stand,rwy) pair's taxi proxy over the last 7 days minus its month mean"),
    ("T", "arrtaxi_stand7d_dev", "arrivals to this stand, taxi-in over the last 7 days minus month mean"),
    # U. lateness net of each airline's habitual lateness
    ("U", "lateexc_apt_past60", "neighbours' lateness minus their operator's month mean, last 60 min"),
    ("U", "lateexc_rwy_past15", "same, this runway, last 15 min"),
    # V. the tail of arrival holding (sibling of round 1's sub-floor arrexc readings)
    ("V", "arrhold_share_both30", "share of arrivals +/-30 min with airborne excess > 600s"),
]
ALL_CANDIDATES = CANDIDATES + CANDIDATES_R2 + CANDIDATES_R3

# the scope each candidate aggregates over, for the scope placebo in the survivor stage
SCOPE = {n: ("op" if "_op_" in n and "opnet" not in n else "ades" if "_ades_" in n or n.startswith("dest") else
             "rwy" if "_rwy" in n else "stand" if "stand" in n else "pier" if "_pier_" in n else "none")
         for _, n, _ in ALL_CANDIDATES}

LOCAL_TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
            "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
            "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich",
            "LTFM": "Europe/Istanbul"}


# ---------------------------------------------------------------------------------------------
# frame loading
# ---------------------------------------------------------------------------------------------
def load_raw(source: str | pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Departures from the ten, arrivals at the ten -- only the columns the candidates read.

    BLOCK/TAXITIME are loaded for ARRIVALS only. The departure frame never carries them.
    """
    lf = pl.scan_parquet(source) if isinstance(source, str) else source
    op = C("FLIGHT_mvt").cast(pl.Utf8).str.extract(r"^([A-Z]{2,3})", 1)
    ep = lambda c: C(c).dt.epoch("s")
    dep = (lf.filter((C("PHASE_mvt") == "DEP") & C("ADEP_mvt").is_in(TEN))
           .select(C("MVT_ID_mvt"), C("FLIGHT_ID_mvt"),
                   C("ADEP_mvt").alias("apt"), C("ADES_mvt").alias("ades"),
                   C("RUNWAY_mvt").cast(pl.Utf8).fill_null("?").alias("rwy"),
                   C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                   C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).alias("atype"),
                   op.alias("op"),
                   C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
                   ep("MVT_TIME_UTC_mvt").alias("t"), ep("SCHED_TIME_UTC_mvt").alias("sched"),
                   ep("AOBT_3_flt").alias("aobt"), ep("EOBT_1_flt").alias("eobt"),
                   ep("LOBT_flt").alias("lobt"), ep("ARVT_1_flt").alias("arvt1"),
                   ep("ARVT_3_flt").alias("arvt3"),
                   C("ARVT_1_flt").dt.strftime("%Y-%m").alias("ym_eta"))
           .collect())
    arr = (lf.filter((C("PHASE_mvt") == "ARR") & C("ADES_mvt").is_in(TEN))
           .select(C("FLIGHT_ID_mvt"), C("ADES_mvt").alias("apt"),
                   C("RUNWAY_mvt").cast(pl.Utf8).fill_null("?").alias("rwy"),
                   C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                   C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).alias("atype"),
                   op.alias("op"),
                   C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
                   ep("MVT_TIME_UTC_mvt").alias("t"), ep("BLOCK_TIME_UTC_mvt").alias("block"),
                   ep("SCHED_TIME_UTC_mvt").alias("sched"),
                   ep("AOBT_3_flt").alias("aobt"), ep("EOBT_1_flt").alias("eobt"),
                   ep("ARVT_1_flt").alias("arvt1"), ep("ARVT_3_flt").alias("arvt3"),
                   C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("taxi"))
           .collect())
    return dep, arr


def f64(s: pl.Series) -> np.ndarray:
    return s.cast(pl.Float64).fill_null(np.nan).to_numpy().astype(np.float64)


def i64(s: pl.Series) -> np.ndarray:
    """Epoch seconds as int64, nulls as a sentinel far outside any window."""
    return s.fill_null(-(10 ** 12)).to_numpy().astype(np.int64)


def joint_codes(*cols: list[pl.Series]) -> list[np.ndarray]:
    """Dense integer codes for string keys, CONSISTENT across several frames. -1 where null."""
    stacked = pl.concat([c.cast(pl.Utf8).rename("k") for c in cols])
    uniq = stacked.drop_nulls().unique().sort()
    table = pl.DataFrame({"k": uniq, "code": np.arange(len(uniq), dtype=np.int64)})
    out = []
    for c in cols:
        j = pl.DataFrame({"k": c.cast(pl.Utf8)}).join(table, on="k", how="left", maintain_order="left")
        out.append(j["code"].fill_null(-1).to_numpy().astype(np.int64))
    return out


def key_series(frame: pl.DataFrame, *cols: str) -> pl.Series:
    return frame.select(pl.concat_str([C(c) for c in cols], separator="|").alias("k"))["k"]


# ---------------------------------------------------------------------------------------------
# vectorised window arithmetic
# ---------------------------------------------------------------------------------------------
class Pool:
    """Valid source rows sorted on key*BIG + (t - T0), with prefix sums of value and value^2."""

    def __init__(self, key: np.ndarray, t: np.ndarray, val: np.ndarray, valid: np.ndarray):
        ok = valid & (key >= 0) & np.isfinite(val)
        comb = key[ok] * BIG + (t[ok] - T0)
        order = np.argsort(comb, kind="stable")
        self.comb = comb[order]
        v = val[ok][order].astype(np.float64)
        self.v = v
        self.cs = np.concatenate([[0.0], np.cumsum(v)])
        self.cs2 = np.concatenate([[0.0], np.cumsum(v * v)])
        self.src = np.flatnonzero(ok)[order]

    def qcomb(self, key, t):
        return np.where(key >= 0, key * BIG + (t - T0), np.int64(-1))

    def span(self, key, t, lo_off, lo_side, hi_off, hi_side):
        q = self.qcomb(key, t)
        lo = np.searchsorted(self.comb, q + lo_off, lo_side)
        hi = np.searchsorted(self.comb, q + hi_off, hi_side)
        bad = key < 0
        lo[bad] = 0
        hi[bad] = 0
        return lo, hi

    def past(self, key, t, w):
        return self.span(key, t, -w, "left", 0, "left")

    def fol(self, key, t, w):
        return self.span(key, t, 0, "right", w, "right")

    def sums(self, spans):
        n = np.zeros(0)
        s = s2 = None
        for lo, hi in spans:
            nn = (hi - lo).astype(np.float64)
            ss = self.cs[hi] - self.cs[lo]
            ss2 = self.cs2[hi] - self.cs2[lo]
            if s is None:
                n, s, s2 = nn, ss, ss2
            else:
                n, s, s2 = n + nn, s + ss, s2 + ss2
        return n, s, s2


def mean_of(n, s, min_n=1):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= min_n, s / np.maximum(n, 1.0), np.nan)


def sd_of(n, s, s2, min_n=3):
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / np.maximum(n, 1.0)
        var = np.maximum(s2 / np.maximum(n, 1.0) - m * m, 0.0)
        return np.where(n >= min_n, np.sqrt(var), np.nan)


def neighbour(pool: Pool, key, t, which: str, max_gap: int):
    """The single next (strictly later) or previous (strictly earlier) valid source's value."""
    q = pool.qcomb(key, t)
    out = np.full(len(key), np.nan)
    if which == "next":
        j = np.searchsorted(pool.comb, q, "right")
        has = (j < len(pool.comb)) & (key >= 0)
        jj = np.where(has, j, 0)
        gap = pool.comb[jj] - q
    else:
        j = np.searchsorted(pool.comb, q, "left") - 1
        has = (j >= 0) & (key >= 0)
        jj = np.where(has, j, 0)
        gap = q - pool.comb[jj]
    ok = has & (gap > 0) & (gap <= max_gap)
    out[ok] = pool.v[jj[ok]]
    return out


def containing(key, start, end, valid, q_key, q_t, self_ok=None, self_start=None, self_end=None):
    """How many valid intervals [start, end) of the same key contain each query time."""
    ok = valid & (key >= 0) & (start < end)
    cs = np.sort(key[ok] * BIG + (start[ok] - T0))
    ce = np.sort(key[ok] * BIG + (end[ok] - T0))
    q = np.where(q_key >= 0, q_key * BIG + (q_t - T0), -1)
    base = np.where(q_key >= 0, q_key * BIG, -1)
    n_start = np.searchsorted(cs, q, "right") - np.searchsorted(cs, base, "left")
    n_end = np.searchsorted(ce, q, "right") - np.searchsorted(ce, base, "left")
    cnt = (n_start - n_end).astype(np.float64)
    if self_ok is not None:
        inside = self_ok & (self_start <= q_t) & (q_t < self_end) & (self_start < self_end)
        cnt = cnt - inside.astype(np.float64)
    cnt[q_key < 0] = np.nan
    return cnt


def demean(v: np.ndarray, group: np.ndarray) -> np.ndarray:
    """v minus its (airport, month) mean over finite rows -- the shipped late_* treatment."""
    out = v.copy()
    ok = np.isfinite(v) & (group >= 0)
    if not ok.any():
        return out
    g = group[ok]
    sums = np.bincount(g, weights=v[ok])
    cnts = np.bincount(g)
    means = np.divide(sums, np.maximum(cnts, 1))
    out[ok] = v[ok] - means[g]
    return out


# ---------------------------------------------------------------------------------------------
# the candidates
# ---------------------------------------------------------------------------------------------
def build(dep: pl.DataFrame, arr: pl.DataFrame, only: set[str] | None = None) -> dict[str, np.ndarray]:
    """Every candidate, RAW (not demeaned), aligned to `dep` row order."""
    want = (lambda n: True) if only is None else (lambda n: n in only)
    out: dict[str, np.ndarray] = {}

    t = i64(dep["t"])
    sched, aobt, eobt = i64(dep["sched"]), i64(dep["aobt"]), i64(dep["eobt"])
    lobt, arvt1, arvt3 = i64(dep["lobt"]), i64(dep["arvt1"]), i64(dep["arvt3"])
    has = lambda x: x > -(10 ** 11)

    # keys -- joint codes so departures and arrivals share one code space per key type
    k_apt_d, k_apt_a = joint_codes(key_series(dep, "apt", "ym"), key_series(arr, "apt", "ym"))
    k_rwy_d, k_rwy_a = joint_codes(key_series(dep, "apt", "ym", "rwy"), key_series(arr, "apt", "ym", "rwy"))
    k_op_d, k_op_a = joint_codes(key_series(dep, "apt", "ym", "op"), key_series(arr, "apt", "ym", "op"))
    (k_ades_d,) = joint_codes(key_series(dep, "apt", "ym", "ades"))
    (k_stand_d,) = joint_codes(key_series(dep, "apt", "ym", "stand"))

    # --- departure proxies, each with its validity bound ---
    late = np.where(has(aobt) & has(eobt), (aobt - eobt).astype(float), np.nan)
    late_ok = np.isfinite(late) & (late >= LATE_LO) & (late <= LATE_HI)
    nm = np.where(has(aobt), (t - aobt).astype(float), np.nan)
    nm_ok = np.isfinite(nm) & (nm >= NM_LO) & (nm <= NM_HI)
    gs = (t - sched).astype(float)
    gs_ok = (gs >= LATE_LO) & (gs <= LATE_HI)

    def dpool(val, ok, key=k_apt_d, tt=t):
        return Pool(key, tt, val, ok)

    def put(name, v):
        if want(name):
            out[name] = v

    # A. take-off punctuality
    if any(want(n) for n in ("gs_apt_past15", "gs_apt_fol15", "gs_apt_past60", "gs_apt_fol60")):
        P = dpool(gs, gs_ok)
        for w, lab in ((900, "15"), (3600, "60")):
            n, s, _ = P.sums([P.past(k_apt_d, t, w)])
            put(f"gs_apt_past{lab}", mean_of(n, s))
            n, s, _ = P.sums([P.fol(k_apt_d, t, w)])
            put(f"gs_apt_fol{lab}", mean_of(n, s))
    if want("gs_rwy_past15") or want("gs_rwy_fol15"):
        P = dpool(gs, gs_ok, k_rwy_d)
        n, s, _ = P.sums([P.past(k_rwy_d, t, 900)])
        put("gs_rwy_past15", mean_of(n, s))
        n, s, _ = P.sums([P.fol(k_rwy_d, t, 900)])
        put("gs_rwy_fol15", mean_of(n, s))

    def cohort(val, ok):
        """Inclusive +/-30 min window on SCHED (flights share round schedule minutes, so ties are
        the cohort, not noise), with the row's own contribution subtracted explicitly."""
        P = Pool(k_apt_d, sched, val, ok)
        lo, hi = P.span(k_apt_d, sched, -1800, "left", 1800, "right")
        n, s, _ = P.sums([(lo, hi)])
        own = ok & np.isfinite(val) & (k_apt_d >= 0)
        n = n - own.astype(float)
        s = s - np.where(own, val, 0.0)
        return mean_of(n, s)

    if want("gs_cohort60"):
        out["gs_cohort60"] = cohort(gs, gs_ok)

    # B. plan-deviation clocks
    def clockdiff(a, b, lo, hi):
        v = np.where(has(a) & has(b), (a - b).astype(float), np.nan)
        return v, np.isfinite(v) & (v >= lo) & (v <= hi)

    sve, sve_ok = clockdiff(sched, eobt, -DEL_LIM, DEL_LIM)
    aol, aol_ok = clockdiff(aobt, lobt, LATE_LO, LATE_HI)
    los, los_ok = clockdiff(lobt, sched, -DEL_LIM, DEL_LIM)
    eol, eol_ok = clockdiff(eobt, lobt, -DEL_LIM, DEL_LIM)
    adr, adr_ok = clockdiff(arvt3, arvt1, -DEL_LIM, DEL_LIM)
    if want("sve_apt_past60") or want("sve_apt_fol60"):
        P = dpool(sve, sve_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        put("sve_apt_past60", mean_of(n, s))
        n, s, _ = P.sums([P.fol(k_apt_d, t, 3600)])
        put("sve_apt_fol60", mean_of(n, s))
    if any(want(n) for n in ("aol_apt_past15", "aol_apt_fol15", "aol_apt_past60")):
        P = dpool(aol, aol_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 900)])
        put("aol_apt_past15", mean_of(n, s))
        n, s, _ = P.sums([P.fol(k_apt_d, t, 900)])
        put("aol_apt_fol15", mean_of(n, s))
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        put("aol_apt_past60", mean_of(n, s))
    if want("aol_rwy_past15"):
        P = dpool(aol, aol_ok, k_rwy_d)
        n, s, _ = P.sums([P.past(k_rwy_d, t, 900)])
        out["aol_rwy_past15"] = mean_of(n, s)
    for name, (v, ok) in (("los_apt_past60", (los, los_ok)), ("eol_apt_past60", (eol, eol_ok)),
                          ("adrift_apt_past60", (adr, adr_ok))):
        if want(name):
            P = dpool(v, ok)
            n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
            out[name] = mean_of(n, s)
    if want("planupd_apt_past60"):
        upd = np.where(eol_ok, (eol != 0).astype(float), np.nan)
        P = dpool(upd, eol_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        out["planupd_apt_past60"] = mean_of(n, s)

    # C. lateness in new scopes
    if any(want(n) for n in ("late_op_past60", "late_op_fol60", "late_op_both120")):
        P = dpool(late, late_ok, k_op_d)
        a = P.past(k_op_d, t, 3600)
        b = P.fol(k_op_d, t, 3600)
        n, s, _ = P.sums([a])
        put("late_op_past60", mean_of(n, s))
        n, s, _ = P.sums([b])
        put("late_op_fol60", mean_of(n, s))
        n, s, _ = P.sums([P.past(k_op_d, t, 7200), P.fol(k_op_d, t, 7200)])
        put("late_op_both120", mean_of(n, s))
    if want("late_ades_both120"):
        P = dpool(late, late_ok, k_ades_d)
        n, s, _ = P.sums([P.past(k_ades_d, t, 7200), P.fol(k_ades_d, t, 7200)])
        out["late_ades_both120"] = mean_of(n, s)
    if want("late_prevstand"):
        P = dpool(late, late_ok, k_stand_d)
        out["late_prevstand"] = neighbour(P, k_stand_d, t, "prev", 6 * 3600)
    if want("late_rwy_next"):
        P = dpool(late, late_ok, k_rwy_d)
        out["late_rwy_next"] = neighbour(P, k_rwy_d, t, "next", 7200)
    if want("late_apt_share900"):
        ind = np.where(late_ok, (late > 900).astype(float), np.nan)
        P = dpool(ind, late_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        out["late_apt_share900"] = mean_of(n, s)
    if want("late_apt_sd60"):
        P = dpool(late, late_ok)
        n, s, s2 = P.sums([P.past(k_apt_d, t, 3600)])
        out["late_apt_sd60"] = sd_of(n, s, s2)
    if want("late_cohort60"):
        out["late_cohort60"] = cohort(late, late_ok)

    # D. taxi proxy in new scopes / statistics
    if want("nm_op_past60") or want("nm_op_fol60"):
        P = dpool(nm, nm_ok, k_op_d)
        n, s, _ = P.sums([P.past(k_op_d, t, 3600)])
        put("nm_op_past60", mean_of(n, s))
        n, s, _ = P.sums([P.fol(k_op_d, t, 3600)])
        put("nm_op_fol60", mean_of(n, s))
    if want("nm_ades_both120"):
        P = dpool(nm, nm_ok, k_ades_d)
        n, s, _ = P.sums([P.past(k_ades_d, t, 7200), P.fol(k_ades_d, t, 7200)])
        out["nm_ades_both120"] = mean_of(n, s)
    if want("nm_apt_share1800_p60") or want("nm_apt_share1800_f60"):
        ind = np.where(nm_ok, (nm > 1800).astype(float), np.nan)
        P = dpool(ind, nm_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        put("nm_apt_share1800_p60", mean_of(n, s))
        n, s, _ = P.sums([P.fol(k_apt_d, t, 3600)])
        put("nm_apt_share1800_f60", mean_of(n, s))
    if want("nm_apt_sd60"):
        P = dpool(nm, nm_ok)
        n, s, s2 = P.sums([P.past(k_apt_d, t, 3600)])
        out["nm_apt_sd60"] = sd_of(n, s, s2)

    # --- arrival proxies ---
    ta = i64(arr["t"])
    a_block, a_sched = i64(arr["block"]), i64(arr["sched"])
    a_aobt, a_eobt = i64(arr["aobt"]), i64(arr["eobt"])
    a_arvt1, a_arvt3 = i64(arr["arvt1"]), i64(arr["arvt3"])
    a_taxi = f64(arr["taxi"])

    def aclock(a, b, lo, hi):
        v = np.where(has(a) & has(b), (a - b).astype(float), np.nan)
        return v, np.isfinite(v) & (v >= lo) & (v <= hi)

    a_del, a_del_ok = aclock(a_block, a_sched, -DEL_LIM, DEL_LIM)
    a_land, a_land_ok = aclock(ta, a_sched, -DEL_LIM, DEL_LIM)
    a_est, a_est_ok = aclock(ta, a_arvt1, -DEL_LIM, DEL_LIM)
    g2g_act = np.where(has(a_aobt), (ta - a_aobt).astype(float), np.nan)
    g2g_pln = np.where(has(a_arvt1) & has(a_eobt), (a_arvt1 - a_eobt).astype(float), np.nan)
    a_exc = g2g_act - g2g_pln
    a_exc_ok = np.isfinite(a_exc) & (np.abs(a_exc) <= EXC_LIM)
    a_orig, a_orig_ok = aclock(a_aobt, a_eobt, LATE_LO, LATE_HI)
    a_clk, a_clk_ok = aclock(ta, a_arvt3, -1800, 1800)
    a_taxi_ok = np.isfinite(a_taxi) & (a_taxi >= 0) & (a_taxi <= 4 * 3600)

    def apool(val, ok, key=k_apt_a):
        return Pool(key, ta, val, ok)

    for name, (v, ok) in (("arrdel_apt_past60", (a_del, a_del_ok)),
                          ("arrland_apt_past60", (a_land, a_land_ok)),
                          ("arrest_apt_past60", (a_est, a_est_ok)),
                          ("arrexc_apt_past60", (a_exc, a_exc_ok)),
                          ("arrorig_apt_past60", (a_orig, a_orig_ok)),
                          ("arrclk_apt_past60", (a_clk, a_clk_ok))):
        if want(name):
            P = apool(v, ok)
            n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
            out[name] = mean_of(n, s)
    if want("arrexc_apt_both30") or want("arrexc_apt_fol60"):
        P = apool(a_exc, a_exc_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 1800), P.fol(k_apt_d, t, 1800)])
        put("arrexc_apt_both30", mean_of(n, s))
        n, s, _ = P.sums([P.fol(k_apt_d, t, 3600)])
        put("arrexc_apt_fol60", mean_of(n, s))

    # F. arrival side, scoped
    if want("arrtaxi_rwy_past60"):
        P = apool(a_taxi, a_taxi_ok, k_rwy_a)
        n, s, _ = P.sums([P.past(k_rwy_d, t, 3600)])
        out["arrtaxi_rwy_past60"] = mean_of(n, s)
    if want("arrexc_rwy_past60"):
        P = apool(a_exc, a_exc_ok, k_rwy_a)
        n, s, _ = P.sums([P.past(k_rwy_d, t, 3600)])
        out["arrexc_rwy_past60"] = mean_of(n, s)
    if want("arrtaxi_op_past60"):
        P = apool(a_taxi, a_taxi_ok, k_op_a)
        n, s, _ = P.sums([P.past(k_op_d, t, 3600)])
        out["arrtaxi_op_past60"] = mean_of(n, s)
    if want("arrdel_op_both120"):
        P = apool(a_del, a_del_ok, k_op_a)
        n, s, _ = P.sums([P.past(k_op_d, t, 7200), P.fol(k_op_d, t, 7200)])
        out["arrdel_op_both120"] = mean_of(n, s)

    # G. the linked inbound
    if any(want(n) for n in ("turn_obt_violation", "turn_in_origin_late", "turn_in_excess")):
        d = dep.select("apt", "stand", "ym", "atype", "op").with_columns(
            pl.Series("_i", np.arange(dep.height)), pl.Series("_t", t))
        a = (arr.select("apt", "stand", "ym", C("atype").alias("a_atype"), C("op").alias("a_op"))
             .with_columns(pl.Series("_ob", a_block), pl.Series("_orig", a_orig),
                           pl.Series("_orig_ok", a_orig_ok), pl.Series("_exc", a_exc),
                           pl.Series("_exc_ok", a_exc_ok))
             .filter(C("_ob") > -(10 ** 11)).drop_nulls(["stand"]).sort("_ob"))
        j = (d.drop_nulls(["stand"]).sort("_t")
             .join_asof(a, left_on="_t", right_on="_ob", by=["apt", "stand", "ym"], strategy="backward")
             .filter(C("_ob").is_not_null()
                     & ((C("_t") - C("_ob")) <= TURN_MAX)
                     & (C("atype") == C("a_atype")) & (C("op") == C("a_op"))))
        idx = j["_i"].to_numpy()
        ob = j["_ob"].to_numpy().astype(np.int64)
        v1 = np.full(dep.height, np.nan)
        own = aobt[idx]
        okk = has(own)
        v1[idx[okk]] = np.minimum((own[okk] - ob[okk]).astype(float), 0.0)
        put("turn_obt_violation", v1)
        v2 = np.full(dep.height, np.nan)
        m2 = j["_orig_ok"].to_numpy()
        v2[idx[m2]] = j["_orig"].to_numpy()[m2]
        put("turn_in_origin_late", v2)
        v3 = np.full(dep.height, np.nan)
        m3 = j["_exc_ok"].to_numpy()
        v3[idx[m3]] = j["_exc"].to_numpy()[m3]
        put("turn_in_excess", v3)

    # H. occupancy at wheels-up
    ok_all = np.ones(len(t), dtype=bool)
    if want("backlog_sched"):
        out["backlog_sched"] = containing(k_apt_d, sched, t, ok_all, k_apt_d, t,
                                          self_ok=ok_all, self_start=sched, self_end=t)
    if want("surface_now") or want("surface_15ago"):
        okb = has(aobt)
        put("surface_now", containing(k_apt_d, aobt, t, okb, k_apt_d, t,
                                      self_ok=okb, self_start=aobt, self_end=t))
        put("surface_15ago", containing(k_apt_d, aobt, t, okb, k_apt_d, t - 900,
                                        self_ok=okb, self_start=aobt, self_end=t))

    # I. clock-regime shares
    if want("mvtsec00_apt_p60"):
        ind = ((t % 60) == 0).astype(float)
        P = dpool(ind, ok_all)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        out["mvtsec00_apt_p60"] = mean_of(n, s)
    if want("arrsec00_apt_p60"):
        okb = has(a_block)
        ind = np.where(okb, ((a_block % 60) == 0).astype(float), np.nan)
        P = apool(ind, okb)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        out["arrsec00_apt_p60"] = mean_of(n, s)
    if want("latezero_apt_p60"):
        ind = np.where(late_ok, (late == 0).astype(float), np.nan)
        P = dpool(ind, late_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        out["latezero_apt_p60"] = mean_of(n, s)
    if want("aobteqlobt_apt_p60"):
        ind = np.where(aol_ok, (aol == 0).astype(float), np.nan)
        P = dpool(ind, aol_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        out["aobteqlobt_apt_p60"] = mean_of(n, s)

    # J. destination side: arrivals at ADES around our first-estimate arrival time, own row excluded
    if want("dest_arrexc_eta"):
        (k_dest_q, k_dest_s) = joint_codes(key_series(dep, "ades", "ym_eta"), key_series(arr, "apt", "ym"))
        P = Pool(k_dest_s, ta, a_exc, a_exc_ok)
        lo, hi = P.span(k_dest_q, arvt1, -1800, "left", 1800, "right")
        n, s, _ = P.sums([(lo, hi)])
        # subtract this flight's OWN arrival row at the destination, found by FLIGHT_ID
        fid_d = f64(dep["FLIGHT_ID_mvt"])
        own = (pl.DataFrame({"fid": fid_d, "ades": dep["ades"], "_i": np.arange(dep.height)})
               .join(pl.DataFrame({"fid": f64(arr["FLIGHT_ID_mvt"]), "ades": arr["apt"],
                                   "_ta": ta, "_v": a_exc, "_ok": a_exc_ok}),
                     on=["fid", "ades"], how="inner"))
        oi = own["_i"].to_numpy()
        inwin = (own["_ok"].to_numpy() & (np.abs(own["_ta"].to_numpy() - arvt1[oi]) <= 1800))
        np.subtract.at(n, oi[inwin], 1.0)
        np.subtract.at(s, oi[inwin], own["_v"].to_numpy()[inwin])
        v = mean_of(n, s)
        v[~np.isfinite(fid_d) | ~has(arvt1)] = np.nan
        out["dest_arrexc_eta"] = v

    return out


def local_clock(dep: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local hour, UTC hour and local weekend flag of each departure's wheels-up."""
    ts = pl.from_epoch(dep["t"], time_unit="s").dt.replace_time_zone("UTC")
    frame = pl.DataFrame({"apt": dep["apt"], "ts": ts})
    lh = None
    wk = None
    for apt, tz in LOCAL_TZ.items():
        loc = C("ts").dt.convert_time_zone(tz)
        lh = (pl.when(C("apt") == apt).then(loc.dt.hour()) if lh is None
              else lh.when(C("apt") == apt).then(loc.dt.hour()))
        wk = (pl.when(C("apt") == apt).then(loc.dt.weekday() >= 6) if wk is None
              else wk.when(C("apt") == apt).then(loc.dt.weekday() >= 6))
    f = frame.select(lh.otherwise(None).alias("lh"), C("ts").dt.hour().alias("uh"),
                     wk.otherwise(None).alias("wk"))
    return (f["lh"].fill_null(-1).to_numpy().astype(np.int64), f["uh"].to_numpy().astype(np.int64),
            f["wk"].fill_null(False).to_numpy().astype(np.int64))


def cell_residual(cell: np.ndarray, resid: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """For each row in fold k, the mean chained residual of its cell over the rows of the OTHER
    folds. Rows outside every fold (months 6 and 12) get NaN. Cells with < 30 rows elsewhere: NaN."""
    out = np.full(len(cell), np.nan)
    in_fold = fold >= 0
    codes, inv = np.unique(cell[in_fold], return_inverse=True)
    rf, ff = resid[in_fold], fold[in_fold]
    nf = int(ff.max()) + 1
    s = np.zeros((len(codes), nf))
    c = np.zeros((len(codes), nf))
    np.add.at(s, (inv, ff), rf)
    np.add.at(c, (inv, ff), 1.0)
    s_tot, c_tot = s.sum(axis=1), c.sum(axis=1)
    so = s_tot[inv] - s[inv, ff]
    co = c_tot[inv] - c[inv, ff]
    val = np.where(co >= 30, so / np.maximum(co, 1.0), np.nan)
    out[np.flatnonzero(in_fold)] = val
    return out


def build_r2(dep: pl.DataFrame, arr: pl.DataFrame, only: set[str] | None = None,
             resid: np.ndarray | None = None, fold: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Round-2 candidates, RAW, aligned to `dep`. The P family needs out-of-fold chained residuals
    and fold ids (from the fold files); without them it is skipped."""
    want = (lambda n: True) if only is None else (lambda n: n in only)
    out: dict[str, np.ndarray] = {}
    t = i64(dep["t"])
    sched, aobt, eobt = i64(dep["sched"]), i64(dep["aobt"]), i64(dep["eobt"])
    arvt1 = i64(dep["arvt1"])
    has = lambda x: x > -(10 ** 11)
    ta = i64(arr["t"])
    a_block, a_sched = i64(arr["block"]), i64(arr["sched"])
    a_aobt, a_eobt, a_arvt1 = i64(arr["aobt"]), i64(arr["eobt"]), i64(arr["arvt1"])
    a_taxi = f64(arr["taxi"])

    late = np.where(has(aobt) & has(eobt), (aobt - eobt).astype(float), np.nan)
    late_ok = np.isfinite(late) & (late >= LATE_LO) & (late <= LATE_HI)
    nm = np.where(has(aobt), (t - aobt).astype(float), np.nan)
    nm_ok = np.isfinite(nm) & (nm >= NM_LO) & (nm <= NM_HI)
    gs = (t - sched).astype(float)
    gs_ok = (gs >= LATE_LO) & (gs <= LATE_HI)
    a_sub = np.where(has(a_block), np.abs(a_block - a_sched).astype(float), np.nan)
    a_sub_ok = np.isfinite(a_sub)
    g2g_act = np.where(has(a_aobt), (ta - a_aobt).astype(float), np.nan)
    g2g_pln = np.where(has(a_arvt1) & has(a_eobt), (a_arvt1 - a_eobt).astype(float), np.nan)
    a_exc = g2g_act - g2g_pln
    a_exc_ok = np.isfinite(a_exc) & (np.abs(a_exc) <= EXC_LIM)
    a_taxi_ok = np.isfinite(a_taxi) & (a_taxi >= 0) & (a_taxi <= 4 * 3600)

    dep_p = dep.with_columns(C("stand").str.slice(0, 1).alias("pier"))
    arr_p = arr.with_columns(C("stand").str.slice(0, 1).alias("pier"))
    k_apt_d, k_apt_a = joint_codes(key_series(dep, "apt", "ym"), key_series(arr, "apt", "ym"))
    k_rwy_d, k_rwy_a = joint_codes(key_series(dep, "apt", "ym", "rwy"), key_series(arr, "apt", "ym", "rwy"))
    k_op_d, k_op_a = joint_codes(key_series(dep, "apt", "ym", "op"), key_series(arr, "apt", "ym", "op"))
    k_pier_d, k_pier_a = joint_codes(key_series(dep_p, "apt", "ym", "pier"), key_series(arr_p, "apt", "ym", "pier"))
    k_net_d, k_net_a = joint_codes(key_series(dep, "ym"), key_series(arr, "ym"))
    (k_opnet_d,) = joint_codes(key_series(dep, "op", "ym"))
    (k_adesnet_d,) = joint_codes(key_series(dep, "ades", "ym_eta"))

    def put(name, v):
        if want(name):
            out[name] = v

    # K. copy regime / broken stamps on the arrival side
    if any(want(n) for n in ("arrsub6_apt_both60", "arrsub6_apt_both180")):
        ind = np.where(a_sub_ok, (a_sub <= 6).astype(float), np.nan)
        P = Pool(k_apt_a, ta, ind, a_sub_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600), P.fol(k_apt_d, t, 3600)])
        put("arrsub6_apt_both60", mean_of(n, s))
        n, s, _ = P.sums([P.past(k_apt_d, t, 10800), P.fol(k_apt_d, t, 10800)])
        put("arrsub6_apt_both180", mean_of(n, s))
    if want("arrsub60_apt_both60"):
        ind = np.where(a_sub_ok, (a_sub <= 60).astype(float), np.nan)
        P = Pool(k_apt_a, ta, ind, a_sub_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600), P.fol(k_apt_d, t, 3600)])
        out["arrsub60_apt_both60"] = mean_of(n, s)
    if want("arrneg_apt_both120"):
        okb = has(a_block)
        ind = np.where(okb, (a_block <= ta).astype(float), np.nan)
        P = Pool(k_apt_a, ta, ind, okb)
        n, s, _ = P.sums([P.past(k_apt_d, t, 7200), P.fol(k_apt_d, t, 7200)])
        out["arrneg_apt_both120"] = mean_of(n, s)
    if want("arrsub6_op_both180"):
        ind = np.where(a_sub_ok, (a_sub <= 6).astype(float), np.nan)
        P = Pool(k_op_a, ta, ind, a_sub_ok)
        n, s, _ = P.sums([P.past(k_op_d, t, 10800), P.fol(k_op_d, t, 10800)])
        out["arrsub6_op_both180"] = mean_of(n, s)
    if want("turn_in_copy6"):
        d = dep.select("apt", "stand", "ym").with_columns(pl.Series("_i", np.arange(dep.height)),
                                                          pl.Series("_t", t))
        a = (arr.select("apt", "stand", "ym")
             .with_columns(pl.Series("_ob", a_block), pl.Series("_sub", a_sub))
             .filter(C("_ob") > -(10 ** 11)).drop_nulls(["stand"]).sort("_ob"))
        j = (d.drop_nulls(["stand"]).sort("_t")
             .join_asof(a, left_on="_t", right_on="_ob", by=["apt", "stand", "ym"], strategy="backward")
             .filter(C("_ob").is_not_null() & ((C("_t") - C("_ob")) <= TURN_MAX)))
        v = np.full(dep.height, np.nan)
        v[j["_i"].to_numpy()] = (j["_sub"].to_numpy() <= 6).astype(float)
        out["turn_in_copy6"] = v

    # L. network scope
    if want("late_net_past60"):
        P = Pool(k_net_d, t, late, late_ok)
        n, s, _ = P.sums([P.past(k_net_d, t, 3600)])
        out["late_net_past60"] = mean_of(n, s)
    if want("late_opnet_both60"):
        P = Pool(k_opnet_d, t, late, late_ok)
        n, s, _ = P.sums([P.past(k_opnet_d, t, 3600), P.fol(k_opnet_d, t, 3600)])
        out["late_opnet_both60"] = mean_of(n, s)
    if want("late_adesnet_eta60"):
        P = Pool(k_adesnet_d, arvt1, late, late_ok & has(arvt1))
        lo, hi = P.span(k_adesnet_d, arvt1, -3600, "left", 3600, "right")
        n, s, _ = P.sums([(lo, hi)])
        own = late_ok & has(arvt1) & (k_adesnet_d >= 0)
        n = n - own.astype(float)
        s = s - np.where(own, late, 0.0)
        v = mean_of(n, s)
        v[~has(arvt1)] = np.nan
        out["late_adesnet_eta60"] = v
    if want("arrexc_net_past60"):
        P = Pool(k_net_a, ta, a_exc, a_exc_ok)
        n, s, _ = P.sums([P.past(k_net_d, t, 3600)])
        out["arrexc_net_past60"] = mean_of(n, s)
    if want("nm_net_past60"):
        P = Pool(k_net_d, t, nm, nm_ok)
        n, s, _ = P.sums([P.past(k_net_d, t, 3600)])
        out["nm_net_past60"] = mean_of(n, s)

    # M. pier scope
    if want("nm_pier_past60"):
        P = Pool(k_pier_d, t, nm, nm_ok)
        n, s, _ = P.sums([P.past(k_pier_d, t, 3600)])
        out["nm_pier_past60"] = mean_of(n, s)
    if want("late_pier_past60"):
        P = Pool(k_pier_d, t, late, late_ok)
        n, s, _ = P.sums([P.past(k_pier_d, t, 3600)])
        out["late_pier_past60"] = mean_of(n, s)
    if want("arrtaxi_pier_past60"):
        P = Pool(k_pier_a, ta, a_taxi, a_taxi_ok)
        n, s, _ = P.sums([P.past(k_pier_d, t, 3600)])
        out["arrtaxi_pier_past60"] = mean_of(n, s)

    # N. queue at the scheduled time
    if want("backlog_at_sched"):
        ok_all = np.ones(len(t), dtype=bool)
        out["backlog_at_sched"] = containing(k_apt_d, sched, t, ok_all, k_apt_d, sched,
                                             self_ok=ok_all, self_start=sched, self_end=t)
    if want("arrbank_at_sched"):
        okb = has(a_block)
        P = Pool(k_apt_a, a_block, np.ones(len(ta)), okb)
        lo, hi = P.span(k_apt_d, sched, -900, "left", 900, "right")
        n, _, _ = P.sums([(lo, hi)])
        n[k_apt_d < 0] = np.nan
        out["arrbank_at_sched"] = n

    # O. siblings of round 1's pass
    if want("late_rwy_prev"):
        P = Pool(k_rwy_d, t, late, late_ok)
        out["late_rwy_prev"] = neighbour(P, k_rwy_d, t, "prev", 7200)
    if want("gs_rwy_next"):
        P = Pool(k_rwy_d, t, gs, gs_ok)
        out["gs_rwy_next"] = neighbour(P, k_rwy_d, t, "next", 7200)

    # P. local time -- a residual encoding, so it needs the fold residuals
    if resid is not None and (want("localhour_cellres") or want("utchour_cellres")):
        lh, uh, wk = local_clock(dep)
        (apt_code,) = joint_codes(dep["apt"])
        put("localhour_cellres", cell_residual(apt_code * 100 + lh * 2 + wk, resid, fold))
        put("utchour_cellres", cell_residual(apt_code * 100 + uh * 2 + wk, resid, fold))

    # Q / X
    if want("arr_rwy_cnt_both10"):
        P = Pool(k_rwy_a, ta, np.ones(len(ta)), np.ones(len(ta), dtype=bool))
        lo, hi = P.span(k_rwy_d, t, -600, "left", 600, "right")
        n, _, _ = P.sums([(lo, hi)])
        n[k_rwy_d < 0] = np.nan
        out["arr_rwy_cnt_both10"] = n
    if want("sched_same_minute"):
        P = Pool(k_apt_d, sched, np.ones(len(t)), np.ones(len(t), dtype=bool))
        lo, hi = P.span(k_apt_d, sched, 0, "left", 0, "right")
        n, _, _ = P.sums([(lo, hi)])
        n = n - 1.0
        n[k_apt_d < 0] = np.nan
        out["sched_same_minute"] = n
    return out


def loo_group_mean(key: np.ndarray, val: np.ndarray, ok: np.ndarray, min_others: int = 5):
    """For each row, the mean of `val` over the OTHER valid rows sharing its key (leave-one-out).
    NaN when fewer than `min_others` other valid rows exist."""
    good = ok & (key >= 0) & np.isfinite(val)
    kk = np.where(key >= 0, key, 0)
    nk = int(kk.max()) + 1 if len(kk) else 1
    s = np.bincount(kk[good], weights=val[good], minlength=nk)
    c = np.bincount(kk[good], minlength=nk).astype(float)
    so = s[kk] - np.where(good, val, 0.0)
    co = c[kk] - good.astype(float)
    out = np.where(co >= min_others, so / np.maximum(co, 1.0), np.nan)
    out[key < 0] = np.nan
    return out


def build_r3(dep: pl.DataFrame, arr: pl.DataFrame, only: set[str] | None = None) -> dict[str, np.ndarray]:
    """Round-3 candidates, RAW, aligned to `dep`."""
    want = (lambda n: True) if only is None else (lambda n: n in only)
    out: dict[str, np.ndarray] = {}
    t = i64(dep["t"])
    aobt, eobt = i64(dep["aobt"]), i64(dep["eobt"])
    has = lambda x: x > -(10 ** 11)
    ta = i64(arr["t"])
    a_aobt, a_eobt, a_arvt1 = i64(arr["aobt"]), i64(arr["eobt"]), i64(arr["arvt1"])
    a_taxi = f64(arr["taxi"])

    late = np.where(has(aobt) & has(eobt), (aobt - eobt).astype(float), np.nan)
    late_ok = np.isfinite(late) & (late >= LATE_LO) & (late <= LATE_HI)
    nm = np.where(has(aobt), (t - aobt).astype(float), np.nan)
    nm_ok = np.isfinite(nm) & (nm >= NM_LO) & (nm <= NM_HI)
    a_taxi_ok = np.isfinite(a_taxi) & (a_taxi >= 0) & (a_taxi <= 4 * 3600)
    g2g_act = np.where(has(a_aobt), (ta - a_aobt).astype(float), np.nan)
    g2g_pln = np.where(has(a_arvt1) & has(a_eobt), (a_arvt1 - a_eobt).astype(float), np.nan)
    a_exc = g2g_act - g2g_pln
    a_exc_ok = np.isfinite(a_exc) & (np.abs(a_exc) <= EXC_LIM)

    k_apt_d, k_apt_a = joint_codes(key_series(dep, "apt", "ym"), key_series(arr, "apt", "ym"))
    (k_rwy_d,) = joint_codes(key_series(dep, "apt", "ym", "rwy"))
    (k_pair_d,) = joint_codes(key_series(dep, "apt", "ym", "stand", "rwy"))
    (k_pair_a,) = joint_codes(key_series(arr, "apt", "ym", "stand", "rwy"))
    k_stand_d, k_stand_a = joint_codes(key_series(dep, "apt", "ym", "stand"), key_series(arr, "apt", "ym", "stand"))
    (k_op_d,) = joint_codes(key_series(dep, "apt", "ym", "op"))

    def put(name, v):
        if want(name):
            out[name] = v

    # S. mix-adjusted congestion
    if want("arrtaxi_exc_apt_past60") or want("arrtaxi_exc_apt_both30"):
        ref = loo_group_mean(k_pair_a, a_taxi, a_taxi_ok)
        exc = a_taxi - ref
        ok = a_taxi_ok & np.isfinite(ref)
        P = Pool(k_apt_a, ta, exc, ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        put("arrtaxi_exc_apt_past60", mean_of(n, s))
        n, s, _ = P.sums([P.past(k_apt_d, t, 1800), P.fol(k_apt_d, t, 1800)])
        put("arrtaxi_exc_apt_both30", mean_of(n, s))
    if any(want(n) for n in ("nm_exc_apt_past60", "nm_exc_rwy_past15", "nm_exc_apt_fol60")):
        ref = loo_group_mean(k_pair_d, nm, nm_ok)
        exc = nm - ref
        ok = nm_ok & np.isfinite(ref)
        P = Pool(k_apt_d, t, exc, ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        put("nm_exc_apt_past60", mean_of(n, s))
        n, s, _ = P.sums([P.fol(k_apt_d, t, 3600)])
        put("nm_exc_apt_fol60", mean_of(n, s))
        P = Pool(k_rwy_d, t, exc, ok)
        n, s, _ = P.sums([P.past(k_rwy_d, t, 900)])
        put("nm_exc_rwy_past15", mean_of(n, s))

    # T. slow time scales
    if want("nm_rwy_lag24h"):
        P = Pool(k_rwy_d, t, nm, nm_ok)
        lo, hi = P.span(k_rwy_d, t - 86400, -1800, "left", 1800, "right")
        n, s, _ = P.sums([(lo, hi)])
        out["nm_rwy_lag24h"] = mean_of(n, s)
    if want("nm_pair7d_dev"):
        P = Pool(k_pair_d, t, nm, nm_ok)
        n, s, _ = P.sums([P.past(k_pair_d, t, 7 * 86400)])
        m7 = mean_of(n, s)
        ref = loo_group_mean(k_pair_d, nm, nm_ok, min_others=5)
        out["nm_pair7d_dev"] = m7 - ref
    if want("arrtaxi_stand7d_dev"):
        P = Pool(k_stand_a, ta, a_taxi, a_taxi_ok)
        n, s, _ = P.sums([P.past(k_stand_d, t, 7 * 86400)])
        m7 = mean_of(n, s)
        # the stand's month mean of arrival taxi-in, from ARRIVALS only (no departure enters it)
        good = a_taxi_ok & (k_stand_a >= 0)
        nk = int(max(k_stand_a.max(), k_stand_d.max())) + 1
        ssum = np.bincount(k_stand_a[good], weights=a_taxi[good], minlength=nk)
        scnt = np.bincount(k_stand_a[good], minlength=nk).astype(float)
        kd = np.where(k_stand_d >= 0, k_stand_d, 0)
        ref = np.where((k_stand_d >= 0) & (scnt[kd] >= 5), ssum[kd] / np.maximum(scnt[kd], 1.0), np.nan)
        out["arrtaxi_stand7d_dev"] = m7 - ref

    # U. lateness net of airline habit
    if want("lateexc_apt_past60") or want("lateexc_rwy_past15"):
        ref = loo_group_mean(k_op_d, late, late_ok)
        exc = late - ref
        ok = late_ok & np.isfinite(ref)
        P = Pool(k_apt_d, t, exc, ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 3600)])
        put("lateexc_apt_past60", mean_of(n, s))
        P = Pool(k_rwy_d, t, exc, ok)
        n, s, _ = P.sums([P.past(k_rwy_d, t, 900)])
        put("lateexc_rwy_past15", mean_of(n, s))

    # V. holding tail
    if want("arrhold_share_both30"):
        ind = np.where(a_exc_ok, (a_exc > 600).astype(float), np.nan)
        P = Pool(k_apt_a, ta, ind, a_exc_ok)
        n, s, _ = P.sums([P.past(k_apt_d, t, 1800), P.fol(k_apt_d, t, 1800)])
        out["arrhold_share_both30"] = mean_of(n, s)
    return out


def group_apt_month(dep: pl.DataFrame) -> np.ndarray:
    (g,) = joint_codes(key_series(dep, "apt", "ym"))
    return g


def group_scope(dep: pl.DataFrame, scope: str) -> np.ndarray:
    if scope == "pier":
        dep = dep.with_columns(C("stand").str.slice(0, 1).alias("pier"))
    cols = {"op": ("apt", "ym", "op"), "ades": ("apt", "ym", "ades"), "rwy": ("apt", "ym", "rwy"),
            "stand": ("apt", "ym", "stand"), "pier": ("apt", "ym", "pier"),
            "none": ("apt", "ym")}[scope]
    (g,) = joint_codes(key_series(dep, *cols))
    return g


def placebo(v: np.ndarray, group: np.ndarray, seed: int) -> np.ndarray:
    """The same construction read at a DIFFERENT flight's time in the same group. Finite values are
    permuted among the rows of each group that already had one, so the missingness pattern is held
    fixed and the group's level is preserved; only the time alignment is broken."""
    rng = np.random.default_rng(seed)
    out = v.copy()
    ok = np.flatnonzero(np.isfinite(v) & (group >= 0))
    if len(ok) < 2:
        return out
    g = group[ok]
    order = np.lexsort((rng.random(len(ok)), g))       # random order within each group
    sorted_idx = ok[np.argsort(g, kind="stable")]      # rows grouped, original order
    shuffled_idx = ok[order]                           # rows grouped, random order
    out[sorted_idx] = v[shuffled_idx]
    return out


def stable_seed(name: str) -> int:
    return 20260921 + sum((i + 1) * ord(c) for i, c in enumerate(name))
