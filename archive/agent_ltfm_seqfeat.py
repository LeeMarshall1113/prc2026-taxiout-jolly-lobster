"""LTFM take-off-SEQUENCE features (importable; no I/O at import). Serve-computable: reads only MVT, LOBT, SCHED,
STAND, RUNWAY of departures and MVT/RUNWAY of arrivals -- all present in ranking.parquet -- plus a (stand, runway)
median-taxi table T fitted on labelled rows the caller chooses (fold-honest when T excludes the test months).

Physics. At LTFM, AOBT_3 is not a measurement (45% of MVT - AOBT_3 sits exactly on 20 min: NM's default taxi), so
the ledger's closure of the queue family ("counts between off-block and wheels-up re-read gap_aobt") does not hold
here: there is no off-block clock. What IS known for every departure is its planned off-block (LOBT) and its
take-off time. Under FIFO at the runway, departure ORDER reveals relative queue-entry order. If flights planned
after me took off before me, I entered the queue late (pushed late -> shorter taxi than my slip suggests); if I
overtook flights planned before me, they pushed late or I was quick. Isotonic regression of planned take-off times
(LOBT + T) onto the observed take-off order pools neighbours' plans into a denoised queue-entry estimate.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

W_OT = 1800          # overtake window (s)
SESSION_BREAK = 3600  # a runway gap longer than this starts a new sequence
BUSY_H = 150         # consecutive take-offs closer than this = saturated runway
RWY_GROUP = {"16L": "W", "34R": "W", "16R": "W", "34L": "W", "17L": "C", "35R": "C", "17R": "C", "35L": "C",
             "18": "E", "36": "E"}

SEQ_FEATURES = ["sq_slip", "sq_ot_ahead_n", "sq_ot_ahead_s", "sq_ot_behind_n", "sq_ot_behind_s", "sq_ot_net",
                "sq_win_n", "sq_iso_dev", "sq_iso_gap", "sq_nb_prev_slip", "sq_nb_next_slip", "sq_own_minus_nb",
                "sq_busy_elapsed", "sq_busy_pos", "sq_busy_rest", "sq_arr_W15", "sq_arr_C15", "sq_arr_E15",
                "sq_dep_other15"]


def fit_T(lab: pl.DataFrame) -> tuple[dict, dict, float]:
    """Median taxi per (stand, runway) with n >= 20, back-off per runway, then global. lab: STAND_mvt, RUNWAY_mvt, y."""
    g = lab.group_by("STAND_mvt", "RUNWAY_mvt").agg(m=pl.col("y").median(), n=pl.len()).filter(pl.col("n") >= 20)
    t_sr = {(a, b): float(m) for a, b, m in g.select("STAND_mvt", "RUNWAY_mvt", "m").iter_rows()}
    r = lab.group_by("RUNWAY_mvt").agg(m=pl.col("y").median())
    t_r = {a: float(m) for a, m in r.iter_rows()}
    return t_sr, t_r, float(lab["y"].median())


LEVEL_FEATURES = ["sq_slip", "sq_iso_gap", "sq_nb_prev_slip", "sq_nb_next_slip"]


def build(deps: pl.DataFrame, arrs: pl.DataFrame, T, demean: bool = False) -> pl.DataFrame:
    """deps: MVT_ID_mvt, MVT_TIME_UTC_mvt, LOBT_flt, SCHED_TIME_UTC_mvt, STAND_mvt, RUNWAY_mvt (all LTFM departures
    in the frame, labelled or not). arrs: MVT_TIME_UTC_mvt, RUNWAY_mvt of LTFM arrivals. Returns MVT_ID_mvt + SEQ_FEATURES."""
    t_sr, t_r, t_g = T
    n = deps.height
    mvt = deps["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(float)
    lobt = deps["LOBT_flt"].dt.epoch("s").to_numpy().astype(float) if deps["LOBT_flt"].null_count() < n else np.full(n, np.nan)
    lobt = np.where(deps["LOBT_flt"].is_null().to_numpy(), np.nan, lobt)
    sched = deps["SCHED_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(float)
    base = np.where(np.isfinite(lobt), lobt, sched)
    stand = deps["STAND_mvt"].fill_null("?").to_numpy().astype(str)
    rwy = deps["RUNWAY_mvt"].fill_null("?").to_numpy().astype(str)
    tt = np.array([t_sr.get((s, r), t_r.get(r, t_g)) for s, r in zip(stand, rwy)])
    pm = base + tt
    slip = mvt - pm

    out = {k: np.full(n, np.nan) for k in SEQ_FEATURES}
    out["sq_slip"] = slip
    for r in np.unique(rwy):
        idx = np.flatnonzero(rwy == r)
        o = idx[np.argsort(mvt[idx], kind="stable")]
        t, p, s = mvt[o], pm[o], slip[o]
        m = len(o)
        # overtakes inside +/- W_OT
        lo = np.searchsorted(t, t - W_OT, "left")
        hi = np.searchsorted(t, t + W_OT, "right")
        a_n = np.zeros(m); a_s = np.zeros(m); b_n = np.zeros(m); b_s = np.zeros(m)
        maxoff = int(max((np.arange(m) - lo).max(initial=0), (hi - 1 - np.arange(m)).max(initial=0)))
        ar = np.arange(m)
        for d in range(1, maxoff + 1):
            j = ar - d
            ok = (j >= lo) & (j >= 0)
            jj = np.clip(j, 0, m - 1)
            diff = p[jj] - p           # planned take-off of the one AHEAD minus mine
            hit = ok & (diff > 0)
            a_n += hit; a_s += np.where(hit, diff, 0.0)
            j = ar + d
            ok = (j < hi) & (j < m)
            jj = np.clip(j, 0, m - 1)
            diff = p - p[jj]           # mine minus the one BEHIND
            hit = ok & (diff > 0)
            b_n += hit; b_s += np.where(hit, diff, 0.0)
        out["sq_ot_ahead_n"][o] = a_n; out["sq_ot_ahead_s"][o] = a_s / 60
        out["sq_ot_behind_n"][o] = b_n; out["sq_ot_behind_s"][o] = b_s / 60
        out["sq_ot_net"][o] = a_n - b_n
        out["sq_win_n"][o] = hi - lo - 1
        # sessions: runway gap > SESSION_BREAK splits the sequence
        head = np.diff(t, prepend=-1e18)
        sess = np.cumsum(head > SESSION_BREAK)
        iso = np.empty(m)
        nb_prev = np.full(m, np.nan); nb_next = np.full(m, np.nan)
        for sid in np.unique(sess):
            q = np.flatnonzero(sess == sid)
            if len(q) >= 2:
                iso[q] = IsotonicRegression(increasing=True).fit_transform(np.arange(len(q)), p[q])
            else:
                iso[q] = p[q]
            sq = s[q]
            k = 4
            for ii in range(len(q)):
                pv = sq[max(0, ii - k):ii]
                nx = sq[ii + 1:ii + 1 + k]
                if len(pv):
                    nb_prev[q[ii]] = np.median(pv)
                if len(nx):
                    nb_next[q[ii]] = np.median(nx)
        out["sq_iso_dev"][o] = iso - p
        out["sq_iso_gap"][o] = t - iso
        out["sq_nb_prev_slip"][o] = nb_prev
        out["sq_nb_next_slip"][o] = nb_next
        both = np.nanmedian(np.vstack([nb_prev, nb_next]), axis=0)
        out["sq_own_minus_nb"][o] = s - both
        # busy periods
        busy_start = np.empty(m); pos = np.zeros(m)
        for ii in range(m):
            if ii > 0 and t[ii] - t[ii - 1] < BUSY_H:
                busy_start[ii] = busy_start[ii - 1]; pos[ii] = pos[ii - 1] + 1
            else:
                busy_start[ii] = t[ii]; pos[ii] = 0
        rest = np.zeros(m)
        for ii in range(m - 2, -1, -1):
            if t[ii + 1] - t[ii] < BUSY_H:
                rest[ii] = rest[ii + 1] + 1
        out["sq_busy_elapsed"][o] = t - busy_start
        out["sq_busy_pos"][o] = pos
        out["sq_busy_rest"][o] = rest
    # arrivals by runway group in the 15 min before wheels-up; departures on OTHER runways, same window
    at = arrs["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(float)
    ag = np.array([RWY_GROUP.get(x, "?") for x in arrs["RUNWAY_mvt"].fill_null("?").to_numpy().astype(str)])
    for gname in ("W", "C", "E"):
        st = np.sort(at[ag == gname])
        out[f"sq_arr_{gname}15"] = (np.searchsorted(st, mvt, "right") - np.searchsorted(st, mvt - 900, "left")).astype(float)
    alld = np.sort(mvt)
    tot = np.searchsorted(alld, mvt, "right") - np.searchsorted(alld, mvt - 900, "left")
    own = np.zeros(n)
    for r in np.unique(rwy):
        idx = np.flatnonzero(rwy == r)
        st = np.sort(mvt[idx])
        own[idx] = np.searchsorted(st, mvt[idx], "right") - np.searchsorted(st, mvt[idx] - 900, "left")
    out["sq_dep_other15"] = (tot - own).astype(float)
    if demean:
        # The slip LEVELS drift 2025->2026 by +-200 s at the median (agent_ltfm_07), the same lateness drift late_* is
        # demeaned against. Subtract each calendar month's mean over this frame's LTFM departures (label-free, and the
        # same arithmetic in a whole-year training frame and the Jan+Jul serve frame). Relative columns are untouched.
        mon = deps["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy().astype(str)
        for c in LEVEL_FEATURES:
            v = out[c]
            for m in np.unique(mon):
                k = mon == m
                mu = np.nanmean(v[k]) if np.isfinite(v[k]).any() else 0.0
                v[k] = v[k] - mu
    return pl.DataFrame({"MVT_ID_mvt": deps["MVT_ID_mvt"], **{k: out[k] for k in SEQ_FEATURES}})
