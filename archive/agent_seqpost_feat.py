"""seqpost: take-off-SEQUENCE features generalised from archive/agent_ltfm_seqfeat.py to all ten airports
(importable, no I/O at import). ONE implementation for the fold test and for serving.

Inputs are serve-computable: departures' MVT, LOBT, SCHED, STAND, RUNWAY (+ airport) and arrivals' MVT, RUNWAY (+ airport),
all present in data/raw/training_*.parquet and data/raw/ranking.parquet, plus a per-row planned-taxi value `tt` from an
(apt, stand, runway) median-taxi table T fitted on labelled rows the caller chooses (fold-honest when T excludes the
row's own month and the test month).

Differences from the LTFM version (disclosed): sequences are keyed by (airport, runway); the LTFM runway-group arrival
counts (W/C/E) are kept for LTFM only (NaN elsewhere) and two generic counts are added for every airport
(arrivals on the SAME runway designator and ALL arrivals in the 15 min before wheels-up); the four level columns are
demeaned by (airport, calendar month) over the frame's departures. Vectorised; same definitions otherwise.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

C = pl.col
ID = "MVT_ID_mvt"
W_OT = 1800
SESSION_BREAK = 3600
BUSY_H = 150
NB_K = 4
LTFM_GROUP = {"16L": "W", "34R": "W", "16R": "W", "34L": "W", "17L": "C", "35R": "C", "17R": "C", "35L": "C",
              "18": "E", "36": "E"}
SEQ = ["sq_slip", "sq_ot_ahead_n", "sq_ot_ahead_s", "sq_ot_behind_n", "sq_ot_behind_s", "sq_ot_net", "sq_win_n",
       "sq_iso_dev", "sq_iso_gap", "sq_nb_prev_slip", "sq_nb_next_slip", "sq_own_minus_nb", "sq_busy_elapsed",
       "sq_busy_pos", "sq_busy_rest", "sq_arr_W15", "sq_arr_C15", "sq_arr_E15", "sq_arr_same15", "sq_arr_all15",
       "sq_dep_other15"]
LEVEL = ["sq_slip", "sq_iso_gap", "sq_nb_prev_slip", "sq_nb_next_slip"]


def load_movements(paths) -> tuple[pl.DataFrame, pl.DataFrame]:
    """deps: id, apt, mvt(s), base(s: LOBT else SCHED), stand, rwy, month_key, hour, y (null when unlabelled).
    arrs: apt, rwy, t(s)."""
    lf = pl.concat([pl.scan_parquet(p) for p in paths], how="vertical_relaxed")
    dep = (lf.filter(C("PHASE_mvt") == "DEP")
           .select(C(ID), C("ADEP_mvt").alias("apt"),
                   C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt"),
                   pl.coalesce(C("LOBT_flt"), C("SCHED_TIME_UTC_mvt")).dt.epoch("s").cast(pl.Float64).alias("base"),
                   C("STAND_mvt").fill_null("?").alias("stand"), C("RUNWAY_mvt").fill_null("?").alias("rwy"),
                   C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
                   C("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int32).alias("month"),
                   C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Int32).alias("hour"),
                   (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("y"))
           .collect())
    arr = (lf.filter(C("PHASE_mvt") == "ARR")
           .select(C("ADES_mvt").alias("apt"), C("RUNWAY_mvt").fill_null("?").alias("rwy"),
                   C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t"))
           .collect())
    return dep, arr


def fit_T(lab: pl.DataFrame):
    """lab: apt, stand, rwy, y. Median per (apt, stand, rwy) with n >= 20, back-off (apt, rwy), then apt."""
    lab = lab.filter(C("y").is_not_null())
    a = lab.group_by("apt", "stand", "rwy").agg(m=C("y").median(), n=pl.len()).filter(C("n") >= 20).drop("n")
    b = lab.group_by("apt", "rwy").agg(mr=C("y").median())
    c = lab.group_by("apt").agg(ma=C("y").median())
    return a, b, c


def apply_T(dep: pl.DataFrame, T) -> np.ndarray:
    a, b, c = T
    j = (dep.select("apt", "stand", "rwy").with_row_index("_i")
         .join(a, on=["apt", "stand", "rwy"], how="left").join(b, on=["apt", "rwy"], how="left")
         .join(c, on="apt", how="left").sort("_i"))
    return j.select(pl.coalesce("m", "mr", "ma").alias("tt"))["tt"].fill_null(900.0).to_numpy().astype(float)


def _shift(x, d, fill=np.nan):
    out = np.full_like(x, fill, dtype=float)
    if d > 0:
        out[d:] = x[:-d]
    elif d < 0:
        out[:d] = x[-d:]
    else:
        out[:] = x
    return out


def build(dep: pl.DataFrame, arr: pl.DataFrame, tt: np.ndarray, demean: bool = True) -> pl.DataFrame:
    """dep rows (any order) with columns from load_movements; tt aligned to dep. Returns id + SEQ."""
    n = dep.height
    apt = dep["apt"].to_numpy().astype(str)
    rwy = dep["rwy"].to_numpy().astype(str)
    mvt = dep["mvt"].to_numpy().astype(float)
    base = dep["base"].to_numpy().astype(float)
    pm = base + tt
    slip = mvt - pm
    grp_lab = np.char.add(np.char.add(apt, "|"), rwy)
    _, grp = np.unique(grp_lab, return_inverse=True)
    OFF = 1e10
    key = grp * OFF + mvt
    o = np.argsort(key, kind="stable")
    k, t, p, s, g = key[o], mvt[o], pm[o], slip[o], grp[o]
    out = {c: np.full(n, np.nan) for c in SEQ}
    out["sq_slip"] = slip.copy()
    # overtakes within +-W_OT on the same (apt, runway)
    lo = np.searchsorted(k, k - W_OT, "left")
    hi = np.searchsorted(k, k + W_OT, "right")
    ar = np.arange(n)
    a_n = np.zeros(n); a_s = np.zeros(n); b_n = np.zeros(n); b_s = np.zeros(n)
    maxoff = int(max((ar - lo).max(initial=0), (hi - 1 - ar).max(initial=0)))
    for d in range(1, maxoff + 1):
        j = ar - d
        ok = j >= lo
        jj = np.clip(j, 0, n - 1)
        diff = p[jj] - p
        hit = ok & (diff > 0)
        a_n += hit; a_s += np.where(hit, diff, 0.0)
        j = ar + d
        ok = j < hi
        jj = np.clip(j, 0, n - 1)
        diff = p - p[jj]
        hit = ok & (diff > 0)
        b_n += hit; b_s += np.where(hit, diff, 0.0)
    res = {"sq_ot_ahead_n": a_n, "sq_ot_ahead_s": a_s / 60, "sq_ot_behind_n": b_n, "sq_ot_behind_s": b_s / 60,
           "sq_ot_net": a_n - b_n, "sq_win_n": (hi - lo - 1).astype(float)}
    # sessions: new group or a runway gap > SESSION_BREAK
    newg = np.r_[True, g[1:] != g[:-1]]
    head = np.r_[np.inf, np.diff(t)]
    sess = np.cumsum(newg | (head > SESSION_BREAK))
    starts = np.flatnonzero(np.r_[True, sess[1:] != sess[:-1]])
    ends = np.r_[starts[1:], n]
    iso = p.copy()
    for a0, a1 in zip(starts, ends):
        if a1 - a0 >= 2:
            iso[a0:a1] = IsotonicRegression(increasing=True).fit_transform(np.arange(a1 - a0), p[a0:a1])
    prevs = np.vstack([np.where(_shift(sess, d, -1) == sess, _shift(s, d), np.nan) for d in range(1, NB_K + 1)])
    nexts = np.vstack([np.where(_shift(sess, -d, -1) == sess, _shift(s, -d), np.nan) for d in range(1, NB_K + 1)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nb_prev = np.nanmedian(prevs, axis=0)
            nb_next = np.nanmedian(nexts, axis=0)
            both = np.nanmedian(np.vstack([nb_prev, nb_next]), axis=0)
    res.update({"sq_iso_dev": iso - p, "sq_iso_gap": t - iso, "sq_nb_prev_slip": nb_prev, "sq_nb_next_slip": nb_next,
                "sq_own_minus_nb": s - both})
    # busy periods: consecutive take-offs on the same (apt, runway) closer than BUSY_H
    link = (~newg) & (head < BUSY_H)
    seg = np.cumsum(~link)
    sst = np.flatnonzero(~link)
    seg_start = sst[seg - 1]
    seg_end = np.r_[sst[1:], n][seg - 1] - 1
    res.update({"sq_busy_elapsed": t - t[seg_start], "sq_busy_pos": (ar - seg_start).astype(float),
                "sq_busy_rest": (seg_end - ar).astype(float)})
    for c, v in res.items():
        out[c][o] = v
    # arrival / departure counts in the 15 min before wheels-up
    aapt = arr["apt"].to_numpy().astype(str)
    arwy = arr["rwy"].to_numpy().astype(str)
    at = arr["t"].to_numpy().astype(float)

    def count(src_lab, src_t, q_lab, q_t):
        labs, inv = np.unique(np.r_[src_lab, q_lab], return_inverse=True)
        si, qi = inv[:len(src_lab)], inv[len(src_lab):]
        sk = np.sort(si * OFF + src_t)
        qk = qi * OFF + q_t
        return (np.searchsorted(sk, qk, "right") - np.searchsorted(sk, qk - 900, "left")).astype(float)

    out["sq_arr_same15"] = count(np.char.add(np.char.add(aapt, "|"), arwy), at, grp_lab, mvt)
    out["sq_arr_all15"] = count(aapt, at, apt, mvt)
    tot = count(apt, mvt, apt, mvt)
    own = count(grp_lab, mvt, grp_lab, mvt)
    out["sq_dep_other15"] = tot - own
    lt = apt == "LTFM"
    ag = np.array([LTFM_GROUP.get(x, "?") for x in arwy])
    for gname in ("W", "C", "E"):
        m = (aapt == "LTFM") & (ag == gname)
        v = count(np.full(int(m.sum()), "x"), at[m], np.full(n, "x"), mvt)
        out[f"sq_arr_{gname}15"] = np.where(lt, v, np.nan)
    if demean:
        # level columns drift 2025->2026 (agent_ltfm_07); subtract each (airport, calendar month) mean over this frame
        ym = dep["ym"].to_numpy().astype(str)
        cell = np.char.add(np.char.add(apt, "|"), ym)
        _, ci = np.unique(cell, return_inverse=True)
        for c in LEVEL:
            v = out[c]
            f = np.isfinite(v)
            sm = np.bincount(ci[f], weights=v[f], minlength=ci.max() + 1)
            cn = np.bincount(ci[f], minlength=ci.max() + 1)
            mu = np.where(cn > 0, sm / np.maximum(cn, 1), 0.0)
            out[c] = v - mu[ci]
    return pl.DataFrame({ID: dep[ID], **{c: out[c].astype(np.float32) for c in SEQ}})  # float32: identical to the LightGBM input
