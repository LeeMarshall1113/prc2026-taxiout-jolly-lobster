"""experiment neighbour: ONE implementation of the neighbour-congestion features (2025 fold build and 2026 serving share it).
docs/LAB-NOTES.md (pre-registered set NBR, 14 columns).

Input frame columns (one row per departure of the airport-days concerned, dump rows AND non-dump rows):
  apt, day, t (MVT epoch s), rwy, and SOURCE values (null where the row is not a source):
  m    = ADS-B measured taxi (mvt - prod_t) minus the bulk prediction p0, winsorised +-900 s; only rows whose production
         estimate is a real off-block measurement (prod_kind stand/appear, class M)
  lvl  = the same measured taxi, clipped to [0, 3600]; class M rows only
  s    = the ADS-B stage's move fin - p0, winsorised +-900 s; every offblock-dump row (classes M and W)
Every window excludes the row itself. No label of any row enters. All windows are within the same airport and UTC day
string `day`, centred on t, except m_day (whole airport-day).
"""
import numpy as np
import polars as pl

C = pl.col
NBR = ["m_c15", "n_c15", "m_c30", "n_c30", "m_c60", "n_c60", "m_r30", "n_r30", "lvl_c30", "m_day",
       "s_c30", "ns_c30", "s_r30", "s_day"]


def _window(t, apt, day, vals, ok, W, key=None, past=False):
    out = np.full(len(t), np.nan)
    cnt = np.zeros(len(t))
    g = pl.DataFrame({"a": pl.Series([str(x) for x in apt], dtype=pl.String),
                      "d": pl.Series([str(x) for x in day], dtype=pl.String),
                      "k": pl.Series([str(x) for x in key] if key is not None else [""] * len(t), dtype=pl.String)})
    g = g.with_row_index()
    for idx in g.group_by("a", "d", "k", maintain_order=True).agg(C("index"))["index"].to_list():
        idx = np.asarray(idx)
        o = np.argsort(t[idx], kind="stable")
        idx = idx[o]
        tt = t[idx]
        ok_ = ok[idx]
        v = np.where(ok_, vals[idx], 0.0)
        cs = np.concatenate([[0.0], np.cumsum(v)])
        cn = np.concatenate([[0], np.cumsum(ok_)])
        lo = np.searchsorted(tt, tt - W, "left")
        hi = np.searchsorted(tt, tt, "left") if past else np.searchsorted(tt, tt + W, "right")
        sm = cs[hi] - cs[lo]
        n = (cn[hi] - cn[lo]).astype(float)
        if not past:
            sm = sm - v
            n = n - ok_
        out[idx] = np.where(n > 0, sm / np.maximum(n, 1), np.nan)
        cnt[idx] = n
    return out, cnt


def features(fr):
    """fr: polars frame with apt, day, t, rwy, m, lvl, s. Returns a polars frame of the NBR columns aligned with fr."""
    t = fr["t"].cast(pl.Float64).to_numpy()
    apt = fr["apt"].to_list()
    day = fr["day"].to_list()
    rwy = fr["rwy"].fill_null("?").to_list()
    num = lambda c: fr[c].cast(pl.Float64).fill_null(np.nan).fill_nan(np.nan).to_numpy()  # noqa: E731
    m, lvl, s = num("m"), num("lvl"), num("s")
    okm, oks = np.isfinite(m), np.isfinite(s)
    f = {}
    for W in (900, 1800, 3600):
        f[f"m_c{W // 60}"], f[f"n_c{W // 60}"] = _window(t, apt, day, m, okm, W)
    f["m_r30"], f["n_r30"] = _window(t, apt, day, m, okm, 1800, key=rwy)
    f["lvl_c30"], _ = _window(t, apt, day, lvl, okm & np.isfinite(lvl), 1800)
    f["m_day"], _ = _window(t, apt, day, m, okm, 2 * 86400)
    f["s_c30"], f["ns_c30"] = _window(t, apt, day, s, oks, 1800)
    f["s_r30"], _ = _window(t, apt, day, s, oks, 1800, key=rwy)
    f["s_day"], _ = _window(t, apt, day, s, oks, 2 * 86400)
    return pl.DataFrame({k: f[k] for k in NBR})


def sources(tx_minus_p0, tx, fin_minus_p0, is_m, in_dump):
    """helper: build (m, lvl, s) numpy arrays from per-row quantities (NaN where undefined)."""
    m = np.where(is_m, np.clip(tx_minus_p0, -900, 900), np.nan)
    lvl = np.where(is_m, np.clip(tx, 0, 3600), np.nan)
    s = np.where(in_dump, np.clip(fin_minus_p0, -900, 900), np.nan)
    return m, lvl, s
