"""experiment sb2 common (docs/LAB-NOTES.md): the NEW recipe tokens of the small bundle, shared by
archive/agent_sb2_restack.py (bx module) and archive/agent_sb2_restack4.py (seq module):
  mw   month-mix sample weight (Jan+Jul carry 50% of the effective weight, split 44.3/55.7; other months share 50%)
  lw   sample weight 2.5 on LIRF rows
  unc  one column unc_w (quantile-head interval width), set on the rows frame by the caller (nested per outer fold)
  osm  osm_dc, osm_dff (agent_sb2_feat_osm.py), joined by MVT_ID from extra["osm"]
  occ  occ_push (archive/agent_sb2_feat.py occ), joined by MVT_ID from extra["occ"]
  nov  one column nov_w (row novelty, archive/agent_nov_01_feat.py), set on the rows frame by the caller (fold: nov_f<k>
       per outer fold k; serving: nov_full for training rows, agent_nov_2026 for 2026 rows) -- amendment s0.8
With none of these tokens every function here is a no-op (weights -> None, no columns, no joins).
"""
import numpy as np
import polars as pl

C = pl.col
SB2_TOKENS = ("mw", "lw", "unc", "osm", "occ", "nov")
OSM = ["osm_dc", "osm_dff"]
OCC = ["occ_push"]
LIRF_W = 2.5
BOARD_JUL, BOARD_JAN = 0.557, 0.443


def join_extra(r, toks, extra):
    extra = extra or {}
    n = r.height
    for tok, cols in (("osm", OSM), ("occ", OCC)):
        if tok in toks:
            x = pl.read_parquet(extra[tok], columns=["MVT_ID_mvt"] + cols).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
            assert x["MVT_ID_mvt"].n_unique() == x.height, f"{extra[tok]}: ids not unique"
            r = r.join(x, on="MVT_ID_mvt", how="left", maintain_order="left")
    assert r.height == n
    return r


def _f(r, c):
    return r[c].cast(pl.Float64).fill_null(np.nan).to_numpy()


def extra_cols(r, toks):
    cols = []
    if "unc" in toks:
        assert "unc_w" in r.columns, "token unc: the caller must set column unc_w"
        cols += [_f(r, "unc_w")]
    if "osm" in toks:
        cols += [_f(r, c) for c in OSM]
    if "occ" in toks:
        cols += [_f(r, c) for c in OCC]
    if "nov" in toks:
        assert "nov_w" in r.columns, "token nov: the caller must set column nov_w"
        cols += [_f(r, "nov_w")]
    return cols


def month_weights(days):
    """days: array of 'YYYY-MM-DD'. Label-free; shares from the rows passed (the re-stack's training frame)."""
    mon = np.array([d[5:7] for d in days])
    jan, jul = mon == "01", mon == "07"
    rest = ~(jan | jul)
    s_jan, s_jul, s_rest = jan.mean(), jul.mean(), rest.mean()
    w = np.where(jan, 0.5 * BOARD_JAN / s_jan, np.where(jul, 0.5 * BOARD_JUL / s_jul, 0.5 / s_rest))
    return w / w.mean()


def weights(r, toks):
    """sample weights for the rows frame r (aligned), or None when no weight token is present."""
    if not (toks & {"mw", "lw"}):
        return None
    w = np.ones(r.height)
    if "mw" in toks:
        w = w * month_weights(np.asarray(r["day"].cast(pl.Utf8).to_list()))
    if "lw" in toks:
        w = w * np.where(np.asarray(r["apt"].to_list()) == "LIRF", LIRF_W, 1.0)
    return w
