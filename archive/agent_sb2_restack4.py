"""experiment sb2: COPY of archive/agent_restack6_restack4.py (docs/LAB-NOTES.md) adding the tokens of
archive/agent_sb2_common.py (mw, lw, unc, osm, occ): load_rows(..., extra=) joins the osm/occ parquets, features()
appends unc/osm/occ columns AFTER every original column, fit(..., w=) passes sample_weight. With none of the new
tokens (and w=None) this is the original bit for bit. Original docstring follows.

experiment restack6: COPY of archive/agent_restack4_restack.py (docs/LAB-NOTES.md) adding THREE new capacity
tokens alongside the existing `big` (1500 trees), the SAME additions as archive/agent_restack6_restack.py (the bx-capable
module) but on the non-bx module used for the LFPG/LIRF (seq, no bx) recipe:
  r1    2500 trees, lr 0.03 (unchanged), 31 leaves (unchanged), min_child 200 (unchanged) -- capacity only
  r2    3000 trees, lr 0.015 (halved), 31 leaves, min_child 200 -- 2x trees AND half lr
  r3    1500 trees (unchanged), lr 0.03 (unchanged), 62 leaves (doubled), min_child 200 -- leaves only
Without r1/r2/r3 this is agent_restack4_restack.py bit for bit. Original docstring follows.

experiment restack4: COPY of archive/agent_neighbour_restack.py (docs/LAB-NOTES.md) adding the tokens
  seq   + the 21 SEQ columns of archive/agent_seqpost_feat.py (exactly as archive/agent_smallbundle_restack.py)
  seqm  + the same 21 columns, set to NaN on rows whose airport is not in SEQ_APTS (LEMD, LFPG, LIRF)
  arr   + the ARR columns of archive/agent_restack4_arr.py (arrival ADS-B), joined by MVT_ID from an --arr parquet
  big   1500 trees on top of the cap parameters (lr 0.03, 31 leaves, min_child 200)
  cb    fit()/predict() use CatBoost (CB_PARAMS) instead of LightGBM on the same matrix (used by the fold script and
        the serve blend; the LightGBM path is untouched when cb is absent)
Column order: base est_v2, evt, trk, seq|seqm, nbr, arr. Without the new tokens this is agent_neighbour_restack bit for
bit, and recipe cap,evt,trk,seq is agent_smallbundle_restack's. Original docstring follows.

experiment restack3: the RE-STACK with the pre-registered increments of docs/LAB-NOTES.md. ONE implementation for
the fold test (archive/agent_restack3_03_fold.py) and serving (archive/agent_restack3_04_serve.py).
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from archive.agent_sb2_common import SB2_TOKENS, extra_cols, join_extra  # noqa: E402

from archive.agent_expensive_restack import APTS, CLIP, PARAMS, ROW_COLS, features as base_features  # noqa: F401
from archive.agent_neighbour_feat import NBR
from archive.agent_seqpost_feat import SEQ
from archive.agent_restack4_arr import ARR

C = pl.col
TOKENS = ("evt", "trk", "cap", "apt", "clip", "bag", "nbr", "seq", "seqm", "arr", "big", "cb", "r1", "r2", "r3") + SB2_TOKENS
SEQ_APTS = ("LEMD", "LFPG", "LIRF")
EVT_RAW = ["n_push", "t_first", "t_last", "t_gfirst", "t_air_prev", "first_m", "last_m", "d_first_last_m", "max_stop",
           "stop_d_first_m", "max_sil", "taxi_between", "d_to"]
ANCH = ["ci_20", "ci_30", "ci_50", "ci_75", "ci_100", "ci_150", "ci_300", "appear_raw", "gp_20", "gp_150", "t_in150",
        "dens", "start_d"]
ANCH_REL = ["ci_20", "ci_30", "ci_50", "ci_75", "ci_100", "ci_150", "ci_300", "appear_raw"]
TRK = ["tf_path_m", "tf_nstops", "tf_stop_s", "tf_maxstop_s", "tf_movegap_s", "tf_holdrun_s", "tf_fmhold_s", "tf_avgspd_kt"]
CAP_PARAMS = dict(PARAMS, n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=200)
BIG_PARAMS = dict(CAP_PARAMS, n_estimators=1500)
R1_PARAMS = dict(CAP_PARAMS, n_estimators=2500)
R2_PARAMS = dict(CAP_PARAMS, n_estimators=3000, learning_rate=0.015)
R3_PARAMS = dict(CAP_PARAMS, n_estimators=1500, num_leaves=62)
CB_PARAMS = dict(loss_function="RMSE", iterations=800, learning_rate=0.1, depth=6, l2_leaf_reg=3, border_count=254,
                 thread_count=4, verbose=0, allow_writing_files=False)
TCLIP = 900.0
NBAG = 5


def parse(recipe):
    toks = {t for t in (recipe or "").split(",") if t and t != "none"}
    bad = toks - set(TOKENS)
    assert not bad, f"unknown recipe tokens {bad}"
    assert not ({"seq", "seqm"} <= toks), "seq and seqm are exclusive"
    assert len(toks & {"big", "r1", "r2", "r3"}) <= 1, "big/r1/r2/r3 are mutually exclusive capacity settings"
    return toks


def load_rows(path, recipe, evt=None, trk=None, nbr=None, seq=None, arr=None, extra=None):
    toks = parse(recipe)
    r = pl.read_parquet(path, columns=ROW_COLS).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
    assert r["MVT_ID_mvt"].n_unique() == r.height, f"{path}: ids not unique"
    n = r.height
    if "evt" in toks:
        e = (pl.read_parquet(evt, columns=["MVT_ID_mvt"] + EVT_RAW).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
             .unique(subset="MVT_ID_mvt", keep="first"))
        r = r.join(e, on="MVT_ID_mvt", how="left")
    if "trk" in toks:
        t = (pl.read_parquet(trk, columns=["MVT_ID_mvt"] + ANCH + TRK).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
             .unique(subset="MVT_ID_mvt", keep="first"))
        r = r.join(t, on="MVT_ID_mvt", how="left")
    if "nbr" in toks:
        b = pl.read_parquet(nbr, columns=["MVT_ID_mvt"] + NBR).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
        assert b["MVT_ID_mvt"].n_unique() == b.height
        r = r.join(b, on="MVT_ID_mvt", how="left")
    if toks & {"seq", "seqm"}:
        q = (pl.read_parquet(seq, columns=["MVT_ID_mvt"] + SEQ).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
             .unique(subset="MVT_ID_mvt", keep="first"))
        r = r.join(q, on="MVT_ID_mvt", how="left")
    if "arr" in toks:
        w = pl.read_parquet(arr, columns=["MVT_ID_mvt"] + ARR).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
        assert w["MVT_ID_mvt"].n_unique() == w.height
        r = r.join(w, on="MVT_ID_mvt", how="left")
    r = join_extra(r, toks, extra)
    assert r.height == n
    return r


def _f(r, c):
    return r[c].cast(pl.Float64).fill_null(np.nan).to_numpy()


def features(r, fin, recipe):
    toks = parse(recipe)
    X = base_features(r, fin, "est_v2")
    cols = []
    if "evt" in toks:
        mvt = _f(r, "mvt_t")
        cols += [_f(r, "n_push")]
        cols += [mvt - _f(r, c) - fin for c in ("t_first", "t_last", "t_gfirst")]
        cols += [mvt - _f(r, "t_air_prev")]
        cols += [_f(r, c) for c in ("first_m", "last_m", "d_first_last_m", "max_stop", "stop_d_first_m", "max_sil",
                                    "taxi_between", "d_to")]
        npush, ms, sd = _f(r, "n_push"), _f(r, "max_stop"), _f(r, "stop_d_first_m")
        hold = np.where(np.isnan(npush), np.nan,
                        ((npush >= 1) & (np.nan_to_num(ms) >= 600) & (np.nan_to_num(sd, nan=1e9) <= 150)).astype(float))
        cols += [hold]
    if "trk" in toks:
        cols += [_f(r, c) - fin if c in ANCH_REL else _f(r, c) for c in ANCH]
        cols += [_f(r, c) for c in TRK]
    if "seq" in toks:
        cols += [_f(r, c) for c in SEQ]
    if "seqm" in toks:
        keep = np.isin(np.asarray(r["apt"].to_list()), SEQ_APTS)
        cols += [np.where(keep, _f(r, c), np.nan) for c in SEQ]
    if "nbr" in toks:
        cols += [_f(r, c) for c in NBR]
    if "arr" in toks:
        cols += [_f(r, c) for c in ARR]
    cols += extra_cols(r, toks)
    return np.column_stack([X] + cols) if cols else X


def _one(X, y, seed, params, w=None):
    return lgb.LGBMRegressor(random_state=seed, **params).fit(X, y, sample_weight=w, categorical_feature=[0])


def _params(toks):
    if "r1" in toks:
        return R1_PARAMS
    if "r2" in toks:
        return R2_PARAMS
    if "r3" in toks:
        return R3_PARAMS
    if "big" in toks:
        return BIG_PARAMS
    if "cap" in toks:
        return CAP_PARAMS
    return PARAMS


def fit(X, y, recipe, seed=0, w=None):
    """returns a model object for predict(); X[:, 0] is the airport index (categorical for LightGBM, numeric for cb)."""
    toks = parse(recipe)
    if "cb" in toks:
        assert w is None, "cb takes no weights here"
        from catboost import CatBoostRegressor
        return {"cb": CatBoostRegressor(random_seed=seed, **CB_PARAMS).fit(X, y)}
    params = _params(toks)
    if "clip" in toks:
        y = np.clip(y, -TCLIP, TCLIP)
    seeds = [seed + i for i in range(NBAG)] if "bag" in toks else [seed]
    if "apt" in toks:
        return {"apt": {a: [_one(X[X[:, 0] == a], y[X[:, 0] == a], s, params,
                                   None if w is None else w[X[:, 0] == a]) for s in seeds]
                        for a in np.unique(X[:, 0])}}
    return {"pooled": [_one(X, y, s, params, w) for s in seeds]}


def predict(m, X):
    if "cb" in m:
        p = m["cb"].predict(X)
    elif "pooled" in m:
        p = np.mean([mm.predict(X) for mm in m["pooled"]], axis=0)
    else:
        p = np.zeros(len(X))
        for a, ms in m["apt"].items():
            k = X[:, 0] == a
            if k.any():
                p[k] = np.mean([mm.predict(X[k]) for mm in ms], axis=0)
    return np.clip(p, -CLIP, CLIP)
