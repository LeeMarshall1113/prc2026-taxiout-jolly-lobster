"""experiment neighbour: COPY of archive/agent_restack3_restack.py adding the token `nbr` (docs/LAB-NOTES.md arm B):
the 14 NBR columns of archive/agent_neighbour_feat.py, joined by MVT_ID from an --nbr parquet. Every other token is
unchanged, so recipe cap,evt,trk must reproduce agent_restack3_CAB.npz bit for bit (N0). Original docstring follows.

experiment restack3: the RE-STACK with the pre-registered increments of docs/LAB-NOTES.md. ONE implementation for
the fold test (archive/agent_restack3_03_fold.py) and serving (archive/agent_restack3_04_serve.py).

Base = archive/agent_expensive_restack.py with feats est_v2 (imported, not re-typed, so the base cannot drift).
A recipe is a set of tokens; the empty recipe is the base, bit for bit:
  evt   + event-reader columns (archive/agent_monster2_01_events.py), joined by MVT_ID from an --evt parquet
  trk   + production anchor columns (ANCHOR_COLS) and track-shape columns (TRACKFEAT_COLS), from a --trk parquet
  cap   900 trees, lr 0.03, 31 leaves, min_child 200
  apt   one model per airport instead of one pooled model
  clip  training target clipped to [-900, 900] s
  bag   mean over seeds seed..seed+4
"""
import lightgbm as lgb
import numpy as np
import polars as pl

from archive.agent_expensive_restack import APTS, CLIP, PARAMS, ROW_COLS, features as base_features

C = pl.col
TOKENS = ("evt", "trk", "cap", "apt", "clip", "bag", "nbr")
from archive.agent_neighbour_feat import NBR
EVT_RAW = ["n_push", "t_first", "t_last", "t_gfirst", "t_air_prev", "first_m", "last_m", "d_first_last_m", "max_stop",
           "stop_d_first_m", "max_sil", "taxi_between", "d_to"]
ANCH = ["ci_20", "ci_30", "ci_50", "ci_75", "ci_100", "ci_150", "ci_300", "appear_raw", "gp_20", "gp_150", "t_in150",
        "dens", "start_d"]
ANCH_REL = ["ci_20", "ci_30", "ci_50", "ci_75", "ci_100", "ci_150", "ci_300", "appear_raw"]
TRK = ["tf_path_m", "tf_nstops", "tf_stop_s", "tf_maxstop_s", "tf_movegap_s", "tf_holdrun_s", "tf_fmhold_s", "tf_avgspd_kt"]
CAP_PARAMS = dict(PARAMS, n_estimators=900, learning_rate=0.03, num_leaves=31, min_child_samples=200)
TCLIP = 900.0
NBAG = 5


def parse(recipe):
    toks = {t for t in (recipe or "").split(",") if t and t != "none"}
    bad = toks - set(TOKENS)
    assert not bad, f"unknown recipe tokens {bad}"
    return toks


def load_rows(path, recipe, evt=None, trk=None, nbr=None):
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
    if "nbr" in toks:
        cols += [_f(r, c) for c in NBR]
    return np.column_stack([X] + cols) if cols else X


def _one(X, y, seed, params):
    return lgb.LGBMRegressor(random_state=seed, **params).fit(X, y, categorical_feature=[0])


def fit(X, y, recipe, seed=0):
    """returns a model object for predict(); X[:, 0] is the airport index (categorical)."""
    toks = parse(recipe)
    params = CAP_PARAMS if "cap" in toks else PARAMS
    if "clip" in toks:
        y = np.clip(y, -TCLIP, TCLIP)
    seeds = [seed + i for i in range(NBAG)] if "bag" in toks else [seed]
    if "apt" in toks:
        return {"apt": {a: [_one(X[X[:, 0] == a], y[X[:, 0] == a], s, params) for s in seeds]
                        for a in np.unique(X[:, 0])}}
    return {"pooled": [_one(X, y, s, params) for s in seeds]}


def predict(m, X):
    if "pooled" in m:
        p = np.mean([mm.predict(X) for mm in m["pooled"]], axis=0)
    else:
        p = np.zeros(len(X))
        for a, ms in m["apt"].items():
            k = X[:, 0] == a
            if k.any():
                p[k] = np.mean([mm.predict(X[k]) for mm in ms], axis=0)
    return np.clip(p, -CLIP, CLIP)
