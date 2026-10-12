"""RE-STACK (docs/LAB-NOTES.md): one LightGBM after the whole ADS-B stage (main stack, IV blend, hard floor,
floor stack), on every ADS-B-matched row, target y - fin, features = the production estimators RELATIVE TO fin (so it sees
the main-stack estimate and the floor together, after the floor stack has moved the row) + apt, fin, taxinm flag, hour.
Optional: the push-back reader's columns (docs/LAB-NOTES.md) as well.

Row features come from the offblock dump format (archive/agent_offblock_01_dump.py: 2025 mode for training,
2026 mode for serving; data/interim/agent_offblock_rows_{2025,2026}.parquet). ONE implementation for fold and serving.
"""
import lightgbm as lgb
import numpy as np
import polars as pl

C = pl.col
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
EST = ["prod_t", "lb_t", "e_appear", "e_start", "e_ci20"]
V2E = ["e_dend", "e_dnext", "e_mv5", "e_mv10", "e_mv20", "e_rev", "e_hdg", "e_rev0", "e_revany", "e_revany_prev",
       "e_pushend", "e_rev0x"]
V2Q = ["P_span", "P_n", "P_gap", "P_dstand", "P_long", "seg_n", "seg_rate", "push", "first_gs", "first_dstand",
       "seg_start_gap", "revany_gap", "rev_dist", "rev_gs", "rev_dstand"]
ROW_COLS = ["MVT_ID_mvt", "day", "apt", "src", "hour", "mvt_t"] + EST + V2E + V2Q
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, n_jobs=4)
CLIP = 1800.0


def load_rows(path):
    r = pl.read_parquet(path, columns=ROW_COLS).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
    assert r["MVT_ID_mvt"].n_unique() == r.height, f"{path}: ids not unique"
    return r


def features(r, fin, feats="est"):
    """r: rows frame (ROW_COLS); fin: np array aligned with r (the ADS-B stage output, seconds)."""
    mvt = r["mvt_t"].cast(pl.Float64).to_numpy()
    apt = np.array([APTS.index(a) if a in APTS else -1 for a in r["apt"].to_list()], float)
    cols = [apt, fin, (r["src"] == "taxinm").cast(pl.Float64).to_numpy(), r["hour"].cast(pl.Float64).to_numpy()]
    cols += [mvt - r[e].cast(pl.Float64).fill_null(np.nan).to_numpy() - fin for e in EST]
    if feats == "est_v2":
        cols += [mvt - r[e].cast(pl.Float64).fill_null(np.nan).to_numpy() - fin for e in V2E]
        cols += [r[q].cast(pl.Float64).fill_null(np.nan).to_numpy() for q in V2Q]
    return np.column_stack(cols)


def fit(X, y, seed=0):
    return lgb.LGBMRegressor(random_state=seed, **PARAMS).fit(X, y, categorical_feature=[0])


def predict(m, X):
    return np.clip(m.predict(X), -CLIP, CLIP)
