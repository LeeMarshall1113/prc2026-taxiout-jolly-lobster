"""Candidate blend members for prc.final --fit-members (2026-09-22).

Each kind here is the serve-time twin of one arm of lean_member_screen.py,
so a member ships exactly as it was screened:

    lgb             arm A  shipped LightGBM (crossval._fit_lightgbm, regression, seed 0)
    lgb_routed      arm B  per-airport LightGBM, 5,000-row floor, reset_index, seed 0
    lgb_huber600    arm C  LightGBM, huber objective, alpha 600
    lgb_leaves63    arm D  LightGBM, num_leaves 63, min_child_samples 200
    lgb_extratrees  arm E  LightGBM, extra_trees=True
    xgb_hist        arm F  XGBoost hist, max_depth 8, native categoricals
    lgb_et_s<k>     extra-trees LightGBM with seed k (round 2: seed bags; extra_trees makes the seed matter)
    lgb_etr_s<k>    per-airport extra-trees LightGBM, seed k (5,000-row floor; global ET seed k below it)

Every fitter returns predictions on the RESIDUAL scale; the caller adds base_rank.
"""
from __future__ import annotations

import copy

import numpy as np

ROUTE_FLOOR = 5000
KINDS = ("lgb", "lgb_routed", "lgb_huber600", "lgb_leaves63", "lgb_extratrees", "xgb_hist")
SEEDED = ("lgb_et_s", "lgb_etr_s")


def valid_kind(kind: str) -> bool:
    return kind in KINDS or any(kind.startswith(p) and kind[len(p):].isdigit() for p in SEEDED)


def _cast_categoricals(train_x, test_x, cat_names):
    """Same category discipline as crossval._fit_lightgbm: levels fixed from train."""
    import pandas as pd

    train_x = train_x.copy()
    test_x = test_x.copy()
    for c in cat_names:
        if c not in train_x.columns:
            continue
        levels = pd.Index(train_x[c].astype("string").unique()).dropna()
        dt = pd.CategoricalDtype(categories=levels)
        train_x[c] = train_x[c].astype("string").astype(dt)
        test_x[c] = test_x[c].astype("string").astype(dt)
    return train_x, test_x


def _fit_lightgbm_extratrees(train_x, fit_y, test_x, cat_names, args, seed):
    import lightgbm as lgb

    train_x, test_x = _cast_categoricals(train_x, test_x, cat_names)
    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        n_estimators=args.iterations, learning_rate=args.lr,
        num_leaves=getattr(args, "num_leaves", 255),
        min_child_samples=getattr(args, "min_child_samples", 40),
        reg_lambda=args.l2, random_state=1113 + seed * 977,
        n_jobs=args.threads, verbose=-1, extra_trees=True,
    )
    model.fit(train_x, fit_y, categorical_feature=[c for c in cat_names if c in train_x.columns])
    return model.predict(test_x)


def _fit_xgb_hist(train_x, fit_y, test_x, cat_names, args, seed):
    import xgboost as xgb

    train_x, test_x = _cast_categoricals(train_x, test_x, cat_names)
    model = xgb.XGBRegressor(
        n_estimators=args.iterations, max_depth=8, learning_rate=args.lr,
        reg_lambda=args.l2, tree_method="hist", enable_categorical=True,
        objective="reg:squarederror", random_state=1113 + seed * 977,
        n_jobs=args.threads, verbosity=0,
    )
    model.fit(train_x, fit_y)
    return model.predict(test_x)


def fit_member(kind, train_x, fit_y, rank_x, cat_names, args, tr_apt, rk_apt, seed=0, log=print):
    """Fit one candidate member on (train_x, fit_y); return its rank_x predictions (residual scale)."""
    from .crossval import _fit_lightgbm

    def lgb_fit(tx, ty, rx, a, **kw):
        m, _, rxx = _fit_lightgbm(tx, ty, rx, cat_names, a, seed, **kw)
        return m.predict(rxx)

    if kind == "lgb":
        return lgb_fit(train_x, fit_y, rank_x, args, objective="regression")
    if kind == "lgb_routed":
        out, sizes = np.full(len(rk_apt), np.nan), []
        for code in sorted(set(tr_apt) & set(rk_apt)):
            tr_m, rk_m = tr_apt == code, rk_apt == code
            if tr_m.sum() < ROUTE_FLOOR or rk_m.sum() == 0:
                sizes.append(f"{code} skipped")
                continue
            out[rk_m] = lgb_fit(train_x[tr_m].reset_index(drop=True), fit_y[tr_m],
                                rank_x[rk_m].reset_index(drop=True), args, objective="regression")
            sizes.append(f"{code} {int(tr_m.sum()):,}")
        log("    lgb_routed: " + "  ".join(sizes))
        if np.isnan(out).any():
            glob = lgb_fit(train_x, fit_y, rank_x, args, objective="regression")
            out = np.where(np.isnan(out), glob, out)
        return out
    if kind == "lgb_huber600":
        return lgb_fit(train_x, fit_y, rank_x, args, objective="huber", huber_alpha=600.0)
    if kind == "lgb_leaves63":
        a = copy.copy(args)
        a.num_leaves, a.min_child_samples = 63, 200
        return lgb_fit(train_x, fit_y, rank_x, a, objective="regression")
    if kind == "lgb_extratrees":
        return _fit_lightgbm_extratrees(train_x, fit_y, rank_x, cat_names, args, seed)
    if kind == "xgb_hist":
        return _fit_xgb_hist(train_x, fit_y, rank_x, cat_names, args, seed)
    if kind.startswith("lgb_et_s"):
        return _fit_lightgbm_extratrees(train_x, fit_y, rank_x, cat_names, args, int(kind[8:]))
    if kind.startswith("lgb_etr_s"):
        k = int(kind[9:])
        out, sizes = np.full(len(rk_apt), np.nan), []
        for code in sorted(set(tr_apt) & set(rk_apt)):
            tr_m, rk_m = tr_apt == code, rk_apt == code
            if tr_m.sum() < ROUTE_FLOOR or rk_m.sum() == 0:
                sizes.append(f"{code} skipped")
                continue
            out[rk_m] = _fit_lightgbm_extratrees(train_x[tr_m].reset_index(drop=True), fit_y[tr_m],
                                                 rank_x[rk_m].reset_index(drop=True), cat_names, args, k)
            sizes.append(f"{code} {int(tr_m.sum()):,}")
        log(f"    {kind}: " + "  ".join(sizes))
        if np.isnan(out).any():
            glob = _fit_lightgbm_extratrees(train_x, fit_y, rank_x, cat_names, args, k)
            out = np.where(np.isnan(out), glob, out)
        return out
    raise SystemExit(f"unknown member kind {kind!r}; known: {', '.join(KINDS)}")
