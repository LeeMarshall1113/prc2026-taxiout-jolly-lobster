"""vfac variant factory: the model hook shared by the fold scripts (agent_vfac_bxnest.py / agent_vfac_fold4.py) and the serve
scripts (agent_vfac_bxserve.py / agent_vfac_serve.py). One call per (variant, fold-or-train) -> the re-stack ADD for the test rows.
A variant = (recipe edit, model kind). `ctrl` = the production LightGBM re-stack (same code path, same process/environment) and is the
paired control: d_v = add_v - add_ctrl is the difference input, built identically on fold and 2026 serve. docs/LAB-NOTES.md"""
import numpy as np

CLIPA = 1800.0
# name: (recipe edit, kind). recipe edit: ("drop", {tokens}) | ("only", {tokens kept from the main recipe}) | None
VARIANTS = {
    "ctrl":    (None, "lgb"),
    "cb":      (None, "cb"),                  # CatBoost depth 8, 800 it, lr .07
    "xgb":     (None, "xgb"),                 # XGBoost hist depth 7, 800 trees
    "dart":    (None, "dart"),                # LightGBM dart, 500 trees lr .06
    "l1":      (None, "l1"),                  # LightGBM objective l1 (conditional median), BIG params
    "logt":    (None, "logt"),                # LightGBM on log1p(y) - log1p(fin)
    "a_noosm": (("drop", {"osm", "occ"}), "lgb"),
    "a_nobx":  (("drop", {"bx"}), "lgb"),
    "a_noseq": (("drop", {"seq"}), "lgb"),
    "a_notrk": (("drop", {"trk", "evt"}), "lgb"),     # no event / track columns (est_v2 base + cap/big only)
    "a_cap":   (("only", {"cap", "big"}), "lgb"),     # base + capacity only
    # ---- wave 2
    "q10": (None, "q10"), "q25": (None, "q25"), "q75": (None, "q75"), "q90": (None, "q90"),   # LightGBM objective=quantile, BIG params
    # ---- wave 3
    "q05": (None, "q05"), "q40": (None, "q40"), "q60": (None, "q60"), "q95": (None, "q95"),
    "xq10": (None, "xq10"), "xq50": (None, "xq50"), "xq90": (None, "xq90"),     # XGBoost quantile (reg:quantileerror)
    "cq10": (None, "cq10"), "cq50": (None, "cq50"), "cq90": (None, "cq90"),     # CatBoost Quantile:alpha
    "hub30": (None, "hub30"), "hub100": (None, "hub100"),                        # LightGBM Huber, delta 30 / 100 s
    "cap15": (None, "cap15"), "cap255": (None, "cap255"),                        # 15 leaves / 255 leaves
    "ridge": (None, "ridge"),   # ridge on median-imputed standardised features + airport one-hot
    "mlp": (None, "mlp"),       # sklearn MLP (128,64), 300k-row subsample
    "knn": (None, "knn"),       # kNN (k=40) on standardised features, 120k-row subsample
    "rf": (None, "rf"),         # random forest 60 trees, leaf 50, max_features .3, max_samples .25
}
ORDER = list(VARIANTS)


def alt_recipe(name, main):
    ed = VARIANTS[name][0]
    toks = [t for t in main.split(",") if t]
    if ed is None:
        return main
    kind, s = ed
    if kind == "drop":
        return ",".join(t for t in toks if t not in s)
    return ",".join(t for t in toks if t in s)


def _cat_df(X):
    import pandas as pd
    df = pd.DataFrame(X.astype(np.float32))
    df[0] = np.nan_to_num(X[:, 0], nan=-1).astype(int)
    df.columns = [f"f{i}" for i in range(X.shape[1])]
    return df


def _prep(Xtr, Xte, nsub=None, seed=0):
    """median-impute (train medians), clip-standardise, airport (col 0) one-hot. Optional train subsample."""
    Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
    idx = None
    if nsub and len(Xtr) > nsub:
        idx = np.random.default_rng(seed).choice(len(Xtr), nsub, replace=False)
        Xtr = Xtr[idx]
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0); mu = np.where(np.isfinite(mu), mu, 0.0); sd = np.where(sd > 1e-6, sd, 1.0)
    def f(X):
        A = np.where(np.isnan(X), med, X)
        A = np.clip((A - mu) / sd, -5.0, 5.0)[:, 1:]
        oh = (X[:, [0]] == np.arange(9)[None, :]).astype(np.float32)
        return np.hstack([A, oh]).astype(np.float32)
    return f(Xtr), f(Xte), idx


def fit_predict(name, rec, Xtr, ttr, fin_tr, Xte, fin_te, seed, R):
    """R = the re-stack module the calling script imported (fit/predict/_params/parse). Returns add (len Xte), clipped like production."""
    kind = VARIANTS[name][1]
    if kind == "lgb":
        return R.predict(R.fit(Xtr, ttr, rec, seed), Xte)
    toks = R.parse(rec)
    P = dict(R._params(toks))
    if kind in ("ridge", "mlp", "knn", "rf"):
        t = np.clip(ttr, -900, 900)
        if kind == "ridge":
            from sklearn.linear_model import Ridge
            A, B, idx = _prep(Xtr, Xte); p = Ridge(alpha=100.0).fit(A, t).predict(B)
        elif kind == "mlp":
            from sklearn.neural_network import MLPRegressor
            A, B, idx = _prep(Xtr, Xte, 300000, seed)
            p = MLPRegressor(hidden_layer_sizes=(128, 64), batch_size=512, learning_rate_init=1e-3, max_iter=25, early_stopping=True, n_iter_no_change=3,
                             random_state=seed).fit(A, t[idx] if idx is not None else t).predict(B)
        elif kind == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            A, B, idx = _prep(Xtr, Xte, 120000, seed)
            m = KNeighborsRegressor(n_neighbors=40, algorithm="brute", n_jobs=4).fit(A, t[idx] if idx is not None else t)
            p = np.concatenate([m.predict(B[i:i + 20000]) for i in range(0, len(B), 20000)])
        else:
            from sklearn.ensemble import RandomForestRegressor
            A = np.nan_to_num(Xtr.astype(np.float32), nan=-99999.0); B = np.nan_to_num(Xte.astype(np.float32), nan=-99999.0)
            p = RandomForestRegressor(n_estimators=60, min_samples_leaf=50, max_features=0.3, max_samples=0.25, n_jobs=4, random_state=seed).fit(A, t).predict(B)
        return np.clip(p, -CLIPA, CLIPA)
    import re as _re
    if kind in ("cap15", "cap255") or kind.startswith("hub"):
        import lightgbm as lgb
        if kind == "cap15":
            P.update(num_leaves=15, min_child_samples=200)
        elif kind == "cap255":
            P.update(num_leaves=255, min_child_samples=100, n_estimators=700)
        else:
            P.update(objective="huber", alpha=float(kind[3:]))
        m = lgb.LGBMRegressor(random_state=seed, **P).fit(Xtr, ttr, categorical_feature=[0])
        return np.clip(m.predict(Xte), -CLIPA, CLIPA)
    if _re.fullmatch(r"xq\d+", kind):
        import xgboost as xgb
        m = xgb.XGBRegressor(tree_method="hist", objective="reg:quantileerror", quantile_alpha=int(kind[2:]) / 100.0, n_estimators=800,
                             learning_rate=0.03, max_depth=7, min_child_weight=200, subsample=0.8, colsample_bytree=0.8, max_bin=128,
                             n_jobs=4, random_state=seed)
        m.fit(Xtr.astype(np.float32), np.clip(ttr, -900, 900))
        return np.clip(m.predict(Xte.astype(np.float32)), -CLIPA, CLIPA)
    if _re.fullmatch(r"cq\d+", kind):
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=800, depth=8, learning_rate=0.07, loss_function=f"Quantile:alpha={int(kind[2:]) / 100.0}", thread_count=4,
                              random_seed=seed, verbose=0, border_count=32, allow_writing_files=False)
        m.fit(Xtr.astype(np.float32), np.clip(ttr, -900, 900))
        return np.clip(m.predict(Xte.astype(np.float32)), -CLIPA, CLIPA)
    if _re.fullmatch(r"q\d+", kind):
        import lightgbm as lgb
        P.update(objective="quantile", alpha=int(kind[1:]) / 100.0)
        m = lgb.LGBMRegressor(random_state=seed, **P).fit(Xtr, ttr, categorical_feature=[0])
        return np.clip(m.predict(Xte), -CLIPA, CLIPA)
    if kind in ("l1", "dart", "logt"):
        import lightgbm as lgb
        y = ttr
        if kind == "l1":
            P["objective"] = "l1"
        if kind == "dart":
            P.update(boosting_type="dart", n_estimators=500, learning_rate=0.06, drop_rate=0.1, skip_drop=0.5)
        if kind == "logt":
            y = np.log1p(np.maximum(ttr + fin_tr, 0.0)) - np.log1p(np.maximum(fin_tr, 0.0))
        m = lgb.LGBMRegressor(random_state=seed, **P).fit(Xtr, y, categorical_feature=[0])
        p = m.predict(Xte)
        if kind == "logt":
            p = (np.maximum(fin_te, 0.0) + 1.0) * np.exp(np.clip(p, -3, 3)) - 1.0 - np.maximum(fin_te, 0.0)
        return np.clip(p, -CLIPA, CLIPA)
    if kind == "cb":
        from catboost import CatBoostRegressor
        m = CatBoostRegressor(iterations=800, depth=8, learning_rate=0.07, loss_function="RMSE", thread_count=4, random_seed=seed,
                              verbose=0, border_count=32, allow_writing_files=False)   # airport index numeric (a categorical CTR made it 10x slower)
        m.fit(Xtr.astype(np.float32), np.clip(ttr, -900, 900))
        return np.clip(m.predict(Xte.astype(np.float32)), -CLIPA, CLIPA)
    if kind == "xgb":
        import xgboost as xgb
        m = xgb.XGBRegressor(tree_method="hist", n_estimators=800, learning_rate=0.03, max_depth=7, min_child_weight=200,
                             subsample=0.8, colsample_bytree=0.8, max_bin=128, n_jobs=4, random_state=seed)
        m.fit(Xtr.astype(np.float32), np.clip(ttr, -900, 900))
        return np.clip(m.predict(Xte.astype(np.float32)), -CLIPA, CLIPA)
    raise ValueError(kind)
