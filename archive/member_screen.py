"""Which model family makes the most DIFFERENT errors from the current blend? (screen, not sizing)

Fits the three current members plus candidates on a subsample of fold (3,9), and reports each one's
RMSE, the correlation of its ERRORS with the current 3-member blend's errors, and the blend RMSE with
it added as a 4th equal-weight member. Subsampled to fit in ~3 GB beside the v93 build, so absolute
gains are inflated (scale-down bias) -- rank by error-decorrelation, confirm survivors at full scale.
"""
import numpy as np, polars as pl, time
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from prc.features import CATEGORICAL, FEATURES, TARGET
from prc.final import DROP
C = pl.col; rng = np.random.default_rng(0)
WAVE2 = "dep_rwy_30min,dep_rwy_headway,arr_taxi_60min,sched_demand_30min".split(",")
WX = "wx_temp_c,wx_dewpoint_c,wx_wind_kt,wx_gust_kt,wx_visibility_km,wx_ceiling_ft,wx_precip,wx_heavy_precip,wx_deice_risk,wx_thunder,wx_snow,wx_freezing,wx_lowvis,wx_age_min".split(",")
from prc.features import NM_NEIGHBOUR_FEATURES, LATE_NEIGHBOUR_FEATURES
keep = set(WAVE2 + WX + ["callsign_op"] + list(NM_NEIGHBOUR_FEATURES) + list(LATE_NEIGHBOUR_FEATURES))
feat = pl.scan_parquet("data/interim/train_features.parquet")
names = set(feat.collect_schema().names())
use = [f for f in FEATURES if f in names and f != TARGET and (f not in DROP or f in keep) and f not in ("doy", "month")]
cats = [c for c in use if c in CATEGORICAL]; nums = [c for c in use if c not in cats]
raw = pl.scan_parquet("data/raw/training_*.parquet").select(C("MVT_ID_mvt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
d = feat.select(["MVT_ID_mvt", TARGET, *use]).join(raw, on="MVT_ID_mvt").filter(C(TARGET).is_not_null()).collect()
tr = d.filter(~C("mo").is_in([3, 9])); te = d.filter(C("mo").is_in([3, 9]))
tr = tr[rng.choice(tr.height, 150000, replace=False)]; te = te[rng.choice(te.height, 75000, replace=False)]
print(f"{len(use)} features ({len(cats)} categorical); train {tr.height:,}  test {te.height:,}", flush=True)
ytr, yte = tr[TARGET].to_numpy().astype(float), te[TARGET].to_numpy().astype(float)
def pdf(f): return f.with_columns([C(c).cast(pl.Utf8).cast(pl.Categorical) for c in cats]).select(use).to_pandas()
Xtr, Xte = pdf(tr), pdf(te)
def num(f):  # ordinal-coded matrix for sklearn families
    m = f.select([C(c).cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Float64).alias(c) for c in cats] + [C(c).cast(pl.Float64) for c in nums])
    return np.nan_to_num(m.to_numpy(), nan=-1.0)
Ntr, Nte = num(tr), num(te)
ci = [use.index(c) for c in cats]
P = {}
def run(name, fn):
    t = time.time(); P[name] = np.asarray(fn(), float); print(f"  fit {name:22} {time.time()-t:6.0f}s", flush=True)
run("catboost_d8", lambda: CatBoostRegressor(iterations=800, depth=8, verbose=0, random_seed=0, cat_features=ci).fit(Xtr, ytr).predict(Xte))
run("catboost_d10", lambda: CatBoostRegressor(iterations=800, depth=10, verbose=0, random_seed=1113, cat_features=ci).fit(Xtr, ytr).predict(Xte))
run("lightgbm", lambda: lgb.LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.05, verbose=-1).fit(Xtr, ytr, categorical_feature=cats).predict(Xte))
run("catboost_huber", lambda: CatBoostRegressor(iterations=800, depth=8, verbose=0, random_seed=0, loss_function="Huber:delta=600", cat_features=ci).fit(Xtr, ytr).predict(Xte))
run("lightgbm_huber", lambda: lgb.LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.05, objective="huber", alpha=600, verbose=-1).fit(Xtr, ytr, categorical_feature=cats).predict(Xte))
run("extratrees", lambda: ExtraTreesRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5, n_jobs=3, random_state=0).fit(Ntr, ytr).predict(Nte))
run("randomforest", lambda: RandomForestRegressor(n_estimators=150, min_samples_leaf=5, max_features=0.5, n_jobs=3, random_state=0).fit(Ntr, ytr).predict(Nte))
sc = StandardScaler().fit(Ntr)
run("mlp", lambda: MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, max_iter=60, random_state=0).fit(sc.transform(Ntr), ytr).predict(sc.transform(Nte)))
run("ridge", lambda: Ridge(alpha=10.0).fit(sc.transform(Ntr), ytr).predict(sc.transform(Nte)))
rmse = lambda p: float(np.sqrt(np.mean((p - yte) ** 2)))
blend3 = (P["catboost_d8"] + P["catboost_d10"] + P["lightgbm"]) / 3.0
e3 = blend3 - yte; b3 = rmse(blend3)
print(f"\ncurrent 3-member blend (cat_d8 + cat_d10 + lightgbm): RMSE {b3:.2f}")
print(f"{'candidate':22} {'alone':>8} {'err-corr w/ blend':>18} {'blend+it (4 eq.)':>17} {'delta':>8}")
for k, p in sorted(P.items(), key=lambda kv: np.corrcoef(kv[1] - yte, e3)[0, 1]):
    r4 = rmse((3 * blend3 + p) / 4.0) if k not in ("catboost_d8", "catboost_d10", "lightgbm") else float("nan")
    print(f"  {k:20} {rmse(p):8.2f} {np.corrcoef(p - yte, e3)[0,1]:18.4f} {r4:17.2f} {r4-b3:+8.2f}")
