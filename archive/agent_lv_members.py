"""experiment lv (docs/LAB-NOTES.md): LTFM-only member / quantile "views", FOLD and SERVE twins from ONE code path.
NEW FILE; prc/ and other experiments' scripts are imported unedited.  Extends archive/agent_ltfmm_fold.py.

    --mode fold   : exactly agent_ltfmm_fold.py (cv.main folds; held months out; members c,p3,r,e checkpointed by bulkr_run_fold) + quantile models
    --mode serve  : the SAME member code, but train = all 2025 LTFM rows (frame rows with month != 99), test = the 2026 LTFM ranking rows, which are
                    appended to the frame with month = 99 and handed to bulkr_run_fold with held=(99,98): fold and serve share every line of the fit.
    --ltfm-seq on|off, --only c,p3,r,e, --ckpt-dir DIR, --quant q10,q50,q90 (optional), --smoke-frac f
Outputs <ckpt-dir>/<tag>_<a>_<b>_<member>.npz (ids, pred; member scale = residual + base, pre-regime) and ..._q10/q50/q90.npz the same way.
Serve checkpoints are tagged with held 99_98.  Never reads 2026 labels (ranking.parquet has none).
"""
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_bulkr_fold as bf  # noqa: E402
from agent_ltfm_seqfeat import SEQ_FEATURES, build as seq_build, fit_T  # noqa: E402

import prc.crossval as cv  # noqa: E402
import prc.features as F  # noqa: E402
from prc.config import RAW_DIR  # noqa: E402
from prc.features import TARGET, build as feat_build  # noqa: E402
from prc.train import load_training_features  # noqa: E402

COLS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt",
        "TAXITIME_SEC_mvt", "LOBT_flt"]
SERVE_HELD = (99, 98)


def _pop(argv, flag, has_val=True):
    if flag not in argv:
        return None
    i = argv.index(flag)
    v = argv[i + 1] if has_val else True
    del argv[i:i + (2 if has_val else 1)]
    return v


QA = {"q05": 0.05, "q10": 0.1, "q25": 0.25, "q50": 0.5, "q75": 0.75, "q90": 0.9, "q95": 0.95}


def quantile_views(frame, held, feats, cat_idx, args, kinds, ckpt):
    """Extra views fitted LTFM-only on the bulk features, residual target (same prelude as bulkr_run_fold).
    kinds: q05..q95 (LightGBM quantile), cbS/cbD (CatBoost depth 6/10), dart, l1 (LightGBM), xgb (prc.members xgb_hist), cs1..cs4 (CatBoost c-recipe, seeds 1114+),
    nowx / nosch (CatBoost c-recipe WITHOUT the weather / schedule-gap columns)."""
    import lightgbm as lgb
    import pandas as pd
    from catboost import CatBoostRegressor, Pool
    from prc import reference
    from prc.weather import FEATURES as WXF
    month = pl.col("month")
    train = frame.filter(~month.is_in(held))
    test = frame.filter(month.is_in(held))
    ids = test["MVT_ID_mvt"].to_numpy()
    todo = [q for q in kinds if not (Path(ckpt) / f"{args.tag}_{held[0]}_{held[1]}_{q}.npz").exists()]
    if not todo:
        print("[lv] extra views all checkpointed", flush=True)
        return
    train, test = reference.attach(train, test)
    y_train = train[cv.TARGET].to_numpy().astype(float)
    train_bulk, fit_y, base_test = cv.prepare_residual(train, y_train, args, apply_to=test)
    train_x = train_bulk.select(feats).to_pandas()
    test_x = test.select(feats).to_pandas()
    cat_names = [feats[i] for i in cat_idx]
    bt = np.broadcast_to(np.asarray(base_test, dtype=float), (len(ids),))
    tr_cat, te_cat = train_x.copy(), test_x.copy()
    for c in cat_names:
        levels = pd.Index(train_x[c].astype("string").unique()).dropna()
        dt = pd.CategoricalDtype(categories=levels)
        tr_cat[c] = train_x[c].astype("string").astype(dt)
        te_cat[c] = test_x[c].astype("string").astype(dt)
    smoke = bool(bf.SMOKE_FRAC)

    def cb(depth, seed, cols):
        kw = cv._catboost_kwargs(args)
        kw["random_seed"] = seed
        kw["depth"] = depth
        if smoke:
            kw["iterations"] = 30
        cn = [c for c in cat_names if c in cols]
        g = CatBoostRegressor(**kw)
        g.fit(Pool(train_x[cols], fit_y, cat_features=cn))
        return g.predict(Pool(test_x[cols], cat_features=cn))

    for q in todo:
        if q in QA:
            g = lgb.LGBMRegressor(objective="quantile", alpha=QA[q], n_estimators=30 if smoke else 600, learning_rate=0.05, num_leaves=63,
                                  min_child_samples=100, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, n_jobs=8, random_state=0, verbose=-1)
            g.fit(tr_cat, fit_y)
            p = g.predict(te_cat)
        elif q in ("dart", "l1"):
            kw = dict(boosting_type="dart", objective="regression", n_estimators=30 if smoke else 400, learning_rate=0.08) if q == "dart" else                 dict(objective="l1", n_estimators=30 if smoke else 800, learning_rate=0.05)
            g = lgb.LGBMRegressor(num_leaves=63, min_child_samples=100, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, n_jobs=8,
                                  random_state=0, verbose=-1, **kw)
            g.fit(tr_cat, fit_y)
            p = g.predict(te_cat)
        elif q == "xgb":
            from prc.members import fit_member
            a2 = args
            p = fit_member("xgb_hist", train_x, fit_y, test_x, cat_names, a2, np.zeros(len(train_x)), np.zeros(len(test_x)))
        elif q == "cbS":
            p = cb(6, 1113, list(feats))
        elif q == "cbD":
            p = cb(10, 1113, list(feats))
        elif q.startswith("cs"):
            p = cb(args.depth, 1113 + int(q[2:]) * 977, list(feats))
        elif q == "nowx":
            p = cb(args.depth, 1113, [f for f in feats if f not in set(WXF)])
        elif q == "nosch":
            p = cb(args.depth, 1113, [f for f in feats if "sched" not in f.lower()])
        else:
            raise SystemExit(f"unknown extra view {q}")
        p = np.asarray(p, dtype=float) + bt
        out = Path(ckpt) / f"{args.tag}_{held[0]}_{held[1]}_{q}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, ids=ids, pred=p)
        tmp.replace(out)
        print(f"[lv] {q} checkpointed -> {out} (mean {p.mean():.1f}s)", flush=True)


def main():
    argv = list(sys.argv[1:])
    mode = _pop(argv, "--mode")
    seq = _pop(argv, "--ltfm-seq")
    only = _pop(argv, "--only")
    ckpt = _pop(argv, "--ckpt-dir")
    quant = _pop(argv, "--quant")
    smoke = _pop(argv, "--smoke-frac")
    if mode not in ("fold", "serve") or seq not in ("on", "off") or not only or not ckpt:
        raise SystemExit("--mode fold|serve, --ltfm-seq on|off, --only, --ckpt-dir are required")
    quants = [s for s in (quant or "").split(",") if s]
    on, serve = seq == "on", mode == "serve"
    bf.ONLY = {s for s in only.split(",") if s}
    bf.CKPT = ckpt
    if smoke:
        bf.SMOKE_FRAC = float(smoke)
        bf.patch_smoke(bf.SMOKE_FRAC)
    sys.argv = ["prc.crossval", *argv]

    frame0 = load_training_features()          # BEFORE the lists are extended: the cache check reads FEATURES
    if smoke:
        frame0 = frame0.gather_every(max(int(round(1 / float(smoke))), 1))
    cv.load_training_features = lambda: frame0

    RANK = None
    if serve:   # built BEFORE the SEQ columns are appended to FEATURES (features.build selects FEATURES)
        RANK = feat_build(pl.read_parquet(RAW_DIR / "ranking.parquet"), with_target=False)
        RANK = RANK.filter(pl.col("ADEP_mvt") == "LTFM")
        print(f"[lv] serve: {RANK.height:,} LTFM ranking rows", flush=True)

    SF_TRAIN = SF_RANK = None
    if on:
        ltfm = (pl.col("ADEP_mvt") == "LTFM") | (pl.col("ADES_mvt") == "LTFM")
        raw = pl.concat([pl.scan_parquet(f).select(COLS).filter(ltfm).collect() for f in sorted(RAW_DIR.glob("training_2025-*.parquet"))])
        deps = raw.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM")).with_columns(
            _m=pl.col("MVT_TIME_UTC_mvt").dt.month(), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
        arrs = raw.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LTFM"))
        lab = deps.filter(pl.col("LOBT_flt").is_not_null())
        if serve:
            parts = []
            for m in range(1, 13):   # leave-own-month-out T, as agent_lks_final.py / agent_kg_ltfm_final.py
                t = fit_T(lab.filter(pl.col("_m") != m))
                f = seq_build(deps, arrs, t, demean=True).join(deps.select("MVT_ID_mvt", "_m"), on="MVT_ID_mvt").filter(pl.col("_m") == m)
                parts.append(f.drop("_m"))
            SF_TRAIN = pl.concat(parts)
            assert SF_TRAIN["MVT_ID_mvt"].n_unique() == SF_TRAIN.height == deps.height
            r26 = pl.scan_parquet(RAW_DIR / "ranking.parquet").select(COLS).filter(ltfm).collect()
            D26 = r26.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM"))
            A26 = r26.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LTFM"))
            SF_RANK = seq_build(D26, A26, fit_T(lab), demean=True)
            print(f"[lv] seq ON serve: train {SF_TRAIN.height:,} deps, rank {SF_RANK.height:,} deps, {A26.height:,} arrivals", flush=True)
        for lst in (F.NUMERIC, F.FEATURES):
            lst.extend(SEQ_FEATURES)
        assert cv.FEATURES is F.FEATURES
        cv.NOPLAN_DROP.extend(SEQ_FEATURES)

    def run_fold(frame, held, feats, cat_idx, args):
        if serve:
            tr = frame.filter(pl.col("ADEP_mvt") == "LTFM")
            rk = RANK
            if on:
                tr = tr.join(SF_TRAIN, on="MVT_ID_mvt", how="left", maintain_order="left")
                rk = rk.join(SF_RANK, on="MVT_ID_mvt", how="left", maintain_order="left")
            rk = rk.with_columns(pl.lit(SERVE_HELD[0]).cast(tr.schema["month"]).alias("month"),
                                 pl.lit(0.0).cast(tr.schema[TARGET]).alias(TARGET))
            missing = [c for c in feats if c not in rk.columns]
            assert not missing, f"ranking frame lacks features {missing}"
            fr = pl.concat([tr, rk.select([c for c in tr.columns if c in rk.columns])], how="diagonal_relaxed")
            held = SERVE_HELD
            print(f"[lv] serve frame: train {tr.height:,} LTFM rows (month != 99), test {rk.height:,}; {len(feats)} features", flush=True)
        else:
            if on:
                parts = []
                for m in range(1, 13):
                    excl = set(held) | ({m} if m not in held else set())
                    t = fit_T(lab.filter(~pl.col("_m").is_in(sorted(excl))))
                    f = seq_build(deps, arrs, t, demean=True).join(deps.select("MVT_ID_mvt", "_m"), on="MVT_ID_mvt") \
                        .filter(pl.col("_m") == m)
                    parts.append(f.drop("_m"))
                sf = pl.concat(parts)
                assert sf["MVT_ID_mvt"].n_unique() == sf.height == deps.height
                frame = frame.join(sf, on="MVT_ID_mvt", how="left", maintain_order="left")
            fr = frame.filter(pl.col("ADEP_mvt") == "LTFM")
            print(f"[lv] fold {held}: LTFM frame {fr.height:,} rows, {len(feats)} features", flush=True)
        res = bf.bulkr_run_fold(fr, held, feats, cat_idx, args)
        if quants:
            quantile_views(fr, held, feats, cat_idx, args, quants, ckpt)
        return res

    cv.run_fold = run_fold
    cv.main()


if __name__ == "__main__":
    main()
