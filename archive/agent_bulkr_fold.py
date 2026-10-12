"""experiment bulkr: fold-wise fit of the SERVED bulk configuration (v94 = v92 flags with the LightGBM member set to
{lgb_routed, lgb_extratrees}) with CURRENT code, for docs/LAB-NOTES.md.

    python archive/agent_bulkr_fold.py --tag bulkr_f1 <prc.crossval flags as v92> --folds 1 [--smoke-frac 0.05]

Runs prc.crossval.main() UNEDITED for argument parsing, feature selection and frame loading, but swaps its run_fold for
bulkr_run_fold below (module attribute patch, nothing on disk is edited). Members, all on crossval's residual scale:
  c   CatBoost depth args.depth, routed per airport (_fit_predict's route block; the global fit is only made if some test
      row is not covered by a route, because routing overwrites it everywhere else), seed 1113
  p3  CatBoost depth args.blend_cat_depth, routed (crossval's d3 block + route_member), seed 1113
  l   global LightGBM (crossval's shipped member, seed 0)            -- v92 analogue / identity only
  r   routed LightGBM (crossval --lgb-routes block, seed 0)          -- v94 member
  e   extra-trees LightGBM (prc.members._fit_lightgbm_extratrees, seed 0) -- v94 member
Then: plan rows = (c + p3 + r + e) / 4 (v94) or (c + p3 + l) / 3 (v92); no-plan rows = crossval.fit_noplan; then
crossval's regime correction, in crossval's order. Saves results/preds_<tag>_<a>_<b>.parquet with bagged (= v94 analogue),
bagged92, members, npred. --skip-members e,l,... skips members (identity runs).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prc.crossval as cv  # noqa: E402
from prc.members import _fit_lightgbm_extratrees  # noqa: E402
from prc.validate import rmse  # noqa: E402

SMOKE_FRAC = 0.0
SKIP: set[str] = set()
ONLY: set[str] = set()   # --only c,p3: fit (or load) just these members, checkpoint them, stop (Modal one-fit containers)
CKPT = ""                # --ckpt-dir DIR: per-member checkpoints <tag>_<a>_<b>_<member>.npz; existing ones are loaded


def _log(*a):
    print("[bulkr]", *a, flush=True)


def bulkr_run_fold(frame, held, feats, cat_idx, args) -> dict:
    from catboost import CatBoostRegressor, Pool

    from prc import reference

    t0 = time.time()
    month = pl.col("month")
    assert not int(getattr(args, "holdout_days", 0) or 0) > 1, "holdout-days not supported"
    assert not getattr(args, "drop_dayoffset", False) and not args.winsor, "unsupported flag"
    assert getattr(args, "learner", "catboost") == "catboost" and args.target == "raw"
    assert not getattr(args, "season_split", False) and not getattr(args, "month_weight", "")
    train = frame.filter(~month.is_in(held))
    test = frame.filter(month.is_in(held))
    train, test = reference.attach(train, test)
    y_train = train[cv.TARGET].to_numpy().astype(float)
    test_x = test.select(feats).to_pandas()
    y = test[cv.TARGET].to_numpy().astype(float)
    plan = test["has_flight_plan"].to_numpy() == 1

    train_bulk, fit_y, base_test_v = cv.prepare_residual(train, y_train, args, apply_to=test)
    train_x = train_bulk.select(feats).to_pandas()
    cat_names = [feats[i] for i in cat_idx]
    sw = cv.month_weights(train_bulk, "")
    tr_apt = train_bulk["ADEP_mvt"].to_numpy()
    te_apt = test["ADEP_mvt"].to_numpy()
    routes = sorted(set(tr_apt) & set(te_apt))
    routed_codes = [c for c in routes if (tr_apt == c).sum() >= 5000 and (te_apt == c).sum() > 0]
    covered = np.isin(te_apt, routed_codes)
    _log(f"fold {held}: train {train.height:,} (bulk {train_bulk.height:,}) test {test.height:,}; "
         f"{len(feats)} features; routes {routed_codes}; uncovered test rows {int((~covered).sum())}")
    bt = np.broadcast_to(np.asarray(base_test_v, dtype=float), (len(te_apt),))

    def apt_path(label, code):
        return Path(CKPT) / "apt" / f"{args.tag}_{held[0]}_{held[1]}_{label}_{code}.npz" if CKPT else None

    def apt_part(label, code, te_m):
        """Per-airport partial checkpoint (bounds the work a preemption can lose to one airport fit)."""
        p = apt_path(label, code)
        if p is None or not p.exists():
            return None
        z = np.load(p)
        assert (z["ids"] == test["MVT_ID_mvt"].to_numpy()[te_m]).all(), f"{p}: row order differs"
        _log(f"{label} {code} loaded from {p}")
        return z["pred"]

    def apt_save(label, code, te_m, pred):
        p = apt_path(label, code)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.stem + ".tmp.npz")
        np.savez(tmp, ids=test["MVT_ID_mvt"].to_numpy()[te_m], pred=np.asarray(pred, dtype=float))
        tmp.replace(p)

    def cat_routed(make, weight, label):
        out = np.full(len(te_apt), np.nan)
        if (~covered).any():
            g = make()
            g.fit(Pool(train_x, fit_y, cat_features=cat_idx, weight=weight))
            out[:] = g.predict(test_x)
            _log(f"{label}: global fitted for {int((~covered).sum())} uncovered rows")
        for code in routed_codes:
            tr_m, te_m = tr_apt == code, te_apt == code
            part = apt_part(label, code, te_m)
            if part is not None:
                out[te_m] = part
                continue
            sub = make()
            sub.fit(Pool(train_x[tr_m], fit_y[tr_m], cat_features=cat_idx,
                         weight=None if weight is None else weight[tr_m]))
            out[te_m] = sub.predict(test_x[te_m])
            apt_save(label, code, te_m, out[te_m])
            _log(f"{label} {code} {int(tr_m.sum()):,} done at {time.time() - t0:.0f}s")
        return out + bt

    mem = {}
    ids_now = test["MVT_ID_mvt"].to_numpy()

    def ck_path(k):
        return Path(CKPT) / f"{args.tag}_{held[0]}_{held[1]}_{k}.npz" if CKPT else None

    def want(k):
        """True = fit k now. Loads k from the checkpoint dir when it is there (preemption-safe restarts)."""
        p = ck_path(k)
        if p is not None and p.exists():
            z = np.load(p)
            assert (z["ids"] == ids_now).all(), f"checkpoint {p} row order differs"
            mem[k] = z["pred"]
            _log(f"{k} loaded from {p}")
            return False
        if k in SKIP or (ONLY and k not in ONLY):
            return False
        return True

    def done(k):
        p = ck_path(k)
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.stem + ".tmp.npz")
            np.savez(tmp, ids=ids_now, pred=mem[k])
            tmp.replace(p)
            _log(f"{k} checkpointed -> {p}")

    # c: _fit_predict's routed CatBoost (seed index 0 -> random_seed 1113); routed subs fitted WITHOUT weight there.
    def make_c():
        kw = cv._catboost_kwargs(args)
        kw["random_seed"] = 1113
        return CatBoostRegressor(**kw)
    if want("c"):
        mem["c"] = cat_routed(make_c, None, "c")
        done("c")

    d3 = int(getattr(args, "blend_cat_depth", 0) or 0)
    assert d3, "--blend-cat-depth required"

    def make_d3():
        return CatBoostRegressor(
            iterations=args.iterations, depth=d3, learning_rate=args.lr,
            loss_function=args.loss, thread_count=args.threads,
            l2_leaf_reg=args.l2, one_hot_max_size=args.one_hot_max_size,
            random_seed=1113, verbose=0)
    if want("p3"):
        mem["p3"] = cat_routed(make_d3, sw, "p3")
        done("p3")

    if want("l"):
        lmodel, _, ltest_x = cv._fit_lightgbm(train_x.copy(), fit_y, test_x.copy(), cat_names, args, 0,
                                              objective="regression")
        mem["l"] = lmodel.predict(ltest_x) + base_test_v
        _log(f"l done at {time.time() - t0:.0f}s")
        done("l")
    if want("r"):
        # every test airport is routed in practice (asserted); the global fallback is only for uncovered rows
        assert covered.all() or "l" in mem, "uncovered test rows need the global LightGBM (member l)"
        out = np.array(mem["l"], dtype=float) if "l" in mem else np.full(len(te_apt), np.nan)
        for code in routed_codes:
            tr_m, te_m = tr_apt == code, te_apt == code
            part = apt_part("r", code, te_m)
            if part is not None:
                out[te_m] = part
                continue
            rmodel, _, rtest_x = cv._fit_lightgbm(
                train_x[tr_m].reset_index(drop=True), fit_y[tr_m],
                test_x[te_m].reset_index(drop=True), cat_names, args, 0, objective="regression")
            out[te_m] = rmodel.predict(rtest_x) + bt[te_m]
            apt_save("r", code, te_m, out[te_m])
            _log(f"r {code} done at {time.time() - t0:.0f}s")
        mem["r"] = out
        _log(f"r done at {time.time() - t0:.0f}s")
        done("r")
    if want("e"):
        mem["e"] = _fit_lightgbm_extratrees(train_x, fit_y, test_x, cat_names, args, 0) + base_test_v
        _log(f"e done at {time.time() - t0:.0f}s")
        done("e")
    if ONLY:
        _log(f"--only {sorted(ONLY)}: members fitted/checkpointed, stopping before the no-plan model and the blend")
        return {"fold": list(held), "n": int(len(y)), "seeds": 1, "rmse": float("nan"), "rmse_bagged": float("nan"),
                "rmse_bulk": float("nan"), "seconds": round(time.time() - t0), "bulkr_col": "only"}

    nmodel, nfeats = cv.fit_noplan(train, feats, cat_names, args)
    npred = nmodel.predict(test.select(nfeats).to_pandas())
    _log(f"no-plan done at {time.time() - t0:.0f}s")

    from prc.config import RAW_DIR
    raw = sorted(RAW_DIR.glob("training_*.parquet"))
    lam = getattr(args, "regime_correction", 0.0) or 0.0
    rf_test = rf_train = None
    if lam:
        rf_test = cv.regime_frame(test, raw, with_target=False)
        rf_train = cv.regime_frame(train, raw, with_target=True)

    def finish(blend):
        split = np.where(plan, blend, npred)
        if lam:
            split, *_ = cv.apply_regime_correction(rf_test, split, rf_train, lam,
                                                   getattr(args, "regime_quantile", 0.99))
        return split

    cols = {"MVT_ID_mvt": test["MVT_ID_mvt"].to_numpy(), "ADEP_mvt": te_apt, "y": y,
            "has_flight_plan": plan.astype(np.int8)}
    if all(k in mem for k in ("c", "p3", "r", "e")):
        cols["bagged"] = finish((mem["c"] + mem["p3"] + mem["r"] + mem["e"]) / 4.0)
    if all(k in mem for k in ("c", "p3", "l")):
        cols["bagged92"] = finish((mem["c"] + mem["p3"] + mem["l"]) / 3.0)
    for k, v in mem.items():
        cols[f"m_{k}"] = v
    cols["npred"] = npred
    path = Path(f"results/preds_{args.tag}_{held[0]}_{held[1]}.parquet")
    path.parent.mkdir(exist_ok=True)
    pl.DataFrame(cols).write_parquet(path)
    _log(f"saved {path} ({sorted(cols)}) at {time.time() - t0:.0f}s")
    main_col = "bagged" if "bagged" in cols else "bagged92"
    b = cols[main_col]
    return {"fold": list(held), "n": int(len(y)), "seeds": 1, "rmse": float("nan"),
            "rmse_bagged": rmse(y, b), "rmse_bulk": rmse(y[plan], b[plan]), "seconds": round(time.time() - t0),
            "bulkr_col": main_col}


def main():
    global SMOKE_FRAC, SKIP, ONLY, CKPT
    argv = list(sys.argv[1:])
    if "--only" in argv:
        i = argv.index("--only")
        ONLY = {s for s in argv[i + 1].split(",") if s}
        del argv[i:i + 2]
    if "--ckpt-dir" in argv:
        i = argv.index("--ckpt-dir")
        CKPT = argv[i + 1]
        del argv[i:i + 2]
    if "--smoke-frac" in argv:
        i = argv.index("--smoke-frac")
        SMOKE_FRAC = float(argv[i + 1])
        del argv[i:i + 2]
    if "--skip-members" in argv:
        i = argv.index("--skip-members")
        SKIP = {s for s in argv[i + 1].split(",") if s}
        del argv[i:i + 2]
    cv.run_fold = bulkr_run_fold
    if SMOKE_FRAC:
        patch_smoke(SMOKE_FRAC)
    sys.argv = ["prc.crossval", *argv]
    cv.main()


def patch_smoke(frac):
    """Local smoke only: deterministic row subsample of the cached feature frame (every Nth row)."""
    orig = cv.load_training_features
    n = max(int(round(1 / frac)), 1)

    def small(*a, **k):
        f = orig(*a, **k)
        return f.gather_every(n)
    cv.load_training_features = small


if __name__ == "__main__":
    main()
