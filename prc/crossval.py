"""Multi-fold validation across month-pairs.

Why this exists: three submissions in a row improved on the Jan+Jul 2025
holdout and got worse on the 2026 leaderboard. Month 11 agreed with the
holdout, so it is not a bad fold -- but one fold cannot tell a real improvement
from a fold-specific one, and this metric is decided by a handful of rows that
differ between any two periods. A change worth shipping should win on most
folds, not on one.

Six folds, each holding out one winter month and one summer month so every fold
has the seasonal mix of the ranking set (January and July 2026):

    fold 0: 1, 7      fold 2: 3, 9      fold 4: 5, 11
    fold 1: 2, 8      fold 3: 4, 10     fold 5: 6, 12

Iterations are FIXED rather than early-stopped. Early stopping needs a
validation month, which would either eat a fold or vary between folds, and the
stopping point itself becomes a source of fold-to-fold variance that has nothing
to do with the change being tested.

    python -m prc.crossval --tag base --drop airport_plan,dep_rwy_30min,...
    python -m prc.crossval --tag wave2 --drop airport_plan
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from types import SimpleNamespace

import numpy as np
import polars as pl

from .features import CATEGORICAL, CLOCK_FEATURES, FEATURES, TARGET
from .train import load_training_features
from .validate import rmse

FOLDS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]
RESULTS = Path("results/crossval.jsonl")


# Features that actually exist for rows with no flight plan. Everything derived
# from a _flt column is null there, so a global model is asking a specialist
# question of a model whose inputs are 98.5% absent for this group.
NOPLAN_DROP = [
    "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt",
    "gap_aobt", "gap_eobt", "gap_lobt", "dep_delay", "sched_vs_eobt",
    "has_flight_plan",  # constant within the group
    # The 20 NM neighbour and following features. They help the BULK model on
    # both folds (-1.74s and -5.06s on flight-plan rows) and wreck the specialist,
    # which fits on 1,000-18,000 rows: n_feat read +54.39s on fold (1,7) no-plan
    # rows against -45.10s on fold (2,8). Excluded here so enabling them reaches
    # the bulk model only.
    "nm_apt_past15_mean",
    "nm_apt_past15_n",
    "nm_apt_fol15_mean",
    "nm_apt_fol15_n",
    "nm_apt_past60_mean",
    "nm_apt_past60_n",
    "nm_apt_fol60_mean",
    "nm_apt_fol60_n",
    "nm_rwy_past15_mean",
    "nm_rwy_past15_n",
    "nm_rwy_fol15_mean",
    "nm_rwy_fol15_n",
    "nm_rwy_past60_mean",
    "nm_rwy_past60_n",
    "nm_rwy_fol60_mean",
    "nm_rwy_fol60_n",
    "nm_rwy_past60_own_minus",
    "nm_rwy_fol60_own_minus",
    "nm_rwy_next_wait",
    "nm_rwy_next_proxy",
    # Explicit NM clock differences (features.CLOCK_FEATURES). All null without a
    # flight plan anyway, and every family added to the specialist so far has
    # overfitted its 1,000-18,000 rows -- so bulk model only, from the start.
    *CLOCK_FEATURES,
]


DAY = 86400.0

# Fold RMSE is not a readable instrument on its own. On fold (1,7) two Paris rows
# whose BLOCK stamp is most of a day stale -- with an ordinary schedule gap, so
# nothing observable says they are coming -- carry 29.5% of the fold's squared
# error, and 34 rows carry 39%. Trim the worst 0.01% and the fold reads 316.61 ->
# 246.82, while fold (2,8) goes 258.36 -> 244.27: two folds that look 58s apart
# are 2.5s apart underneath. That variance is why v18 was estimated at -3.46s on
# a fold and landed at -0.12s on the board.
#
# So `rmse_core` is reported beside the shipped number. The cut is on the TRUTH,
# not on our error: trimming each arm's own worst rows would let an arm win by
# concentrating error into fewer rows. 10,000s is above any real taxi -- nothing
# pushes back and reaches a runway three hours later -- so rows above it are
# stale timestamps rather than slow departures. It removes 0.012% of training
# rows (259 of 2,085,047) and about a third of fold (1,7)'s squared error, and
# deliberately KEEPS the 3,000-10,000s band, which is where the disruption-day
# headroom lives and must stay measurable.
#
# `rmse_clipped` remains the number that decides what ships. `rmse_core` is not a
# second gate on it -- measured against the shipped gate arm it reads +0.84s and
# +0.67s, so it would have rejected the change that took 12.0s off the board. The
# two see different work: a bulk change is nearly invisible in the shipped number
# and shows up here; a tail change is invisible here by construction. Which one
# reads an arm is decided by what the arm targets, before it runs.
CORE_MAX_Y = 10000.0


def drop_day_offsets(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove training rows whose label is a confirmed one-day timestamp error.

    Identified in docs/LAB-NOTES.md: about 15 rows a year
    where removing a whole number of days leaves a plausible taxi time (median
    951s against a global median of 912s). Every one is at LIRF or LSZH.

    This is NOT winsorising, which was measured at 78s worse. Winsorising
    falsifies the label on genuinely long taxis, so the model learns to
    underpredict real ones. This deletes rows we have proven are corrupt, and
    only those -- under squared loss each carries on the order of a thousand
    times an ordinary row's gradient weight, so a handful of them steers a
    large share of the fit.

    Applied to TRAINING only. Never to anything we score on: the evaluation
    labels are corrupt in the same way and predicting them is the job.
    """
    y = pl.col(TARGET).cast(pl.Float64)
    residual = y - (y / DAY).round() * DAY
    corrupt = (y >= 20000) & (residual >= 0) & (residual < 7200)
    return frame.filter(~corrupt.fill_null(False))


def _catboost_kwargs(args) -> dict:
    """Loss, and the categorical-encoding knobs that have never been touched.

    ``--loss Huber:delta=2000`` keeps eval_metric on RMSE, which CatBoost allows
    and which costs us nothing since we fix iterations rather than early-stop.
    Huber is the continuous form of dropping outliers: residuals past delta get
    linear rather than quadratic weight, so extreme rows stop dominating split
    selection without their labels being falsified.

    ``one_hot_max_size`` defaults to 2, so today every categorical -- including
    FLIGHT_RULE_mvt and WK_TBL_CAT_flt, which have a handful of levels -- goes
    through target-statistic encoding and its prior shrinkage. For a
    low-cardinality feature that shrinkage is noise, not regularisation.
    """
    kwargs = dict(
        iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
        l2_leaf_reg=args.l2, thread_count=args.threads, random_seed=1113, verbose=False,
    )
    # Five standard axes that were never exposed, let alone tested. An audit of
    # all 68 recorded runs found --loss has NEVER been set in any of them, so
    # every model this campaign has fitted used plain RMSE loss -- on a target
    # where 76 training labels a year are corrupt and carry hundreds of times an
    # ordinary row's gradient.
    #
    #   max_ctr_complexity  how many categoricals get COMBINED in target
    #                       statistics. Default 4. Taxi time is plausibly a
    #                       stand-by-runway-by-hour effect, which is a 3-way.
    #   rsm                 feature subsampling per split. Decorrelates trees,
    #                       which is the same mechanism that makes our blend pay.
    #   bootstrap_type      MVS by default for regression; Bernoulli with
    #                       subsample is the usual alternative.
    #   random_strength     noise added to split scores; the standard lever
    #                       against overfitting high-cardinality categoricals,
    #                       of which we have several with 1,899 levels.
    for name, attr in (("max_ctr_complexity", "max_ctr_complexity"),
                       ("rsm", "rsm"),
                       ("bootstrap_type", "bootstrap_type"),
                       ("subsample", "subsample"),
                       ("random_strength", "random_strength")):
        v = getattr(args, attr, None)
        if v is not None:
            kwargs[name] = v
    # CatBoost grows OBLIVIOUS trees by default: every split at a given depth
    # uses the same feature for every node. That is why it is fast, and it is a
    # real constraint on representing per-airport structure -- LIRF's bulk RMSE
    # is 404 against 176-261 everywhere else, and an airport-specific
    # interaction has to be spent at a whole level rather than in one branch.
    # Lossguide and Depthwise both split per node instead. The 2024 winner of
    # this challenge family used LightGBM, which is leaf-wise by default.
    grow = getattr(args, "grow_policy", "SymmetricTree")
    if grow != "SymmetricTree":
        kwargs["grow_policy"] = grow
        # depth means different things per policy: for Lossguide it is a cap
        # alongside max_leaves, and CatBoost rejects depth > 16 there anyway.
        if grow == "Lossguide":
            kwargs["max_leaves"] = getattr(args, "max_leaves", 64)
            kwargs["depth"] = min(args.depth, 16)

    loss = getattr(args, "loss", "RMSE")
    kwargs["loss_function"] = loss
    if loss != "RMSE":
        kwargs["eval_metric"] = "RMSE"
    ohms = getattr(args, "one_hot_max_size", 0)
    if ohms:
        kwargs["one_hot_max_size"] = ohms
    return kwargs


BASELINE_COLS = ["gap_aobt", "gap_eobt", "gap_lobt", "gap_sched"]


def _baseline_matrix(frame: pl.DataFrame) -> np.ndarray:
    cols = [np.nan_to_num(frame[c].to_numpy().astype(float), nan=0.0,
                          posinf=0.0, neginf=0.0) for c in BASELINE_COLS]
    cols.append(np.ones(frame.height))
    return np.column_stack(cols)


LOBT_SWITCH_S = 4200.0
# Minimum training rows before an (airport, hour) baseline cell is fitted on
# its own rather than inheriting the airport's. Five coefficients, so this is
# 100 rows each.
BASELINE_CELL_MIN = 500


def fit_baseline(train: pl.DataFrame, mode: str):
    """Return a callable giving the baseline to residualise against.

    ``aobt``  the raw Network Manager off-block gap. What v7 ships. Alone it
              scores 384.9s against 546s for a constant, and residualising
              against it moved fold (1,7)'s bulk from 272.4 to 240.2.
    ``blend`` a least-squares combination of all four timestamp gaps, fitted on
              training rows only. gap_eobt and gap_lobt score 676.7s and 740.1s
              alone -- worse than gap_aobt but not redundant with it, since each
              is a different revision of the same estimate. If a better level
              buys what a good level bought, this is where the next gain is.

    Fitted on training rows and applied outward, like prc.reference: the
    coefficients see no row they are later used to predict.
    """
    if mode == "lobt_switch":
        # Where the last-calculated off-block sits more than LOBT_SWITCH_S after
        # the actual one, the airport's BLOCK_TIME tracks LOBT, so residualise
        # against gap_lobt instead of gap_aobt.
        #
        # Measured as a pure arithmetic substitution on the raw gaps, with the
        # threshold chosen out of fold each time (it converged on 4,200s for all
        # six independently): wins 6/6, mean -18.79s, fold (1,7) 339.27 -> 304.15.
        # 256 scored rows clear the threshold, at 7.4 per 10k against 5.5 per 10k
        # in training, so the population is present in 2026 -- which is the check
        # the LIRF pocket failed.
        def _switch(f):
            ga = np.nan_to_num(f["gap_aobt"].to_numpy().astype(float), nan=0.0,
                               posinf=0.0, neginf=0.0)
            gl = np.nan_to_num(f["gap_lobt"].to_numpy().astype(float), nan=0.0,
                               posinf=0.0, neginf=0.0)
            return np.where(gl - ga > LOBT_SWITCH_S, gl, ga)
        return _switch

    if mode not in ("blend", "blend_apt", "blend_apt_hour"):
        return lambda f: np.nan_to_num(f["gap_aobt"].to_numpy().astype(float),
                                       nan=0.0, posinf=0.0, neginf=0.0)

    x = _baseline_matrix(train)
    y = train[TARGET].to_numpy().astype(float)
    # Fit on flight-plan rows only: elsewhere every gap is zero and the rows are
    # routed to the dedicated model anyway, so they would only drag the fit.
    keep = train["has_flight_plan"].to_numpy() == 1
    apt_train = train["ADEP_mvt"].to_numpy()

    def _solve(mask):
        return np.linalg.lstsq(x[mask], y[mask], rcond=None)[0]

    global_coef = _solve(keep)
    print("    baseline blend: " + "  ".join(
        f"{n}={c:+.3f}" for n, c in zip(BASELINE_COLS + ["const"], global_coef)))

    # blend_apt_hour strata the fit by airport AND hour of day rather than by
    # airport alone. Fold (1,7), RMSE of the baseline itself over all held-out
    # plan rows: global 346.0, per-airport 315.3 (what ships), per-(airport,hour)
    # 310.8. The cells are not thin -- the median airport-hour holds 7,712
    # training rows against five coefficients -- and the minimum hardly matters:
    # 310.6 at min 200, 310.9 at min 2,000, 311.2 at min 5,000. 500 is 100 rows
    # per coefficient, picked for that ratio rather than off the sweep.
    #
    # Note this is the baseline's own error, NOT the model's. The trees see
    # `hour` and all four gaps already, so some of this is structure they can
    # recover themselves; the fold is what says how much is actually new.
    #
    # Four things that did NOT work, measured at the same time, so nobody
    # retries them: trimming the corrupt-label rows out of the fit changes the
    # residual sd by nothing at all (315.1 either way -- 76 rows in 1.7M cannot
    # move the coefficients however heavily least squares weights them); a Huber
    # fit is WORSE at 319.0, for the same reason Huber loss on the model itself
    # was worse, robust fitting chases the median and the metric wants the mean;
    # four coarse time-of-day buckets instead of 24 hours gives back most of it
    # at 313.9; and interacting the gaps with hour harmonics, which needs no
    # cell minimum at all, tops out at 311.2 with three harmonics -- more
    # elegant, measurably worse than just fitting the cells.
    hour_train = (train["MVT_TIME_UTC_mvt"].dt.hour().to_numpy()
                  if "MVT_TIME_UTC_mvt" in train.columns else train["hour"].to_numpy())

    per_airport = {}
    if mode in ("blend_apt", "blend_apt_hour"):
        # Ten airports with different layouts and procedures have no reason to
        # share one linear relation between a filed estimate and actual taxi.
        for a in np.unique(apt_train[keep]):
            m = keep & (apt_train == a)
            if m.sum() >= 5000:
                per_airport[a] = _solve(m)
        print(f"    per-airport baselines fitted for {len(per_airport)} of "
              f"{len(np.unique(apt_train))} airports")
        if mode == "blend_apt_hour" and len(per_airport) == 0:
            raise RuntimeError("blend_apt_hour: no per-airport rung fitted; the "
                               "hour cells would fall back to a global fit")

    per_hour = {}
    if mode == "blend_apt_hour":
        for a in np.unique(apt_train[keep]):
            for h in range(24):
                m = keep & (apt_train == a) & (hour_train == h)
                if m.sum() >= BASELINE_CELL_MIN:
                    per_hour[(a, h)] = _solve(m)
        print(f"    per-(airport,hour) baselines fitted for {len(per_hour)} of 240 cells")

    def apply(frame: pl.DataFrame) -> np.ndarray:
        mat = _baseline_matrix(frame)
        out = mat @ global_coef
        # Cells fall back: (airport,hour) -> airport -> global. A cell under
        # BASELINE_CELL_MIN training rows keeps the coarser fit rather than a
        # noisy own one. The airport rung is worth 0.5s over falling straight to
        # global, which is what the first version of this did.
        if per_airport:
            apt = frame["ADEP_mvt"].to_numpy()
            for a, coef in per_airport.items():
                at = apt == a
                if at.any():
                    out[at] = mat[at] @ coef
        if per_hour:
            apt = frame["ADEP_mvt"].to_numpy()
            hh = (frame["MVT_TIME_UTC_mvt"].dt.hour().to_numpy()
                  if "MVT_TIME_UTC_mvt" in frame.columns else frame["hour"].to_numpy())
            for (a, h), coef in per_hour.items():
                at = (apt == a) & (hh == h)
                if at.any():
                    out[at] = mat[at] @ coef
        # no flight plan -> no baseline; those rows go to the dedicated model
        return np.where(frame["has_flight_plan"].to_numpy() == 1, out, 0.0)

    return apply


def _baseline(frame: pl.DataFrame) -> np.ndarray:
    """Network Manager's own off-block-to-wheels-up gap, as a physical baseline.

    One option is to model the RESIDUAL from this baseline
    rather than the target directly. The idea is that `MVT_TIME - AOBT_3_flt` is
    already a measurement of roughly the right quantity from an independent
    source -- it scores 384.9s used alone, worse than our model but far better
    than the 546s a constant gives -- so asking the trees to learn only the
    correction should be easier than asking them to reconstruct the level and
    the correction together.

    It is emphatically not a label leak: correlation with the target is 0.536,
    it matches exactly on 0.65% of rows and within a minute on 21%. Verified
    directly rather than taken on trust, because a claim that it reconstructs
    the label was circulating and is wrong.

    Zero where the flight plan is absent, so the residual reduces to the target
    for those rows -- they are routed to the dedicated model regardless.
    """
    gap = frame["gap_aobt"].to_numpy().astype(float)
    return np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_residual(frame, y, args, apply_to=None):
    """The residual setup, in ONE place, because order matters and drifts.

    --bulk-plan-only filters the training rows; --residual fits a baseline on
    them. Keeping both paths in one function is still worth it -- crossval and
    final have drifted apart repeatedly -- but the incident this was written
    for is NOT an example of it.

    CORRECTED 2026-09-11. The original docstring said the two paths ordered
    these differently and that this cost v10 10.8s. It did not. fit_baseline
    masks to flight-plan rows unconditionally before solving, so the order
    cannot change the coefficients, and v10's and v11's bulk predictions are
    bit-identical on all scored plan rows. The two runs differ in one
    flag, --noplan-routes LIRF,LFPG,LSZH against LIRF, and every bit of the
    19-second move is in the 5,290 no-plan rows, LFPG carrying 4.0e9 of the
    squared movement against LIRF's exact zero.

    The real lesson is about routing, not ordering: a specialist fitted on a
    0.25% slice memorises the corrupt-BLOCK rows inside it and then bets tens of
    thousands of seconds on ordinary departures. It also means --bulk-plan-only
    has never been isolated; it shipped inside v10, which regressed, and was
    then credited with v11's gain.
    """
    if getattr(args, "bulk_plan_only", False):
        keep = (frame["has_flight_plan"] == 1).to_numpy()
        frame = frame.filter(pl.col("has_flight_plan") == 1)
        y = y[keep]

    if getattr(args, "residual", False):
        make_base = fit_baseline(frame, getattr(args, "baseline", "aobt"))
        base_train = make_base(frame)
        base_other = make_base(apply_to) if apply_to is not None else 0.0
    else:
        base_train = base_other = 0.0

    fit_y = y - base_train
    if getattr(args, "target", "raw") == "log":
        fit_y = np.log1p(np.maximum(fit_y, 0.0))
    return frame, fit_y, base_other


def _fit_lightgbm(train_x, fit_y, test_x, cat_names, args, seed,
                  objective=None, huber_alpha=None):
    """LightGBM, optionally with a linear model in each leaf.

    `linear_tree=True` fits a ridge regression per leaf instead of a constant.
    The idea comes from another team's public write-up (see the README credits); our own fold tests decide whether it ships.

    Two things it needs that CatBoost handles for us:

    * **Categoricals.** LightGBM wants pandas `category` dtype, and the levels
      must be the SAME set in train and test or the integer codes mean different
      things in each. The categories are therefore fixed from the training frame
      and applied to the test frame, so a level only seen at serve time becomes
      NaN rather than silently colliding with a different level's code.
    * **Extrapolation.** Linear leaves are unbounded, and the team above measured
      one row predicted at 28,561s against a true 1,390s. Their fix was to give
      the model signed-log copies of the wide-range features alongside the raw
      ones, so a leaf's linear term acts on a compressed scale.

    ``objective``/``huber_alpha`` override ``args.lgb_objective`` /
    ``args.lgb_huber_alpha`` when given, and are ONLY used by run_fold's
    "members" blend-eval path (2026-09-22) to fit a second, Huber-objective
    LightGBM member alongside the shipped regression-objective one from the
    SAME call site. prc/final.py's blend calls this function with neither
    kwarg and no `lgb_objective` attribute on its args, so `objective`
    resolves to "regression" there exactly as before -- this is additive and
    does not change final.py's or crossval's own default behaviour, which
    hardcoded "regression" here until now.
    """
    import lightgbm as lgb
    import pandas as pd

    objective = objective or getattr(args, "lgb_objective", "regression") or "regression"
    if objective not in ("regression", "huber"):
        raise SystemExit(f"_fit_lightgbm: unknown objective {objective!r}")

    train_x = train_x.copy()
    test_x = test_x.copy()
    for c in cat_names:
        if c not in train_x.columns:
            continue
        levels = pd.Index(train_x[c].astype("string").unique()).dropna()
        dt = pd.CategoricalDtype(categories=levels)
        train_x[c] = train_x[c].astype("string").astype(dt)
        test_x[c] = test_x[c].astype("string").astype(dt)

    if getattr(args, "linear_tree", False):
        # compress the wide-range numerics for the leaf regressions
        for c in train_x.columns:
            if c in cat_names or not pd.api.types.is_numeric_dtype(train_x[c]):
                continue
            if float(np.nanmax(np.abs(train_x[c].to_numpy(dtype=float)))) > 5000:
                for f in (train_x, test_x):
                    v = f[c].to_numpy(dtype=float)
                    f[f"{c}__slog"] = np.sign(v) * np.log1p(np.abs(v))

    params = dict(
        objective=objective, metric="rmse",
        n_estimators=args.iterations, learning_rate=args.lr,
        num_leaves=getattr(args, "num_leaves", 255),
        min_child_samples=getattr(args, "min_child_samples", 40),
        reg_lambda=args.l2, random_state=1113 + seed * 977,
        n_jobs=args.threads, verbose=-1,
    )
    if objective == "huber":
        alpha = huber_alpha if huber_alpha is not None else getattr(args, "lgb_huber_alpha", None)
        if alpha is not None:
            params["alpha"] = alpha
    if getattr(args, "linear_tree", False):
        params["linear_tree"] = True
        params["linear_lambda"] = getattr(args, "linear_lambda", 1.0)
    model = lgb.LGBMRegressor(**params)
    model.fit(train_x, fit_y, categorical_feature=[c for c in cat_names
                                                   if c in train_x.columns])
    return model, train_x, test_x


def month_weights(train_frame, spec: str):
    """Per-row sample weights from a "month:weight,..." spec. None if empty.

    We train on twelve months of 2025 and are scored on January and July 2026,
    and every row has counted equally throughout. Whether that is right has
    never been tested.

    A caveat the fold structure forces, and it is not small: fold (1,7) HOLDS
    OUT months 1 and 7, so "upweight the months we are scored on" is untestable
    there by construction. The in-fold analogue is to upweight the seasonally
    adjacent months -- 12 and 2 for January, 6 and 8 for July -- which asks the
    same question (does seasonal matching beat data volume?) without asking it
    about the exact months. A win in the fold is evidence FOR the real thing,
    not a measurement of it.
    """
    import numpy as np
    if not spec:
        return None
    table = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        mo, _, w = part.partition(":")
        table[int(mo)] = float(w)
    if not table:
        return None
    months = train_frame["month"].to_numpy()
    w = np.ones(len(months), dtype=float)
    for mo, val in table.items():
        w[months == mo] = val
    return w


def _fit_predict(train_frame, test_frame, test_x, feats, cat_idx, y_train, args, seed):
    """Fit on the chosen target scale and return predictions on the SECONDS scale.

    ``--target log`` fits log1p(seconds). A right-skewed duration is the textbook
    case for it: on the log scale the bulk stops being distorted by a handful of
    multi-hour rows, and unlike winsorising the model can still emit large values,
    because the back-transform is exponential rather than clipped.

    The back-transform uses Duan's smearing estimator. exp(E[log y]) is the
    conditional *median*, not the mean, and RMSE wants the mean -- so naively
    exponentiating biases every prediction low. Smearing corrects it
    non-parametrically by the mean of exp(training residuals), which needs no
    assumption that those residuals are normal. Getting this wrong is the usual
    reason a log-target model scores worse than it should.
    """
    from catboost import CatBoostRegressor, Pool

    # The global model's predictions on no-flight-plan rows are thrown away --
    # `np.where(has_plan, bagged, npred)` replaces every one of them with the
    # dedicated model's. But those rows are still in its training set, and on
    # the residual scale they are enormous (sd ~3,400s against a few hundred for
    # plan rows), so they steer a large share of every split. CatBoost's trees
    # are oblivious: one level splitting on has_flight_plan does not stop those
    # rows shaping the other levels, or the ordered target statistics for
    # STAND_mvt, which are pooled over all rows.
    #
    # This is not the np_all experiment reversed. That asked whether the no-plan
    # model does better seeing all rows (it does not, 0/3 -- the specialisation
    # is the value). This asks the same question of the other model.
    train_frame, fit_y, base_test_v = prepare_residual(
        train_frame, y_train, args, apply_to=test_frame)
    train_x = train_frame.select(feats).to_pandas()
    if getattr(args, "learner", "catboost") == "lightgbm":
        cat_names = [feats[i] for i in cat_idx]
        model, train_x, test_x = _fit_lightgbm(
            train_x, fit_y, test_x, cat_names, args, seed)
    else:
        kwargs = _catboost_kwargs(args)
        kwargs["random_seed"] = 1113 + seed * 977
        model = CatBoostRegressor(**kwargs)
        sw = month_weights(train_frame, getattr(args, "month_weight", ""))
        model.fit(Pool(train_x, fit_y, cat_features=cat_idx, weight=sw))

    # PER-AIRPORT BULK MODELS. One model is fitted for each named airport on
    # that airport's rows alone; everything else keeps the shared model above.
    #
    # The case: bulk RMSE is 393 at LIRF against 171-259 elsewhere, and LTFM,
    # LIRF and EGLL together hold 46% of bulk squared error. Ten airports at
    # ~200k rows each is ample data to fit on.
    #
    # Routing is on airport codes FIXED BEFORE THE FIT, never chosen by score,
    # so this is not the per-subgroup model selection that cost a 2024 team
    # 2,066 -> 2,276. Same justification as the no-plan routing -- and the same
    # warning applies: that routing worked at LIRF and cost 19s live when it was
    # extended to LFPG and LSZH on a floor analysis that turned out unreliable.
    # Two bulk models split on season. January bulk RMSE is 206.41 against
    # July's 249.30 -- a 43-second gap that says the two months are close to
    # different problems. Month weighting (reweighting one training set) was
    # measured dead and monotonically worse; fitting two models is a different
    # question, the same way routing was different from reweighting.
    if getattr(args, "season_split", False):
        WINTER = {11, 12, 1, 2, 3}
        tr_w = np.isin(train_frame["month"].to_numpy(), list(WINTER))
        te_w = np.isin(test_frame["month"].to_numpy(), list(WINTER))
        pred = model.predict(test_x)
        for lab, tm, em in (("winter", tr_w, te_w), ("summer", ~tr_w, ~te_w)):
            if tm.sum() < 5000 or em.sum() == 0:
                continue
            kw = _catboost_kwargs(args)
            kw["random_seed"] = 1113 + seed * 977
            sub = CatBoostRegressor(**kw)
            sub.fit(Pool(train_x[tm], fit_y[tm], cat_features=cat_idx))
            pred[em] = sub.predict(test_x[em])
            print(f"    season split: {lab} trained on {int(tm.sum()):,}, "
                  f"predicting {int(em.sum()):,}")
        if args.target != "log":
            return pred + base_test_v
        sm = float(np.mean(np.exp(fit_y - model.predict(train_x))))
        return np.expm1(pred) * sm + base_test_v

    routes = [c.strip().upper() for c in
              (getattr(args, "bulk_routes", "") or "").split(",") if c.strip()]
    if routes:
        pred = model.predict(test_x)
        tr_apt = train_frame["ADEP_mvt"].to_numpy()
        te_apt = test_frame["ADEP_mvt"].to_numpy()
        if routes == ["ALL"]:
            routes = sorted(set(tr_apt) & set(te_apt))
        sizes = []
        for code in routes:
            tr_m, te_m = tr_apt == code, te_apt == code
            if tr_m.sum() < 5000 or te_m.sum() == 0:
                sizes.append(f"{code} skipped")
                continue
            kw = _catboost_kwargs(args)
            kw["random_seed"] = 1113 + seed * 977
            sub = CatBoostRegressor(**kw)
            sub.fit(Pool(train_x[tr_m], fit_y[tr_m], cat_features=cat_idx))
            pred[te_m] = sub.predict(test_x[te_m])
            sizes.append(f"{code} {int(tr_m.sum()):,}")
        print("    bulk routing: " + "  ".join(sizes))
        if args.target != "log":
            return pred + base_test_v
        sm = float(np.mean(np.exp(fit_y - model.predict(train_x))))
        return np.expm1(pred) * sm + base_test_v

    base_test = base_test_v
    if args.target != "log":
        return model.predict(test_x) + base_test

    smearing = float(np.mean(np.exp(fit_y - model.predict(train_x))))
    return np.expm1(model.predict(test_x)) * smearing + base_test


class _SchedBlend:
    """No-plan predictions softly blended toward MVT_TIME - SCHED_TIME.

    When an airport's system has no actual off-block reading it writes the
    SCHEDULED time into the block field. Measured on 2025: for departures with a
    taxi-out over two hours, the recorded block time equals the scheduled time to
    within a minute in 74% of cases, and 82.5% of the no-flight-plan rows over
    two hours have |y - gap_sched| under 60s. For those rows the target is
    EXACTLY gap_sched -- an identity, not a correlation.

    A tree cannot represent a coefficient-1 dependence on a continuous feature
    except as a staircase, which is why gap_sched being available as a feature
    has never been enough. The identity has to be supplied, the way residualising
    against the NM gap supplied it for the bulk.

    Blended, never switched. A classifier gives p = P(this row is
    schedule-substituted) and the prediction is (1-p)*model + p*gap_sched. A hard
    threshold is brittle; a soft blend degrades gracefully
    when p is wrong, which on a mixture it often is.
    """

    def __init__(self, model, clf, feats, cfeats):
        self._m, self._c, self._f, self._cf = model, clf, feats, cfeats

    def predict(self, frame):
        base = self._m.predict(frame[self._f])
        p = self._c.predict_proba(frame[self._cf])[:, 1]
        # polars Series.to_numpy() takes no dtype kwarg in this version, unlike
        # pandas'. The two identical-looking calls in _fit_lightgbm operate on
        # pandas frames and are fine; these two are polars and raised TypeError
        # on every call, making --sched-blend unusable from either entry point.
        sched = np.nan_to_num(
            frame["gap_sched"].to_numpy().astype(float), nan=0.0)
        return (1.0 - p) * base + p * sched


def fit_sched_blend(train, model, nfeats, ncats, args):
    """Train the substitution classifier on the no-plan training rows."""
    from catboost import CatBoostClassifier, Pool

    sub = train.filter(pl.col("has_flight_plan") == 0)
    y = sub[TARGET].to_numpy().astype(float)
    gs = np.nan_to_num(sub["gap_sched"].to_numpy().astype(float), nan=0.0)
    label = (np.abs(y - gs) <= 60).astype(int)
    if label.sum() < 100 or label.sum() == len(label):
        return model  # nothing to learn from; leave the model alone

    cf = [f for f in nfeats if f != "gap_sched"] + ["gap_sched"]
    ci = [cf.index(c) for c in cf if c in {nfeats[i] for i in ncats}] if ncats else []
    clf = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05, thread_count=args.threads,
        random_seed=1113, verbose=False,
    )
    clf.fit(Pool(sub.select(cf).to_pandas(), label, cat_features=ci))
    print(f"    schedule-substitution classifier: {label.mean()*100:.1f}% positives "
          f"on {len(label):,} no-plan training rows")
    return _SchedBlend(model, clf, nfeats, cf)


def apply_regime_correction(frame: pl.DataFrame, pred: np.ndarray, train_frame: pl.DataFrame,
                            lam: float, quantile: float = 0.99, min_hour_flights: int = 10,
                            max_raise: float = 1500.0):
    """Correct predictions on disruption days from the day's visible taxi regime.

    Returns (new_pred, plan_rows_shifted, noplan_rows_raised, hot_airport_days). lam 0 is a no-op.

    Why. On a disruption day taxi times run far above normal, and a model trained mostly on
    ordinary days shrinks back toward them. The ranking set is retrospective and whole-day, so
    the day's actual regime is visible at serve through other departures' NM taxi proxy,
    gap_aobt = MVT - AOBT_3. On Istanbul's February 2025 snow, held out in fold (2,8), plan rows
    were underpredicted by +400 to +724s on the four worst days and no-plan rows by +1,300 to
    +3,300s, while the label tracked the proxy within a few hundred seconds.

    What it does. Thresholds and offsets come from train_frame only: per airport, the given
    quantile of its daily mean proxy (valid proxies 0..7,200s), and the median of label minus
    proxy over valid rows with labels in (0, 20,000). On frame an airport-day is hot when its mean
    proxy exceeds that quantile. Flight-plan predictions on a hot day shift up by lam times the
    day's mean visible shortfall -- mean(proxy + offset - prediction) over its plan rows with a
    valid proxy -- when that is positive. The shift is per DAY because raising plan rows one by
    one to the proxy made them worse (+1.41s on fold (2,8)): the proxy is too noisy per row.
    No-plan predictions on a hot day, which have no clock of their own, rise to the airport-day-
    hour mean proxy plus the offset when that is higher.

    Safeguard. A no-plan row is raised only when at least min_hour_flights departures support
    its hour average, and by at most max_raise seconds. Without it the 2026 ranking set got 43
    raises above 2,000s and 14 above 3,000s, the largest from hour averages of 2-4 flights, while
    no held-out raise ever exceeded 1,293s (fold (2,8)) or 1,485s (fold (1,7)). The cap sits just
    above the largest raise actually validated, so every raise stays inside tested territory.

    Measured on gate's saved fold predictions, clipped, lam 1, quantile 0.99, with the defaults:
    -3.46s on fold (2,8), the only held-out disruption episode, and +0.03s on fold (1,7), which has
    none (-3.49s and +0.04s without the safeguard). The
    2026 ranking set holds 48 hot airport-days, Schiphol on 3-7 January beyond any 2025 day. One
    held-out episode is thin evidence; docs/LAB-NOTES.md.
    """
    if not lam:
        return pred, 0, 0, 0
    need = ("ADEP_mvt", "MVT_TIME_UTC_mvt", "gap_aobt", "has_flight_plan")
    for f, name in ((frame, "frame"), (train_frame, "train_frame")):
        missing = [c for c in need if c not in f.columns]
        if missing:
            raise SystemExit(f"--regime-correction: {name} lacks {missing}")
    if TARGET not in train_frame.columns:
        raise SystemExit(f"--regime-correction: train_frame lacks {TARGET}")

    col = pl.col
    ok = col("gap_aobt").is_between(0, 7200)
    tr = (train_frame.select("ADEP_mvt", col("MVT_TIME_UTC_mvt").dt.date().alias("_day"), "gap_aobt",
                             col(TARGET).alias("_y")).filter(ok))
    thr = (tr.group_by(["ADEP_mvt", "_day"]).agg(col("gap_aobt").mean().alias("_m"))
             .group_by("ADEP_mvt").agg(col("_m").quantile(quantile).alias("_thr")))
    off = (tr.filter((col("_y") > 0) & (col("_y") < 20000)).group_by("ADEP_mvt")
             .agg((col("_y") - col("gap_aobt")).median().alias("_off")))

    raw_pred = np.asarray(pred, dtype=float)
    base = np.clip(raw_pred, 0, None)
    f = (frame.select("ADEP_mvt", col("MVT_TIME_UTC_mvt").dt.date().alias("_day"),
                      col("MVT_TIME_UTC_mvt").dt.hour().alias("_hh"), "gap_aobt",
                      (col("has_flight_plan") == 1).alias("_plan"))
         .with_columns(pl.Series("_row", np.arange(frame.height)), pl.Series("_p", base)))
    day = f.filter(ok).group_by(["ADEP_mvt", "_day"]).agg(col("gap_aobt").mean().alias("_day_px"))
    hour = f.filter(ok).group_by(["ADEP_mvt", "_day", "_hh"]).agg(
        col("gap_aobt").mean().alias("_hour_px"), pl.len().alias("_hour_n"))
    f = (f.join(day, on=["ADEP_mvt", "_day"], how="left")
          .join(hour, on=["ADEP_mvt", "_day", "_hh"], how="left")
          .join(thr, on="ADEP_mvt", how="left").join(off, on="ADEP_mvt", how="left"))
    short = (f.filter(col("_plan") & ok).group_by(["ADEP_mvt", "_day"])
               .agg((col("gap_aobt") + col("_off") - col("_p")).mean().alias("_short")))
    f = f.join(short, on=["ADEP_mvt", "_day"], how="left").sort("_row")

    hot = (f["_day_px"] > f["_thr"]).fill_null(False).to_numpy()
    plan = f["_plan"].to_numpy()
    shift = np.maximum(f["_short"].fill_null(0.0).to_numpy().astype(float), 0.0)
    target = (f["_hour_px"] + f["_off"]).to_numpy().astype(float)
    hour_n = f["_hour_n"].fill_null(0).to_numpy()
    out = raw_pred.copy()
    mp = hot & plan
    out[mp] = base[mp] + lam * shift[mp]
    mn = hot & ~plan & np.isfinite(target) & (hour_n >= min_hour_flights)
    out[mn] = base[mn] + np.minimum(np.maximum(target[mn] - base[mn], 0.0), max_raise)
    n_days = int(f.filter(pl.Series(hot)).select(["ADEP_mvt", "_day"]).unique().height)
    return out, int((mp & (shift > 0)).sum()), int((mn & (target > base)).sum()), n_days


REGIME_COLUMNS = ("MVT_ID_mvt", "ADEP_mvt", "gap_aobt", "has_flight_plan")


def regime_frame(frame: pl.DataFrame, raw_paths, with_target: bool) -> pl.DataFrame:
    """The slim frame apply_regime_correction reads, with MVT_TIME_UTC_mvt joined back on.

    features.build keeps ids and model columns only, so neither the training cache nor the
    ranking matrix carries the wheels-up timestamp the correction groups by day and hour.
    v18 was produced by a side script on raw frames, so nothing ran this path end to end until
    v19, which trained for 3h30m and then died on "frame lacks ['MVT_TIME_UTC_mvt']" with no
    file written. The join keeps the frame's own row order, because predictions are
    positional, and refuses to return a frame that gained, lost or failed to time a row.
    """
    cols = [*REGIME_COLUMNS, *([TARGET] if with_target else [])]
    slim = frame.select(cols)
    if "MVT_TIME_UTC_mvt" in frame.columns:
        return slim.with_columns(frame["MVT_TIME_UTC_mvt"])
    paths = [str(p) for p in (raw_paths if isinstance(raw_paths, (list, tuple)) else [raw_paths])]
    times = (pl.scan_parquet(paths).filter(pl.col("PHASE_mvt") == "DEP")
               .select("MVT_ID_mvt", "MVT_TIME_UTC_mvt").collect())
    out = (slim.with_row_index("_ord").join(times, on="MVT_ID_mvt", how="left")
               .sort("_ord").drop("_ord"))
    untimed = out["MVT_TIME_UTC_mvt"].null_count()
    if out.height != frame.height or untimed:
        raise SystemExit(f"regime_frame: {out.height:,} rows out for {frame.height:,} in, "
                         f"{untimed:,} without MVT_TIME_UTC_mvt")
    return out


def apply_lirf_sched_floor(frame: pl.DataFrame, pred: np.ndarray, threshold: int):
    """Raise LIRF no-plan predictions to gap_sched where gap_sched > threshold.

    Returns (new_pred, rows_raised). A threshold of 0 is a no-op.

    Why. At LIRF, a no-plan departure whose schedule gap exceeds 60,000s has a
    target that sits near that gap: all 11 such rows of 2025 land within
    30,000s of gap_sched (3 substitutions, where y == gap_sched, and 8 of the
    y ~ 87,400 kind). The specialist undershoots them -- by 14-30k on fold
    (1,7), by 54k on fold (2,8)'s one -- because it has to reach those
    magnitudes from features and shrinks toward the pocket mean. At the nine
    other airports the same rule is catastrophic: 0 of 30 such rows are near
    gap_sched, which there is a stale long-haul schedule.

    Measured on saved out-of-fold predictions, clipped as shipped, max() form,
    threshold 60,000: +4.54s and +16.52s on q_ctrl folds (1,7) and (2,8);
    +5.57s, +0.09s and +0.00s on sp_ctrl folds (1,7), (3,9), (5,11). Never
    negative. Replacing the prediction outright instead read -1.85s and -1.01s
    on (3,9) and (5,11), which is why this only ever raises.

    EXPLORATORY. The threshold and these rows were inspected before the rule
    was measured, and under v16 it touches 2 ranking rows. Treat a fold gain as
    an upper bound.
    """
    if not threshold:
        return pred, 0
    missing = [c for c in ("ADEP_mvt", "has_flight_plan", "gap_sched") if c not in frame.columns]
    if missing:
        raise SystemExit(f"--lirf-sched-floor needs {missing}, which this frame does not carry")
    gs = np.nan_to_num(frame["gap_sched"].to_numpy().astype(float),
                       nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    at = ((frame["has_flight_plan"].to_numpy() == 0)
          & (frame["ADEP_mvt"].to_numpy() == "LIRF") & (gs > threshold))
    out = np.where(at, np.maximum(pred, gs), pred)
    return out, int((out != pred).sum())


class _GateDelay:
    """Predict the GATE DELAY and subtract it from a known quantity.

    The target is `y = MVT - BLOCK`. Both `MVT - SCHED` and `BLOCK - SCHED` are
    differences from the same schedule, so exactly:

        y = D - G        D = MVT - SCHED  (this is `gap_sched`, serve-available)
                         G = BLOCK - SCHED (the training target)

    Verified on all 1,488 LIRF no-plan rows: max |y - (D - G)| = 0.000000.

    Why G is the better thing to regress on. Inside that population y mixes
    three processes with wildly different scales -- an ordinary taxi, a schedule
    substitution, a day rollover -- and spans 68 to 131,167s. G is exactly 0 on
    the 48.5% of rows that are substitutions. sd falls 11,215 -> 5,005.

    CORRECTED 2026-09-15. This used to say G is also "about -86,400 on a
    rollover". It is not: over all 1,488 LIRF no-plan rows of 2025 no G falls
    below -80,000, and the twelve y ~ 87,400 rows sit between -75,474 and
    +6,365. What they share is a gap_sched nearly as large as y, and gate's
    default answer IS gap_sched (|G| < 20,000 on 1,474 of 1,488 rows) -- so it
    helps them by starting from the right magnitude, not by making them a
    constant. The same default is why it hurts ordinary rows, whose gap_sched
    overstates y (median 5,941 against 1,030). Measured in
    compare_gate.py.

    That is also why our substitution classifier captured nothing: it was asked
    to identify a mixture, where this asks the regressor for a target in which
    the substitution class is already a point.

    Found in a public repository by another participant (see the README credits). They tested it at LFPG and rejected it there,
    corr(T,D) being near zero -- matching our own finding that Rome is singular.
    """

    def __init__(self, model, col="gap_sched"):
        self._m, self._col = model, col

    def predict(self, frame):
        d = np.nan_to_num(np.asarray(frame[self._col], dtype=float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        return d - self._m.predict(frame)


class _Bagged:
    """Mean of several fits differing only in seed."""

    def __init__(self, models):
        self._models = models

    def predict(self, frame):
        return np.mean([m.predict(frame) for m in self._models], axis=0)


def _fit_one_noplan(sub, nfeats, ncats, args, weights=None):
    from catboost import CatBoostRegressor, Pool

    y = sub[TARGET].to_numpy().astype(float)
    mode = getattr(args, "noplan_target", "raw")
    log_scale = mode == "log"
    gate = mode == "gate"
    if gate:
        # G = D - y, with D = gap_sched. Rows without a schedule gap cannot be
        # expressed this way; they keep a zero gate, so the model predicts D.
        d = np.nan_to_num(sub["gap_sched"].to_numpy().astype(float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        fit_y = d - y
    else:
        fit_y = np.log1p(np.maximum(y, 0.0)) if log_scale else y
    x = sub.select(nfeats).to_pandas()
    # Variance knobs, all at CatBoost defaults until now. A residual autopsy on
    # real out-of-fold predictions found this group's error is 92% instability
    # (seeds disagree) against 7% bias (seeds agree and are wrong), with
    # log-corr(seed variance, squared error) 0.37-0.44 across all three folds --
    # two to three times the bulk's. The 2026-09-06 grid varied only depth,
    # iterations and feature set, never regularisation, and it ran before
    # routing existed, so it tuned ONE model on 22,470 rows. LIRF's specialist
    # now fits about 1,200, with min_data_in_leaf at CatBoost's default of 1.
    kw = dict(
        iterations=args.noplan_iterations, depth=getattr(args, "noplan_depth", 6),
        learning_rate=0.05, loss_function="RMSE",
        thread_count=args.threads, verbose=False,
    )
    l2 = getattr(args, "noplan_l2", None)
    if l2 is not None:
        kw["l2_leaf_reg"] = l2
    # min_data_in_leaf is IGNORED under CatBoost's default SymmetricTree policy
    # -- verified: minleaf_20 returned a bit-identical 1926.2 to the control on
    # fold (1,7). It needs Depthwise or Lossguide. That pairing is worth having
    # for a ~1,200-row specialist even though Lossguide lost 0/3 on the GLOBAL
    # model, because the constraint that hurt 2M rows is the one that helps
    # 1,200.
    grow = getattr(args, "noplan_grow_policy", "SymmetricTree")
    if grow != "SymmetricTree":
        kw["grow_policy"] = grow
    mdl = getattr(args, "noplan_min_data", None)
    if mdl:
        if grow == "SymmetricTree":
            raise SystemExit(
                "--noplan-min-data needs --noplan-grow-policy Depthwise or "
                "Lossguide; CatBoost silently ignores it on symmetric trees")
        kw["min_data_in_leaf"] = mdl
    rsm = getattr(args, "noplan_rsm", None)
    if rsm:
        kw["rsm"] = rsm

    # Bag the specialist. Instability is what dominates here, and averaging is
    # the direct answer to variance -- but the same autopsy bounds 3 -> infinity
    # seeds at about 10s on this group, so this is a small lever, not the lever.
    nseeds = max(int(getattr(args, "noplan_seeds", 1) or 1), 1)
    models = []
    for k in range(nseeds):
        m = CatBoostRegressor(random_seed=1113 + k * 977, **kw)
        m.fit(Pool(x, fit_y, cat_features=ncats, weight=weights))
        models.append(m)
    model = models[0] if nseeds == 1 else _Bagged(models)
    if gate:
        return _GateDelay(model)
    if not log_scale:
        return model
    smear = float(np.mean(np.exp(fit_y - model.predict(x))))
    raw = model.predict

    class _Smeared:
        @staticmethod
        def predict(frame):
            return np.expm1(raw(frame)) * smear

    return _Smeared


# Every setting fit_noplan honours. Kept explicit because the failure mode is
# silence: a caller that hands over an object missing one of these gets the
# getattr default and no warning, which is how --noplan-split-lirf came to parse
# in final.py, print nothing, and never route a single row. The 5/6-fold win it
# names could not reach a submission for two days.
NOPLAN_SETTINGS = (
    "noplan_iterations", "noplan_depth", "threads",
    "noplan_routes", "noplan_split_lirf", "noplan_target",
    "noplan_train", "noplan_weight", "sched_blend",
    "noplan_l2", "noplan_min_data", "noplan_rsm", "noplan_seeds",
    "noplan_grow_policy", "noplan_gate_routes", "noplan_clip_train",
)


def require_noplan_settings(args) -> None:
    missing = [k for k in NOPLAN_SETTINGS if not hasattr(args, k)]
    if missing:
        raise SystemExit(
            "fit_noplan was handed an args object missing " + repr(missing) + "; "
            "it would silently use defaults for them. Pass the real parsed args, "
            "or add these to whatever stand-in is being used."
        )


def noplan_routes(args) -> list[str]:
    """Which airports get their own no-plan model.

    One reader for both --noplan-split-lirf (kept: it names a shipped result)
    and --noplan-routes. crossval and final have drifted apart four times over
    exactly this kind of setting, so they share this function rather than each
    parsing the flags.
    """
    explicit = getattr(args, "noplan_routes", "") or ""
    codes = [c.strip().upper() for c in explicit.split(",") if c.strip()]
    if codes:
        return codes
    return ["LIRF"] if getattr(args, "noplan_split_lirf", False) else []


class _RoutedNoPlan:
    """Two no-plan models, routed on whether the airport is LIRF.

    Inside the no-flight-plan group the two halves are barely the same problem:
    at LIRF the mean is 6,531s and essentially every row also lacks an aircraft
    type, while everywhere else the mean is 1,019s against a global 991s -- an
    ordinary departure that happens to be missing its flight plan.

    An earlier test showed a per-half CONSTANT loses badly to the single model
    (3,722 against 2,154), which is why this was left alone. But that measured
    whether a constant beats a model, not whether two models beat one, and those
    are different questions. Routing is on an airport code fixed in advance, not
    chosen by score, so it is not the per-subgroup selection that cost a 2024
    team 2,066 -> 2,276.

    Generalised on 2026-09-09 from a hard-coded LIRF flag to a list of airports,
    each getting its own model. A floor analysis of the scored set found that
    LFPG and LSZH no-plan predictions are dispersed at less than half the
    residual sd achievable within tight strata (687 against a 1,370 floor at
    LFPG; 522 against 1,075 at LSZH), while LIRF -- the one already routed --
    is dispersed correctly. Routing is on airport codes fixed before the fit,
    never chosen by score.
    """

    def __init__(self, models: dict, other, nfeats):
        self._models, self._other, self._nfeats = models, other, nfeats

    def predict(self, frame):
        # frame is a pandas DataFrame carrying ADEP_mvt among nfeats
        apt = frame["ADEP_mvt"].to_numpy()
        out = self._other.predict(frame)
        for code, model in self._models.items():
            if model is None:
                continue
            m = apt == code
            if m.any():
                out = np.where(m, model.predict(frame), out)
        return out


def fit_noplan(train, feats, cat_features, args):
    """A model for rows with no flight plan, on the columns they actually have.

    ``args.noplan_train`` picks what it learns from:

    ``group``   only the ~22k no-flight-plan rows. What v5 shipped.
    ``all``     every departure, with the flight-plan-derived columns removed.
                The feature vector is identical either way -- stand, runway,
                time, congestion, weather all exist for all 2.08M rows -- so
                restricting to 22k throws away two orders of magnitude of data
                about how those features map to taxi time, for no reason beyond
                how the split was first written.
    ``weighted`` same as ``all`` but with no-flight-plan rows upweighted, so the
                extra data informs the shape without drowning the population we
                actually predict.

    This is not per-subgroup model *selection* -- the thing that cost a 2024
    team 2,066 -> 2,276. Nothing is chosen by score; the group is defined by
    which columns exist, fixed before measurement, and both halves scored
    together on folds not used for training.
    """
    require_noplan_settings(args)
    from catboost import CatBoostRegressor, Pool

    mode = getattr(args, "noplan_train", "group")
    nfeats = [f for f in feats if f not in NOPLAN_DROP]
    ncats = [nfeats.index(c) for c in cat_features if c in nfeats]

    if mode == "group":
        sub = train.filter(pl.col("has_flight_plan") == 0)
        weights = None
    else:
        sub = train
        if mode == "weighted":
            w = getattr(args, "noplan_weight", 20.0)
            weights = np.where(sub["has_flight_plan"].to_numpy() == 0, w, 1.0)
        else:
            weights = None

    # Drop extreme labels from the TRAINING set only. 76 departures in all of
    # 2025 carry y > 20,000s, five and a half hours of taxi. They are not long
    # taxis: 91-96% of them have BLOCK == SCHED, meaning no off-block was
    # recorded and the schedule was substituted, so the "taxi time" is really
    # how late the aircraft was against a stale schedule. 74 of the 76 are
    # no-flight-plan rows, so in this model's ~22k training rows they are 74
    # labels whose residual is tens of thousands of seconds against a typical
    # 200s. Under an RMSE gradient each is worth hundreds of ordinary rows.
    #
    # UNTESTED AND PROBABLY WRONG TO SHIP. This flag exists to measure that
    # trade, not because the class is safe to discard: the scored months have
    # a larger share of these rows than our folds do, not a smaller one. Measured
    # directly on gap_sched, which is readable for departures in ranking.parquet,
    # the 2026 scored months hold more no-plan rows with gap_sched >= 7,200s than
    # Jan+Jul 2025, and about twice as many above 20,000s. Clipping them out of
    # training buys accuracy on the 95% at the cost of the 5%, and which way
    # that nets is an empirical question. Filtering is training-side only and
    # uses no held-out information either way.
    clip = getattr(args, "noplan_clip_train", 0) or 0
    if clip:
        sub = sub.filter(pl.col(TARGET) < clip)
        if weights is not None:
            weights = np.where(sub["has_flight_plan"].to_numpy() == 0,
                               getattr(args, "noplan_weight", 20.0), 1.0)

    y = sub[TARGET].to_numpy().astype(float)
    # The global --target flag never reached this model, so a log scale was
    # rejected for the bulk while going untested where it is most motivated:
    # this group's wild half has a standard deviation of 11,166s.
    log_scale = getattr(args, "noplan_target", "raw") == "log"
    fit_y = np.log1p(np.maximum(y, 0.0)) if log_scale else y

    x = sub.select(nfeats).to_pandas()
    model = CatBoostRegressor(
        iterations=args.noplan_iterations, depth=getattr(args, "noplan_depth", 6),
        learning_rate=0.05, loss_function="RMSE",
        thread_count=args.threads, random_seed=1113, verbose=False,
    )
    if getattr(args, "sched_blend", False):
        model.fit(Pool(x, fit_y, cat_features=ncats, weight=weights))
        return fit_sched_blend(train, model, nfeats, ncats, args), nfeats

    routes = noplan_routes(args)
    if routes:
        models, sizes = {}, []
        # The gate reparameterisation applies ONLY where it is named, because
        # applying it everywhere failed by +667s. It needs gap_sched to be a
        # live schedule; at the nine other airports it is often a stale
        # long-haul one, and y = gap_sched - G is then meaningless. The team
        # whose live score it moved (316.97 -> 288.90) applied it at Rome alone.
        gate_at = {c.strip().upper()
                   for c in (getattr(args, "noplan_gate_routes", "") or "").split(",")
                   if c.strip()}
        for code in routes:
            rows = sub.filter(pl.col("ADEP_mvt") == code)
            a = args
            if gate_at:
                a = SimpleNamespace(**{**vars(args)} if hasattr(args, "__dict__")
                                    else {k: getattr(args, k) for k in NOPLAN_SETTINGS})
                a.noplan_target = "gate" if code in gate_at else "raw"
            models[code] = (_fit_one_noplan(rows, nfeats, ncats, a)
                            if rows.height > 200 else None)
            sizes.append(f"{code} {rows.height:,}"
                         + (" [gate]" if code in gate_at else "")
                         + ("" if models[code] else " (too thin)"))
        other_rows = sub.filter(~pl.col("ADEP_mvt").is_in(list(routes)))
        a_other = args
        if gate_at:
            a_other = SimpleNamespace(**{**vars(args)} if hasattr(args, "__dict__")
                                      else {k: getattr(args, k) for k in NOPLAN_SETTINGS})
            a_other.noplan_target = "raw"
        other = _fit_one_noplan(other_rows, nfeats, ncats, a_other)
        print(f"    no-plan routed: {', '.join(sizes)}, other {other_rows.height:,}")
        return _RoutedNoPlan(models, other, nfeats), nfeats

    model.fit(Pool(x, fit_y, cat_features=ncats, weight=weights))
    if not log_scale:
        return model, nfeats

    smear = float(np.mean(np.exp(fit_y - model.predict(x))))
    raw_predict = model.predict

    class _Smeared:  # same interface, seconds scale out
        feature_names_ = getattr(model, "feature_names_", None)

        @staticmethod
        def predict(frame):
            return np.expm1(raw_predict(frame)) * smear

    return _Smeared, nfeats


def run_fold(frame, held: tuple[int, int], feats, cat_idx, args) -> dict:
    from catboost import CatBoostRegressor, Pool

    from . import reference

    month = pl.col("month")
    # Holding out whole months means fold (1,7) never sees a single January or
    # July row, while the submitted model trains on all twelve. final.py's own
    # docstring has said so since it was written -- "we were withholding the only
    # January and July the model could learn from, 24.3% of the data" -- and that
    # is the most likely cause of the ~-35s gap between what fold (1,7) reads and
    # what the board returns, which has held across four submissions.
    #
    # --holdout-days N holds out every Nth DAY inside the fold months instead, so
    # the model trains on the rest of those months and sees the season it is
    # being asked to predict. If that reads near the board, the offset is season
    # exposure and our validation becomes calibrated rather than merely ordinal.
    nth = int(getattr(args, "holdout_days", 0) or 0)
    if nth > 1:
        dom = pl.col("MVT_TIME_UTC_mvt").dt.day() if "MVT_TIME_UTC_mvt" in frame.columns             else pl.col("doy")
        in_months = month.is_in(held)
        is_test = in_months & (dom % nth == 0)
        train = frame.filter(~is_test)
        test = frame.filter(is_test)
        print(f"    --holdout-days {nth}: testing on every {nth}th day of {held}; "
              f"train {train.height:,} (includes {train.filter(in_months).height:,} "
              f"rows from those months), test {test.height:,}")
    else:
        train = frame.filter(~month.is_in(held))
        test = frame.filter(month.is_in(held))
    train, test = reference.attach(train, test)
    if getattr(args, "drop_dayoffset", False):
        before = train.height
        train = drop_day_offsets(train)
        print(f"    dropped {before - train.height} confirmed day-offset training rows")

    y_train = train[TARGET].to_numpy().astype(float)
    if args.winsor:
        y_train = np.minimum(y_train, args.winsor)

    train_pool = Pool(train.select(feats).to_pandas(), y_train, cat_features=cat_idx)
    test_x = test.select(feats).to_pandas()
    y = test[TARGET].to_numpy().astype(float)
    plan = test["has_flight_plan"].to_numpy() == 1

    started = time.time()
    preds = [_fit_predict(train, test, test_x, feats, cat_idx, y_train, args, s)
             for s in range(args.seeds)]
    stack = np.vstack(preds)
    bagged = stack.mean(axis=0)

    # BLEND MEMBERS, mirroring prc.final's shipped 3-way blend bit for bit
    # (prc/final.py ~lines 457-528: the `--blend-lightgbm > 0` block). Added
    # 2026-09-22 for the "members" full-scale fold test -- this harness never
    # implemented final.py's blend before, so no prior fold number here ever
    # measured the shipped model. At the DEFAULTS (blend_lightgbm=0.0), none
    # of this runs and `bagged` is untouched, i.e. today's behaviour.
    #
    # `cat_bulk_pred` is the routed+bagged CatBoost bulk member on its own --
    # exactly what `bagged` already holds at this point -- and is always
    # available for --save-fold-members, independent of the blend flags.
    cat_bulk_pred = bagged.copy()

    blend_w = float(getattr(args, "blend_lightgbm", 0.0) or 0.0)
    p3 = lpred = huber_lpred = lpred_routed = None
    if blend_w > 0.0:
        # Recomputed here rather than threaded out of _fit_predict, the same
        # way prc.final computes train_x/fit_y/base_rank ONCE, above its own
        # seed loop, and reuses them for every member. prepare_residual is a
        # pure function of (train, y_train, args, test) -- no randomness --
        # so recomputing it gives numerically identical values to what each
        # _fit_predict(..., s) call above already computed internally.
        train_bulk, fit_y, base_test_v = prepare_residual(train, y_train, args, apply_to=test)
        train_x = train_bulk.select(feats).to_pandas()
        cat_names = [feats[i] for i in cat_idx]
        sw = month_weights(train_bulk, getattr(args, "month_weight", ""))

        # Twin of prc.final's route() closure: per-airport CatBoost bulk
        # models for the depth-N third member, same 5,000-row floor, same
        # ALL expansion. The shipped LightGBM member stays global -- never
        # routed, in final.py or here.
        routes = [c.strip().upper() for c in
                  (getattr(args, "bulk_routes", "") or "").split(",") if c.strip()]
        tr_apt = train_bulk["ADEP_mvt"].to_numpy()
        te_apt = test["ADEP_mvt"].to_numpy()
        if routes == ["ALL"]:
            routes = sorted(set(tr_apt) & set(te_apt))

        def route_member(make_model, global_pred, weight):
            if not routes:
                return global_pred
            out, sizes = global_pred.copy(), []
            for code in routes:
                tr_m, te_m = tr_apt == code, te_apt == code
                if tr_m.sum() < 5000 or te_m.sum() == 0:
                    sizes.append(f"{code} skipped")
                    continue
                sub = make_model()
                sub.fit(Pool(train_x[tr_m], fit_y[tr_m], cat_features=cat_idx,
                             weight=None if weight is None else weight[tr_m]))
                out[te_m] = sub.predict(test_x[te_m])
                sizes.append(f"{code} {int(tr_m.sum()):,}")
            print("    bulk routing (blend member): " + "  ".join(sizes))
            return out

        _lp = []
        for _ls in range(max(1, int(getattr(args, "lgb_seeds", 1)))):
            # objective="regression" EXPLICIT here, regardless of what
            # --lgb-objective says -- this is the shipped member prc.final's
            # --blend-lightgbm fits, and --lgb-objective huber (below) must
            # only add a SEPARATE member, never silently change this one.
            lmodel, _, ltest_x = _fit_lightgbm(
                train_x.copy(), fit_y, test_x.copy(), cat_names, args, _ls,
                objective="regression")
            _lp.append(lmodel.predict(ltest_x) + base_test_v)
        lpred = np.mean(_lp, axis=0)
        if len(_lp) > 1:
            print(f"    LightGBM bagged over {len(_lp)} seeds")
        cpred = cat_bulk_pred

        # ROUTED LightGBM (crossval-only candidate, 2026-09-22). The two CatBoost
        # members are routed per airport; the LightGBM member never has been.
        # Same routes, same 5,000-row floor, same seeds as the shipped member;
        # the global seed-k prediction is kept for any airport that is skipped.
        # Saved as `lightgbm_routed` and NEVER blended into `bagged`.
        if getattr(args, "lgb_routes", False) and routes:
            _bt = np.broadcast_to(np.asarray(base_test_v, dtype=float), (len(te_apt),))
            _lr = []
            for _ls, _glob in enumerate(_lp):
                out, sizes = np.array(_glob, dtype=float), []
                for code in routes:
                    tr_m, te_m = tr_apt == code, te_apt == code
                    if tr_m.sum() < 5000 or te_m.sum() == 0:
                        sizes.append(f"{code} skipped")
                        continue
                    rmodel, _, rtest_x = _fit_lightgbm(
                        train_x[tr_m].reset_index(drop=True), fit_y[tr_m],
                        test_x[te_m].reset_index(drop=True), cat_names, args, _ls,
                        objective="regression")
                    out[te_m] = rmodel.predict(rtest_x) + _bt[te_m]
                    sizes.append(f"{code} {int(tr_m.sum()):,}")
                _lr.append(out)
                print(f"    LightGBM routing (seed {_ls}): " + "  ".join(sizes))
            lpred_routed = np.mean(_lr, axis=0)
            print(f"    routed LightGBM mean {lpred_routed.mean():,.1f}s "
                  f"(global {lpred.mean():,.1f}s)")

        d3 = int(getattr(args, "blend_cat_depth", 0) or 0)
        if d3:
            print(f"  third member: CatBoost depth {d3}, equal thirds")
            _p3 = []
            for _ds in range(max(1, int(getattr(args, "d3_seeds", 1)))):
                _rs = 1113 + _ds
                m3 = CatBoostRegressor(
                    iterations=args.iterations, depth=d3, learning_rate=args.lr,
                    loss_function=args.loss, thread_count=args.threads,
                    l2_leaf_reg=args.l2, one_hot_max_size=args.one_hot_max_size,
                    random_seed=_rs, verbose=250)
                m3.fit(Pool(train_x, fit_y, cat_features=cat_idx, weight=sw))

                def make_d3_model(_rs=_rs):
                    return CatBoostRegressor(
                        iterations=args.iterations, depth=d3, learning_rate=args.lr,
                        loss_function=args.loss, thread_count=args.threads,
                        l2_leaf_reg=args.l2, one_hot_max_size=args.one_hot_max_size,
                        random_seed=_rs, verbose=0)
                _p3.append(route_member(make_d3_model, m3.predict(test_x), sw) + base_test_v)
            p3 = np.mean(_p3, axis=0)
            if len(_p3) > 1:
                print(f"    depth-{d3} member bagged over {len(_p3)} seeds")
            blended = (cpred + p3 + lpred) / 3.0
            print(f"    catboost {cpred.mean():,.1f}s  cat_d{d3} {p3.mean():,.1f}s  "
                  f"lightgbm {lpred.mean():,.1f}s  blended {blended.mean():,.1f}s")
        else:
            blended = blend_w * cpred + (1.0 - blend_w) * lpred
            print(f"    catboost mean {cpred.mean():,.1f}s  lightgbm mean "
                  f"{lpred.mean():,.1f}s  blended {blended.mean():,.1f}s")
        bagged = blended

        # Huber-objective LightGBM: crossval-only, no prc/final.py equivalent
        # (final.py has no --lgb-objective flag at all). Fit from the SAME
        # train_x/fit_y/test_x as the shipped (regression-objective) member
        # above, bagged over the same --lgb-seeds, so it is comparable to it.
        # NEVER blended into `bagged` -- it exists only to be saved by
        # --save-fold-members as a candidate 4th member or LightGBM
        # replacement for archive/agent_members_blend_eval.py to score.
        if getattr(args, "lgb_objective", "regression") == "huber":
            _hlp = []
            for _ls in range(max(1, int(getattr(args, "lgb_seeds", 1)))):
                hmodel, _, htest_x = _fit_lightgbm(
                    train_x.copy(), fit_y, test_x.copy(), cat_names, args, _ls,
                    objective="huber",
                    huber_alpha=getattr(args, "lgb_huber_alpha", None))
                _hlp.append(hmodel.predict(htest_x) + base_test_v)
            huber_lpred = np.mean(_hlp, axis=0)
            if len(_hlp) > 1:
                print(f"    Huber-LightGBM bagged over {len(_hlp)} seeds")
            print(f"    Huber-LightGBM (alpha={getattr(args, 'lgb_huber_alpha', None)}) "
                  f"mean {huber_lpred.mean():,.1f}s")

    if getattr(args, "save_fold_members", ""):
        # Twin of --save-preds below, but for the blend MEMBERS rather than
        # the final bagged/routed prediction -- pre-noplan, pre-lirf-floor,
        # pre-regime-correction, matching what prc.final's blend members
        # represent (final.py substitutes the no-plan model into the BLENDED
        # `bagged`, never into a member on its own).
        member_cols = {
            "MVT_ID_mvt": test["MVT_ID_mvt"].to_numpy(),
            "ADEP_mvt": test["ADEP_mvt"].to_numpy(),
            "y": y, "has_flight_plan": plan.astype(np.int8),
            "cat_bulk": cat_bulk_pred,
        }
        if p3 is not None:
            member_cols["cat_d3"] = p3
        if lpred is not None:
            member_cols["lightgbm"] = lpred
        if huber_lpred is not None:
            member_cols["lightgbm_huber"] = huber_lpred
        if lpred_routed is not None:
            member_cols["lightgbm_routed"] = lpred_routed
        mbase = Path(args.save_fold_members)
        mpath = mbase.with_name(f"{mbase.stem}_{held[0]}_{held[1]}{mbase.suffix}")
        mpath.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(member_cols).write_parquet(mpath)
        print(f"    saved fold members ({sorted(member_cols)}) -> {mpath}")

    if args.noplan_model:
        cat_names = [feats[i] for i in cat_idx]
        nmodel, nfeats = fit_noplan(train, feats, cat_names, args)
        npred = nmodel.predict(test.select(nfeats).to_pandas())
        split = np.where(plan, bagged, npred)
        print(f"    dedicated no-plan model: global {rmse(y[~plan], bagged[~plan]):8.1f} -> "
              f"{rmse(y[~plan], npred[~plan]):8.1f} on {int((~plan).sum()):,} rows | "
              f"overall {rmse(y, bagged):7.2f} -> {rmse(y, split):7.2f}")
        bagged_global = bagged
        bagged = split

    floor_s = getattr(args, "lirf_sched_floor", 0) or 0
    if floor_s:
        before = rmse(y, bagged)
        bagged, n_raised = apply_lirf_sched_floor(test, bagged, floor_s)
        print(f"    --lirf-sched-floor {floor_s:,}: raised {n_raised} predictions, "
              f"overall {before:7.2f} -> {rmse(y, bagged):7.2f}")

    lam = getattr(args, "regime_correction", 0.0) or 0.0
    if lam:
        before = rmse(y, bagged)
        from .config import RAW_DIR
        raw = sorted(RAW_DIR.glob("training_*.parquet"))
        bagged, n_ps, n_nr, n_days = apply_regime_correction(
            regime_frame(test, raw, with_target=False), bagged,
            regime_frame(train, raw, with_target=True), lam, getattr(args, "regime_quantile", 0.99))
        print(f"    --regime-correction {lam}: {n_days} hot airport-days, {n_ps:,} plan rows shifted, "
              f"{n_nr:,} no-plan rows raised, overall {before:7.2f} -> {rmse(y, bagged):7.2f}")

    core = y <= CORE_MAX_Y
    out = {
        "fold": list(held), "n": int(len(y)), "seeds": args.seeds,
        "rmse": rmse(y, preds[0]),
        "rmse_bagged": rmse(y, bagged),
        # What the submission actually ships: final.py clips at 0 and this file
        # never did, so every fold number before 2026-09-15 is unclipped. On
        # q_ctrl fold (1,7) that is 0.80s, and it read the gate arm as +2.93s
        # where the like-for-like figure was +5.31s. Kept beside rmse_bagged
        # rather than replacing it, so history stays comparable.
        "rmse_clipped": rmse(y, np.clip(bagged, 0.0, None)),
        # Same predictions, same clip, scored on the rows a model can actually
        # reach (see CORE_MAX_Y). Low variance, so small true effects are visible.
        "rmse_core": rmse(y[core], np.clip(bagged[core], 0.0, None)),
        "n_core": int(core.sum()),
        "rmse_bulk": rmse(y[plan], bagged[plan]),
        "rmse_noplan": rmse(y[~plan], bagged[~plan]) if (~plan).any() else None,
        "seconds": round(time.time() - started),
    }
    # NOT a bagging gain. `preds[0]` is the bulk model over EVERY row, including
    # the no-plan rows it was never meant to serve; `bagged` is taken after the
    # specialist has been substituted in. So the difference is dominated by the
    # routing -- 572 -> 333 on fold (1,7), where bagging three seeds is worth
    # about 3. Bagging is only separable on plan rows, where no substitution
    # happens; `rmse_bulk` is the number to watch.
    gain = out["rmse"] - out["rmse_bagged"]
    print(
        f"  fold {held}: bulk-only-everywhere {out['rmse']:8.2f}  routed+bagged({args.seeds}) "
        f"{out['rmse_bagged']:8.2f}  routing+bagging {gain:+6.2f}  "
        f"bulk {out['rmse_bulk']:7.2f}  clipped-as-shipped {out['rmse_clipped']:8.2f}  "
        f"core {out['rmse_core']:7.2f} on {out['n_core']:,} ({int(len(y)) - out['n_core']} cut)  "
        f"({out['seconds']}s)"
    )

    if args.save_preds:
        # Carry the row id, the airport and the timestamp. Without them the file
        # cannot be joined to anything and every later analysis has to guess at
        # row order. Twice this week I estimated headroom against a strawman
        # predictor -- a per-airport mean, then a raw timestamp gap -- because
        # the model's own out-of-fold predictions were not joinable. Both
        # estimates were near-zero in reality.
        cols = {
            "MVT_ID_mvt": test["MVT_ID_mvt"].to_numpy(),
            "ADEP_mvt": test["ADEP_mvt"].to_numpy(),
            "y": y, "has_flight_plan": plan.astype(np.int8), "bagged": bagged,
        }
        cols.update({f"seed{i}": p for i, p in enumerate(preds)})
        path = Path(f"results/preds_{args.tag}_{held[0]}_{held[1]}.parquet")
        path.parent.mkdir(exist_ok=True)
        pl.DataFrame(cols).write_parquet(path)
        print(f"    saved {path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--winsor", type=float, default=0.0)
    parser.add_argument("--drop", default="")
    parser.add_argument("--baseline", choices=["aobt", "blend", "blend_apt", "blend_apt_hour", "lobt_switch"], default="aobt")
    parser.add_argument("--residual", action="store_true",
                        help="model the correction to the NM baseline, not the target")
    parser.add_argument("--loss", default="RMSE", help='e.g. "Huber:delta=2000"')
    parser.add_argument("--one-hot-max-size", type=int, default=0)
    parser.add_argument("--drop-dayoffset", action="store_true")
    parser.add_argument("--holdout-days", type=int, default=0,
                        help="hold out every Nth day INSIDE the fold months "
                             "instead of the whole months, so the model sees "
                             "the season it is scored on")
    parser.add_argument("--target", choices=["raw", "log"], default="raw")
    parser.add_argument("--noplan-model", action="store_true")
    parser.add_argument("--noplan-split-lirf", action="store_true")
    parser.add_argument("--bulk-plan-only", action="store_true",
                        help="train the global model only on rows that have a "
                             "flight plan -- its predictions on the others are "
                             "discarded anyway")
    parser.add_argument("--noplan-clip-train", type=int, default=0,
                        help="drop training rows with target >= this many "
                             "seconds from the no-plan specialist. 0 disables. "
                             "20000 removes the corrupt-BLOCK class (76 rows in "
                             "all of 2025, 74 of them no-plan) that cannot be "
                             "in the scored set")
    parser.add_argument("--regime-correction", type=float, default=0.0,
                        help="on disruption days, shift plan predictions by this fraction of the "
                             "day's visible taxi shortfall and raise no-plan ones to the hour's "
                             "regime; 0 = off. See crossval.apply_regime_correction")
    parser.add_argument("--regime-quantile", type=float, default=0.99,
                        help="an airport-day counts as a disruption day above this quantile of its "
                             "airport's daily mean NM taxi proxy in training")
    parser.add_argument("--lirf-sched-floor", type=int, default=0,
                        help="raise LIRF no-plan predictions to gap_sched where gap_sched "
                             "exceeds this many seconds; 0 = off. Exploratory, see "
                             "crossval.apply_lirf_sched_floor")
    parser.add_argument("--noplan-gate-routes", default="",
                        help="airports whose no-plan model regresses the GATE "
                             "DELAY instead of the target; needs gap_sched to be "
                             "a live schedule, so LIRF and not much else")
    parser.add_argument("--noplan-routes", default="",
                        help="comma-separated ICAO codes that each get their own "
                             "no-plan model, e.g. LIRF,LFPG,LSZH; overrides "
                             "--noplan-split-lirf")
    parser.add_argument("--sched-blend", action="store_true",
                        help="blend no-plan predictions toward MVT-SCHED where substitution is likely")
    parser.add_argument("--noplan-target", choices=["raw", "log", "gate"],
                        default="raw",
                        help="gate: regress BLOCK-SCHED and subtract it from "
                             "gap_sched, so substitutions and rollovers become "
                             "constants in the target")
    parser.add_argument("--noplan-train", choices=["group", "all", "weighted"], default="group")
    parser.add_argument("--noplan-weight", type=float, default=20.0)
    parser.add_argument("--noplan-iterations", type=int, default=600)
    # final.py has always had this; crossval read it off args and got the
    # getattr default instead. Harmless until require_noplan_settings made the
    # mismatch fatal -- which is the guard doing its job, one caller late.
    parser.add_argument("--noplan-depth", type=int, default=6)
    parser.add_argument("--noplan-l2", type=float, default=None,
                        help="l2_leaf_reg for the no-plan specialist; CatBoost "
                             "defaults to 3.0 and this has never been varied")
    parser.add_argument("--noplan-min-data", type=int, default=0,
                        help="min_data_in_leaf; CatBoost defaults to 1, which on "
                             "a ~1,200-row specialist allows single-row leaves")
    parser.add_argument("--noplan-rsm", type=float, default=None)
    parser.add_argument("--noplan-grow-policy",
                        choices=["SymmetricTree", "Depthwise", "Lossguide"],
                        default="SymmetricTree")
    parser.add_argument("--noplan-seeds", type=int, default=1,
                        help="bag the specialist over this many seeds")
    parser.add_argument("--grow-policy",
                        choices=["SymmetricTree", "Depthwise", "Lossguide"],
                        default="SymmetricTree",
                        help="CatBoost tree growth; the default grows oblivious "
                             "trees, which cannot spend a split on one airport")
    parser.add_argument("--max-leaves", type=int, default=64,
                        help="Lossguide only")
    # The LightGBM path has existed since 2026-09-11 and _fit_predict dispatches
    # on args.learner, but the flag was only ever added to prc/final.py -- which
    # cannot execute it. So crossval could run LightGBM and had no way to ask,
    # while final could ask and had no way to run. A Modal arm died on
    # "unrecognized arguments: --learner lightgbm" before this was noticed.
    # Fifth instance in this project of a capability and its switch living in
    # different files; leakcheck.parser_parity() compares flag NAMES across the
    # two parsers and is blind to every one of them.
    parser.add_argument("--learner", choices=["catboost", "lightgbm"],
                        default="catboost",
                        help="which gradient booster fits the bulk model")
    parser.add_argument("--linear-tree", action="store_true",
                        help="lightgbm only: a ridge regression in each leaf")
    parser.add_argument("--num-leaves", type=int, default=255, help="lightgbm only")
    parser.add_argument("--min-child-samples", type=int, default=40, help="lightgbm only")
    parser.add_argument("--linear-lambda", type=float, default=1.0, help="lightgbm only")
    parser.add_argument("--max-ctr-complexity", type=int, default=None,
                        help="how many categoricals CatBoost combines in target "
                             "statistics (default 4); never tested here")
    parser.add_argument("--rsm", type=float, default=None,
                        help="feature subsample per split; never tested here")
    parser.add_argument("--bootstrap-type", default=None,
                        choices=["Bayesian", "Bernoulli", "MVS", "No"])
    parser.add_argument("--subsample", type=float, default=None,
                        help="row subsample; needs --bootstrap-type Bernoulli")
    parser.add_argument("--random-strength", type=float, default=None,
                        help="noise on split scores; the standard lever against "
                             "overfitting 1,899-level categoricals")
    parser.add_argument("--season-split", action="store_true",
                        help="two bulk models, winter (Nov-Mar) and summer "
                             "(Apr-Oct). January bulk RMSE is 206 against July's "
                             "249 -- arguably two different problems")
    parser.add_argument("--bulk-routes", default="",
                        help="airports getting their own BULK model, or ALL. "
                             "Routing is fixed before the fit, never chosen by "
                             "score")
    parser.add_argument("--month-weight", default="",
                        help="per-month training weights, e.g. 12:3,2:3,6:3,8:3 "
                             "-- everything unnamed stays at 1")
    # BLEND MEMBERS. Mirrors prc/final.py's --blend-lightgbm / --blend-cat-depth
    # / --lgb-seeds / --d3-seeds EXACTLY (same code path, same seed formulas,
    # same routing, same features -- see run_fold). This harness fit only the
    # single bulk learner before 2026-09-22, so no fold number on record before
    # then measured the shipped 3-member blend. All four default to what
    # final.py calls "SHIPPED" / disabled, so omitting them is bit-identical
    # to every fold run before this change.
    parser.add_argument("--blend-lightgbm", type=float, default=0.0,
                        help="mirrors prc.final --blend-lightgbm: weight on the "
                             "CatBoost bulk member in a CatBoost/LightGBM blend "
                             "(equal thirds instead when --blend-cat-depth is "
                             "also set, and then the weight VALUE is unused, "
                             "same as final.py); 0 disables the whole blend")
    parser.add_argument("--blend-cat-depth", type=int, default=0,
                        help="mirrors prc.final --blend-cat-depth: a THIRD "
                             "blend member, CatBoost at this depth, equal "
                             "thirds with the bulk and LightGBM members. Only "
                             "takes effect when --blend-lightgbm is ALSO > 0 "
                             "-- final.py nests it inside that guard and this "
                             "mirrors it exactly rather than 'fixing' it. "
                             "0 disables")
    parser.add_argument("--lgb-seeds", type=int, default=1,
                        help="mirrors prc.final --lgb-seeds: bag the LightGBM "
                             "blend member (and the Huber-LightGBM member, if "
                             "--lgb-objective huber) over this many seeds "
                             "(0..N-1). 1 = SHIPPED, bit-identical")
    parser.add_argument("--d3-seeds", type=int, default=1,
                        help="mirrors prc.final --d3-seeds: bag the depth-N "
                             "CatBoost third member over this many seeds "
                             "(1113..). 1 = SHIPPED")
    # Crossval-only extensions, no prc/final.py equivalent: a second,
    # Huber-objective LightGBM member for the "members" fold test, saved
    # alongside the shipped members by --save-fold-members but never blended
    # into `bagged`. Needs --blend-lightgbm > 0 (see run_fold) -- it is not a
    # standalone flag, on purpose, matching this project's rule against a flag
    # that parses and is silently never read.
    parser.add_argument("--lgb-objective", choices=["regression", "huber"],
                        default="regression",
                        help="regression is what _fit_lightgbm has always "
                             "done and is what --blend-lightgbm ships. huber "
                             "ALSO fits a second LightGBM member with a Huber "
                             "objective, for --save-fold-members")
    parser.add_argument("--lgb-huber-alpha", type=float, default=None,
                        help="alpha (Huber delta, seconds) for --lgb-objective "
                             "huber; unset uses LightGBM's own default (0.9). "
                             "archive/member_screen.py's screen used 600")
    parser.add_argument("--save-fold-members", default="",
                        help="write every blend member's per-fold HOLDOUT "
                             "predictions to PATH, with the fold months "
                             "appended before the extension (the members' "
                             "twin of --save-preds): cat_bulk always; cat_d3, "
                             "lightgbm, lightgbm_huber when their flags "
                             "request them. Plus MVT_ID_mvt, ADEP_mvt, y, "
                             "has_flight_plan. For archive/agent_members_blend_eval.py")
    parser.add_argument("--lgb-routes", action="store_true",
                        help="ALSO fit the LightGBM member per airport over the "
                             "--bulk-routes airports (same 5,000-row floor and "
                             "seeds) and save it as lightgbm_routed for "
                             "--save-fold-members. Never blended into the fold "
                             "score. Needs --blend-lightgbm > 0 and --bulk-routes")
    parser.add_argument("--seeds", type=int, default=1, help="models per fold, averaged")
    parser.add_argument("--save-preds", action="store_true")
    parser.add_argument("--folds", default="", help="1-based fold indices, e.g. 1,3,5 (default all)")
    args = parser.parse_args()

    if args.lgb_objective == "huber" and not (float(args.blend_lightgbm or 0.0) > 0.0):
        raise SystemExit(
            "--lgb-objective huber has nothing to fit alongside without "
            "--blend-lightgbm > 0 -- it is not a standalone flag, and this "
            "guard exists so it cannot parse and silently do nothing (see "
            "the --learner and --noplan-split-lirf incidents in prc/final.py)."
        )

    if args.lgb_routes and not (float(args.blend_lightgbm or 0.0) > 0.0 and args.bulk_routes):
        raise SystemExit("--lgb-routes needs --blend-lightgbm > 0 and --bulk-routes; "
                         "otherwise it would parse and silently do nothing.")

    dropped = [c.strip() for c in args.drop.split(",") if c.strip()]
    feats = [f for f in FEATURES if f not in dropped]
    cat_idx = [feats.index(c) for c in CATEGORICAL if c not in dropped]

    chosen = FOLDS
    if args.folds:
        chosen = [FOLDS[int(i) - 1] for i in args.folds.split(",")]

    frame = load_training_features()
    print(f"[{args.tag}] {len(feats)} features, folds {chosen}, "
          f"{args.iterations} iterations" + (f", winsor {args.winsor:,.0f}s" if args.winsor else ""))

    folds = [run_fold(frame, held, feats, cat_idx, args) for held in chosen]
    scores = np.array([f["rmse_bagged"] for f in folds])
    bulk = np.array([f["rmse_bulk"] for f in folds])
    print(f"\n  mean RMSE {scores.mean():8.2f}  sd {scores.std(ddof=1):6.2f}  "
          f"range {scores.min():.1f}..{scores.max():.1f}")
    print(f"  mean bulk {bulk.mean():8.2f}  sd {bulk.std(ddof=1):6.2f}")

    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "tag": args.tag, "dropped": dropped, "winsor": args.winsor,
            "iterations": args.iterations, "depth": args.depth, "lr": args.lr,
            "mean_rmse": float(scores.mean()), "sd_rmse": float(scores.std(ddof=1)),
            "mean_bulk": float(bulk.mean()), "folds": folds,
        }) + "\n")
    print(f"appended to {RESULTS}")


if __name__ == "__main__":
    main()
