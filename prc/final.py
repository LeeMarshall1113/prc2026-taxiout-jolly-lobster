"""Train the submission model on ALL of 2025 and score the ranking set.

Every submission so far (v1, v2, v3) was fitted on months 2-6, 8-10 and 12 --
1,578,296 of 2,085,047 rows. Month 11 went to early stopping and months 1 and 7
were held out. That is the right discipline for *measuring* and the wrong thing
to *ship*: the ranking set is January and July 2026, and we were withholding the
only January and July the model could learn from, 24.3% of the data in total.

So this module deliberately has no holdout. It is for producing a submission
after a decision has already been made elsewhere, on `prc.crossval` folds. It
cannot tell you whether a model is good and must never be used to decide that.

Iterations are fixed at a value near where v1's early stopping landed (1057),
since there is no validation month left to stop against.

    python -m prc.final --seeds 3
    python -m prc.final --seeds 3 --transform "soften@8000x0.5" --upload
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from .config import INTERIM_DIR, RAW_DIR, TEAM_NAME, ensure_dirs
from .features import CATEGORICAL, CLOCK_FEATURES, FEATURES, TARGET, build
from .submit import ID_COL, TARGET_COL, build as build_submission
from .train import load_training_features

# Dropped by default: everything rejected on the folds, plus anything not yet
# validated. The default configuration is exactly what is currently shipped, and
# additions are opt-in via --keep. A feature that has not won 3/3 folds must not
# reach a submission because it happened to exist in the feature list.
from .weather import FEATURES as _WEATHER

DROP = [
    # NM neighbour and following context (prc.features._nm_neighbours), added
    # 2026-09-15 and not yet fold-tested. Opt in with --keep, or remove them
    # from a run's drop list.
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
    # Off-block LATENESS neighbours (prc.features._late_neighbours), added 2026-09-21.
    # Screened strongly (family LOFO -1.341s, all five folds negative, -1.231s after deleting
    # the hundred best rows, negative in both scored months) but NOT yet fold-tested through
    # build(). Opt in with --keep once it wins its folds.
    "late_apt_past15_mean",
    "late_apt_fol15_mean",
    "late_rwy_past15_mean",
    "late_rwy_fol15_mean",
    "late_apt_past60_mean",
    "late_rwy_past60_mean",
    "late_apt_past60_own_minus",
    "airport_plan",
    "dep_rwy_30min", "dep_rwy_headway", "arr_taxi_60min", "sched_demand_30min",
    "ref_taxi_s", "ref_level",
    # measured but not yet fold-validated; opt in with --keep once they win
    "prev_stand_gap", "prev_stand_headway", "prev_rwy_gap",
    "mvt_sec_00",
    "callsign_op", "lobt_minus_aobt",
    "arr_sub_day", "arr_sub_stand",
    "turn_in_delay", "turn_sched_slack", "turn_rwy_match",
    "wake_sep_req", "wake_prev_heavy", "wake_heavy_60min",
    "cfg_shift", "cfg_new",
    "arr_taxi_180min", "arr_taxi_trend", "arr_slow_60min", "arr_slow_180min",
    *_WEATHER,
    # Explicit NM clock differences, 2026-09-16 -- see features.CLOCK_FEATURES.
    # Measured only as a residual pretest (-0.33s / -0.31s); no fold test yet.
    *CLOCK_FEATURES,
]

# Every feature that has actually won a fold test and is cleared to ship.
#
# Four times now a feature has been added to FEATURES, left out of DROP, and
# silently enrolled in the next submission without ever being validated. DROP
# is opt-out, so forgetting is the default and the failure is silent. This is
# the opt-in half: a new feature is in neither list, the check below fails at
# import, and the build refuses to start until someone decides which it is.
VALIDATED = frozenset({
    "ADEP_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt",
    "FLIGHT_RULE_mvt", "FLIGHT_TYPE_flt", "MARKET_SEGMENT_flt", "RUNWAY_mvt",
    "STAND_mvt", "WK_TBL_CAT_flt", "arr_30min", "arr_60min", "dep_30min",
    "dep_60min", "dep_delay", "dow", "doy", "gap_aobt", "gap_eobt", "gap_lobt",
    "gap_sched", "has_flight_plan", "hour", "is_weekend", "minute_of_day",
    "month", "rwy_arr_active", "rwy_dep_active", "rwy_dep_share",
    "rwy_mixed_mode", "sched_vs_eobt", "stand_runway_pair_n",
    # Won 3/3 folds in run_confirm.sh on 2026-09-12: bulk deltas
    # -4.15s, -3.92s, -3.31s against a matched control, mean -3.80s, every
    # fold negative and every one clear of the 1.09s seed band. Holds under
    # every cut: -3.88s and -3.27s restricted to ordinary taxis under an
    # hour, and still -2.43s on fold (1,7) after deleting the hundred rows
    # it helps most -- which is what separates it from the arm measured the
    # same day whose gain was 65% in ten rows and negative after fifty.
    "turn_age", "turn_taxi_in", "turn_type_match", "turn_op_match",
    # Won both folds at FULL SCALE on 2026-09-21 (run_late_fold2500.sh, tags
    # late2500_{ctrl,treat}_f{3,4}): 2500 iterations, control = exactly the shipped v90 set of 73,
    # treatment = those 73 plus these seven, same seed and pinned threads. Fold (3,9) -3.429s,
    # fold (4,10) -3.419s, pooled -3.421s, paired bootstrap [-4.073, -2.749], P(<0) = 1.000.
    # Kept 100% of its 100-iteration gain (-3.418s), so capacity does not substitute for it.
    # Survives deleting its top 100 (-2.371s), 500 (-1.241s) and 1000 (-0.471s) helped rows.
    # Negative at 9 of 10 airports, led by LTFM -10.58s and LEBL -8.45s; none harmed.
    "late_apt_past15_mean", "late_apt_fol15_mean", "late_rwy_past15_mean",
    "late_rwy_fol15_mean", "late_apt_past60_mean", "late_rwy_past60_mean",
    "late_apt_past60_own_minus",
})


def _check_feature_registry() -> None:
    shipping = {f for f in FEATURES if f not in DROP}
    unclassified = shipping - VALIDATED
    if unclassified:
        raise SystemExit(
            f"these features would ship without ever having won a fold test: "
            f"{sorted(unclassified)}" + chr(10) +
            f"Add each one to prc.final.DROP (untested -- opt in later with "
            f"--keep) or to VALIDATED (it won its folds; say which run)."
        )
    stale = VALIDATED - set(FEATURES)
    if stale:
        raise SystemExit(f"VALIDATED names features that no longer exist: {sorted(stale)}")


_check_feature_registry()


def resolve_transform(spec: str):
    """Turn a name from prc.postprocess's sweep into the function it names."""
    from . import postprocess

    if not spec or spec == "identity":
        return postprocess.identity
    for fn in postprocess.candidates():
        if fn.__name__ == spec:
            return fn
    raise SystemExit(f"unknown transform {spec!r}; run `python -m prc.postprocess` to see names")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1100)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--baseline", choices=["aobt", "blend", "blend_apt", "blend_apt_hour", "lobt_switch"], default="aobt")
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
    # The rest of what fit_noplan honours. crossval could measure all of these
    # and final could ship none of them; require_noplan_settings now refuses to
    # start rather than defaulting in silence.
    parser.add_argument("--sched-blend", action="store_true",
                        help="blend toward gap_sched by predicted probability "
                             "that the off-block time is a schedule substitution")
    parser.add_argument("--noplan-target", choices=["raw", "log", "gate"],
                        default="raw",
                        help="gate: regress BLOCK-SCHED and subtract it from "
                             "gap_sched, so substitutions and rollovers become "
                             "constants in the target")
    parser.add_argument("--noplan-train", choices=["group", "all", "weighted"],
                        default="group")
    parser.add_argument("--noplan-weight", type=float, default=20.0)
    parser.add_argument("--grow-policy",
                        choices=["SymmetricTree", "Depthwise", "Lossguide"],
                        default="SymmetricTree",
                        help="CatBoost tree growth; the default grows oblivious "
                             "trees, which cannot spend a split on one airport")
    parser.add_argument("--max-leaves", type=int, default=64,
                        help="Lossguide only")
    parser.add_argument("--target", choices=["raw", "log"], default="raw")
    parser.add_argument("--noplan-model", action="store_true")
    parser.add_argument("--noplan-iterations", type=int, default=600)
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
    parser.add_argument("--keep", default="",
                        help="comma-separated features to restore to the drop list, "
                             "e.g. the wave-2 columns if they win on folds")
    parser.add_argument("--bulk-routes", default="",
                        help="airports getting their own BULK CatBoost models, or ALL (the LightGBM "
                             "member stays global). Twin of crossval --bulk-routes")
    parser.add_argument("--loss", default="RMSE",
                        help="CatBoost loss_function; crossval can measure "
                             "Huber:delta=N and this could not ship it")
    parser.add_argument("--l2", type=float, default=3.0)
    parser.add_argument("--lgb-seeds", type=int, default=1,
                        help="bag the LightGBM blend member over this many seeds (0..N-1). 1 = SHIPPED, bit-identical.")
    parser.add_argument("--d3-seeds", type=int, default=1,
                        help="bag the depth-N CatBoost third member over this many seeds (1113..). 1 = SHIPPED.")
    parser.add_argument("--save-members", default="",
                        help="write the three blend members (cpred, p3, lpred) and base_rank to this .npz, so they "
                             "can be re-bagged later without a retrain.")
    parser.add_argument("--load-members", default="",
                        help="MEMBER SWAP (2026-09-22): take the CatBoost bulk (cpred) and depth-N (p3) members "
                             "from a --save-members .npz instead of refitting them, and fit only the LightGBM "
                             "member(s). base_rank must match this build's to 1e-6 or it stops. Everything after "
                             "the blend (no-plan model, regime correction, transform) runs exactly as normal.")
    parser.add_argument("--lgb-routes", action="store_true",
                        help="route the LightGBM member per airport over --bulk-routes (same 5,000-row floor and "
                             "seeds as crossval --lgb-routes; global fit only for airports below the floor). "
                             "Default off = SHIPPED.")
    parser.add_argument("--lgb-huber-member", type=float, default=0.0,
                        help="if > 0, add a Huber-objective LightGBM (alpha = this value, global, bagged over "
                             "--lgb-seeds) as a 4th EQUAL member: (cpred + p3 + lpred + hpred) / 4, matching "
                             "crossval's lightgbm_huber. 0 = SHIPPED.")
    parser.add_argument("--fit-members", default="",
                        help="FIT-MEMBERS MODE (2026-09-22): comma list of prc.members kinds (lgb, lgb_routed, "
                             "lgb_huber600, lgb_leaves63, lgb_extratrees, xgb_hist). Fits ONLY those members on the "
                             "production training matrix, writes their ranking predictions (+ base_rank) to "
                             "--members-out, and exits -- no CatBoost, no no-plan model, no submission.")
    parser.add_argument("--members-out", default="",
                        help="the .npz --fit-members writes")
    parser.add_argument("--one-hot-max-size", type=int, default=2)
    parser.add_argument("--drop-dayoffset", action="store_true")
    parser.add_argument("--month-weight", default="",
                        help="per-month training weights, e.g. 1:3,7:3 -- at "
                             "serve time these ARE the scored months, unlike in "
                             "any fold, where they are held out")
    parser.add_argument("--drop-extra", default="",
                        help="additional features to drop, beyond DROP -- for "
                             "testing a REMOVAL on the board when no fold can "
                             "measure it, e.g. doy,month")
    parser.add_argument("--blend-cat-depth", type=int, default=0,
                        help="add a THIRD blend member: one CatBoost at this "
                             "depth. With --blend-lightgbm, all three are "
                             "averaged equally. 0 disables")
    parser.add_argument("--blend-lightgbm", type=float, default=0.0,
                        help="weight on CatBoost in a CatBoost/LightGBM blend; "
                             "0 disables. 0.70 measured -1.35s on fold (1,7)")
    parser.add_argument("--learner", choices=["catboost", "lightgbm"],
                        default="catboost")
    parser.add_argument("--linear-tree", action="store_true",
                        help="lightgbm only: a ridge regression in each leaf "
                             "instead of a constant")
    parser.add_argument("--num-leaves", type=int, default=255)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--linear-lambda", type=float, default=1.0)
    parser.add_argument("--transform", default="identity")
    parser.add_argument("--team", default=TEAM_NAME)
    parser.add_argument("--version", type=int)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    from catboost import CatBoostRegressor, Pool

    ensure_dirs()
    keep = {c.strip() for c in args.keep.split(',') if c.strip()}
    unknown = keep - set(DROP)
    if unknown:
        raise SystemExit(f'--keep names features that are not dropped: {sorted(unknown)}')
    # One rule for what is dropped, used everywhere below. Filtering the matrix
    # on `DROP minus keep` but the categorical index on `DROP` alone silently
    # hands CatBoost a string column it was told is numeric, and it dies on
    # "LSZH|1" -- which is exactly --keep airport_plan, the feature this whole
    # module's docstring is written around. crossval.py has always used a single
    # list; this is the fourth time the two paths have disagreed.
    # --drop-extra is applied to the SAME `dropped` set that feeds both the
    # feature list and the categorical index. Splitting them is how this module
    # previously handed CatBoost a string column it had been told was numeric.
    extra = {c.strip() for c in getattr(args, "drop_extra", "").split(",") if c.strip()}
    unknown_extra = extra - set(FEATURES)
    if unknown_extra:
        raise SystemExit(f"--drop-extra names unknown features: {sorted(unknown_extra)}")
    dropped = (set(DROP) - keep) | extra
    feats = [f for f in FEATURES if f not in dropped]
    if extra:
        print(f"  --drop-extra removing {sorted(extra)}: "
              f"{len(FEATURES) - len(dropped)} features remain")

    # The import-time registry check inspects FEATURES minus DROP -- which by
    # construction excludes everything --keep restores. So the guard built to
    # stop unvalidated features shipping is blind to the mechanism actually used
    # to ship them: v16 carries 19 such features and NONE is in VALIDATED.
    # Several have no clean read at all; the weather block's only fold test
    # predates the residual baseline, no-plan routing and the turnaround
    # features, so what it is worth in the current model is simply unknown.
    #
    # Not fatal -- these have shipped for many submissions and blocking them now
    # would be a bigger change than the problem -- but it must be visible in the
    # log of every build rather than discoverable only by auditing the source.
    unvalidated = sorted(set(feats) - VALIDATED)
    if unvalidated:
        print(f"  NOTE: {len(unvalidated)} of {len(feats)} shipping features are "
              f"NOT in prc.final.VALIDATED:")
        print(f"        {unvalidated}")
        print(f"        (--keep bypasses the import-time registry check)")
    if keep:
        print(f'restored to the model: {sorted(keep)}')
    cat_idx = [feats.index(c) for c in CATEGORICAL if c not in dropped]

    train = load_training_features()
    if args.drop_dayoffset:
        from .crossval import drop_day_offsets
        before = train.height
        train = drop_day_offsets(train)
        print(f"  --drop-dayoffset: {before - train.height:,} rows removed")
    print(f"training on ALL {train.height:,} rows of 2025 ({len(feats)} features, {args.seeds} seeds)")

    rank = build(pl.read_parquet(RAW_DIR / "ranking.parquet"), with_target=False)
    rank_x = rank.select(feats).to_pandas()
    print(f"scoring {rank.height:,} ranking rows")

    preds = []
    y_train = train[TARGET].to_numpy().astype(float)
    # Model the correction to Network Manager's own off-block gap rather than
    # the target. Won 3/3 folds at -6.44s, and -16.22s on the Jan+Jul fold that
    # matches the ranking months. Zero where no flight plan exists, so those
    # rows reduce to the plain target and are routed to the dedicated model.
    from .crossval import prepare_residual

    # One call, shared with crossval, so the filter-then-baseline order cannot
    # drift again. v10 shipped the opposite order and lost 10.8s.
    n_before = train.height
    train_bulk, fit_y, base_rank = prepare_residual(
        train, y_train, args, apply_to=rank)
    if args.bulk_plan_only:
        print(f"  --bulk-plan-only: global model fits {train_bulk.height:,} of "
              f"{n_before:,} rows")
    if args.residual:
        print(f"  residual mode ({args.baseline}): baseline mean "
              f"{np.mean(base_rank):.1f}s on the ranking set")

    # crossval can fit LightGBM (crossval.py:361 dispatches on args.learner);
    # this file cannot -- the loop below builds a CatBoostRegressor
    # unconditionally. Until that is implemented, a --learner the submission
    # path cannot honour must abort rather than silently ship a different model
    # than the one that won its folds. This is the fourth flag in this project
    # to be parsed and never read: --noplan-split-lirf routed nothing for four
    # submissions, and leakcheck.parser_parity() cannot catch it because it
    # compares flag NAMES across the two parsers, not behaviour.
    # --blend-lightgbm is implemented below; --learner lightgbm alone still is
    # not, and must keep aborting rather than silently shipping CatBoost.
    if getattr(args, "learner", "catboost") != "catboost":
        raise SystemExit(
            f"--learner {args.learner} is not implemented in prc/final.py; it "
            "would silently ship CatBoost. Implement the dispatch here (mirror "
            "crossval._fit_lightgbm) before shipping a run that measured it."
        )
    if getattr(args, "linear_tree", False):
        raise SystemExit("--linear-tree is LightGBM-only and unimplemented here")

    train_x = train_bulk.select(feats).to_pandas()

    # --bulk-routes: per-airport CatBoost bulk models, the serve-time twin of the block in
    # crossval._fit_predict (same 5,000-row floor, same seed per member, the global model kept for
    # airports below the floor). Routed here for the CatBoost members only -- the learner Suite R
    # validated (fold (3,9): -2.15s all, -2.50s core; fold (1,7), older configuration: -1.27s);
    # the LightGBM member stays global. Not refactored into a shared helper while a crossval run
    # of the original is queued, so the evaluated code cannot change under it.
    routes = [c.strip().upper() for c in (getattr(args, "bulk_routes", "") or "").split(",") if c.strip()]
    tr_apt = train_bulk["ADEP_mvt"].to_numpy()
    rk_apt = rank["ADEP_mvt"].to_numpy()
    if routes == ["ALL"]:
        routes = sorted(set(tr_apt) & set(rk_apt))

    if getattr(args, "fit_members", ""):
        from .members import fit_member, valid_kind
        kinds = [k.strip() for k in args.fit_members.split(",") if k.strip()]
        bad = [k for k in kinds if not valid_kind(k)]
        if bad or not args.members_out:
            raise SystemExit(f"--fit-members: unknown kinds {bad} or no --members-out")
        cat_names = [feats[i] for i in cat_idx]
        out = {"base_rank": np.broadcast_to(np.asarray(base_rank, dtype=float), (rank.height,)).copy(),
               "mvt_id": rank[ID_COL].to_numpy(), "has_plan": rank["has_flight_plan"].to_numpy()}
        # What archive/member_delta.py needs to move the regime correction with the members:
        # hot-day plan rows, the day's mean visible target M (the correction applied to an
        # all-zero prediction), which rows have a valid proxy, and the airport-day key.
        if args.regime_correction:
            from .crossval import apply_regime_correction, regime_frame
            rf_rank = regime_frame(rank, RAW_DIR / "ranking.parquet", with_target=False)
            z, *_ = apply_regime_correction(
                rf_rank, np.zeros(rank.height),
                regime_frame(train, sorted(RAW_DIR.glob("training_*.parquet")), with_target=True),
                args.regime_correction, args.regime_quantile)
            out["regime_zero"] = np.asarray(z, dtype=float)
            out["hot_plan"] = (out["has_plan"] == 1) & (out["regime_zero"] > 0)
            out["proxy_ok"] = rf_rank["gap_aobt"].is_between(0, 7200).fill_null(False).to_numpy()
            out["day_key"] = (rf_rank["ADEP_mvt"] + "_" + rf_rank["MVT_TIME_UTC_mvt"].dt.date()
                              .cast(pl.Utf8)).to_numpy().astype(str)
            print(f"  regime: {int(out['hot_plan'].sum()):,} hot-day plan rows on "
                  f"{len(set(out['day_key'][out['hot_plan']])):,} airport-days")
        import time as _time
        for k in kinds:
            _t0 = _time.time()
            out[k] = np.asarray(fit_member(k, train_x, fit_y, rank_x, cat_names, args, tr_apt, rk_apt),
                                dtype=float) + out["base_rank"]
            np.savez_compressed(args.members_out, **out)
            print(f"  member {k}: {(_time.time() - _t0) / 60:.1f} min, mean {out[k].mean():,.1f}s "
                  f"-> {args.members_out}")
        print("FIT-MEMBERS DONE")
        return

    def route(make_model, global_pred, weight):
        if not routes:
            return global_pred
        out, sizes = global_pred.copy(), []
        for code in routes:
            tr_m, rk_m = tr_apt == code, rk_apt == code
            if tr_m.sum() < 5000 or rk_m.sum() == 0:
                sizes.append(f"{code} skipped")
                continue
            sub = make_model()
            sub.fit(Pool(train_x[tr_m], fit_y[tr_m], cat_features=cat_idx,
                         weight=None if weight is None else weight[tr_m]))
            out[rk_m] = sub.predict(rank_x[rk_m])
            sizes.append(f"{code} {int(tr_m.sum()):,}")
        print("    bulk routing: " + "  ".join(sizes))
        return out

    _loaded = None
    if getattr(args, "load_members", ""):
        _loaded = np.load(args.load_members)
        _lb = np.asarray(_loaded["base_rank"], dtype=float)
        _nb = np.broadcast_to(np.asarray(base_rank, dtype=float), _lb.shape)
        if _lb.shape != (rank.height,) or not np.allclose(_lb, _nb, atol=1e-6):
            raise SystemExit(f"--load-members: base_rank in {args.load_members} does not match this build "
                             f"(shape {_lb.shape} vs {rank.height}) -- different data or flags; refusing")
        if not int(getattr(args, "blend_cat_depth", 0) or 0) or not float(getattr(args, "blend_lightgbm", 0.0) or 0.0):
            raise SystemExit("--load-members needs --blend-lightgbm > 0 and --blend-cat-depth (it replaces those fits)")
        print(f"  --load-members: cpred and p3 loaded from {args.load_members} (base_rank verified); "
              f"skipping {args.seeds} bulk and {args.d3_seeds} depth-N fits")
    for seed in range(0 if _loaded is not None else args.seeds):
        model = CatBoostRegressor(
            iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
            loss_function=args.loss, thread_count=args.threads,
            l2_leaf_reg=args.l2, one_hot_max_size=args.one_hot_max_size,
            **({} if args.grow_policy == "SymmetricTree" else
               {"grow_policy": args.grow_policy,
                **({"max_leaves": args.max_leaves} if args.grow_policy == "Lossguide" else {})}),
            random_seed=1113 + seed * 977, verbose=250,
        )
        # Sixth flag in this project parsed on one side and ignored on the
        # other. crossval.month_weights is imported rather than reimplemented
        # so the two paths cannot drift the way --learner did.
        from .crossval import month_weights
        sw = month_weights(train_bulk, getattr(args, "month_weight", ""))
        if sw is not None and seed == 0:
            import numpy as _np
            print(f"    month weights active: {_np.unique(sw).tolist()} "
                  f"over {len(sw):,} training rows")
        model.fit(Pool(train_x, fit_y, cat_features=cat_idx, weight=sw))
        if routes:
            if args.target == "log":
                raise SystemExit("--bulk-routes with --target log is not implemented in prc/final.py")

            def make_seed_model(seed=seed):
                return CatBoostRegressor(
                    iterations=args.iterations, depth=args.depth, learning_rate=args.lr,
                    loss_function=args.loss, thread_count=args.threads,
                    l2_leaf_reg=args.l2, one_hot_max_size=args.one_hot_max_size,
                    **({} if args.grow_policy == "SymmetricTree" else
                       {"grow_policy": args.grow_policy,
                        **({"max_leaves": args.max_leaves} if args.grow_policy == "Lossguide" else {})}),
                    random_seed=1113 + seed * 977, verbose=0)
            preds.append(route(make_seed_model, model.predict(rank_x), sw) + base_rank)
            print(f"  seed {seed} done (routed)")
            continue
        if args.target == "log":
            # Duan smearing: exp of a log-scale prediction is the conditional
            # median, and RMSE wants the mean. See prc.crossval._fit_predict.
            smear = float(np.mean(np.exp(fit_y - model.predict(train_x))))
            preds.append(np.expm1(model.predict(rank_x)) * smear + base_rank)
            print(f"  seed {seed} done (log target, smearing {smear:.4f})")
        else:
            preds.append(model.predict(rank_x) + base_rank)
            print(f"  seed {seed} done")

    # CatBoost and LightGBM disagree enough to be worth averaging. Their errors
    # correlate at rho = 0.9249 on fold (1,7), which by the optimal-blend
    # identity puts the best achievable combination at -1.36s, and a measured
    # sweep put it at -1.35s. The weight surface is flat from 0.60 to 0.80, so
    # 0.70 is chosen a priori rather than fitted, and the cherry-picking penalty
    # is about zero.
    #
    # LightGBM ALONE is far worse (+8.27s), so this is not a learner switch. It
    # is the decorrelation that pays.
    if float(getattr(args, "blend_lightgbm", 0.0) or 0.0) > 0.0:
        from .crossval import _fit_lightgbm
        w = float(args.blend_lightgbm)
        cat_names = [feats[i] for i in cat_idx]
        print(f"  blending LightGBM at weight {1 - w:.2f} "
              f"({w:.2f} on the CatBoost ensemble)")
        _lp = []
        _lroutes = routes if getattr(args, "lgb_routes", False) else []
        if getattr(args, "lgb_routes", False) and not routes:
            raise SystemExit("--lgb-routes needs --bulk-routes")
        for _ls in range(max(1, int(getattr(args, "lgb_seeds", 1)))):
            if not _lroutes:
                lmodel, _, lrank_x = _fit_lightgbm(
                    train_x.copy(), fit_y, rank_x.copy(), cat_names, args, _ls)
                _lp.append(lmodel.predict(lrank_x) + base_rank)
                continue
            # Twin of crossval's --lgb-routes block: same masks, same seeds, same reset_index.
            _out, _sizes = np.full(rank.height, np.nan), []
            for code in _lroutes:
                tr_m, rk_m = tr_apt == code, rk_apt == code
                if tr_m.sum() < 5000 or rk_m.sum() == 0:
                    _sizes.append(f"{code} skipped")
                    continue
                rmodel, _, rrank_x = _fit_lightgbm(
                    train_x[tr_m].reset_index(drop=True), fit_y[tr_m],
                    rank_x[rk_m].reset_index(drop=True), cat_names, args, _ls)
                _out[rk_m] = rmodel.predict(rrank_x)
                _sizes.append(f"{code} {int(tr_m.sum()):,}")
            if np.isnan(_out).any():
                lmodel, _, lrank_x = _fit_lightgbm(
                    train_x.copy(), fit_y, rank_x.copy(), cat_names, args, _ls)
                _out = np.where(np.isnan(_out), lmodel.predict(lrank_x), _out)
            print(f"    LightGBM routing (seed {_ls}): " + "  ".join(_sizes))
            _lp.append(_out + base_rank)
        lpred = np.mean(_lp, axis=0)
        if len(_lp) > 1:
            print(f"    LightGBM bagged over {len(_lp)} seeds" + (" (routed)" if _lroutes else ""))
        hpred = None
        if float(getattr(args, "lgb_huber_member", 0.0) or 0.0) > 0.0:
            _hp = []
            for _ls in range(max(1, int(getattr(args, "lgb_seeds", 1)))):
                hmodel, _, hrank_x = _fit_lightgbm(
                    train_x.copy(), fit_y, rank_x.copy(), cat_names, args, _ls,
                    objective="huber", huber_alpha=float(args.lgb_huber_member))
                _hp.append(hmodel.predict(hrank_x) + base_rank)
            hpred = np.mean(_hp, axis=0)
            print(f"    Huber LightGBM 4th member (alpha {args.lgb_huber_member:g}, {len(_hp)} seeds) "
                  f"mean {hpred.mean():,.1f}s")
        cpred = np.mean(preds, axis=0) if _loaded is None else np.asarray(_loaded["cpred"], dtype=float)

        # A THIRD member, when asked for. Two CatBoosts at different depths
        # correlate at rho = 0.9858 -- nearly the same model -- while CatBoost
        # against LightGBM is 0.9249, so almost all the decorrelation comes from
        # the different library. The extra depth-10 member still measured an
        # additional -0.53s on fold (1,7): -1.35s for the pair, -1.88s at the
        # grid optimum, -1.72s at equal thirds.
        #
        # EQUAL weights, deliberately. Another team's public write-up advises equal weights because fitted weights lost on holdout. The grid optimum here is
        # only 0.16s better than equal thirds on the fold, which after the
        # haircut the board has been applying is under a tenth of a second --
        # not worth fitting a weight for.
        d3 = int(getattr(args, "blend_cat_depth", 0) or 0)
        if d3:
            print(f"  third member: CatBoost depth {d3}, equal thirds")
            _p3 = [] if _loaded is None else [np.asarray(_loaded["p3"], dtype=float)]
            for _ds in range(0 if _loaded is not None else max(1, int(getattr(args, "d3_seeds", 1)))):
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
                _p3.append(route(make_d3_model, m3.predict(rank_x), sw) + base_rank)
            p3 = np.mean(_p3, axis=0)
            if len(_p3) > 1:
                print(f"    depth-{d3} member bagged over {len(_p3)} seeds")
            blended = (cpred + p3 + lpred) / 3.0 if hpred is None else (cpred + p3 + lpred + hpred) / 4.0
            if getattr(args, "save_members", ""):
                np.savez_compressed(args.save_members, cpred=cpred, p3=p3, lpred=lpred, base_rank=base_rank,
                                    **({} if hpred is None else {"hpred": hpred}))
                print(f"    saved blend members to {args.save_members}")
            print(f"    catboost {cpred.mean():,.1f}s  cat_d{d3} {p3.mean():,.1f}s  "
                  f"lightgbm {lpred.mean():,.1f}s  blended {blended.mean():,.1f}s")
        else:
            blended = w * cpred + (1.0 - w) * lpred
            print(f"    catboost mean {cpred.mean():,.1f}s  lightgbm mean "
                  f"{lpred.mean():,.1f}s  blended {blended.mean():,.1f}s")
        preds = [blended]

    bagged = np.vstack(preds).mean(axis=0)

    if args.noplan_model:
        from .crossval import NOPLAN_DROP, fit_noplan, noplan_routes

        # Pass the real args. This used to build a stand-in object restating
        # three settings by hand, which meant fit_noplan's getattr defaults
        # answered for everything else -- so --noplan-split-lirf parsed here,
        # printed nothing, and never routed anything. The 5/6-fold, -10.20s
        # routing win could not reach a submission, and would have looked like
        # a failure to transfer. Anything fit_noplan learns to read in future
        # now arrives on its own.
        cat_names = [c for c in CATEGORICAL if c not in dropped]
        nmodel, nfeats = fit_noplan(train, feats, cat_names, args)
        if noplan_routes(args):
            print(f"  no-plan routing: {', '.join(noplan_routes(args))}")
        npred = nmodel.predict(rank.select(nfeats).to_pandas())
        route = rank["has_flight_plan"].to_numpy() == 0
        print(f"  dedicated no-plan model: {len(nfeats)} features, routing {int(route.sum()):,} rows")
        print(f"    global mean on those rows {bagged[route].mean():8.1f}s -> "
              f"dedicated {npred[route].mean():8.1f}s")
        bagged = np.where(route, npred, bagged)

    # Checkpoint the trained-and-routed predictions before any post-processing. v19 trained
    # for 3h30m, then a post-processing step crashed and nothing reached disk; with this file
    # a failure below costs a rerun of the step, not of the build.
    ckpt = INTERIM_DIR / "ranking_predictions_prepost.parquet"
    pl.DataFrame({ID_COL: rank[ID_COL], "pred": bagged}).write_parquet(ckpt)
    print(f"  checkpoint: pre-post-processing predictions -> {ckpt.name}")

    # Serve-time twin of crossval's --lirf-sched-floor, through the same function
    # so the two cannot drift -- flag and implementation living in different
    # files is the bug this project has shipped eight times.
    if args.lirf_sched_floor:
        from .crossval import apply_lirf_sched_floor
        bagged, n_raised = apply_lirf_sched_floor(rank, bagged, args.lirf_sched_floor)
        print(f"  --lirf-sched-floor {args.lirf_sched_floor:,}: raised {n_raised} LIRF "
              f"no-plan predictions to gap_sched")

    # Serve-time twin of crossval's --regime-correction, through the same function.
    if args.regime_correction:
        from .crossval import apply_regime_correction, regime_frame
        bagged, n_ps, n_nr, n_days = apply_regime_correction(
            regime_frame(rank, RAW_DIR / "ranking.parquet", with_target=False), bagged,
            regime_frame(train, sorted(RAW_DIR.glob("training_*.parquet")), with_target=True),
            args.regime_correction, args.regime_quantile)
        print(f"  --regime-correction {args.regime_correction}: {n_days} hot airport-days, "
              f"{n_ps:,} plan rows shifted, {n_nr:,} no-plan rows raised")

    transform = resolve_transform(args.transform)
    shaped = transform(bagged)
    changed = int((np.abs(shaped - bagged) > 1e-6).sum())
    print(f"\ntransform {transform.__name__}: changed {changed:,} of {len(bagged):,} predictions")

    final = np.clip(shaped, 0.0, None)
    print(f"  mean={final.mean():.1f}s median={np.median(final):.1f}s max={final.max():.1f}s")

    out = INTERIM_DIR / "ranking_predictions.parquet"
    pl.DataFrame({ID_COL: rank[ID_COL], TARGET_COL: np.rint(final).astype(np.int32)}).write_parquet(out)
    submission = build_submission(out, RAW_DIR / "submitting.parquet",
                                  team=args.team, version=args.version)
    if args.upload:
        from .data import upload_submission

        upload_submission(submission)


if __name__ == "__main__":
    main()
