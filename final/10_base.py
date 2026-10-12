"""Stage 1 for the final phase: the shipped base model (v92 recipe), fitted on all of 2025 and serving a chosen set of months.

Derived from prc/final.py (the script behind jolly-lobster_v92, build_v92model.sh); main() is copied from it with
only the changes listed here, so `diff prc/final.py final/10_base.py` shows all of them.

  1. The ranking file, the template and the month set are parameters (--ranking, --template, --months).
     The original reads data/raw/ranking.parquet and scores every row in it.
  2. Features are built ONE CALENDAR MONTH AT A TIME (month of the movement time, arrivals included), exactly as the
     training features are built (prc.train.load_training_features builds each month file alone). Windowed features
     (turnaround up to 24 h, headways, 30/60 minute neighbour counts) then never see the previous or next month, so
     serving February and January in one run cannot change January's features. The original built the whole file at once.
  3. Every fitted blend member is cached on disk as soon as it is fitted (--cache-dir) and reused on restart, so a crash
     resumes at the member that was running instead of at the start. The cache is keyed by a fingerprint of the flags,
     the served ids and the training-feature hash; a mismatch stops the run instead of reusing stale members.
  4. Outputs go to --out-dir (default data/final), never to data/interim or submissions/. The result is joined to the
     template and written as <out-dir>/<tag>_base.parquet. There is no upload option.
  5. The training-feature cache is read only. If it is stale the run stops (the original rebuilt and overwrote it).

Everything else (flags, models, seeds, residual baseline, no-plan model, regime correction) is the original code.
The shipped v92 flags are in final/10_base.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from prc.config import INTERIM_DIR, RAW_DIR, ensure_dirs  # noqa: E402
from prc import features as _features_module  # noqa: E402
from prc.features import CATEGORICAL, FEATURES, TARGET, build  # noqa: E402
from prc.final import DROP, VALIDATED, resolve_transform  # noqa: E402  (the shipped feature lists, not copied)
from prc.submit import ID_COL, TARGET_COL  # noqa: E402
from prc.train import load_training_features  # noqa: E402


def build_by_month(raw: pl.DataFrame, months: list[int]) -> pl.DataFrame:
    """Feature matrix for the departures of the chosen months, built one calendar month at a time."""
    month = raw["MVT_TIME_UTC_mvt"].dt.month()
    parts = []
    for m in sorted(months):
        part = build(raw.filter(month == m), with_target=False)
        print(f"  month {m:02d}: {part.height:,} departures")
        parts.append(part)
    return pl.concat(parts)


def training_cache_is_fresh() -> bool:
    """True when data/interim/train_features.parquet was built from the current prc/features.py."""
    digest = hashlib.sha256(Path(_features_module.__file__).read_bytes()).hexdigest()[:16]
    stamp = INTERIM_DIR / "train_features.featurehash"
    return (INTERIM_DIR / "train_features.parquet").exists() and stamp.exists() and stamp.read_text().strip() == digest


class MemberCache:
    """Pickle-free cache of fitted predictions: one .npy per member, valid only for one fingerprint."""

    def __init__(self, directory: Path, fingerprint: dict):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "fingerprint.json"
        text = json.dumps(fingerprint, sort_keys=True)
        if meta.exists() and meta.read_text() != text:
            raise SystemExit(f"{self.dir} holds members from a different run (flags, served ids or features changed). "
                             f"Use a new --cache-dir or delete it.")
        meta.write_text(text)

    def peek(self, name: str):
        path = self.dir / f"{name}.npy"
        if path.exists():
            print(f"    cache hit: {name}")
            return np.load(path)
        return None

    def put(self, name: str, value) -> None:
        path = self.dir / f"{name}.npy"
        if not path.exists():
            np.save(path.with_suffix(".tmp.npy"), np.asarray(value, dtype=float))
            path.with_suffix(".tmp.npy").replace(path)   # whole file or nothing


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
    # Parameters added for the final phase (the original had --team/--version/--upload and fixed paths).
    parser.add_argument("--ranking", type=Path, default=RAW_DIR / "final_ranking.parquet",
                        help="movements file to serve (departures and arrivals)")
    parser.add_argument("--template", type=Path, default=RAW_DIR / "final_submitting.parquet",
                        help="submission template; fixes the id order of the output")
    parser.add_argument("--months", default="1,2,6,7",
                        help="calendar months (of the movement time) to serve, features built one month at a time")
    parser.add_argument("--out-dir", type=Path, default=REPO / "data" / "final")
    parser.add_argument("--tag", default="base", help="output name: <out-dir>/<tag>_base.parquet")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="fitted members are saved here as they finish; default <out-dir>/<tag>_members")
    parser.add_argument("--max-rows-per-month", type=int, default=0,
                        help="DRY RUN ONLY: keep the first N movements of each month before building (0 = all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the matrices and print shapes, then stop before any model is fitted")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    served_months = [int(t) for t in args.months.split(",") if t.strip()]
    if not training_cache_is_fresh():
        raise SystemExit("data/interim/train_features.parquet is missing or stale for the current prc/features.py; "
                         "this script does not rebuild it (the shared cache is read-only here)")

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

    if args.dry_run:
        # Do not load 2 million training rows: read 100k, and report what the full frame would occupy.
        _cache = INTERIM_DIR / "train_features.parquet"
        _n_train = pl.scan_parquet(_cache).select(pl.len()).collect().item()
        _head = pl.scan_parquet(_cache).head(100_000).collect()
        _scale = _n_train / _head.height
        print(f"DRY RUN sizes: training frame {_n_train:,} rows; polars about {_head.estimated_size() * _scale / 2**30:.2f} GB; "
              f"the {len(feats)}-feature pandas matrix about "
              f"{_head.select(feats).to_pandas().memory_usage(deep=True).sum() * _scale / 2**30:.2f} GB")
        train = _head
    else:
        train = load_training_features()
    if args.drop_dayoffset:
        from prc.crossval import drop_day_offsets
        before = train.height
        train = drop_day_offsets(train)
        print(f"  --drop-dayoffset: {before - train.height:,} rows removed")
    print(f"training on ALL {_n_train if args.dry_run else train.height:,} rows of 2025 ({len(feats)} features, {args.seeds} seeds)")

    raw_rank = pl.read_parquet(args.ranking)
    if args.max_rows_per_month:
        _mo = raw_rank["MVT_TIME_UTC_mvt"].dt.month()
        raw_rank = pl.concat([raw_rank.filter(_mo == m).head(args.max_rows_per_month) for m in served_months])
    rank = build_by_month(raw_rank, served_months)
    del raw_rank
    rank_x = rank.select(feats).to_pandas()
    print(f"scoring {rank.height:,} ranking rows (months {served_months}) from {args.ranking.name}")
    if args.dry_run:
        print(f"DRY RUN: serve matrix {rank_x.shape}; stopping before any fit")
        return
    # Fingerprint for the member cache: anything that would make cached predictions wrong for this run.
    cache = MemberCache(args.cache_dir or (args.out_dir / f"{args.tag}_members"), {
        "argv": [a for a in sys.argv[1:] if not a.startswith(("--cache-dir", "--out-dir", "--tag"))],
        "ids": hashlib.sha256(rank[ID_COL].to_numpy().tobytes()).hexdigest(),
        "train_rows": train.height,
        "features_py": hashlib.sha256(Path(_features_module.__file__).read_bytes()).hexdigest()[:16],
    })

    preds = []
    y_train = train[TARGET].to_numpy().astype(float)
    # Model the correction to Network Manager's own off-block gap rather than
    # the target. Won 3/3 folds at -6.44s, and -16.22s on the Jan+Jul fold that
    # matches the ranking months. Zero where no flight plan exists, so those
    # rows reduce to the plain target and are routed to the dedicated model.
    from prc.crossval import prepare_residual

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
        from prc.members import fit_member, valid_kind
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
            from prc.crossval import apply_regime_correction, regime_frame
            rf_rank = regime_frame(rank, args.ranking, with_target=False)
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
        # Resume: members already in an earlier output for the same ids are kept, not refitted.
        if Path(args.members_out).exists():
            _prev = np.load(args.members_out, allow_pickle=False)
            if np.array_equal(_prev["mvt_id"], out["mvt_id"]):
                out.update({k: _prev[k] for k in kinds if k in _prev.files})
        for k in kinds:
            if k in out:
                print(f"  member {k}: already in {args.members_out}, kept")
                continue
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
        _hit = cache.peek(f"cat_seed{seed}")
        if _hit is not None:
            preds.append(_hit)
            continue
        if preds:
            cache.put(f"cat_seed{len(preds) - 1}", preds[-1])
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
        from prc.crossval import month_weights
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

    if preds and _loaded is None:
        cache.put(f"cat_seed{len(preds) - 1}", preds[-1])

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
        from prc.crossval import _fit_lightgbm
        w = float(args.blend_lightgbm)
        cat_names = [feats[i] for i in cat_idx]
        print(f"  blending LightGBM at weight {1 - w:.2f} "
              f"({w:.2f} on the CatBoost ensemble)")
        _lp = []
        _lroutes = routes if getattr(args, "lgb_routes", False) else []
        if getattr(args, "lgb_routes", False) and not routes:
            raise SystemExit("--lgb-routes needs --bulk-routes")
        _lhit = cache.peek("lgb_member")      # the seed-mean LightGBM prediction of an earlier, interrupted run
        if _lhit is not None:
            _lp = [_lhit]
        for _ls in range(0 if _lhit is not None else max(1, int(getattr(args, "lgb_seeds", 1)))):
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
        cache.put("lgb_member", lpred)
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
        # EQUAL weights, deliberately. Another team's public write-up advises equal weights because
        # fitted weights lost on holdout. The grid optimum here is
        # only 0.16s better than equal thirds on the fold, which after the
        # haircut the board has been applying is under a tenth of a second --
        # not worth fitting a weight for.
        d3 = int(getattr(args, "blend_cat_depth", 0) or 0)
        if d3:
            print(f"  third member: CatBoost depth {d3}, equal thirds")
            _p3 = [] if _loaded is None else [np.asarray(_loaded["p3"], dtype=float)]
            _phit = cache.peek("depth_member")
            if _phit is not None:
                _p3 = [_phit]
            for _ds in range(0 if (_loaded is not None or _phit is not None) else max(1, int(getattr(args, "d3_seeds", 1)))):
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
            cache.put("depth_member", p3)
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
        from prc.crossval import NOPLAN_DROP, fit_noplan, noplan_routes

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
    ckpt = args.out_dir / f"{args.tag}_prepost.parquet"
    pl.DataFrame({ID_COL: rank[ID_COL], "pred": bagged}).write_parquet(ckpt)
    print(f"  checkpoint: pre-post-processing predictions -> {ckpt.name}")

    # Serve-time twin of crossval's --lirf-sched-floor, through the same function
    # so the two cannot drift -- flag and implementation living in different
    # files is the bug this project has shipped eight times.
    if args.lirf_sched_floor:
        from prc.crossval import apply_lirf_sched_floor
        bagged, n_raised = apply_lirf_sched_floor(rank, bagged, args.lirf_sched_floor)
        print(f"  --lirf-sched-floor {args.lirf_sched_floor:,}: raised {n_raised} LIRF "
              f"no-plan predictions to gap_sched")

    # Serve-time twin of crossval's --regime-correction, through the same function.
    if args.regime_correction:
        from prc.crossval import apply_regime_correction, regime_frame
        bagged, n_ps, n_nr, n_days = apply_regime_correction(
            regime_frame(rank, args.ranking, with_target=False), bagged,
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

    # Join to the template (its id order and answer type). Whole file when every template id was served; otherwise
    # only the served ids are written, which is a partial file and not a candidate submission.
    served = pl.DataFrame({ID_COL: rank[ID_COL], TARGET_COL: np.rint(final).astype(np.int32)})
    tpl = pl.read_parquet(args.template, columns=[ID_COL])
    joined = tpl.join(served, on=ID_COL, how="left", maintain_order="left")
    missing = joined[TARGET_COL].null_count()
    out = args.out_dir / f"{args.tag}_base.parquet"
    if missing:
        print(f"  NOTE: {missing:,} of {tpl.height:,} template ids were not served (months {served_months}); "
              f"writing the served rows only -- PARTIAL, not a submission")
        joined = joined.drop_nulls(TARGET_COL)
    joined.write_parquet(out)
    print(f"wrote {out}: {joined.height:,} rows")


if __name__ == "__main__":
    main()
