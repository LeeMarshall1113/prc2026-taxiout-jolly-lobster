"""Sweep the no-flight-plan specialist without fitting a global model at all.

`rmse_noplan` is computed only on rows where `has_flight_plan == 0`, and every
one of those predictions comes from the specialist — the global model's output
for them is discarded by `np.where(plan, bagged, npred)`. So the global fit,
which is 2,500 iterations on 2.06M rows and the expensive half of a fold, has no
bearing on this metric and can be skipped. `prc.tune_noplan` still fits it once
per fold to score combined RMSE; this trades that for dozens of configurations
in minutes on a small slot.

WHY THIS SWEEP EXISTS. A residual autopsy on real out-of-fold predictions found
the no-plan group's error is **92% instability and 7% bias** — 92% of squared
error sits where the three seeds disagree with each other, only 7% where they
agree and are jointly wrong. `log-corr(seed variance, squared error)` is
0.37-0.44 in this group across all three folds, two to three times the bulk's.
Variance, not missing signal, is what is costing us here.

And nothing has ever been aimed at it. The 2026-09-06 grid varied depth,
iterations and feature set — twelve configurations, no winner — and never
touched regularisation. It also ran *before* airport routing existed, so it
tuned a single model over 22,470 rows. Today LIRF has its own specialist fit on
about 1,200, with `min_data_in_leaf` at CatBoost's default of 1, which permits
single-row leaves.

MULTIPLE COMPARISONS. With N configurations over 3 folds, about N/8 win all
three by chance. A 3/3 record is necessary and not sufficient. Keep the grid
small, and require a margin that is large against the spread between folds.
A 2024 team lost 2,066 -> 2,276 selecting a per-subgroup model this way.

    python -m prc.sweep_noplan --folds 1,3,5
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from .crossval import FOLDS, NOPLAN_SETTINGS, fit_noplan
from .features import CATEGORICAL, FEATURES, TARGET
from .final import DROP
from .train import load_training_features
from .weather import FEATURES as _WX
from .validate import rmse

# CORRECTED 2026-09-11. This list, minus `final.DROP`, was claimed to be "v8's
# shipped feature set, so the control here is the model actually running". It
# was not. It handed the specialist 22 features where v11 ships it 41, because
# `final.DROP` gates the METAR block, the wave-2 congestion columns and
# callsign_op, and every shipped build restores them with `--keep`.
#
# So every result this rig has ever produced was measured on a model that does
# not exist: depth-4's -59.8s, the l2 grid, Depthwise, the gate's +667s, the
# bagging arms. All void. They tuned a 22-feature specialist while a 41-feature
# one shipped, on the segment carrying roughly a third of board squared error
# from 1.5% of rows.
#
# The keep list is now read from one place and matches run_v11.sh exactly.
_SHIPPED_KEEP = set(_WX) | {
    "dep_rwy_30min", "dep_rwy_headway", "arr_taxi_60min",
    "sched_demand_30min", "callsign_op",
}
EXTRA_DROP = [
    "prev_stand_gap", "prev_stand_headway", "prev_rwy_gap", "mvt_sec_00",
    "arr_sub_day", "arr_sub_stand", "lobt_minus_aobt",
]


def settings(**over) -> SimpleNamespace:
    """An args-like object carrying every setting fit_noplan reads."""
    base = dict(
        noplan_iterations=600, noplan_depth=6, threads=6,
        noplan_routes="LIRF", noplan_split_lirf=False, noplan_target="raw",
        noplan_train="group", noplan_weight=20.0, sched_blend=False,
        noplan_l2=None, noplan_min_data=0, noplan_rsm=None, noplan_seeds=1,
        noplan_grow_policy="SymmetricTree", noplan_gate_routes="",
        noplan_clip_train=0,
    )
    base.update(over)
    missing = [k for k in NOPLAN_SETTINGS if k not in base]
    if missing:
        raise SystemExit(f"settings() is missing {missing}; add them here too")
    return SimpleNamespace(**base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="1,3,5")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="results/sweep_noplan.json")
    args = ap.parse_args()

    frame = load_training_features()
    dropped = (set(DROP) - _SHIPPED_KEEP) | set(EXTRA_DROP)
    feats = [f for f in FEATURES if f not in dropped]
    cats = [c for c in CATEGORICAL if c in feats]
    chosen = [FOLDS[int(i) - 1] for i in args.folds.split(",")]

    # Deliberately small: four one-at-a-time variance knobs against the control,
    # plus one combination. Five arms over three folds is well short of the
    # N/8 spurious-winner rate that would make a 3/3 record meaningless.
    # SMALL AND RE-MOTIVATED, because the feature set under it just changed
    # from 22 columns to 41 and every previous result on this rig is void.
    # The docstring's own warning applies: with N configurations over 3 folds,
    # about N/8 win all three by chance, so six arms is the budget.
    GRID = {
        "ctrl":        {},
        # 600 iterations was chosen for a 22-feature specialist. It now has 41,
        # including the whole METAR block, and may simply be underfitting.
        "it1500":      {"noplan_iterations": 1500},
        # The residual autopsy found this group's error is 92% instability and
        # 7% bias, and log-corr(seed variance, squared error) is 0.37-0.44 here
        # against the bulk's 0.15. Seed bagging is the single most motivated
        # knob and it has never been measured on the right feature set.
        "bag5":        {"noplan_seeds": 5},
        "l2_10":       {"noplan_l2": 10.0},
        "bag5_it1500": {"noplan_seeds": 5, "noplan_iterations": 1500},
        # Kept only as a control. depth4 "won" -59.8s on the old rig; that was
        # measured on all no-plan rows, where fold (1,7) holds 26 whose labels
        # are substituted schedules rather than taxi times. Scoring below now
        # reports with and without them, so this arm either survives or dies.
        "depth4":      {"noplan_depth": 4},
    }
    print(f"{len(chosen)} folds x {len(GRID)} configurations "
          f"(~{len(GRID)/8:.1f} spurious 3/3 records expected by chance)")
    print("scoring rmse on no-plan rows only; no global model is fitted\n")

    results: dict[str, list[float]] = {}
    clean: dict[str, list[float]] = {}
    for held in chosen:
        month = pl.col("month")
        train = frame.filter(~month.is_in(held))
        test = frame.filter(month.is_in(held))
        te = test.filter(pl.col("has_flight_plan") == 0)
        y = te[TARGET].to_numpy().astype(float)
        print(f"fold {held}: {te.height:,} no-plan test rows")

        for name, over in GRID.items():
            t0 = time.time()
            model, nfeats = fit_noplan(
                train, feats, cats, settings(threads=args.threads, **over))
            pred = model.predict(te.select(nfeats).to_pandas())
            r = rmse(y, pred)
            # The same number with the corrupt-label class removed. 76 rows in
            # all of 2025 carry y > 20,000s and 74 of them are no-plan, so in
            # THIS population they are a fifth of a percent of rows and can be
            # most of the score. A config that only wins on the full number is
            # fitting them, which is exactly what depth-4 did.
            keep = y < 20000
            rc = rmse(y[keep], pred[keep])
            results.setdefault(name, []).append(r)
            clean.setdefault(name, []).append(rc)
            print(f"    {name:12s} {r:9.1f}   excl>=20k {rc:8.1f}   ({time.time()-t0:.0f}s)")
        print()

    for label, res in (("ALL no-plan rows", results),
                       ("EXCLUDING y >= 20,000 (the substituted-label class)", clean)):
        base = np.array(res["ctrl"])
        print("")
        print(f"=== {label} ===")
        print(f"{'config':<14}{'mean':>10}{'vs ctrl':>10}   per-fold delta{'':>12}{'wins':>7}")
        rows = []
        for name, scores in res.items():
            d = np.array(scores) - base
            rows.append((name, float(np.mean(scores)), d, int((d < 0).sum())))
        for name, mean, d, wins in sorted(rows, key=lambda r: r[1]):
            per = "  ".join(f"{x:+8.1f}" for x in d)
            print(f"{name:<14}{mean:10.1f}{np.mean(d):+10.1f}   {per}   {wins}/{len(d)}")
    print("")
    print("Ship nothing that wins only on the first table.")

    print("\nA win here is worth roughly a third as much on the board: the group is")
    print("1.6% of rows, and -100s of no-plan RMSE is about -5s overall.")
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"all": results, "clean": clean}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
