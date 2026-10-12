"""Rome's NO-PLAN rows: the capped, calibrated, half-blended mixture, with two new signals.

    .venv/Scripts/python.exe archive/rome_noplan_gate.py eval
    .venv/Scripts/python.exe archive/rome_noplan_gate.py apply --in X.parquet --out Y.parquet

WHY THIS IS NOT THE MIXTURE THAT ALREADY FAILED. A Rome no-plan mixture was tried once (notes
section 6, +0.45s / -0.16s, collapsing under top-100 removal) -- BEFORE the recipe that works on
plan rows existed. That recipe is: a copy classifier and a genuine-taxi regressor, mixed as
P * gap + (1 - P) * taxi, LINEARLY CALIBRATED on a holdout, CAPPED to the gap band where a wrong
call cannot cost the whole gap, and HALF-BLENDED with the model rather than replacing it. The cap
and the half blend are what make a sub-97-per-cent-precision classifier profitable: section 2
measured the break-even for a hard override at ~97 per cent, and both of those move it.

TWO SIGNALS THE EARLIER ATTEMPT DID NOT HAVE, measured on 2025 before this was written:
  always_np   flights whose number NEVER files a flight plan all year are 88.6 per cent copies at
              LIRF, against 42.1 per cent for flights that sometimes file one (158 of 1,423 rows).
              Rome-specific: elsewhere the same split is 0-13 per cent either way.
  arr_copy    the same day's ARRIVAL copy rate at Rome, which is labelled even in the scored file.
              Rome no-plan copy rate runs 40 / 41 / 49 / 60 per cent across quartiles of it; at
              LTFM and EHAM the same tabulation is flat, so this is Rome's reporting pipeline
              wobbling, not a general effect.

PASS RULE, FIXED BEFORE THE FIRST RUN, on the five folds' out-of-fold predictions:
  1. pooled all-rows change <= -0.15s;
  2. at most one fold positive, and no fold worse than +0.10s;
  3. pooled change still <= 0 after deleting the 20 rows it helps most.

RUN 1 FAILED IT, and the way it failed says what to change. Pooled read -0.801s, but deleting the
twenty rows it helped most took that to +0.961s, and three of the five folds were positive. With a
48.5 per cent base copy rate the half blend moves EVERY row in the band, so the half of them that
are genuine taxi get dragged toward the gap; the win was a handful of large-gap copies paying for
that damage. So run 2 fires only where the classifier is informative -- P >= P_HI (call it a copy)
or P <= P_LO (call it genuine) -- and leaves the ambiguous middle to the model untouched.

P_HI AND P_LO ARE SET A PRIORI, NOT TUNED ON THE FOLDS. Section 2 measured the break-even precision
for a HARD override at about 97 per cent. A half blend inside a capped band halves the cost of a
wrong call, which puts break-even near 0.9; 0.90 / 0.10 is that number, not a swept one. The
measured out-of-fold precision at both thresholds is printed next to the verdict so the assumption
is checkable rather than assumed.
"""
import argparse
import sys

sys.path.insert(0, ".")

import lightgbm as lgb
import numpy as np
import polars as pl

C = pl.col
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
GAP_CAP = 25000.0
P_HI, P_LO = 0.90, 0.10   # fire only outside this band; see the header -- fixed a priori        # above this the day-early rule and the schedule floor own the rows
DAY = 86400.0
FOLDS = {"1_7": ([1, 7], "results/preds_n_feat_1_7.parquet"), "2_8": ([2, 8], "results/preds_n_feat_2_8.parquet"),
         "3_9": ([3, 9], "results/preds_n3_bulk_3_9.parquet"), "4_10": ([4, 10], "results/preds_s4_ctrl_4_10.parquet"),
         "5_11": ([5, 11], "results/preds_c_ctrl_5_11.parquet")}
PARAMS = dict(learning_rate=0.05, num_leaves=15, min_child_samples=30, verbose=-1)


def rome_rows(src: str, labelled: bool) -> pl.DataFrame:
    sel = [C(ID), C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
           C("MVT_TIME_UTC_mvt").dt.month().alias("mo"), C("MVT_TIME_UTC_mvt").dt.date().alias("date"),
           C("MVT_TIME_UTC_mvt").dt.hour().alias("hour"), C("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),
           C("STAND_mvt").cast(pl.Utf8).alias("stand"),
           (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap")]
    if labelled:
        sel += [(C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y"),
                ((C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60).alias("copy")]
    return (pl.scan_parquet(src).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_null())
            .select(sel).collect())


def arrival_copy_rate(src: str) -> pl.DataFrame:
    """Rome's arrival copy rate per day -- labelled in the scored file too, so it serves."""
    return (pl.scan_parquet(src).filter((C("PHASE_mvt") == "ARR") & (C("ADES_mvt") == "LIRF"))
            .select(C("MVT_TIME_UTC_mvt").dt.date().alias("date"),
                    ((C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60).alias("ac"))
            .group_by("date").agg(pl.len().alias("n_arr"), C("ac").mean().alias("arr_copy")).collect())


def plan_rate(src: str) -> pl.DataFrame:
    """Per flight number at LIRF: how often it files a flight plan at all."""
    return (pl.scan_parquet(src).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF"))
            .select(C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
                    C("AOBT_3_flt").is_null().alias("np"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"))
            .collect())


FEATS = ["gap", "hour", "dow", "always_np", "seen", "arr_copy", "n_arr"]


def featurise(rows: pl.DataFrame, hist: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    return (rows.join(hist, on="flight", how="left").join(arr, on="date", how="left")
                .with_columns(C("always_np").fill_null(0.5), C("seen").fill_null(0),
                              C("arr_copy").fill_null(arr["arr_copy"].mean()), C("n_arr").fill_null(0)))


def fit_gate(train: pl.DataFrame, seed: int = 0):
    """Classifier, genuine-taxi regressor, and the linear calibration of their mixture."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(train.height)
    hold, fit = idx[: max(train.height // 5, 1)], idx[max(train.height // 5, 1):]
    X = train.select(FEATS).to_pandas()
    copy = train["copy"].to_numpy()
    y = train["y"].to_numpy().astype(float)
    gap = train["gap"].to_numpy().astype(float)
    clf_h = lgb.LGBMClassifier(n_estimators=200, **PARAMS).fit(X.iloc[fit], copy[fit])
    nfit = np.intersect1d(fit, np.flatnonzero(~copy))
    reg_h = lgb.LGBMRegressor(n_estimators=200, **PARAMS).fit(X.iloc[nfit], np.clip(y[nfit], 0, 7200))
    ph = clf_h.predict_proba(X.iloc[hold])[:, 1]
    mix_h = ph * gap[hold] + (1 - ph) * reg_h.predict(X.iloc[hold])
    coef = np.linalg.lstsq(np.column_stack([np.ones(len(hold)), mix_h]), np.clip(y[hold], -3600, 60000), rcond=None)[0]
    clf = lgb.LGBMClassifier(n_estimators=200, **PARAMS).fit(X, copy)
    reg = lgb.LGBMRegressor(n_estimators=200, **PARAMS).fit(X[~copy], np.clip(y[~copy], 0, 7200))
    return clf, reg, coef


def mixture(frame: pl.DataFrame, clf, reg, coef):
    X = frame.select(FEATS).to_pandas()
    p = clf.predict_proba(X)[:, 1]
    raw = p * frame["gap"].to_numpy().astype(float) + (1 - p) * reg.predict(X)
    return coef[0] + coef[1] * raw, p


def cmd_eval(arm="both", gap_floor=0.0):
    print(f"ARM: {arm}  gap floor: {gap_floor:,.0f}s")
    train_all = rome_rows("data/raw/training_*.parquet", True)
    arr = arrival_copy_rate("data/raw/training_*.parquet")
    pr = plan_rate("data/raw/training_*.parquet")
    print(f"2025 Rome no-plan rows: {train_all.height:,}; copy {train_all['copy'].mean():.1%}; "
          f"in scope (gap <= {GAP_CAP:,.0f}s): {train_all.filter(C('gap') <= GAP_CAP).height:,}")
    pooled_y, pooled_p, pooled_q, per_fold, conf = [], [], [], [], []
    for fold, (months, path) in FOLDS.items():
        tr = train_all.filter(~C("mo").is_in(months) & (C("gap") <= GAP_CAP))
        te = train_all.filter(C("mo").is_in(months) & (C("gap") <= GAP_CAP))
        hist = (pr.filter(~C("mo").is_in(months)).group_by("flight")
                  .agg(pl.len().alias("seen"), C("np").mean().alias("always_np")))
        trf, tef = featurise(tr, hist, arr), featurise(te, hist, arr)
        clf, reg, coef = fit_gate(trf)
        P = pl.read_parquet(path).select(ID, "y", "bagged")
        j = tef.join(P, on=ID, how="inner")
        if not j.height:
            continue
        mix, prob = mixture(j, clf, reg, coef)
        fire = np.asarray(j["gap"].to_numpy(), float) >= gap_floor
        fire &= {"both": (prob >= P_HI) | (prob <= P_LO), "copy": prob >= P_HI, "genuine": prob <= P_LO}[arm]
        truth = j["copy"].to_numpy()
        conf.append((int((prob >= P_HI).sum()), int(truth[prob >= P_HI].sum()),
                     int((prob <= P_LO).sum()), int((~truth[prob <= P_LO]).sum())))
        p_old = np.clip(j["bagged"].to_numpy().astype(float), 0, None)
        p_new = np.where(fire, np.clip(0.5 * p_old + 0.5 * mix, 0, None), p_old)
        full_y = P["y"].to_numpy().astype(float)
        full_p = np.clip(P["bagged"].to_numpy().astype(float), 0, None)
        upd = dict(zip(j[ID].to_list(), p_new))
        full_q = np.array([upd.get(m, v) for m, v in zip(P[ID].to_list(), full_p)])
        d = float(np.sqrt(((full_y - full_q) ** 2).mean()) - np.sqrt(((full_y - full_p) ** 2).mean()))
        per_fold.append((fold, int(fire.sum()), d))
        pooled_y.append(full_y)
        pooled_p.append(full_p)
        pooled_q.append(full_q)
        print(f"  fold {fold}: {int(fire.sum())} of {j.height} in-scope rows fired, fold RMSE {d:+.3f}s")
    y = np.concatenate(pooled_y)
    p = np.concatenate(pooled_p)
    q = np.concatenate(pooled_q)
    pooled = float(np.sqrt(((y - q) ** 2).mean()) - np.sqrt(((y - p) ** 2).mean()))
    gain = (y - p) ** 2 - (y - q) ** 2
    keep = np.ones(len(y), bool)
    keep[np.argsort(gain)[::-1][:20]] = False
    minus = float(np.sqrt(((y - q)[keep] ** 2).mean()) - np.sqrt(((y - p)[keep] ** 2).mean()))
    pos = [f for f, _, d in per_fold if d > 0]
    worst = max(d for _, _, d in per_fold)
    checks = [("1 pooled <= -0.15s", pooled <= -0.15, f"{pooled:+.3f}"),
              ("2a at most one fold positive", len(pos) <= 1, f"{len(pos)} positive: {pos}"),
              ("2b no fold worse than +0.10s", worst <= 0.10, f"worst {worst:+.3f}"),
              ("3 pooled <= 0 minus top-20", minus <= 0, f"{minus:+.3f}")]
    print()
    for name, ok, val in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<30} {val}")
    nh, ch, nl, cl = (sum(c[i] for c in conf) for i in range(4))
    print()
    print(f"  classifier out-of-fold: P >= {P_HI}: {ch}/{nh} really copies"
          f"{f' ({ch / nh:.1%})' if nh else ''};  P <= {P_LO}: {cl}/{nl} really genuine"
          f"{f' ({cl / nl:.1%})' if nl else ''}")
    print("VERDICT: " + ("PASS" if all(o for _, o, _ in checks) else "FAIL"))


def cmd_apply(args):
    train = rome_rows("data/raw/training_*.parquet", True).filter(C("gap") <= GAP_CAP)
    arr_tr = arrival_copy_rate("data/raw/training_*.parquet")
    pr = plan_rate("data/raw/training_*.parquet")
    hist = pr.group_by("flight").agg(pl.len().alias("seen"), C("np").mean().alias("always_np"))
    clf, reg, coef = fit_gate(featurise(train, hist, arr_tr))
    sub = pl.read_parquet(args.inp)
    rank = rome_rows("data/raw/ranking.parquet", False).filter(C("gap") <= GAP_CAP)
    arr_rk = arrival_copy_rate("data/raw/ranking.parquet")
    rk = featurise(rank, hist, arr_rk).join(sub, on=ID, how="inner")
    mix, prob = mixture(rk, clf, reg, coef)
    old = rk[TARGET].to_numpy().astype(float)
    new = np.where((prob >= P_HI) | (prob <= P_LO), np.clip(0.5 * old + 0.5 * mix, 0, None), old)
    print(f"2026 Rome no-plan rows in scope: {rk.height}; move median {np.median(new - old):+.0f}s, "
          f"mean {np.mean(new - old):+.0f}s, max {np.max(np.abs(new - old)):.0f}s")
    upd = rk.select(ID).with_columns(pl.Series("new", np.rint(new)))
    out = (sub.join(upd, on=ID, how="left")
              .with_columns(pl.coalesce(C("new"), C(TARGET)).cast(sub.schema[TARGET]).alias(TARGET))
              .select(ID, TARGET))
    assert out.height == sub.height and out[ID].equals(sub[ID])
    out.write_parquet(args.out)
    print(f"wrote {args.out}: {int((out[TARGET] != sub[TARGET]).sum())} rows changed")


ap = argparse.ArgumentParser()
sub_ap = ap.add_subparsers(dest="cmd", required=True)
e = sub_ap.add_parser("eval")
e.add_argument("--arm", choices=("both", "copy", "genuine"), default="both")
e.add_argument("--gap-floor", type=float, default=0.0)
a = sub_ap.add_parser("apply")
a.add_argument("--in", dest="inp", required=True)
a.add_argument("--out", required=True)
args = ap.parse_args()
cmd_eval(args.arm, args.gap_floor) if args.cmd == "eval" else cmd_apply(args)
