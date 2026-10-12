"""Screen a candidate POST-PROCESSING rule against the chain we ship, on the folds, under the gate rule.

    .venv/Scripts/python.exe archive/screen_rule.py --rule path/to/rule.parquet [--name my_rule]

The rule file is a parquet with MVT_ID_mvt and new_pred: the prediction the rule wants for the rows it
fires on (2025 rows only; rows it does not name are left alone). Everything else is done here, the same
way every time, because the two ways this project has fooled itself are both in the plumbing:

  * v32 (docs/LAB-NOTES.md): a rule judged against RAW fold predictions passed six checks and then cost
    +0.48s on the board, because production applies the rule LAST, after the gap clamp, the LOBT box
    and the ADS-B step had already corrected 96% of the rows it touched. So both arms here go through
    production's chain, and the rule is applied at the END of it.
  * Rome tests judged on fold files that were built WITHOUT the Rome no-plan gate. Only
    results/preds_q_gate_{1_7,2_8}.parquet are gate-inclusive, so LIRF rows are scored on those two
    folds only; the other three folds contribute their non-LIRF rows.
  * KNOWN GAP (found 2026-09-20, NOT yet fixed): the caveat above covers rome_noplan_gate.py.
    It does NOT cover apply_rome_plan_gate.py, which is chain step 0 -- build_v38.sh and
    build_v40.sh both start from S0=data/interim/v29_rome_gate_step.parquet, i.e. the plan
    gate is already applied before step 1. That gate replaces every 2026 LIRF row with a
    flight plan (AOBT_3_flt not null) and schedule gap <= 7,200s by
    0.5*pred + 0.5*calibrated P(copy)*gap. The baseline built below (gap clamp + LOBT box)
    does not include it.
    => A rule firing on LIRF PLAN rows inside gap <= 7,200s is credited here for work the
       shipped chain already does. The bias is OPTIMISTIC. A kill under it is safe (the rule
       failed despite being flattered); a PASS under it must be re-verified on the full chain
       (compare_v39_v40.py) before shipping. Measured instance: of 138 fires from the
       substitution detector, 110 were inside this gate's scope and were being double-credited.
  * SECOND KNOWN GAP (found 2026-09-21, NOT yet fixed): the baseline below also omits
    apply_regime_correction (prc/crossval.py:677, wired at prc/final.py:527). results/preds_*.parquet
    are PRE-correction. Measured: the fold (2,8) baseline here is 3.46s WORSE than production, all
    of it Istanbul -- 10,004 rows across the five folds, and 17 airport-days where the shipped chain
    already raises the level (LTFM 6 days at +155s mean, EGLL 4 at +98s, EHAM 4 at +63s).
    => Any rule firing on DISRUPTION-DAY rows is over-credited here. Same asymmetry as above: a kill
       is safe, a pass must be re-verified on a built submission.
    data/interim/deice_folds.parquet inherits this.

PASS RULE (the project's standing rule for a narrow post-processing rule, applied here automatically):
  1. pooled all-rows change <= -0.10s;
  2. at most one fold positive, and none worse than +0.10s;
  3. still negative after deleting the 20 rows it helps most.
A rule that fires on very few rows will pass 3 only if the gain is not one lucky row.
"""
import argparse
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, W = "MVT_ID_mvt", 3606.0
FOLDS = [("1_7", [1, 7], "results/preds_q_gate_1_7.parquet", False),
         ("2_8", [2, 8], "results/preds_q_gate_2_8.parquet", False),
         ("3_9", [3, 9], "results/preds_n3_bulk_3_9.parquet", True),
         ("4_10", [4, 10], "results/preds_s4_ctrl_4_10.parquet", True),
         ("5_11", [5, 11], "results/preds_c_ctrl_5_11.parquet", True)]

ap = argparse.ArgumentParser()
ap.add_argument("--rule", required=True, help="parquet with MVT_ID_mvt and new_pred")
ap.add_argument("--name", default="rule")
ap.add_argument("--mode", choices=["replace", "raise", "lower"], default="replace",
                help="replace: the rule's value stands. raise: a LOWER bound -- only rows predicted below it move up. "
                     "lower: an UPPER bound -- only rows predicted above it move down. A bound applied as a replace "
                     "also drags correct predictions to the bound, which is a different (worse) rule.")
args = ap.parse_args()

rule = pl.read_parquet(args.rule).select(C(ID), C("new_pred").cast(pl.Float64))
want = dict(zip(rule[ID].to_list(), rule["new_pred"].to_list()))
raw = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C(ID), C("ADEP_mvt").alias("adep"), C("AOBT_3_flt").is_not_null().alias("plan"),
               (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
               (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl")).collect())
print(f"rule {args.name} ({args.mode}): fires on {rule.height:,} of 2025's departures")

per, ys, bs, ns = [], [], [], []
for label, months, path, drop_lirf in FOLDS:
    P = pl.read_parquet(path).select(ID, "y", "bagged").join(raw, on=ID, how="left")
    if drop_lirf:  # these fold files are pre-gate at Rome, so their LIRF rows are not what production ships
        P = P.filter(C("adep") != "LIRF")
    y = P["y"].to_numpy().astype(float)
    gs = P["gs"].fill_null(np.nan).to_numpy().astype(float)
    gl = P["gl"].fill_null(np.nan).to_numpy().astype(float)
    plan = P["plan"].fill_null(False).to_numpy()
    p = np.clip(P["bagged"].to_numpy().astype(float), 0, None)
    p = np.where(np.isfinite(gs) & (gs >= 5000) & (gs < 50000) & (p > gs), gs, p)          # gap clamp
    box = plan & np.isfinite(gl)
    p = np.where(box, np.clip(p, np.where(box, gl - W, -np.inf), np.where(box, gl + W, np.inf)), p)  # LOBT box
    base = np.clip(p, 0, None)
    new = base.copy()
    ids = P[ID].to_list()
    hit = np.array([m in want for m in ids])
    if hit.any():
        v = np.clip(np.array([want[m] for m, h in zip(ids, hit) if h]), 0, None)
        cur = new[hit]
        new[hit] = v if args.mode == "replace" else (np.maximum(cur, v) if args.mode == "raise" else np.minimum(cur, v))
    d = float(np.sqrt(((y - new) ** 2).mean()) - np.sqrt(((y - base) ** 2).mean()))
    per.append((label, d, int(hit.sum())))
    ys.append(y), bs.append(base), ns.append(new)
    print(f"  fold {label:5}: fires on {int(hit.sum()):5} of {len(y):,} rows   all-rows {d:+.3f}s")

y, b, n = (np.concatenate(a) for a in (ys, bs, ns))
mse = ((y - b) ** 2).mean()
gain = (y - b) ** 2 - (y - n) ** 2
pooled = float(np.sqrt(mse - gain.sum() / len(y)) - np.sqrt(mse))
keep = np.ones(len(y), bool)
keep[np.argsort(gain)[::-1][:20]] = False
minus20 = float(np.sqrt(mse - gain[keep].sum() / len(y)) - np.sqrt(mse))
pos = [d for _, d, _ in per if d > 0]
checks = [("pooled <= -0.10s", pooled <= -0.10, f"{pooled:+.3f}"),
          ("<= 1 fold positive, none > +0.10s", len(pos) <= 1 and all(d <= 0.10 for d in pos),
           " ".join(f"{l} {d:+.3f}" for l, d, _ in per)),
          ("negative minus top-20", minus20 < 0, f"{minus20:+.3f}")]
for nm, ok, val in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {nm:<34} {val}")
print(f"VERDICT {args.name}: {'PASS' if all(c[1] for c in checks) else 'FAIL'}")
