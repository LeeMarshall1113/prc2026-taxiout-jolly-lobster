"""Post-process a submission: keep flight-plan predictions inside the NM matching window.

    .venv/Scripts/python.exe archive/apply_lobt_box.py --in submissions/X.parquet --out submissions/Y.parquet

RULE. A movement is matched to its Network Manager flight only when its off-block lies within
an hour of the flight's LOBT: in 2025, |BLOCK - LOBT| <= 3,606s on every one of 2,062,577
flight-plan departures, zero exceptions. So for a flight-plan row
    y = MVT - BLOCK  lies in  [MVT - LOBT - 3,606,  MVT - LOBT + 3,606]
and a prediction outside that box moves to its nearer edge. With truth inside the box the move
cannot increase any row's error. W carries 14s of slack over the observed maximum, for
timestamp jitter; the slack costs nothing measurable.

EVIDENCE (2025, scratch directory lobt_box.py, pass rule fixed before the run: premise on every plan
row and no fold worse): held-out predictions outside the box 94 / 46 / 61 per fold, fold RMSE
-1.20s (1,7), -0.15s (2,8), -0.98s (3,9). On the 2026 scored set v18 had 61 rows outside (60
below, 1 above) at a squared distance of 8.1e7 s^2 -- a floor of about -0.43s on the board,
since truth sits at or beyond the edge.

HARD BINDING. When the lower edge L = MVT - LOBT - 3,606 exceeds NM's own taxi (MVT - AOBT_3)
by more than 1,800s, the recorded off-block cannot be the push-back NM saw, and truth does not
sit at the edge: on the 359 such 2025 rows, y - L has median 2,771s (p10 666s, p90 5,396s) and
only 9.5% are schedule copies -- the stamp lands ~14 minutes after LOBT while the aircraft
leaves much later. Those rows move to min(L + median, upper edge) instead of the edge, with the
median recomputed from 2025 at run time. Leave-fold-out (scratch directory box_binding.py, rule fixed
before the run: beat the edge on all three folds): edge -0.86s / -0.09s / -0.88s, median
target -1.33s / -0.20s / -1.42s. The copy value (y = gap) lost on two folds.
"""
import argparse

import numpy as np
import polars as pl

C = pl.col
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
W = 3620.0
W_EXACT = 3606.0      # the observed window, used where the rule was measured with it
HARD_S = 1800.0       # edge above NM's own taxi by more than this = hard binding
MAX_CHANGED = 400

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--hard-binding-blend", type=float, default=1.0,
                help="fraction of the median lift to apply when hard binding is ON. 1.0 is the "
                     "original full-strength behaviour; 0.5 is the half-strength version measured "
                     "on 2026-09-20 at -0.281s pooled, all five folds negative, against the FULL "
                     "current seven-step chain. Ignored entirely when --no-hard-binding is passed, "
                     "so the shipped v39 build is bit-identical whatever this is set to.")
ap.add_argument("--no-hard-binding", action="store_true",
                help="move predictions to the box edge only. v21 shipped the hard-binding lift (edge + median) "
                     "together with the Rome identity override and schedule floor, and scored 270.66 against "
                     "v20's 270.44: those uncertain parts cost at least 1.8e8 s^2 net of the certain ones, "
                     "and they cannot be told apart without probing. Edge-only keeps what holds by construction.")
args = ap.parse_args()

sub = pl.read_parquet(args.inp)
cols = [C(ID), C("ADEP_mvt").alias("apt"), C("AOBT_3_flt").is_not_null().alias("plan"),
        (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
        (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga")]
rk = pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP").select(cols).collect()
j = sub.join(rk, on=ID, how="left", maintain_order="left")
if j.height != sub.height:
    raise SystemExit(f"join changed the row count: {j.height:,} vs {sub.height:,}")

# median of (truth - exact lower edge) over 2025 plan rows that bind hard
tr = (pl.scan_parquet("data/raw/training_*.parquet")
      .filter((C("PHASE_mvt") == "DEP") & C("AOBT_3_flt").is_not_null() & C("LOBT_flt").is_not_null())
      .select((C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
              (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
              (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y"))
      .filter(C("gl") - W_EXACT - C("ga") > HARD_S).collect())
if tr.height < 200:
    raise SystemExit(f"only {tr.height} hard-binding rows in 2025; expected ~359")
M = float((tr["y"] - (tr["gl"] - W_EXACT)).median())

p = j[TARGET].to_numpy().astype(float)
gl = j["gl"].to_numpy().astype(float)
ga = j["ga"].to_numpy().astype(float)
plan = j["plan"].fill_null(False).to_numpy() & ~np.isnan(gl)
lo = np.where(plan, np.ceil(gl - W), -np.inf)
hi = np.where(plan, np.floor(gl + W), np.inf)
new = np.clip(np.clip(p, lo, hi), 0, None)
hard = plan & ~np.isnan(ga) & (gl - W_EXACT - ga > HARD_S)
target = np.minimum(np.rint(gl - W_EXACT + M * args.hard_binding_blend), hi)
lift = hard & (new < target) & (not args.no_hard_binding)
new = np.where(lift, target, new)
print(f"hard binding: {tr.height} rows in 2025, median truth above the edge {M:,.0f}s; "
      f"{int(hard.sum())} scored rows bind hard, {int(lift.sum())} lifted to edge + "
      f"{args.hard_binding_blend:.2f} x median")
use = new != p
n = int(use.sum())
if n > MAX_CHANGED:
    raise SystemExit(f"refusing: {n} rows would change, more than {MAX_CHANGED}")
d = (new - p)[use]
by_apt = j.filter(pl.Series(use)).group_by("apt").len().sort("apt")
print(f"LOBT box +/-{W:,.0f}s: moved {n} flight-plan predictions ({int((d > 0).sum())} up, {int((d < 0).sum())} down); "
      f"total move {np.abs(d).sum():,.0f}s, squared {float((d ** 2).sum()):,.0f} s^2; by airport "
      + ", ".join(f"{a} {k}" for a, k in by_apt.iter_rows()))
out = sub.with_columns(pl.Series(TARGET, np.rint(new).astype(np.int64)).cast(sub[TARGET].dtype))
out.write_parquet(args.out)
print(f"wrote {args.out}")
