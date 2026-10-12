"""Pick the v94 member set from the lean screen, by a rule fixed BEFORE fold 4 was read.

Stand-in blend on plan rows: (2*cat + sum members) / (2 + n), cat = late2500 'bagged'.
Candidate sets (the shipped LightGBM 'lgb_global' is dropped unless listed):
    set:lgb_routed | set:lgb_global,lgb_extratrees | set:lgb_routed,lgb_extratrees | set:lgb_extratrees
RULE: a set qualifies only if its board estimate is <= -0.10 s on BOTH folds; ship the qualifier
with the best mean board estimate; otherwise NONE. Board estimate = the lean screen's own
blind 0.77x / covered 0.31x conversion (agent_members_blend_eval.expected_board_delta).
Prints the chosen mode (e.g. set:lgb_routed,lgb_extratrees) on the last line.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BLIND = {"LTFM", "LIRF", "LFPG"}
SETS = {
    "set:lgb_routed": ["lgb_routed"],
    "set:lgb_global,lgb_extratrees": ["lgb_global", "lgb_extratrees"],
    "set:lgb_routed,lgb_extratrees": ["lgb_routed", "lgb_extratrees"],
    "set:lgb_extratrees": ["lgb_extratrees"],
}
FOLDS = {3: "results/preds_late2500_treat_f3_3_9.parquet", 4: "results/preds_late2500_treat_f4_4_10.parquet"}

spec = importlib.util.spec_from_file_location("mbe", ROOT / "archive/agent_members_blend_eval.py")
mbe = importlib.util.module_from_spec(spec); spec.loader.exec_module(mbe)

res = {k: [] for k in SETS}
for f, ref in FOLDS.items():
    m = pl.read_parquet(ROOT / f"results/lean_members_f{f}.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64))
    r = pl.read_parquet(ROOT / ref).select(pl.col("MVT_ID_mvt").cast(pl.Int64), pl.col("bagged").alias("cat"))
    j = m.join(r, on="MVT_ID_mvt", how="inner").filter(pl.col("has_flight_plan") == 1)
    y, cat = j["y"].to_numpy().astype(float), j["cat"].to_numpy().astype(float)
    blind = np.isin(j["ADEP_mvt"].to_numpy(), list(BLIND))
    prod = (2 * cat + j["lgb_global"].to_numpy()) / 3
    print(f"fold {f}: n={j.height:,}  stand-in production RMSE {np.sqrt(((prod - y) ** 2).mean()):.3f}")
    for name, mem in SETS.items():
        new = (2 * cat + sum(j[k].to_numpy().astype(float) for k in mem)) / (2 + len(mem))
        rm = float(np.sqrt(((new - y) ** 2).mean()))
        b = mbe.expected_board_delta(y, prod, new, blind)
        b = float(b[0])  # (board, blind_contrib, covered_contrib, actual_pooled)
        res[name].append(b)
        print(f"  {name:34} RMSE {rm:8.3f}  board~{b:+.3f}s")
ok = {k: float(np.mean(v)) for k, v in res.items() if len(v) == 2 and max(v) <= -0.10}
choice = min(ok, key=ok.get) if ok else "NONE"
print(f"qualifiers (both folds <= -0.10): {ok}")
print(choice)
