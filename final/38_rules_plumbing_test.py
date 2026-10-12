"""Plumbing test of the rule layers on the fallback file: rows moved per rule and airport-month for February and June 2026,
next to the same months of 2025.

Only row counts and the absence of a crash matter here: the predictions are placeholders (final/00_fallback.py's file F0 for
2026, the airport x calendar-month mean of the 2025 label for the 2025 side), so what a rule does to a row is not meaningful;
which rows fall in a rule's scope is.

  python final/38_rules_plumbing_test.py [--months 2,6] [--pred data/final/F0.parquet]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fcommon import ID_COL, OUT, RAW, TARGET_COL, training_label_stats  # noqa: E402

C = pl.col


def load(name: str, key: str):
    spec = importlib.util.spec_from_file_location(key, HERE / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--months", default="2,6")
    ap.add_argument("--pred", type=Path, default=OUT / "F0.parquet")
    ap.add_argument("--stage", default="post,rows,last")
    ap.add_argument("--sample", type=float, default=None)
    a = ap.parse_args()
    months = [int(t) for t in a.months.split(",")]
    g30 = load("30_general_rules.py", "mod_30")
    g32 = load("32_published_daily.py", "mod_32")
    stages = [s for s in a.stage.split(",") if s]

    results = {}
    # 2026 side: the fallback file, final ranking file, with the +12 s jitter allowance
    pred26 = pl.read_parquet(a.pred)
    ctx = g30.Ctx(RAW / "final_ranking.parquet", months, a.sample, jitter_allow=12.0)
    ctx.inp = ctx.inp.filter(C("id").is_in(pred26[ID_COL].implode()))
    p0 = g30.align(pred26, ctx.inp)
    print("== 2026, final_ranking.parquet, placeholder file")
    p = g30.apply_stages(ctx, p0.copy(), stages)
    ctx.counters.print(months)
    results["2026"] = ctx.counters.table(months)
    # published daily layer on the same file (the 32 pieces that need no upstream outputs)
    out, notes = g32.run(pred26.filter(C(ID_COL).is_in(ctx.inp["id"].implode())), RAW / "final_ranking.parquet", pieces={"level", "types", "ltfm", "lfpg"},
                         months=months, sample=None, lam_cap=None, own_bias=False, guard=True,
                         level_airports=["EDDF", "EDDM", "EGLL", "LEBL", "LEMD", "LIRF", "LTFM", "LFPG"], counters=None, window=3618.0)
    for n in notes:
        print("  note:", n)

    # 2025 side: airport x month mean of the 2025 label as the placeholder
    glob2025 = RAW / "training_2025-*.parquet"
    st = training_label_stats()
    ctx25 = g30.Ctx(glob2025, months, a.sample, jitter_allow=12.0)
    mean = {(r["apt"], r["month"]): r["mean"] for r in st.iter_rows(named=True)}
    p25 = np.array([mean[(a_, m)] for a_, m in zip(ctx25.inp["apt"].to_list(), ctx25.inp["month"].to_list())])
    pred25 = ctx25.inp.select(C("id").alias(ID_COL)).with_columns(pl.Series(TARGET_COL, np.rint(p25)).cast(pl.Int32))
    print("\n== 2025, same months, placeholder = airport x month mean of the label")
    g30.apply_stages(ctx25, np.rint(p25), stages)
    ctx25.counters.print(months)
    results["2025"] = ctx25.counters.table(months)

    both = (results["2026"].rename({"rows": "rows_2026"}).join(results["2025"].rename({"rows": "rows_2025"}), on=["rule", "apt", "month"], how="full", coalesce=True)
            .fill_null(0).sort("rule", "month", "apt"))
    print("\nrows moved per rule and airport-month, 2026 next to 2025 (same month):")
    for rule in both["rule"].unique(maintain_order=True):
        s = both.filter(C("rule") == rule)
        print(f"  {rule}: 2026 {int(s['rows_2026'].sum()):,}   2025 {int(s['rows_2025'].sum()):,}")
        for m in months:
            t = s.filter(C("month") == m).filter((C("rows_2026") > 0) | (C("rows_2025") > 0))
            if t.height:
                print(f"     m{m:02d}  " + "  ".join(f"{r['apt']} {r['rows_2026']}/{r['rows_2025']}" for r in t.iter_rows(named=True)))
    both.write_csv(OUT / "_gen" / "rules_plumbing_2026_vs_2025.csv")
    print(f"wrote {OUT / '_gen' / 'rules_plumbing_2026_vs_2025.csv'}")


if __name__ == "__main__":
    main()
