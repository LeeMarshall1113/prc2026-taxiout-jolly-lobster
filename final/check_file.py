"""Label-free checks that every candidate final file must pass.

Reads only the candidate, the template's ids, the ranking file's INPUT columns, and 2025 training labels.
Exit status 1 if any hard check fails. Soft findings (range flags, violation counts) are printed as FLAG/REPORT.

Hard checks: columns, row count equals the template, ids unique and equal to the template's in template order,
no nulls, no negatives, same integer type as the template.
Reported: per airport-month mean and median against the range over the twelve 2025 months of the same airport;
plan-window violations (plan rows whose prediction lies outside take-off minus LOBT, plus or minus the window);
gap-clamp violations (rows with a schedule gap in [lo, hi) whose prediction exceeds floor(gap)).

  python final/check_file.py data/final/F0.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from fcommon import ID_COL, RAW, TARGET_COL, departure_inputs, template_ids, training_label_stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--template", type=Path, default=RAW / "final_submitting.parquet")
    ap.add_argument("--window", type=float, default=3606 + 12, help="plan window in s: 3606 plus the 12 s take-off jitter")
    ap.add_argument("--clamp-lo", type=float, default=5000.0)
    ap.add_argument("--clamp-hi", type=float, default=50000.0)
    a = ap.parse_args()

    bad = 0

    def hard(ok: bool, msg: str) -> None:
        nonlocal bad
        print(f"{'PASS' if ok else 'FAIL'}  {msg}")
        bad += 0 if ok else 1

    cand = pl.read_parquet(a.file)
    ids, dtype = template_ids(a.template)
    hard(cand.columns == [ID_COL, TARGET_COL], f"columns {cand.columns}")
    if bad:
        sys.exit(1)
    hard(cand.height == ids.height, f"row count {cand.height:,} vs template {ids.height:,}")
    hard(cand[ID_COL].n_unique() == cand.height, "ids unique")
    hard(cand.height == ids.height and bool((cand[ID_COL] == ids[ID_COL]).all()), "ids equal the template's, in template order")
    hard(cand[TARGET_COL].null_count() == 0 and cand[ID_COL].null_count() == 0, "no nulls")
    hard(cand.schema[TARGET_COL] == dtype, f"answer dtype {cand.schema[TARGET_COL]} (template {dtype})")
    y = cand[TARGET_COL].cast(pl.Float64).to_numpy()
    hard(int((y < 0).sum()) == 0, f"no negatives (min {np.nanmin(y):,.0f})")
    if bad:
        print(f"{bad} hard check(s) failed; the soft checks need matching ids, stopping.")
        sys.exit(1)

    dep = departure_inputs(a.ranking)
    j = cand.join(dep, on=ID_COL, how="left", maintain_order="left")
    if j["month"].null_count():
        print("FAIL  some ids are not ranking-file departures")
        sys.exit(1)

    # per airport-month level against the 2025 range of that airport
    st = training_label_stats()
    rng = st.group_by("apt").agg(pl.col("mean").min().alias("mlo"), pl.col("mean").max().alias("mhi"),
                                 pl.col("median").min().alias("dlo"), pl.col("median").max().alias("dhi"))
    lvl = (j.group_by(pl.col("ADEP_mvt").alias("apt"), "month")
             .agg(pl.col(TARGET_COL).mean().alias("mean"), pl.col(TARGET_COL).median().alias("median"), pl.len().alias("n"))
             .join(rng, on="apt").sort("apt", "month"))
    print("\nairport-month level (candidate mean/median vs the 2025 12-month range)")
    flags = 0
    for r in lvl.iter_rows(named=True):
        out_m = not (r["mlo"] <= r["mean"] <= r["mhi"])
        out_d = not (r["dlo"] <= r["median"] <= r["dhi"])
        flags += out_m + out_d
        print(f"  {r['apt']} m{r['month']:02d} n={r['n']:6,}  mean {r['mean']:7.1f} [{r['mlo']:6.1f},{r['mhi']:6.1f}]"
              f"  median {r['median']:7.1f} [{r['dlo']:6.1f},{r['dhi']:6.1f}]" + ("  FLAG" if out_m or out_d else ""))
    print(f"{'FLAG' if flags else 'PASS'}  {flags} airport-month statistic(s) outside the 2025 range")

    # plan window: plan rows only (AOBT_3 present); gap clamp: schedule gap in [lo, hi)
    s = j.select(pl.col(TARGET_COL).cast(pl.Float64).alias("p"),
                 (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().alias("gl"),
                 (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                 pl.col("AOBT_3_flt").is_not_null().alias("plan"), pl.col("month"))
    inwin = (s["plan"] & s["gl"].is_not_null()).fill_null(False)
    viol = inwin & ((s["p"] < s["gl"] - a.window) | (s["p"] > s["gl"] + a.window)).fill_null(False)
    print(f"\nREPORT  plan-window violations (|p - (take-off - LOBT)| > {a.window:,.0f} s): {int(viol.sum()):,} of {int(inwin.sum()):,} plan rows;"
          f" by month {s.filter(viol).group_by('month').len().sort('month').rows()}")
    clamp = ((s["gs"] >= a.clamp_lo) & (s["gs"] < a.clamp_hi) & (s["p"] > s["gs"].floor())).fill_null(False)
    print(f"REPORT  gap-clamp violations (schedule gap in [{a.clamp_lo:,.0f}, {a.clamp_hi:,.0f}) and p > floor(gap)): {int(clamp.sum()):,};"
          f" by month {s.filter(clamp).group_by('month').len().sort('month').rows()}")
    print("\nALL HARD CHECKS PASSED")


if __name__ == "__main__":
    main()
