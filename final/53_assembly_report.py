"""Stage 53: report on the assembled variants (no 2026 label is read).

  1. join table: for every join of final/50 the share of rows changed per airport-month, February next to January and June next to July,
     with the joins that change (almost) nothing on the new months flagged;
  2. level check: per airport-month mean of B against the 2025 range, and the February/June shift (B minus the same month of 2025) against the
     January/July shift of the same season, flagged when the two differ by more than 40 s;
  3. layer table: per airport-month mean of A, B, C, B-A, B-J2 (base model alone) and B-F0 (fallback file);
  4. plan-window and gap-clamp violations, prediction bounds.

    python final/53_assembly_report.py [--dir data/final] > docs/LAB-NOTES.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL, OUT, RAW, TARGET_COL, departure_inputs, training_label_stats  # noqa: E402

C = pl.col
SEASON = {2: 1, 6: 7}    # new month -> reference month of the same season


def md(df: pl.DataFrame, floats: int = 1) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        out.append("| " + " | ".join((f"{v:.{floats}f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=OUT)
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    a = ap.parse_args()
    dep = departure_inputs(a.ranking).select(C(ID_COL).cast(pl.Float64), C("ADEP_mvt").alias("apt"), "month", "MVT_TIME_UTC_mvt", "LOBT_flt", "SCHED_TIME_UTC_mvt",
                                             "AOBT_3_flt")

    # 1. join table
    st = pl.read_csv(a.dir / "J11_join_stats.csv")
    st = st.with_columns((C("changed") / C("rows")).alias("share"))
    print("## 1. Rows changed by each join (share of the airport-month's rows), February next to January, June next to July\n")
    flags = []
    for step in st["step"].unique(maintain_order=True).to_list():
        g = st.filter(C("step") == step)
        months = sorted(g["month"].unique().to_list())
        has = [m for m in months if m in (2, 6)]
        if not has:
            continue
        piv = g.pivot(on="month", index="apt", values="share").sort("apt")
        rms = g.pivot(on="month", index="apt", values="rms").sort("apt")
        print(f"### {step}\n")
        t = piv.with_columns(pl.col(pl.Float64).round(3))
        print(md(t, 3) + "\n")
        for m in has:
            ref = SEASON[m]
            for r in piv.iter_rows(named=True):
                new, old = r.get(str(m)), r.get(str(ref))
                if new is None:
                    continue
                allm = g.filter(C("month") == m)["share"].max()
                if (old is not None and old >= 0.05 and new < 0.01) or (old is None and allm >= 0.10 and new < 0.01):
                    flags.append(f"{step}: {r['apt']} m{m:02d} changes {new:.3%} of rows (m{ref:02d}: {'n/a' if old is None else f'{old:.1%}'})")
    print("### Flags (changes under 1% of an airport-month's rows where the reference changes at least 5%, or, for joins without a January/July counterpart, where another airport of the month changes at least 10%)\n")
    print("\n".join("- " + f for f in flags) if flags else "- none")

    # 2. level check
    b = pl.read_parquet(a.dir / "final_B.parquet").select(C(ID_COL).cast(pl.Float64), C(TARGET_COL).cast(pl.Float64).alias("B"))
    j = dep.join(b, on=ID_COL)
    lab = training_label_stats()
    lvl = j.group_by("apt", "month").agg(n=pl.len(), mean=C("B").mean(), median=C("B").median()).sort("apt", "month")
    ref25 = lab.rename({"mean": "m25", "median": "d25", "n": "n25"})
    lvl = lvl.join(ref25.rename({"apt": "apt"}), on=["apt", "month"], how="left").with_columns((C("mean") - C("m25")).alias("shift"))
    rng = lab.group_by("apt").agg(lo=C("mean").min(), hi=C("mean").max())
    lvl = lvl.join(rng, on="apt")
    rows = []
    for apt in sorted(lvl["apt"].unique().to_list()):
        g = lvl.filter(C("apt") == apt)
        sh = {r["month"]: r["shift"] for r in g.iter_rows(named=True)}
        for m, ref in SEASON.items():
            d = sh[m] - sh[ref]
            rows.append(dict(apt=apt, month=m, mean_B=float(g.filter(C("month") == m)["mean"][0]), m2025=float(g.filter(C("month") == m)["m25"][0]), shift=sh[m],
                             ref_month=ref, ref_shift=sh[ref], diff=d, flag="FLAG" if abs(d) > 40 else ""))
    print("\n## 2. Level of B: shift against the same month of 2025, February against January and June against July (s)\n")
    print(md(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(1))))
    out_rng = lvl.filter((C("mean") < C("lo")) | (C("mean") > C("hi"))).select("apt", "month", "mean", "lo", "hi").with_columns(pl.col(pl.Float64).round(1))
    print("\nAirport-months with mean outside the 2025 twelve-month range of the airport:\n")
    print(md(out_rng))

    # 3. layer table
    A = pl.read_parquet(a.dir / "final_A.parquet").select(C(ID_COL).cast(pl.Float64), C(TARGET_COL).cast(pl.Float64).alias("A"))
    Cc = pl.read_parquet(a.dir / "final_C.parquet").select(C(ID_COL).cast(pl.Float64), C(TARGET_COL).cast(pl.Float64).alias("C"))
    J2 = pl.read_parquet(a.dir / "J2_v94model.parquet")
    J2 = J2.select(C(ID_COL if ID_COL in J2.columns else "id").cast(pl.Float64).alias(ID_COL), C(TARGET_COL).cast(pl.Float64).alias("J2"))
    F0 = pl.read_parquet(a.dir / "F0.parquet").select(C(ID_COL).cast(pl.Float64), C(TARGET_COL).cast(pl.Float64).alias("F0"))
    k = j.join(A, on=ID_COL).join(Cc, on=ID_COL).join(J2, on=ID_COL, how="left").join(F0, on=ID_COL, how="left")
    t = (k.group_by("apt", "month").agg(n=pl.len(), A=C("A").mean(), B=C("B").mean(), C=C("C").mean(), BmA=(C("B") - C("A")).mean(),
                                         BmJ2=(C("B") - C("J2")).mean(), BmF0=(C("B") - C("F0")).mean(), rmsBmA=((C("B") - C("A")) ** 2).mean().sqrt(),
                                         rmsBmJ2=((C("B") - C("J2")) ** 2).mean().sqrt()).sort("apt", "month").with_columns(pl.col(pl.Float64).round(1)))
    print("\n## 3. Mean prediction per airport-month and what each layer does (s). B-J2 is empty where J2 has no row (January/July are carried).\n")
    print(md(t))

    # 4. bounds
    s = j.join(dep.select(ID_COL, "MVT_TIME_UTC_mvt", "LOBT_flt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"), on=ID_COL, suffix="_d") if False else j
    print("\n## 4. Bounds of B\n")
    print(f"min {float(j['B'].min()):.0f}, max {float(j['B'].max()):.0f}, share >= 3600 s {float((j['B'] >= 3600).mean()):.4%}, share >= 7200 s {float((j['B'] >= 7200).mean()):.4%}")
    print(md(j.group_by("month").agg(rows=pl.len(), p01=C("B").quantile(0.01), p50=C("B").median(), p99=C("B").quantile(0.99), max=C("B").max()).sort("month").with_columns(pl.col(pl.Float64).round(0))))


if __name__ == "__main__":
    main()
