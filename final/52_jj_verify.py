"""Stage 52: January/July half of the assembled variants against the last uploaded file, with the difference split by decision.

Inputs: the assembly's pre-variant file (data/final/J11_pre_variants.parquet or the _ph / _sample version, written by final/50_assemble.py),
the variant file to check (default data/final/J11_jj_C.parquet) and submissions/jolly-lobster_v162.parquet. For the January and July rows:

  S  the same pre-variant file pushed through the day layer and row fixes in the SHIPPED form (final/32 --shipped --cell-split with the
     stored coverage census, final/31 --match string), then the plan-clock window: what the January/July run did, on today's rules.
  C  the variant under test (capped day layer, no cell split, dated row fixes).

Per airport: rms and mean of (C - v162), split into (C - S) = the day-layer decisions (capped form, own-month bias, snow stop days,
null flight strings updated, no cell split) and (S - v162) = everything else (12 s plan-window allowance, rules re-run on the stored file,
Zurich day layer not ported exactly, the one never-uploaded decision). Rows are classified by the first stage that explains them.

  python final/52_jj_verify.py [--variant data/final/J11_jj_C.parquet] [--pre data/final/J11_pre_variants.parquet]
Nothing here reads a 2026 departure label; v162 is an uploaded file of predictions.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL, OUT, RAW, REPO, TARGET_COL  # noqa: E402

HERE = Path(__file__).resolve().parent
C = pl.col


def run(script: str, args: list[str]) -> None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage")
    r = subprocess.run([sys.executable, str(HERE / script)] + args, env=env, cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        print(r.stdout[-1500:], r.stderr[-1500:])
        raise SystemExit(f"{script} failed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", type=Path, default=OUT / "J11_jj_C.parquet")
    ap.add_argument("--pre", type=Path, default=OUT / "J11_pre_variants.parquet")
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--v162", type=Path, default=REPO / "submissions" / "jolly-lobster_v162.parquet")
    ap.add_argument("--census", type=Path, default=REPO / "data" / "interim" / "agent_coverage_census_2026.parquet")
    ap.add_argument("--months", default="1,7")
    ap.add_argument("--window", default="3618", help="half-width used by the day layer on rows it touched (3606 for a jitter-allowance-0 test)")
    ap.add_argument("--jitter-allow", default="12", help="allowance of the last plan-window pass (0 for the reproduction test)")
    ap.add_argument("--with-lszh", action="store_true", help="S also runs the Zurich IFR day layer (an approximation of the shipped joint solve)")
    ap.add_argument("--tag", default="", help="suffix of the temporary files and of the csv")
    a = ap.parse_args()
    months = [int(x) for x in a.months.split(",")]
    tmp = OUT / "_t5x"
    tmp.mkdir(parents=True, exist_ok=True)

    pre = pl.read_parquet(a.pre)
    s_raw, s_fix, s_fin = tmp / f"J11_verify{a.tag}_S_daily.parquet", tmp / f"J11_verify{a.tag}_S_fix.parquet", tmp / f"J11_verify{a.tag}_S.parquet"
    run("32_published_daily.py", ["--pred", str(a.pre), "--ranking", str(a.ranking), "--out", str(s_raw), "--shipped", "--cell-split",
                                   "--census", str(a.census), "--window", a.window, "--months", a.months,
                                   "--pieces", "level,types,ltfm,lfpg" + (",lszh" if a.with_lszh else "")])
    run("31_scored_row_fixes.py", ["--before", str(a.pre), "--pred", str(s_raw), "--ranking", str(a.ranking), "--out", str(s_fix), "--with-row-fixes", "--match", "string"])
    run("30_general_rules.py", ["--pred", str(s_fix), "--ranking", str(a.ranking), "--out", str(s_fin), "--stage", "last", "--jitter-allow", a.jitter_allow])

    dep = (pl.scan_parquet(a.ranking).filter(C("PHASE_mvt") == "DEP")
           .select(C(ID_COL).cast(pl.Float64), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("month")).collect())
    rd = lambda p, n: pl.read_parquet(p).select(C(ID_COL).cast(pl.Float64), C(TARGET_COL).cast(pl.Float64).alias(n))  # noqa: E731
    j = (dep.filter(C("month").is_in(months)).join(rd(a.variant, "c"), on=ID_COL).join(rd(s_fin, "s"), on=ID_COL)
         .join(rd(a.v162, "v"), on=ID_COL).join(rd(a.pre, "pre"), on=ID_COL))
    j = j.with_columns((C("c") - C("v")).alias("d_total"), (C("c") - C("s")).alias("d_daylayer"), (C("s") - C("v")).alias("d_other"))
    # first explanation of a row
    j = j.with_columns(pl.when(C("d_total") == 0).then(pl.lit("equal"))
                       .when(C("c") != C("s")).then(pl.lit("day layer decision"))
                       .when(C("pre") != C("v")).then(pl.lit("other (rules / window / Zurich / lineage)"))
                       .otherwise(pl.lit("other, row untouched by the pipeline")).alias("why"))
    out = (j.group_by("apt").agg(rows=pl.len(), differ=(C("d_total") != 0).sum(), mean=C("d_total").mean(), rms=(C("d_total") ** 2).mean().sqrt(),
                                 rms_daylayer=(C("d_daylayer") ** 2).mean().sqrt(), rms_other=(C("d_other") ** 2).mean().sqrt()).sort("apt")
           .with_columns(pl.col(pl.Float64).round(2)))
    tot = j.select(pl.len().alias("rows"), (C("d_total") != 0).sum().alias("differ"), C("d_total").mean().round(3).alias("mean"),
                   (C("d_total") ** 2).mean().sqrt().round(2).alias("rms"), (C("d_daylayer") ** 2).mean().sqrt().round(2).alias("rms_daylayer"),
                   (C("d_other") ** 2).mean().sqrt().round(2).alias("rms_other"))
    with pl.Config(tbl_rows=40, tbl_cols=10):
        print(f"January/July rows of {a.variant.name} against v162 (s):")
        print(out)
        print("all airports:")
        print(tot)
        print("rows by explanation:")
        print(j.group_by("why").agg(rows=pl.len(), rms=(C("d_total") ** 2).mean().sqrt().round(2), max_abs=C("d_total").abs().max()).sort("rows", descending=True))
        un = j.filter(C("why").str.starts_with("other, row untouched"))
        if un.height:
            print(f"rows with a difference that no stage of the pipeline touched ({un.height}): largest")
            print(un.sort(C("d_total").abs(), descending=True).head(10).select("apt", ID_COL, "pre", "v", "c", "s"))
    out.write_csv(OUT / f"J11_jj_verify{a.tag}.csv")


if __name__ == "__main__":
    main()
