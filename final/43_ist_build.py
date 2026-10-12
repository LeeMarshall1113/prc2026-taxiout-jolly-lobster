"""Stage 43b: Istanbul rows for the served months from the control arm and the two snow arms.

Parameterised combination of archive/agent_kg_ltfm_delta.py (v113: treat - ctrl) and archive/agent_ltreg_build.py (v137: snow - treat).
The two shipped steps telescope: base + (treat - ctrl) + (snow - treat) = base + (snow - ctrl). So this script takes ONE delta,
snow minus control, and the treatment arm (8.5 h on Kaggle) is not run. Differences from the two-step chain: one rounding instead of
two and the plan-window box applied once. The shipped control file of ltreg_build ("treat_s3") plays the role of `ctrl` here.

  ctrl   preds file of final/ist_kg_ltfm_final.py --ltfm-seq on|off ... (columns MVT_ID_mvt, ADEP_mvt, prepost, final)
         For the shipped January/July numbers the reference was the sequence-feature arm ("on") for ltreg_build and the
         sequence-free arm ("off") for kg_ltfm_delta; for Feb/Jun both telescope into: snow minus the sequence-free arm.
  snow   k0 (seed 1113) and k12 (seeds 2090 and 3067) preds files of final/ist_ltreg_final.py; pp = (k0 + 2*k12)/3, then the
         disruption-day regime correction is re-applied (lam 1.0, quantile 0.99), exactly as ltreg_build.correct().

    python final/43_ist_build.py validate --ctrl X --ranking data/raw/final_ranking.parquet
    python final/43_ist_build.py build --base J7_base.parquet --ctrl X --k0 A --k12 B --months 2,6 --out data/final/J3_ist.parquet
Only Istanbul rows change; every other row is copied from --base. Prints rows per airport-month.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from prc.crossval import apply_regime_correction  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def regime_rows(paths, ids=None, with_target=False):
    q = pl.scan_parquet(paths).filter(pl.col("PHASE_mvt") == "DEP").select(
        ID, "ADEP_mvt", "MVT_TIME_UTC_mvt",
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("AOBT_3_flt")).dt.total_seconds().alias("gap_aobt"),
        pl.col("AOBT_3_flt").is_not_null().cast(pl.Int8).alias("has_flight_plan"),
        *([pl.col(T)] if with_target else []))
    f = q.collect()
    if ids is not None:
        f = pl.DataFrame({ID: ids}).join(f, on=ID, how="left", maintain_order="left")
        assert f.height == len(ids) and f["MVT_TIME_UTC_mvt"].null_count() == 0
    return f


def correct(prepost, ids, ranking):
    rank = regime_rows(ranking, ids)
    train = regime_rows(sorted(glob.glob(str(REPO / "data/raw/training_2025-*.parquet"))), with_target=True)
    out, n_ps, n_nr, n_days = apply_regime_correction(rank, prepost, train, 1.0, 0.99)
    print(f"  regime correction: {n_days} hot airport-days, {n_ps:,} plan rows shifted, {n_nr:,} no-plan raised")
    return out


def validate(a):
    c = pl.read_parquet(a.ctrl)
    got = correct(c["prepost"].to_numpy(), c[ID].to_numpy(), a.ranking)
    d = np.abs(got - c["final"].to_numpy())
    print("VALIDATE the local regime correction reproduces the arm's own final from its prepost: max abs", float(d.max()), "PASS" if d.max() < 1e-6 else "FAIL")


def build(a):
    months = [int(m) for m in a.months.split(",")]
    c = pl.read_parquet(a.ctrl)
    k0 = pl.read_parquet(a.k0).select(ID, pl.col("prepost").alias("p0"))
    k12 = pl.read_parquet(a.k12).select(ID, pl.col("prepost").alias("p12"))
    j = c.join(k0, on=ID, how="left", maintain_order="left").join(k12, on=ID, how="left", maintain_order="left")
    assert j["p0"].null_count() == 0 and j["p12"].null_count() == 0, "arms were not served on the same rows"
    lt = (j["ADEP_mvt"] == "LTFM").to_numpy()
    pp = (j["p0"].to_numpy() + 2.0 * j["p12"].to_numpy()) / 3.0
    treat = correct(pp, j[ID].to_numpy(), a.ranking)
    delta = np.where(lt, treat - j["final"].to_numpy(), 0.0)
    print(f"delta on {int(lt.sum()):,} Istanbul rows: mean {delta[lt].mean():+.2f} rms {np.sqrt((delta[lt] ** 2).mean()):.2f} p1/p99 {np.percentile(delta[lt], 1):+.0f}/{np.percentile(delta[lt], 99):+.0f}")
    base = pl.read_parquet(a.base)
    bj = base.select(ID).join(pl.DataFrame({ID: j[ID], "d": delta, "lt": lt}), on=ID, how="left", maintain_order="left")
    d = bj["d"].fill_null(0.0).to_numpy()
    isl = bj["lt"].fill_null(False).to_numpy()
    b = base[T].to_numpy().astype(float)
    new = b + d
    rb = (pl.scan_parquet(a.ranking).filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col(ID), pl.col("AOBT_3_flt").is_not_null().alias("plan"),
                  (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("gl"),
                  pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"), pl.col("ADEP_mvt").alias("apt")).collect())
    bx = base.select(ID).join(rb, on=ID, how="left", maintain_order="left")
    gl = bx["gl"].to_numpy()
    box = bx["plan"].fill_null(False).to_numpy() & np.isfinite(gl) & isl
    clipped = np.where(box, np.clip(new, np.ceil(gl - 3620.0), np.floor(gl + 3620.0)), new)
    print(f"plan-window box re-applied on Istanbul rows: {int((np.abs(clipped - new) > 0).sum())} rows clipped")
    new = np.rint(np.maximum(clipped, 0.0))
    out = base.with_columns(pl.Series(T, new).cast(base[T].dtype))
    chg = np.abs(new - b) > 0
    assert not (chg & ~isl).any(), "a non-Istanbul row changed"
    print(f"changed rows {int(chg.sum()):,} of {int(isl.sum()):,} Istanbul rows; mean shift {np.mean(new[isl] - b[isl]):+.3f}")
    for m in months:
        s = isl & (bx["mo"].to_numpy() == m)
        if s.any():
            print(f"  LTFM month {m:02d}: rows {int(s.sum()):,} mean shift {np.mean(new[s] - b[s]):+.2f} rms {np.sqrt(np.mean((new[s] - b[s]) ** 2)):.2f}")
    out.write_parquet(a.out)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "build"])
    ap.add_argument("--ctrl", required=True)
    ap.add_argument("--k0")
    ap.add_argument("--k12")
    ap.add_argument("--base")
    ap.add_argument("--months", default="2,6")
    ap.add_argument("--ranking", default=str(REPO / "data/raw/final_ranking.parquet"))
    ap.add_argument("--out", default=str(REPO / "data/final/J3_ist.parquet"))
    a = ap.parse_args()
    validate(a) if a.cmd == "validate" else build(a)


if __name__ == "__main__":
    main()
