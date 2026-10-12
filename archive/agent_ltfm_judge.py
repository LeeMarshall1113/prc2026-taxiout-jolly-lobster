"""Judge the LTFM take-off-sequence fold test (agent_ltfm_app.py::ltfm_seq). Rule PRE-REGISTERED 2026-09-23
in docs/LAB-NOTES.md BEFORE any arm ran.

Inputs: results/preds_ltfmseq_{ctrl,treat}_f{1,2,3}_{a}_{b}.parquet (folds (1,7) (2,8) (3,9) = all Jan-Mar + Jul-Sep).
Prediction = `bagged` (raw fold output; LTFM's chain is the regime correction, already inside crossval via
--regime-correction 1.0, plus two LOBT bounds touching ~50 LTFM rows a year).

PRIMARY (gating) = the LTFM SPLICE: treatment predictions on LTFM rows, control predictions everywhere else -- the
shipping design (as v45 spliced late_* at the blind airports), which also removes single-seed tree noise at the
nine untouched airports from the verdict. d = RMSE(splice) - RMSE(control), all rows of the three folds pooled.
  1. all rows d <= -0.10 s
  2. no airport worse than +0.10 s (RMSE delta within the airport's rows)
  3. d < 0 after deleting the 50 rows the splice helps most
  4. d < 0 on Jan-Mar rows AND on Jul-Sep rows
REPORTED, not gating: LTFM-only RMSE delta and per fold/month; the UNSPLICED treatment (whole model) with the same
four clauses, per airport; first-order board conversion.
"""
import glob
import sys

import numpy as np
import polars as pl


def load(arm):
    fs = sorted(glob.glob(f"results/preds_ltfmseq_{arm}_f[123]_*_*.parquet"))
    assert len(fs) == 3, f"{arm}: expected 3 fold files, got {fs}"
    return pl.concat([pl.read_parquet(f).select("MVT_ID_mvt", "ADEP_mvt", "y", "bagged") for f in fs]).sort("MVT_ID_mvt")


TREAT = sys.argv[1] if len(sys.argv) > 1 else "treat"   # "treat" = the pre-registered (demeaned) arm
c, t = load("ctrl"), load(TREAT)
print(f"treatment arm: {TREAT}")
assert (c["MVT_ID_mvt"] == t["MVT_ID_mvt"]).all() and (c["y"] == t["y"]).all(), "arms score different rows"
ids = c["MVT_ID_mvt"].to_numpy()
raw = pl.concat([pl.scan_parquet(f).select("MVT_ID_mvt", "MVT_TIME_UTC_mvt").collect()
                 for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
mon = c.join(raw, on="MVT_ID_mvt", how="left", maintain_order="left")["MVT_TIME_UTC_mvt"].dt.month().to_numpy()
y = c["y"].to_numpy().astype(float)
apts = c["ADEP_mvt"].to_numpy().astype(str)
pc = c["bagged"].to_numpy().astype(float)
pt_full = t["bagged"].to_numpy().astype(float)
lt = apts == "LTFM"
pt_splice = np.where(lt, pt_full, pc)
rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean()))  # noqa: E731
allm = np.ones(len(y), bool)
jm, js = np.isin(mon, [1, 2, 3]), np.isin(mon, [7, 8, 9])


def judge(pt, label):
    d_of = lambda m: rm(pt, m) - rm(pc, m)  # noqa: E731
    dsq = (pt - y) ** 2 - (pc - y) ** 2
    top = np.sort(dsq)[:50].sum()
    d50 = float(np.sqrt(((pc - y) ** 2).mean() + (dsq.sum() - top) / len(y)) - rm(pc, allm))
    byapt = {a: d_of(apts == a) for a in sorted(set(apts))}
    checks = {"1. all rows <= -0.10 s": (d_of(allm), d_of(allm) <= -0.10),
              "2. no airport worse than +0.10 s": (max(byapt.values()), max(byapt.values()) <= 0.10),
              "3. negative minus top-50": (d50, d50 < 0),
              "4a. Jan-Mar rows negative": (d_of(jm), d_of(jm) < 0),
              "4b. Jul-Sep rows negative": (d_of(js), d_of(js) < 0)}
    print(f"\n== {label}: {len(y):,} rows; RMSE ctrl {rm(pc, allm):.3f} -> {rm(pt, allm):.3f}")
    for k, (v, ok) in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k:<34} {v:+.3f}")
    print("  by airport: " + "  ".join(f"{a} {v:+.2f}" for a, v in byapt.items()))
    print("  by month:   " + "  ".join(f"{m} {d_of(mon == m):+.2f}" for m in sorted(set(mon))))
    print(f"  VERDICT ({label}): {'PASS' if all(ok for _, ok in checks.values()) else 'FAIL'}")
    return dsq


dsq = judge(pt_splice, "PRIMARY: LTFM splice")
f = ((pt_full[lt] - y[lt]) ** 2).sum() / ((pc[lt] - y[lt]) ** 2).sum() - 1
print(f"\nLTFM only: RMSE {rm(pc, lt):.2f} -> {rm(pt_full, lt):.2f} ({rm(pt_full, lt) - rm(pc, lt):+.2f} s), SSE {100 * f:+.2f}%")
for m in sorted(set(mon[lt])):
    k = lt & (mon == m)
    print(f"  month {m:>2}: LTFM {rm(pt_full, k) - rm(pc, k):+.2f} s (n={k.sum():,})")
# board: LTFM ~47,581 of scored rows at an assumed LTFM board RMSE ~225 s -> SSE 2.41e9; board RMSE 234.97
print(f"  first-order board: {f:+.4f} x 2.41e9 / (2 x 234.97 x 344,841) = {f * 2.41e9 / (2 * 234.97 * 344841):+.3f} s raw, "
      f"x0.77 blind-airport transfer = {0.77 * f * 2.41e9 / (2 * 234.97 * 344841):+.3f} s")
judge(pt_full, "REPORTED: unspliced treatment (whole model)")
