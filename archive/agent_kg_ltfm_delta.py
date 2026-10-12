"""LTFM-only serving delta from the Kaggle final arms (tag "kg-ltfm", 2026-09-25).

    python archive/agent_kg_ltfm_delta.py --ctrl results/preds_ltfmfin_ctrl_sK.parquet \
        --treat results/preds_ltfmfin_treat_sK.parquet --base submissions/jolly-lobster_v108.parquet \
        --out data/interim/v1xx_ltfmrefit_final.parquet

delta = treat.final - ctrl.final (post-regime-correction floats of the SAME prc.final config, SAME seeds, both on
Kaggle) on LTFM rows only; new = max(base + delta, 0), rounded; every other row copied from base. Checks: row count equal to the base,
ids and order identical to base, only LTFM rows changed, corr > 0.99, |mean shift| < 5 s, no negatives.
"""
import argparse

import numpy as np
import polars as pl

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ap = argparse.ArgumentParser()
ap.add_argument("--ctrl", required=True)
ap.add_argument("--treat", required=True)
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prepost", default="", help="REPLACE variant: LTFM rows of --base are first reset to this file's "
                "values (the pre-post-step LTFM predictions, e.g. v107 == v103 on every LTFM row), removing the "
                "LTFM post-step before the refit delta is added")
a = ap.parse_args()

c = pl.read_parquet(a.ctrl).select(ID, "ADEP_mvt", c="final")
t = pl.read_parquet(a.treat).select(ID, t="final")
d = c.join(t, on=ID, how="inner")
assert d.height == c.height == t.height, (d.height, c.height, t.height)
d = d.with_columns(delta=pl.when(pl.col("ADEP_mvt") == "LTFM").then(pl.col("t") - pl.col("c")).otherwise(0.0))
off = d.filter(pl.col("ADEP_mvt") != "LTFM")
print(f"non-LTFM ctrl vs treat max |diff| (should be ~0 if bulk-only features change nothing else): "
      f"{(off['t'] - off['c']).abs().max():.3f} s over {off.height:,} rows")
lt = d.filter(pl.col("ADEP_mvt") == "LTFM")["delta"].to_numpy()
print(f"LTFM delta: n {len(lt):,}, mean {lt.mean():+.2f}, rms {np.sqrt((lt ** 2).mean()):.1f}, "
      f"p1/p99 {np.percentile(lt, 1):+.1f}/{np.percentile(lt, 99):+.1f}, max|.| {np.abs(lt).max():.0f}, zero {int((lt == 0).sum()):,}")

base = pl.read_parquet(a.base)
j = base.with_row_index("_o").join(d.select(ID, "ADEP_mvt", "delta"), on=ID, how="left").sort("_o")
assert j.height == base.height == 344841, j.height
miss = j["delta"].null_count()
print(f"base rows without a model row: {miss} (kept unchanged)")
b = base[T].to_numpy().astype(float)
b0 = b.copy()
if a.prepost:
    pp = base.select(ID).join(pl.read_parquet(a.prepost).select(ID, _pp=T), on=ID, how="left", maintain_order="left")
    assert pp.height == base.height and pp["_pp"].null_count() == 0
    isl0 = (j["ADEP_mvt"] == "LTFM").fill_null(False).to_numpy()
    ppv = pp["_pp"].to_numpy().astype(float)
    print(f"--prepost {a.prepost}: resetting LTFM rows ({int(isl0.sum()):,}); rows it moves back: "
          f"{int((np.abs(ppv - b) > 0)[isl0].sum()):,}, rms {np.sqrt(((ppv - b)[isl0] ** 2).mean()):.1f} s")
    b = np.where(isl0, ppv, b)
dl = j["delta"].fill_null(0.0).to_numpy()
isl = (j["ADEP_mvt"] == "LTFM").fill_null(False).to_numpy()
dl = np.where(isl, dl, 0.0)
new = b + dl
# the chain's LOBT box (plan rows with LOBT: [gl - 3,620, gl + 3,620]) re-applied to moved rows, as apply_seq_residual.py does
rk = (pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
      .select(pl.col(ID), pl.col("AOBT_3_flt").is_not_null().alias("plan"),
              (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("gl")).collect())
bx = base.select(ID).join(rk, on=ID, how="left", maintain_order="left")
gl = bx["gl"].to_numpy()
box = bx["plan"].fill_null(False).to_numpy() & np.isfinite(gl) & isl
clipped = np.where(box, np.clip(new, np.ceil(gl - 3620.0), np.floor(gl + 3620.0)), new)
print(f"LOBT box re-applied on LTFM rows: {int((np.abs(clipped - new) > 0).sum())} rows clipped")
new = np.rint(np.maximum(clipped, 0.0))
out = base.with_columns(pl.Series(T, new).cast(base[T].dtype))
b_in = b
b = b0   # every check below is against the file as given (--base)
chg = np.abs(out[T].to_numpy().astype(float) - b) > 0
checks = {
    "rows == 344,841": out.height == 344841,
    "ids identical, same order": bool((out[ID] == base[ID]).all()),
    "only LTFM rows changed": bool(not (chg & ~isl).any()),
    "corr > 0.99": float(np.corrcoef(out[T].to_numpy().astype(float), b)[0, 1]) > 0.99,
    "|mean shift| < 5 s": abs(float((out[T].to_numpy() - b).mean())) < 5,
    "no negatives": int((out[T].to_numpy() < 0).sum()) == 0,
    "no nulls": out[T].null_count() == 0,
    "inside the LOBT box (LTFM plan rows)": int((box & ((out[T].to_numpy() < np.ceil(gl - 3620)) | (out[T].to_numpy() > np.floor(gl + 3620)))).sum()) == 0,
}
dp = out[T].to_numpy().astype(float) - b
print(f"changed rows {int(chg.sum()):,} (LTFM {int((chg & isl).sum()):,} of {int(isl.sum()):,}); corr "
      f"{np.corrcoef(out[T].to_numpy().astype(float), b)[0, 1]:.6f}; mean shift all {dp.mean():+.4f} s, LTFM "
      f"{dp[isl].mean():+.3f} s; rms dp on LTFM {np.sqrt((dp[isl] ** 2).mean()):.2f} s; sum dp^2/344,841 {(dp ** 2).sum() / 344841:.1f}")
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
assert all(checks.values()), "checks failed -- not written"
out.write_parquet(a.out)
print(f"wrote {a.out}")
