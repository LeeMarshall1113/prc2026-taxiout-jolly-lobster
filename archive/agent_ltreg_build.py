"""experiment ltreg: build the LTFM snow serving delta on a base file (docs/LAB-NOTES.md, serving section).

    python archive/agent_ltreg_build.py validate
    python archive/agent_ltreg_build.py build --base data/interim/v122_stage9s_final.parquet --out data/interim/v137_ltfmsnow_final.parquet

Treatment prepost = (k0 + 2 * k12) / 3 (the 3-seed bagged routed prediction, seeds 1113 / 2090 / 3067), then the disruption-day
regime correction re-applied through prc.crossval.apply_regime_correction (lam 1.0, quantile 0.99 -- the refit's flags).
Control = results/preds_ltfmfin_treat_s3.parquet `final` (the shipped v113 refit, same seeds). delta = treat - ctrl on LTFM rows.
`validate` first proves the local correction reproduces the control's own final from its prepost.
"""
import argparse
import glob
import json

import numpy as np
import polars as pl

from prc.crossval import apply_regime_correction

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
CTRL = "results/preds_ltfmfin_treat_s3.parquet"


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


def correct(prepost, ids):
    rank = regime_rows("data/raw/ranking.parquet", ids)
    train = regime_rows(sorted(glob.glob("data/raw/training_2025-*.parquet")), with_target=True)
    out, n_ps, n_nr, n_days = apply_regime_correction(rank, prepost, train, 1.0, 0.99)
    print(f"  regime correction: {n_days} hot airport-days, {n_ps:,} plan rows shifted, {n_nr:,} no-plan raised")
    return out


def validate():
    c = pl.read_parquet(CTRL)
    got = correct(c["prepost"].to_numpy(), c[ID].to_numpy())
    d = np.abs(got - c["final"].to_numpy())
    lt = (c["ADEP_mvt"] == "LTFM").to_numpy()
    res = {"rows": c.height, "max_abs": float(d.max()), "ltfm_max_abs": float(d[lt].max()), "pass": bool(d.max() < 1e-6)}
    print("VALIDATE", res)
    json.dump(res, open("logs/agent_ltreg_build_validate.json", "w"), indent=1)


def build(base_path, out_path, k0p="results/preds_lrfin_snow_k0.parquet", k12p="results/preds_lrfin_snow_k12.parquet",
          dout="data/interim/agent_ltreg_snow_delta26.parquet"):
    c = pl.read_parquet(CTRL)
    k0 = pl.read_parquet(k0p).select(ID, pl.col("prepost").alias("p0"))
    k12 = pl.read_parquet(k12p).select(ID, pl.col("prepost").alias("p12"))
    j = c.join(k0, on=ID, how="left", maintain_order="left").join(k12, on=ID, how="left", maintain_order="left")
    assert j.height == c.height == 344841 and j["p0"].null_count() == 0 and j["p12"].null_count() == 0
    lt = (j["ADEP_mvt"] == "LTFM").to_numpy()
    for col in ("p0", "p12"):   # non-LTFM rows: the snow features are NaN there, but seeds differ -> only LTFM is used
        print(f"  {col} vs ctrl prepost on LTFM: rms {np.sqrt(((j[col] - j['prepost']).to_numpy()[lt] ** 2).mean()):.2f}")
    pp = (j["p0"].to_numpy() + 2.0 * j["p12"].to_numpy()) / 3.0
    treat = correct(pp, j[ID].to_numpy())
    delta = np.where(lt, treat - j["final"].to_numpy(), 0.0)
    rk = pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        ID, pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo")).collect()
    mo = j.select(ID).join(rk, on=ID, how="left", maintain_order="left")["mo"].to_numpy()
    print(f"delta on LTFM ({lt.sum():,} rows): mean {delta[lt].mean():+.2f} rms {np.sqrt((delta[lt] ** 2).mean()):.2f} "
          f"p1/p99 {np.percentile(delta[lt], 1):+.0f}/{np.percentile(delta[lt], 99):+.0f} max|.| {np.abs(delta[lt]).max():.0f}")
    for m in (1, 7):
        s = lt & (mo == m)
        print(f"  2026-{m:02d} LTFM: n {s.sum():,} mean {delta[s].mean():+.2f} rms {np.sqrt((delta[s] ** 2).mean()):.2f}")
    pl.DataFrame({ID: j[ID], "ADEP_mvt": j["ADEP_mvt"], "delta": delta}).write_parquet(dout)
    print(f"2026 sum dp^2/row (raw delta, all rows): {(delta ** 2).sum() / 344841:.2f}")

    base = pl.read_parquet(base_path)
    bj = base.select(ID).join(pl.DataFrame({ID: j[ID], "d": delta, "lt": lt}), on=ID, how="left", maintain_order="left")
    assert base.height == 344841 and bj["d"].null_count() == 0
    b = base[T].to_numpy().astype(float)
    isl = bj["lt"].to_numpy()
    new = b + bj["d"].to_numpy()
    rb = (pl.scan_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
          .select(pl.col(ID), pl.col("AOBT_3_flt").is_not_null().alias("plan"),
                  (pl.col("MVT_TIME_UTC_mvt") - pl.col("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("gl")).collect())
    bx = base.select(ID).join(rb, on=ID, how="left", maintain_order="left")
    gl = bx["gl"].to_numpy()
    box = bx["plan"].fill_null(False).to_numpy() & np.isfinite(gl) & isl
    clipped = np.where(box, np.clip(new, np.ceil(gl - 3620.0), np.floor(gl + 3620.0)), new)
    print(f"LOBT box re-applied on LTFM rows: {int((np.abs(clipped - new) > 0).sum())} rows clipped")
    new = np.rint(np.maximum(clipped, 0.0))
    out = base.with_columns(pl.Series(T, new).cast(base[T].dtype))
    dp = new - b
    chg = np.abs(dp) > 0
    print(f"changed rows {int(chg.sum()):,} (LTFM {int((chg & isl).sum()):,} of {int(isl.sum()):,}); non-LTFM changed "
          f"{int((chg & ~isl).sum())}; mean shift all {dp.mean():+.4f}, LTFM {dp[isl].mean():+.3f}; served sum dp^2/344,841 "
          f"{(dp ** 2).sum() / 344841:.2f}")
    assert not (chg & ~isl).any()
    out.write_parquet(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--k0", default="results/preds_lrfin_snow_k0.parquet")
    ap.add_argument("--k12", default="results/preds_lrfin_snow_k12.parquet")
    ap.add_argument("--delta-out", default="data/interim/agent_ltreg_snow_delta26.parquet")
    a = ap.parse_args()
    validate() if a.cmd == "validate" else build(a.base, a.out, a.k0, a.k12, a.delta_out)
