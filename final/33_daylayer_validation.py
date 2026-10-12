"""Validation (a) of the published-daily day layer, as written in docs/LAB-NOTES.md before it was run.

Day-level calibration of delayed flights with lam capped at 0.5 and the bias refitted on each month's own published series,
judged leave-one-month-out on all twelve 2025 months (EDDF EDDM EGLL LEBL LEMD LFPG LIRF). Gain in board seconds (sum of squared
error saved / 1.5e8, positive = better). Construction is archive/agent_genaudit_05_dayeq.py with two changes:

  lam   best of 0.0..0.5 by squared error on the other frame months (each under its own-month bias)
  bias  median of the finite day shifts of the held-out month's own days (label-free)

Base: months 1,2,3,7,8,9,10,11 the faithful held-out chain (agent_waithedge_tbl.served); months 4,5,6,12 the stage + re-stack +
12-month corrector value (agent_genaudit_12m.pk), which has no rules, Rome handling, blend, tail or published-figure pieces.
2025 labels are training data. Run once:  bash final/2x_run.sh "why" final/33_daylayer_validation.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "archive"))
from agent_dayeq2_lib import C, DN_MAX, FRAME_MONTHS, I, R, SE_SCALE, load_pub, solve_delta  # noqa: E402

APTS = ["LFPG", "EGLL", "LEBL", "LEMD", "EDDF", "EDDM", "LIRF"]
LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ROWRULES = ["R1", "R2", "R3a", "R3b"]
PASS_MONTH, PASS_SUM_MONTHS = -0.05, [4, 5, 6, 12]


def one_airport(a: str, sv: pl.DataFrame, g12: pl.DataFrame) -> list[dict]:
    """The whole construction for one airport (one at a time keeps the job under 1 GB)."""
    files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
    raw = (pl.scan_parquet(files).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == a))
           .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y")).collect())
    base = g12.join(sv, on="id", how="left").with_columns(pl.coalesce(C("served"), C("pk")).alias("pred"), C("rule").fill_null(""))
    f = raw.with_columns(C("id").cast(pl.Float64)).join(base.select("id", "pred", "rule"), on="id", how="inner")
    f = f.with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs"), (C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day_pred"))
    f = f.filter(C("gs").is_not_null() & C("y").is_not_null())
    f = f.with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"), C("rule").is_in(ROWRULES).alias("rr"), C("day_pred").dt.month().alias("dmon"))
    f = f.with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60.0).alias("dm"), (~C("amb") & ~C("rr")).alias("sh"))
    pub = load_pub(2025, [a])
    days = (f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum()).rename({"day_pred": "day"})
            .join(pub, on=["apt", "day"], how="inner"))
    days = days.filter(((C("n_pub") - C("n")).abs() <= DN_MAX))
    print(a, "rows", f.height, "gated days", days.height)
    parts = f.partition_by("apt", "day_pred", as_dict=True)
    rows = []
    for r in days.iter_rows(named=True):
        g = parts.get((r["apt"], r["day"]))
        if g is None:
            continue
        m = g["sh"].to_numpy()
        rows.append({"apt": r["apt"], "day": r["day"],
                     "d": solve_delta(g["gs"].to_numpy()[m], g["pred"].to_numpy()[m], r["pub"] - r["fixed"]) if (r["pub"] is not None and np.isfinite(r["pub"])) else np.nan})
    f = f.join(pl.DataFrame(rows), left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
    out = []
    fa = f
    yh, y, d, gs, mon = (fa["pred"].to_numpy(), fa["y"].to_numpy(), fa["d"].to_numpy(), fa["gs"].to_numpy(), fa["dmon"].to_numpy())
    pos = (gs > yh) & fa["sh"].to_numpy() & np.isfinite(d)
    own = {m: float(np.median(d[(mon == m) & np.isfinite(d)])) for m in range(1, 13) if ((mon == m) & np.isfinite(d)).any()}
    bvec = np.array([own.get(int(m), np.nan) for m in mon])
    for m in range(1, 13):
        te = mon == m
        if not te.any() or m not in own:
            continue
        trn = np.isin(mon, FRAME_MONTHS) & (mon != m) & pos & np.isfinite(bvec)
        best = None
        for lam in LAMS:
            sse = ((np.maximum(yh[trn] + lam * (d[trn] - bvec[trn]), 0) - y[trn]) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, lam)
        lam = best[1]
        s_ = te & pos
        new = yh.copy()
        new[s_] = np.maximum(yh[s_] + lam * (d[s_] - own[m]), 0)
        gain = float((((yh - y) ** 2)[te].sum() - ((new - y) ** 2)[te].sum()) / SE_SCALE)
        out.append(dict(apt=a, mon=m, n=int(te.sum()), moved=int((te & (new != yh)).sum()), gain=gain, lam=lam, bias=round(own[m], 1)))
    return out


def main() -> None:
    pl.Config.set_tbl_rows(100)
    sv = pl.read_parquet(I + "agent_waithedge_tbl.parquet", columns=["id", "served", "rule"]).with_columns(C("id").cast(pl.Float64))
    g12 = pl.read_parquet(I + "agent_genaudit_12m.parquet", columns=["id", "pk"]).with_columns(C("id").cast(pl.Float64))
    out = []
    for a in APTS:
        out += one_airport(a, sv, g12)
    res = pl.DataFrame(out)
    tot = res.group_by("mon").agg(gain=C("gain").sum(), moved=C("moved").sum(), n=C("n").sum()).sort("mon")
    print("\ngain by airport and month (board-s)")
    print(res.pivot(on="mon", index="apt", values="gain").sort("apt").with_columns(pl.col(pl.Float64).round(3)))
    print("\nlam used"); print(res.pivot(on="mon", index="apt", values="lam").sort("apt"))
    print("\nown-month bias (s)"); print(res.pivot(on="mon", index="apt", values="bias").sort("apt"))
    print("\nTOTAL by month"); print(tot.with_columns(pl.col("gain").round(3)))
    ctx = pl.read_parquet(I + "agent_genaudit_dayeq.parquet").group_by("mon").agg(shipped_form=C("gain").sum(), own_bias_lam_to_1=C("gain_ob").sum()).sort("mon")
    print("\ncontext, not judged (audit table, no rescoring): shipped form (nested bias, lam up to 1.0) and own-month bias with lam up to 1.0")
    print(tot.join(ctx, on="mon").select("mon", C("gain").alias("capped_own_bias"), "shipped_form", "own_bias_lam_to_1").with_columns(pl.col(pl.Float64).round(3)))
    worst = float(tot["gain"].min())
    s4 = float(tot.filter(C("mon").is_in(PASS_SUM_MONTHS))["gain"].sum())
    c1, c2 = worst >= PASS_MONTH, s4 >= 0
    print(f"\nclause 1: worst month {worst:+.3f} (bar {PASS_MONTH}) -> {'PASS' if c1 else 'FAIL'}")
    print(f"clause 2: sum over months {PASS_SUM_MONTHS} {s4:+.3f} (bar 0) -> {'PASS' if c2 else 'FAIL'}")
    print(f"RESULT: {'PASS' if c1 and c2 else 'FAIL'};  total over twelve months {float(tot['gain'].sum()):+.3f}, months 1+7 {float(tot.filter(C('mon').is_in([1, 7]))['gain'].sum()):+.3f}, months 2+6 {float(tot.filter(C('mon').is_in([2, 6]))['gain'].sum()):+.3f}")
    out_dir = REPO / "data" / "final" / "_gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.write_csv(out_dir / "daylayer_validation_a.csv")


if __name__ == "__main__":
    main()
