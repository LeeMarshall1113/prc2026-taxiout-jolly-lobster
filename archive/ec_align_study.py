"""EUROCONTROL monthly taxi-out alignment: 2025 leave-one-month-out validation + 2026 sizing (2026-09-23).

EUROCONTROL's open "additional taxi-out time" file (organiser-cleared AIU data) gives, per airport-month, the number of
departures TF, the valid ones VALID_FL (TF - VALID_FL = excluded: taxi-out > 120 min, de-icing, missing, helicopters),
and total taxi-out minutes over the TOTAL_REF_NB_FL flights with a reference. Its mean ec = total / nref tracks the TRUE
mean of y <= 7200 within a per-airport offset (2025: EGLL -0.8+-0.6 s, LIRF 9.3+-4.0 s ...).

Alignment: for airport a, month m, target T = ec(a,m) + off(a,m), off = mean over the OTHER 2025 months of the same
season-type of (true mean y<=7200 - ec). Rows: all rows except the K highest predictions, K = the number of true y > 7200
rows implied (EC excluded count where excl tracks monsters, else our own count of predictions > 7200). Shift those rows by
T - mean(pred on them). Validation (2025): apply to the late2500 fold predictions of months 3,4,9,10 (pre-chain,
CatBoost bulk) and measure the RMSE change per airport-month with the offset estimated leave-one-month-out.
"""
import numpy as np
import polars as pl

from prc.config import RAW_DIR

C = pl.col
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
MONSTER_EXCL = {"LIRF", "EGLL"}          # airports where 2025 EC excluded count == count of y > 7200


def ec(year):
    e = pl.read_csv(f"data/external/eurocontrol/taxi_out_additional_time_{year}.csv", infer_schema_length=0)
    return e.filter(C("APT_ICAO").is_in(APTS)).select(
        C("MONTH_NUM").cast(int).alias("mo"), C("APT_ICAO").alias("apt"),
        (C("TF").cast(int) - C("VALID_FL").cast(int)).alias("excl"),
        ((C("TF").cast(int) - C("VALID_FL").cast(int)) / C("TF").cast(int)).alias("xf"),
        ((C("TOTAL_REF_TIME_MIN").cast(float) + C("TOTAL_ADD_TIME_MIN").cast(float)) * 60
         / C("TOTAL_REF_NB_FL").cast(int)).alias("ec_mean"))


tr = (pl.scan_parquet(sorted(str(p) for p in RAW_DIR.glob("training_*.parquet"))).filter(C("PHASE_mvt") == "DEP")
      .select("MVT_ID_mvt", C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
              C("TAXITIME_SEC_mvt").alias("y")).collect())
gap = (tr.filter(C("y") <= 7200).group_by("apt", "mo").agg(C("y").mean().alias("true_mean"))
       .join(ec(2025), on=["apt", "mo"]).with_columns((C("true_mean") - C("ec_mean")).alias("gap")))
SKIP = {"LTFM"}   # offset not predictable (LOMO sd 149 s)


def offset(apt, xf, exclude_mo=None):
    """EC-to-truth offset from a per-airport linear fit on the excluded (de-icing etc.) fraction, 2025 months."""
    g = gap.filter((C("apt") == apt) & (C("mo") != (exclude_mo or -1)))
    x, yv = g["xf"].to_numpy(), g["gap"].to_numpy()
    b, a = np.polyfit(x, yv, 1)
    res = yv - (a + b * x)
    return float(a + b * xf), float(np.std(res))


def align(frame, pred_col, emap, lomo):
    """frame: MVT_ID_mvt, apt, mo, pred. Returns new predictions and a per-cell log."""
    out = frame[pred_col].to_numpy().astype(float).copy()
    log = []
    for apt, mo, idx in frame.with_row_index("_i").group_by("apt", "mo").agg(C("_i")).rows():
        if (apt, mo) not in emap:
            continue
        if apt in SKIP:
            continue
        ecm, excl, xf = emap[(apt, mo)]
        off, sd = offset(apt, xf, exclude_mo=mo if lomo else None)
        idx = np.asarray(idx)
        p = out[idx]
        K = excl if apt in MONSTER_EXCL else int((p > 7200).sum())
        keep = idx[np.argsort(p)[: max(len(idx) - K, 0)]]
        shift = (ecm + off) - out[keep].mean()
        out[keep] += shift
        log.append((apt, mo, len(idx), K, round(off, 1), round(sd, 1), round(shift, 1)))
    return np.clip(out, 0, None), log


# ---- 2025 validation on the late2500 fold predictions (months 3,9 and 4,10) ----
e25 = {(r["apt"], r["mo"]): (r["ec_mean"], r["excl"], r["xf"]) for r in ec(2025).iter_rows(named=True)}
print("2025 LEAVE-ONE-MONTH-OUT validation on fold predictions (pre-chain):")
tot_before = tot_after = 0.0
rows = []
for f in ("results/preds_late2500_treat_f3_3_9.parquet", "results/preds_late2500_treat_f4_4_10.parquet"):
    d = pl.read_parquet(f).select(C("MVT_ID_mvt"), "y", "bagged").join(tr.select("MVT_ID_mvt", "apt", "mo"), on="MVT_ID_mvt")
    new, log = align(d, "bagged", e25, lomo=True)
    y, old = d["y"].to_numpy().astype(float), d["bagged"].to_numpy().astype(float)
    for (apt, mo, n, K, off, sd, sh) in log:
        m = ((d["apt"] == apt) & (d["mo"] == mo)).to_numpy()
        b, a = ((old[m] - y[m]) ** 2).sum(), ((new[m] - y[m]) ** 2).sum()
        rows.append((apt, mo, n, sh, np.sqrt(b / m.sum()), np.sqrt(a / m.sum()), (a - b)))
    tot_before += ((old - y) ** 2).sum(); tot_after += ((new - y) ** 2).sum()
    N = len(y)
print(f"{'apt':5}{'mo':>3}{'n':>8}{'shift':>8}{'rmse_old':>10}{'rmse_new':>10}{'dSSE':>14}")
for r in sorted(rows):
    print(f"{r[0]:5}{r[1]:>3}{r[2]:>8,}{r[3]:>8.1f}{r[4]:>10.2f}{r[5]:>10.2f}{r[6]:>14,.0f}")
nrow = sum(r[2] for r in rows)
print(f"pooled RMSE {np.sqrt(tot_before / nrow):.3f} -> {np.sqrt(tot_after / nrow):.3f}  "
      f"(board-scale d ~ {(tot_after - tot_before) / (2 * nrow * 238):+.3f} s at 1x); cells better "
      f"{sum(r[6] < 0 for r in rows)}/{len(rows)}")

# ---- 2026 sizing on v98 ----
e26 = {(r["apt"], r["mo"]): (r["ec_mean"], r["excl"], r["xf"]) for r in ec(2026).iter_rows(named=True) if r["mo"] in (1, 7)}
rk = (pl.scan_parquet(RAW_DIR / "ranking.parquet").filter(C("PHASE_mvt") == "DEP")
      .select("MVT_ID_mvt", C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo")).collect())
v = pl.read_parquet("submissions/jolly-lobster_v98.parquet").select("MVT_ID_mvt", C("TAXITIME_SEC_mvt").alias("p"))
d = rk.join(v, on="MVT_ID_mvt")
new, log = align(d, "p", e26, lomo=False)
print("\n2026 shifts on v98 (apt, month, rows, K excluded, offset, offset sd, SHIFT s, gain if exact n*shift^2 -> board s):")
for (apt, mo, n, K, off, sd, sh) in sorted(log):
    g = (n - K) * (sh ** 2 - sd ** 2)
    print(f"  {apt} {mo}: n {n:,} K {K} off {off:+.1f}+-{sd:.1f}  shift {sh:+.1f}  board ~{-max(g, 0) / (2 * 344841 * 238):+.3f}")
