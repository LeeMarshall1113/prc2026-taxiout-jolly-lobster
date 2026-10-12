"""agent_big_06: LFPG taxi-phase ADS-B -- the candidate the production pipeline cannot see.

Facts that motivate it (agent_big_05): adsb.lol sees 0.00 LFPG aircraft on the ground per departure in
Jan-Apr 2025, 0.51-0.75 from Sep 2025, 0.63-0.79 in the scored months. Production calibrates ADS-B on
Jan-Sep 2025 (177 days) and requires a detected lift-off (a ground run followed by an airborne fix within
600 s), which at LFPG "fails" (ledger); the 2026 step moves 1.5% of LFPG rows by an RMS 12.5 s.

This test needs no lift-off: each LFPG departure on the Sep 16 - Oct 31 2025 days (fold months 9 and 10,
where the current model's out-of-fold prediction exists) is matched to the aircraft that carries its NM
callsign within [MVT - 90 min, MVT + 15 min]; its ground fixes in [MVT - 90 min, MVT + 60 s] after the last
silence > 20 min give: first ground fix, first fix >= 5 kt, last ground fix, fix count, first speed.
All label-free. Judged against BLOCK, and as an incremental correction to the fold prediction
(results/preds_late2500_treat_f3_3_9 / f4_4_10 'bagged'), 2-fold by day, with the floor
y >= MVT - t(first fix >= 5 kt) checked for violations.
"""
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

C = pl.col
ID = "MVT_ID_mvt"
APT = sys.argv[1] if len(sys.argv) > 1 else "LFPG"
ELEV = {"LFPG": 392, "EGLL": 83, "LEMD": 2001, "LIRF": 15, "EDDF": 364}[APT]
N_BOARD, RMSE_BOARD = 344_841, 239.0145
SSE_BOARD = N_BOARD * RMSE_BOARD ** 2
STEP = N_BOARD * (238.8197 ** 2 - 235.3074 ** 2)
have = sorted(p[:-8] for p in os.listdir("data/adsb/points"))
DAYS = [d for d in have if ("2025-09-16" <= d <= "2025-10-31")]
t0 = time.time()

fold = pl.concat([pl.read_parquet(f"results/preds_late2500_treat_{f}.parquet").select(ID, "bagged", "y")
                  for f in ("f3_3_9", "f4_4_10")]).unique(subset=ID)
deps = (pl.scan_parquet("data/raw/training_*.parquet").filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == APT))
        .select(C(ID), C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day"),
                C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"),
                C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("block_t"),
                C("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("sched_t"),
                C("MVT_TIME_UTC_mvt").dt.hour().alias("hour"),
                C("AOBT_3_flt").is_not_null().alias("plan"))
        .filter(C("day").is_in(DAYS)).collect()).join(fold, on=ID, how="inner")
print(f"{APT}: {deps.height:,} departures with an out-of-fold prediction on {len(DAYS)} days")

rows = []
for day in DAYS:
    p = (pl.scan_parquet(f"data/adsb/points/{day}.parquet").filter(C("apt") == APT)
         .select("icao", "callsign", "t", "alt", "gs").collect())
    p = p.with_columns(((C("alt") == -1) | (((C("alt").cast(pl.Float64) - ELEV).abs() <= 800) & (C("gs") <= 40)))
                       .fill_null(False).alias("ground"),
                       C("callsign").str.strip_chars().str.to_uppercase().alias("cs"))
    d = deps.filter(C("day") == day)
    # callsign -> icao candidates with a fix near the departure's take-off
    cs = (p.filter(C("cs").is_not_null()).select("cs", "icao", "t")
          .join(d.select(ID, C("callsign").alias("cs"), "mvt_t"), on="cs", how="inner")
          .filter((C("t") >= C("mvt_t") - 5400) & (C("t") <= C("mvt_t") + 900))
          .group_by([ID, "icao"]).len().sort("len", descending=True).unique(subset=ID, keep="first"))
    g = p.filter(C("ground")).select("icao", "t", "gs").sort(["icao", "t"])
    tr = {k[0]: (v["t"].to_numpy(), v["gs"].fill_null(0).to_numpy()) for k, v in g.partition_by("icao", as_dict=True).items()}
    icao_of = dict(zip(cs[ID].to_list(), cs["icao"].to_list()))
    for r in d.iter_rows(named=True):
        ic = icao_of.get(r[ID])
        rec = [r[ID], day, r["mvt_t"], r["block_t"], r["hour"], r["plan"], r["bagged"], r["y"], ic is not None]
        feats = [np.nan] * 6
        if ic is not None and ic in tr:
            t, gs = tr[ic]
            k = np.flatnonzero((t >= r["mvt_t"] - 5400) & (t <= r["mvt_t"] + 60))
            if len(k):
                tt, gg = t[k], gs[k]
                brk = np.flatnonzero(np.diff(tt) > 1200)
                if len(brk):
                    tt, gg = tt[brk[-1] + 1:], gg[brk[-1] + 1:]
                mv = np.flatnonzero(gg >= 5)
                feats = [tt[0], tt[mv[0]] if len(mv) else np.nan, tt[-1], float(len(tt)), float(gg[0]),
                         float(np.median(np.diff(tt))) if len(tt) > 1 else np.nan]
        rows.append(rec + feats)
    del p, g, tr
print(f"built in {time.time() - t0:.0f}s")
df = pl.DataFrame(rows, schema=[ID, "day", "mvt_t", "block_t", "hour", "plan", "p", "y", "cs_match",
                                "t_first", "t_mv", "t_last", "n_fix", "gs_first", "dt_med"], orient="row", infer_schema_length=None)
y, p = df["y"].to_numpy(), np.clip(df["p"].to_numpy(), 0, None)
has = np.isfinite(df["t_first"].to_numpy())
print(f"callsign-matched {df['cs_match'].mean():.1%}; with ground fixes before take-off {has.mean():.1%}")
tf, tm, tl = df["t_first"].to_numpy(), df["t_mv"].to_numpy(), df["t_last"].to_numpy()
blk, mvt = df["block_t"].to_numpy(), df["mvt_t"].to_numpy()
for name, a in (("first ground fix", tf), ("first fix >= 5 kt", tm)):
    ok = np.isfinite(a)
    off = a[ok] - blk[ok]
    med = np.median(off)
    print(f"  {name:18s}: n {ok.sum():,} ({ok.mean():.1%})  offset vs BLOCK median {med:+.0f}s  IQR {np.percentile(off, 25):+.0f}..{np.percentile(off, 75):+.0f}"
          f"  RMSE|med {np.sqrt(np.mean((off - med) ** 2)):.0f}s  within 90 s {np.mean(np.abs(off - med) <= 90):.1%}")
print(f"  last ground fix vs MVT: median {np.nanmedian(tl - mvt):+.0f}s (lift-off seen if near 0)")
print(f"  model RMSE on rows with fixes {np.sqrt(np.mean((y[has] - p[has]) ** 2)):.1f}s, all rows {np.sqrt(np.mean((y - p) ** 2)):.1f}s")
# the floor: an aircraft moving at >= 5 kt has left its stand
okm = np.isfinite(tm)
lb = mvt[okm] - tm[okm]
viol = y[okm] < lb - 30
print(f"  floor y >= MVT - t(first >= 5 kt): violated (by > 30 s) on {viol.mean():.2%} of {okm.sum():,} rows; "
      f"binds (p < lb) on {(p[okm] < lb).mean():.1%}")
fl = p.copy()
fl[okm] = np.maximum(p[okm], lb)
print(f"  hard floor alone: SSE saved {np.sum((y - p) ** 2) - np.sum((y - fl) ** 2):.3e} over {df.height:,} rows")

# incremental learner, 2-fold by day
X = np.column_stack([mvt - tf - p, mvt - tm - p, mvt - tl, df["n_fix"].to_numpy(), df["gs_first"].to_numpy(),
                     df["dt_med"].to_numpy(), p, df["hour"].to_numpy().astype(float), mvt - tf, mvt - tm])
tgt = np.clip(y - p, -1800, 1800)
days = sorted(df["day"].unique().to_list())
fo = np.array([days.index(d) % 2 for d in df["day"].to_list()])
new = p.copy()
for f in (0, 1):
    a, b = fo != f, fo == f
    m = lgb.LGBMRegressor(n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.04, min_child_samples=80,
                          subsample=0.8, subsample_freq=1, verbose=-1).fit(X[a], tgt[a])
    new[b] = p[b] + np.clip(m.predict(X[b]), -1800, 1800)
gain = (y - p) ** 2 - (y - new) ** 2
per = gain.sum() / df.height
print(f"\nINCREMENTAL LightGBM over the fold prediction (2-fold by day): RMSE {np.sqrt(np.mean((y - p) ** 2)):.2f} -> "
      f"{np.sqrt(np.mean((y - new) ** 2)):.2f}; {per:+.0f} s^2 per {APT} departure; rows with fixes {gain[has].sum() / has.sum():+.0f}/row")
print(f"  minus top-50 helped: {np.sort(gain)[:-50].sum() / df.height:+.0f}/dep; minus top-200: {np.sort(gain)[:-200].sum() / df.height:+.0f}/dep")
for f in (0, 1):
    print(f"  fold {f}: {gain[fo == f].sum() / (fo == f).sum():+.0f}/dep")
n26 = {"LFPG": 39_872, "EGLL": 39_840, "LEMD": 36_954, "LIRF": 26_899, "EDDF": 36_317}[APT]
tot = per * n26
print(f"  scaled to {n26:,} scored {APT} rows at the same coverage, 1x: {tot:.2e} s^2 = {tot / STEP:.2f} leader steps = "
      f"{np.sqrt((SSE_BOARD - tot) / N_BOARD) - RMSE_BOARD:+.2f}s board")
tot2 = np.sort(gain)[:-50].sum() / df.height * n26
print(f"  same, minus top-50: {np.sqrt((SSE_BOARD - tot2) / N_BOARD) - RMSE_BOARD:+.2f}s board")
