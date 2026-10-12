"""romenear 02b: Item 1 widened. The shuffled control of 02 said LOBT = SCHED does not single out the rows: every Rome hard-binding plan row
(lower window edge > thr) has its label near take-off minus LOBT. Judge the rule on ALL Rome hard-binding plan rows, with and without the
signal, by month / operator / regime folds, per-row gain distribution, and against the other airports (is it Rome-specific?)."""
import numpy as np, polars as pl
C = pl.col
R = ""; I = R + "data/interim/"; OUT = R + "data/interim/local/romenear/"
N26, RMSE = 344841.0, 217.28; W = 3606.0
f = pl.read_parquet(I + "agent_rome1_frame.parquet")
v146 = pl.read_parquet(I + "agent_bundle_v146_final.parquet").select(C("MVT_ID_mvt"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("v146"))
r26 = pl.read_parquet(I + "agent_rome1c_rows.parquet").select("MVT_ID_mvt", C("P").alias("P26"))
f = f.join(v146, on="MVT_ID_mvt", how="left").join(r26, on="MVT_ID_mvt", how="left")
f = f.filter(~C("np") & C("gl").is_not_null() & C("aobt").is_not_null()).with_columns(
    (C("gl") - W).alias("lo"), (C("lobt") - C("sched")).alias("ls"), pl.when(C("mo") <= 4).then(pl.lit("dark")).otherwise(pl.lit("lit")).alias("reg"),
    (C("gl") - W - C("ga")).alias("hard_s"))


def fitw(y, a, t):
    d = t - a; r = y - a
    return 0.0 if (d ** 2).sum() == 0 else float(np.clip((d * r).sum() / (d ** 2).sum(), 0, 1.2))


def sse(y, v): return float(((y - v) ** 2).sum())


for thr in (1800, 2800, 3600):
    hb = f.filter((C("yr") == 2025) & (C("lo") > thr) & C("ship12").is_not_null() & C("y").is_not_null())
    y = hb["y"].to_numpy().astype(float); a = hb["ship12"].to_numpy().astype(float); gl = hb["gl"].to_numpy().astype(float); lo = hb["lo"].to_numpy().astype(float)
    op = hb["op"].fill_null("NA").to_numpy().astype(str); reg = hb["reg"].to_numpy().astype(str); mo = hb["mo"].to_numpy(); sig = (hb["ls"].abs() <= 60).to_numpy()
    n = len(y); pos = (y - lo) / W; cpos = (a - lo) / W
    print(f"\nthr {thr}: Rome hard-binding plan rows n {n} (LOBT=SCHED {sig.sum()}); label pos mean {pos.mean():.2f} med {np.median(pos):.2f} p10 {np.percentile(pos,10):.2f} p90 {np.percentile(pos,90):.2f}; chain pos mean {cpos.mean():.2f}; at LOBT {(np.abs(y-gl)<=60).sum()}; at edge {(np.abs(y-lo)<=60).sum()}")
    print("   by month: " + ", ".join(f"{m}: n {int((mo==m).sum())} pos {pos[mo==m].mean():.2f} chain {cpos[mo==m].mean():.2f}" for m in np.unique(mo)))
    w_in = fitw(y, a, gl); gm = np.zeros(n); pm = a.copy()
    for m in np.unique(mo):
        tr = mo != m; te = ~tr; w = fitw(y[tr], a[tr], gl[tr]); pm[te] = a[te] + w * (gl[te] - a[te])
    gm = (y - a) ** 2 - (y - pm) ** 2
    pl_ = a.copy()
    for o in np.unique(op):
        tr = op != o; te = ~tr; w = fitw(y[tr], a[tr], gl[tr]); pl_[te] = a[te] + w * (gl[te] - a[te])
    go = (y - a) ** 2 - (y - pl_) ** 2
    pt = a.copy(); wt = {}
    for trr, ter in (("lit", "dark"), ("dark", "lit")):
        tr = reg == trr; te = reg == ter; w = fitw(y[tr], a[tr], gl[tr]); wt[ter] = w; pt[te] = a[te] + w * (gl[te] - a[te])
    gt = (y - a) ** 2 - (y - pt) ** 2
    srt = np.sort(gm)[::-1]
    print(f"   w in-sample {w_in:.2f}; LOMO gain {gm.sum()/1e6:.1f}M months better {sum(gm[mo==m].sum()>0 for m in np.unique(mo))}/{len(np.unique(mo))}; LOAO {go.sum()/1e6:.1f}M ops better {sum(go[op==o].sum()>0 for o in np.unique(op))}/{len(np.unique(op))}; temporal lit->dark w {wt['dark']:.2f} gain {gt[reg=='dark'].sum()/1e6:.1f}M on {(reg=='dark').sum()} rows, dark->lit w {wt['lit']:.2f} gain {gt[reg=='lit'].sum()/1e6:.1f}M on {(reg=='lit').sum()} rows")
    print(f"   LOMO: rows worse {(gm<0).sum()}, minus top-5 {(gm.sum()-srt[:5].sum())/1e6:.1f}M, top-10 {(gm.sum()-srt[:10].sum())/1e6:.1f}M, top-20 {(gm.sum()-srt[:20].sum())/1e6:.1f}M; top-5 share {srt[:5].sum()/gm.sum():.2f}; median per-row gain {np.median(gm)/1e3:.0f}k; signal rows gain {gm[sig].sum()/1e6:.1f}M, non-signal {gm[~sig].sum()/1e6:.1f}M")
    print(f"   where the chain already sits above pos 0.9 ({(cpos>0.9).sum()} rows) gain {gm[cpos>0.9].sum()/1e6:.1f}M; below 0.9 ({(cpos<=0.9).sum()}) gain {gm[cpos<=0.9].sum()/1e6:.1f}M")
    # scored
    s26 = f.filter((C("yr") == 2026) & (C("lo") > thr))
    a26 = s26["v146"].to_numpy().astype(float); gl26 = s26["gl"].to_numpy().astype(float); lo26 = s26["lo"].to_numpy().astype(float); sig26 = (s26["ls"].abs() <= 60).to_numpy()
    new26 = a26 + w_in * (gl26 - a26)
    ysim = lo26[:, None] + pos[None, :] * W; eg = (((ysim - a26[:, None]) ** 2) - ((ysim - new26[:, None]) ** 2)).mean(1)
    ysim2 = lo26[:, None] + cpos[None, :] * W; lw = (((ysim2 - a26[:, None]) ** 2) - ((ysim2 - new26[:, None]) ** 2)).mean(1)
    print(f"   scored {s26.height} (Jan {(s26['mo']==1).sum()}, Jul {(s26['mo']==7).sum()}; LOBT=SCHED {sig26.sum()}); served pos {((a26-lo26)/W).mean():.2f}; new pos {((new26-lo26)/W).mean():.2f}; expected gain {eg.sum()/(2*N26*RMSE):+.3f} s (minus top 5 {(eg.sum()-np.sort(eg)[::-1][:5].sum())/(2*N26*RMSE):+.3f}); loss if chain positions {lw.sum()/(2*N26*RMSE):+.3f}; all at edge {(((lo26-a26)**2)-((lo26-new26)**2)).sum()/(2*N26*RMSE):+.3f}")
    s26.select("MVT_ID_mvt", "mo", "op", "FLIGHT_mvt", "gs", "gl", "lo", "ga", "ls", "v146", "P26").with_columns(pl.Series("new", np.rint(new26)), pl.Series("eg_s2", eg)).write_csv(OUT + f"item1_wide_scored_thr{thr}.csv")

# other airports (2025 raw): label position on hard-binding plan rows, for the Rome-specific claim
print("\n=== other airports, 2025 hard-binding plan rows (lo > 2800): label position ===")
tr = (pl.scan_parquet(R + "data/raw/training_*.parquet").filter((C("PHASE_mvt") == "DEP") & C("AOBT_3_flt").is_not_null() & C("LOBT_flt").is_not_null())
      .select("ADEP_mvt", (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"), (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
              (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y"), (C("LOBT_flt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("ls"))
      .filter(C("gl") - W > 2800).collect())
tr = tr.with_columns(((C("y") - (C("gl") - W)) / W).alias("pos"), ((C("y") - C("gl")).abs() <= 60).alias("at_lobt"), (C("ls").abs() <= 60).alias("sig"))
print(tr.group_by("ADEP_mvt", "sig").agg(pl.len().alias("n"), C("pos").mean().round(2).alias("pos_mean"), C("pos").median().round(2).alias("pos_med"), C("at_lobt").mean().round(2).alias("at_lobt")).sort("ADEP_mvt", "sig"))
print("done")
