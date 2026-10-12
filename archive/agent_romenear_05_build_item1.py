"""romenear 05: build the Item 1 candidate on v146. Rome plan rows whose lower window edge (take-off - LOBT - 3,606) is more than 2,800 s
(so an ordinary taxi is excluded by the plan window and the stamp cannot be the push the plan saw): the 2025 label sits at take-off minus
LOBT (window position 0.93 mean, 1.00 median, 50 rows, 24 within 60 s of LOBT, 0 at the edge) while the served chain sits at 0.70.
new = v146 + w x (take-off - LOBT - v146), w = least squares on the 50 rows of 2025 (0.94). Hedged rows untouched (none are plan rows).
Also prints the box-rule definition of hard binding at Rome (lower edge above NM's own taxi by 1,800 s) for the record."""
import numpy as np, polars as pl, subprocess, sys
C = pl.col
R = ""; I = R + "data/interim/"; OUT = R + "data/interim/local/romenear/"
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; W = 3606.0; THR = 2800.0; HARD = 1800.0
# family = lower edge > THR AND the plan's own push (AOBT_3 reading) below the edge by more than HARD (the box rule's hard-binding test):
# excludes rows like ITY340 whose AOBT_3 reading is inside the window (the label then sits at take-off minus AOBT_3, logic audit L1).
f = pl.read_parquet(I + "agent_rome1_frame.parquet").filter(~C("np") & C("gl").is_not_null() & C("aobt").is_not_null()).with_columns((C("gl") - W).alias("lo"), (C("lobt") - C("sched")).alias("ls"))
HB = (C("lo") > THR) & (C("lo") - C("ga") > HARD)
t25 = f.filter((C("yr") == 2025) & HB & C("ship12").is_not_null() & C("y").is_not_null())
y = t25["y"].to_numpy().astype(float); a = t25["ship12"].to_numpy().astype(float); gl = t25["gl"].to_numpy().astype(float); lo = t25["lo"].to_numpy().astype(float)
d = gl - a; w_ls = float(np.clip((d * (y - a)).sum() / (d ** 2).sum(), 0, 1.2)); w = min(w_ls, 1.0)  # capped at 1: serve take-off minus LOBT, never beyond it (the mechanism is a stamp AT LOBT)
sig = (t25["ls"].abs() <= 60).to_numpy(); pos = (y - lo) / W
mo = t25["mo"].to_numpy(); op = t25["op"].fill_null("NA").to_numpy().astype(str)
def fw(m):
    dd = d[m]; return 0.0 if (dd ** 2).sum() == 0 else float(np.clip((dd * (y[m] - a[m])).sum() / (dd ** 2).sum(), 0, 1.2))
pm = a.copy(); po = a.copy()
for k in np.unique(mo): te = mo == k; pm[te] = a[te] + fw(~te) * d[te]
for o in np.unique(op): te = op == o; po[te] = a[te] + fw(~te) * d[te]
gm = (y - a) ** 2 - (y - pm) ** 2; go = (y - a) ** 2 - (y - po) ** 2; srt = np.sort(gm)[::-1]
print(f"LOMO gain {gm.sum()/1e6:.1f}M (months better {sum(gm[mo==k].sum()>0 for k in np.unique(mo))}/{len(np.unique(mo))}, rows worse {(gm<0).sum()}, minus top-5 {(gm.sum()-srt[:5].sum())/1e6:.1f}M, minus top-10 {(gm.sum()-srt[:10].sum())/1e6:.1f}M, minus top-20 {(gm.sum()-srt[:20].sum())/1e6:.1f}M); LOAO gain {go.sum()/1e6:.1f}M (ops better {sum(go[op==o].sum()>0 for o in np.unique(op))}/{len(np.unique(op))}); dark rows {(mo<=4).sum()} gain {gm[mo<=4].sum()/1e6:.1f}M, lit rows {(mo>=5).sum()} gain {gm[mo>=5].sum()/1e6:.1f}M; by month " + ", ".join(f"{k}: n {(mo==k).sum()} pos {pos[mo==k].mean():.2f} chain {((a[mo==k]-lo[mo==k])/W).mean():.2f}" for k in np.unique(mo)))
print(f"2025 rows {len(y)}, w least-squares {w_ls:.3f} used {w:.2f} (in-sample gain at w {((y-a)**2-(y-(a+w*d))**2).sum()/1e6:.1f}M vs at w_ls {((y-a)**2-(y-(a+w_ls*d))**2).sum()/1e6:.1f}M); label pos: LOBT=SCHED rows mean {pos[sig].mean():.2f} (n {sig.sum()}), other rows mean {pos[~sig].mean():.2f} (n {(~sig).sum()}), other rows pos quantiles {np.round(np.percentile(pos[~sig], [10, 25, 50, 75, 90]), 2)}")
box = f.filter((C("yr") == 2025) & (C("gl") - W - C("ga") > 1800) & C("y").is_not_null())
bp = ((box["y"] - (box["gl"] - W)) / W).to_numpy()
print(f"box-rule definition at Rome (lower edge > NM taxi + 1,800): {box.height} rows, label pos mean {bp.mean():.2f} median {np.median(bp):.2f}; overlap with lo > {THR:.0f}: {box.filter(C('lo') > THR).height}")
s26 = f.filter((C("yr") == 2026) & HB).select(ID, "mo", "op", "FLIGHT_mvt", "gs", "gl", "lo", "ga", "ls")
v = pl.read_parquet(I + "agent_bundle_v146_final.parquet")
j = s26.join(v.select(ID, C(T).cast(pl.Float64).alias("cur")), on=ID, how="left")
hedged = ["WMTT3508", "SVII3006", "ITYTY672", "NOSOS431", "QFAA7006", "TAY9LA"]
assert not j["FLIGHT_mvt"].is_in(hedged).any(), "hedged row in scope"
cur = j["cur"].to_numpy(); g26 = j["gl"].to_numpy().astype(float); lo26 = j["lo"].to_numpy().astype(float)
new = np.rint(np.clip(cur + w * (g26 - cur), lo26, g26 + W))
j = j.with_columns(pl.Series("new", new), ((C("cur") - C("lo")) / W).round(2).alias("served_pos"), pl.Series("new_pos", np.round((new - lo26) / W, 2)))
print(j.select("mo", "op", "FLIGHT_mvt", "gs", "gl", "ga", "ls", "cur", "new", "served_pos", "new_pos"))
N26, RMSE = 344841.0, 217.28
ysim = lo26[:, None] + pos[None, :] * W; eg = (((ysim - cur[:, None]) ** 2) - ((ysim - new[:, None]) ** 2)).mean(1)
cpos = (a - lo) / W; ysim2 = lo26[:, None] + cpos[None, :] * W; lw = (((ysim2 - cur[:, None]) ** 2) - ((ysim2 - new[:, None]) ** 2)).mean(1)
print(f"expected gain under the 2025 label positions {eg.sum()/(2*N26*RMSE):+.3f} s (minus top 5 rows {(eg.sum()-np.sort(eg)[::-1][:5].sum())/(2*N26*RMSE):+.3f}; Jan {eg[j['mo'].to_numpy()==1].sum()/(2*N26*RMSE):+.3f}, Jul {eg[j['mo'].to_numpy()==7].sum()/(2*N26*RMSE):+.3f}); loss if the labels follow the chain's 2025 positions {lw.sum()/(2*N26*RMSE):+.3f} s; if every label sat at the late edge {(((lo26-cur)**2)-((lo26-new)**2)).sum()/(2*N26*RMSE):+.3f} s; per-row expected gain " + ", ".join(f"{fl} {g/(2*N26*RMSE):+.3f}" for fl, g in zip(j['FLIGHT_mvt'].to_list(), eg)))
print(f"scored rows {j.height}: mean move {float((new - cur).mean()):+.0f} s, rms {np.sqrt(((new - cur) ** 2).mean()):.0f}, max {np.abs(new - cur).max():.0f}; sum of squared moves {((new - cur) ** 2).sum():.3e}")
upd = pl.DataFrame({ID: j[ID], "_n": new})
out = v.join(upd, on=ID, how="left", maintain_order="left").with_columns(pl.coalesce(C("_n"), C(T).cast(pl.Float64)).round(0).cast(v.schema[T]).alias(T)).drop("_n").select(v.columns)
assert out.height == v.height
path = I + "agent_romenear_hardbind_final.parquet"; out.write_parquet(path); print("written", path)
j.write_csv(OUT + "item1_candidate_rows.csv")
subprocess.run([sys.executable, R + "archive/agent_smallbundle_06_check.py", path, I + "agent_bundle_v146_final.parquet"], cwd=R)
