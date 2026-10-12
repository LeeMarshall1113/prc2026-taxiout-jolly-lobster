"""romenear 01: Item 1 census. Rome plan rows whose plan clock equals the schedule and whose window binds hard.
Where does the 2025 label sit (at LOBT = position 1 of the window; at the late edge = 0; elsewhere), by family definition, month and
regime; the chain's held-out value (ship12 / ship8) against 'serve take-off minus LOBT' and the optimal mixture. Also the copy table for
plan rows with gap 0.5-2 h: what the 'copy' is (stamp at SCHED / LOBT / EOBT / IOBT) and the copy rate by input signal.
Inputs only for 2026 rows."""
import sys, numpy as np, polars as pl
C = pl.col
R = ""
I = R + "data/interim/"
OUT = R + "data/interim/local/romenear/"
import os; os.makedirs(OUT, exist_ok=True)
f = pl.read_parquet(I + "agent_rome1_frame.parquet")
v146 = pl.read_parquet(I + "agent_bundle_v146_final.parquet").select(C("MVT_ID_mvt"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("v146"))
f = f.join(v146, on="MVT_ID_mvt", how="left")
W = 3606
f = f.filter(~C("np") & C("gl").is_not_null() & C("aobt").is_not_null())
f = f.with_columns(
    (C("gl") - W).alias("lo"),
    (C("lobt") - C("sched")).alias("ls"), (C("eobt") - C("sched")).alias("es"), (C("iobt") - C("sched")).alias("is_"),
    (C("iobt") - C("lobt")).alias("il"), (C("eobt") - C("lobt")).alias("el"), (C("aobt") - C("lobt")).alias("al"),
    pl.when(C("mo") <= 4).then(pl.lit("dark")).otherwise(pl.lit("lit")).alias("reg"),
)
# AOBT_3 stand-default flag (label-free): ga equals the modal take-off-minus-AOBT_3 of the stand (rounded to the minute) and that mode is frequent
g = f.with_columns((C("ga") // 60).alias("gam"))
md = g.group_by("STAND_mvt", "gam").agg(pl.len().alias("k")).sort("k", descending=True).group_by("STAND_mvt").agg(C("gam").first().alias("gmode"), C("k").first().alias("kmode"), C("k").sum().alias("kall"))
f = g.join(md, on="STAND_mvt", how="left").with_columns(((C("gam") == C("gmode")) & (C("kmode") / C("kall") > 0.15)).fill_null(False).alias("aobt_def"))
f = f.with_columns(((C("al").abs() <= 60)).alias("aobt_eq_lobt"))
y = f["y"]
f = f.with_columns(
    ((y - C("gl")).abs() <= 60).alias("at_lobt"), ((y - C("lo")).abs() <= 60).alias("at_edge"), ((y - C("gs")).abs() <= 60).alias("at_sched"),
    ((y - C("ge")).abs() <= 60).alias("at_eobt"), ((y - C("gi")).abs() <= 60).alias("at_iobt"), ((y - C("ga")).abs() <= 60).alias("at_aobt"),
    ((y - C("lo")) / W).alias("pos"),
)
t25 = f.filter(C("yr") == 2025); t26 = f.filter(C("yr") == 2026)
print("plan rows 2025", t25.height, "2026", t26.height)


def fam(d, eq, thr, tol=0):
    e = {"ls": (C("ls").abs() <= tol), "es": (C("es").abs() <= tol), "is": (C("is_").abs() <= tol), "any": pl.lit(True)}[eq]
    return d.filter(e & (C("lo") > thr))


print("\n=== A. hard-binding families: where the 2025 label sits ===")
rows = []
for eq in ["ls", "es", "is", "any"]:
    for tol in [0, 60]:
        if eq == "any" and tol: continue
        for thr in [1800, 2800, 3600]:
            d = fam(t25, eq, thr, tol)
            d26 = fam(t26, eq, thr, tol)
            if d.height == 0: continue
            pos = d["pos"].to_numpy(); yy = d["y"].to_numpy(); gl = d["gl"].to_numpy(); lo = d["lo"].to_numpy()
            s12 = d["ship12"].to_numpy(); s8 = d["ship8"].to_numpy()
            def sse(v, m=None):
                m = np.ones(len(yy), bool) if m is None else m
                return float(((yy[m] - v[m]) ** 2).sum())
            # optimal mixture between chain and gl
            best = min(np.linspace(0, 1, 21), key=lambda w: sse(s12 + w * (gl - s12)))
            r = dict(eq=eq, tol=tol, thr=thr, n=d.height, n26=d26.height, at_lobt=int(d["at_lobt"].sum()), at_edge=int(d["at_edge"].sum()),
                     at_aobt=int(d["at_aobt"].sum()), pos_mean=float(np.mean(pos)), pos_med=float(np.median(pos)),
                     dark_n=int((d["reg"] == "dark").sum()), dark_at_lobt=int(d.filter(C("reg") == "dark")["at_lobt"].sum()),
                     lit_n=int((d["reg"] == "lit").sum()), lit_at_lobt=int(d.filter(C("reg") == "lit")["at_lobt"].sum()),
                     sse_chain12=sse(s12), sse_gl=sse(gl), sse_edge=sse(lo), w_best=float(best), sse_best=sse(s12 + best * (gl - s12)),
                     chain_pos=float(np.mean((s12 - lo) / W)), served26_pos=float(((d26["v146"] - d26["lo"]) / W).mean()) if d26.height else np.nan)
            rows.append(r)
            print(f"{eq:3s} tol{tol:2d} thr{thr}: n {r['n']:4d} (2026 {r['n26']:3d}) at_lobt {r['at_lobt']} at_edge {r['at_edge']} at_aobt {r['at_aobt']} pos mean {r['pos_mean']:.2f} med {r['pos_med']:.2f} | dark {r['dark_at_lobt']}/{r['dark_n']} lit {r['lit_at_lobt']}/{r['lit_n']} | SSE chain {r['sse_chain12']/1e6:.1f}M gl {r['sse_gl']/1e6:.1f}M edge {r['sse_edge']/1e6:.1f}M best w {best:.2f} {r['sse_best']/1e6:.1f}M | chain pos {r['chain_pos']:.2f} served26 pos {r['served26_pos']:.2f}")
pl.DataFrame(rows).write_csv(OUT + "item1_families.csv")

print("\n=== A2. ls==0 & lo>2800 by month (2025): n, at_lobt, at_edge, pos mean, chain pos ===")
d = fam(t25, "ls", 2800, 0)
print(d.group_by("mo").agg(pl.len().alias("n"), C("at_lobt").sum(), C("at_edge").sum(), C("pos").mean().round(2), ((C("ship12") - C("lo")) / W).mean().round(2).alias("chain_pos"), C("seen").mean().round(2)).sort("mo"))
print("by operator (n>=2):")
print(d.group_by("op").agg(pl.len().alias("n"), C("at_lobt").sum(), C("pos").mean().round(2)).filter(C("n") >= 2).sort("n", descending=True))
print("rows (2025):")
print(d.select("mo", "op", "FLIGHT_mvt", "gs", "gl", "lo", "ga", "y", "pos", "at_lobt", "ship12", "aobt_def", "il", "el", "seen").sort("mo"))
print("scored rows (2026, inputs + served):")
d26 = fam(t26, "ls", 2800, 0)
print(d26.select("mo", "op", "FLIGHT_mvt", "gs", "gl", "lo", "ga", "v146", ((C("v146") - C("lo")) / W).round(2).alias("served_pos"), "aobt_def", "il", "el").sort("mo"))

print("\n=== B. plan rows generally: does the stamp sit at a plan clock? (2025, gs >= 1800) ===")
inf = t25.filter(C("gs") >= 1800)
for nm in ["at_sched", "at_lobt", "at_eobt", "at_iobt", "at_aobt", "at_edge"]:
    print(nm, int(inf[nm].sum()), "of", inf.height, "rate %.4f" % inf[nm].mean())
# stamps at LOBT where LOBT != SCHED (so it is not the schedule copy)
x = inf.filter(C("ls").abs() > 60)
print("LOBT != SCHED rows", x.height, "at_lobt", int(x["at_lobt"].sum()), "at_sched", int(x["at_sched"].sum()), "at_eobt(eobt!=sched)", int(x.filter(C("es").abs() > 60)["at_eobt"].sum()), "of", x.filter(C("es").abs() > 60).height)
# density of y - gl near zero against flanks (spike test), on LOBT != SCHED rows with gl > 1800
z = x.filter(C("gl") >= 1800)
dgl = (z["y"] - z["gl"]).to_numpy()
for lo_, hi_ in [(-30, 30), (-90, -30), (30, 90), (-300, -90), (90, 300)]:
    m = (dgl >= lo_) & (dgl < hi_); print(f"  y-gl in [{lo_},{hi_}): {int(m.sum())} ({m.sum()/(hi_-lo_):.2f}/s)")
# the same for y - gs on LOBT != SCHED rows (schedule copy)
dgs = (z["y"] - z["gs"]).to_numpy()
for lo_, hi_ in [(-30, 30), (-90, -30), (30, 90), (-300, -90), (90, 300)]:
    m = (dgs >= lo_) & (dgs < hi_); print(f"  y-gs in [{lo_},{hi_}): {int(m.sum())} ({m.sum()/(hi_-lo_):.2f}/s)")

print("\n=== C. plan rows gap 1,800-7,200 s: copy rate by input signal, tracked / untracked months ===")
mid = t25.filter((C("gs") >= 1800) & (C("gs") < 7200)).with_columns(
    (C("type") == "copy").alias("copy"),
    (C("ls").abs() <= 60).alias("sig_ls"), (C("es").abs() <= 60).alias("sig_es"), (C("is_").abs() <= 60).alias("sig_is"),
    (C("il").abs() <= 60).alias("sig_il"), (C("el").abs() <= 60).alias("sig_el"), C("aobt_def").alias("sig_adef"), C("aobt_eq_lobt").alias("sig_al"),
    ((C("ls").abs() <= 60) & (C("il").abs() <= 60)).alias("sig_ls_il"), ((C("ls").abs() <= 60) & (C("el").abs() <= 60)).alias("sig_ls_el"),
    ((C("ls").abs() <= 60) & C("aobt_def")).alias("sig_ls_adef"), ((C("ls").abs() <= 60) & (C("il").abs() <= 60) & (C("el").abs() <= 60)).alias("sig_static"),
    (C("ls").abs() > 60).alias("sig_ls_moved"),
)
print("copy among mid plan rows: what is copied? at_sched", int(mid["at_sched"].sum()), "at_lobt", int(mid["at_lobt"].sum()), "at_eobt", int(mid["at_eobt"].sum()), "at_iobt", int(mid["at_iobt"].sum()), "n", mid.height)
print("  at_lobt & !at_sched", int((mid["at_lobt"] & ~mid["at_sched"]).sum()), " at_eobt & !at_sched", int((mid["at_eobt"] & ~mid["at_sched"]).sum()), " at_iobt & !at_sched", int((mid["at_iobt"] & ~mid["at_sched"]).sum()))
for s in ["sig_ls", "sig_es", "sig_is", "sig_il", "sig_el", "sig_adef", "sig_al", "sig_ls_il", "sig_ls_el", "sig_ls_adef", "sig_static", "sig_ls_moved"]:
    t = mid.group_by("reg", s).agg(pl.len().alias("n"), C("copy").mean().round(4).alias("copy_rate")).sort("reg", s)
    vals = {(r["reg"], r[s]): (r["n"], r["copy_rate"]) for r in t.iter_rows(named=True)}
    print(f"{s:12s} dark: on {vals.get(('dark', True), (0, 0))}  off {vals.get(('dark', False), (0, 0))} | lit: on {vals.get(('lit', True), (0, 0))}  off {vals.get(('lit', False), (0, 0))}")
# within-operator expected rate: does the signal lift above the operator habit?
op_rate = mid.group_by("op").agg(C("copy").mean().alias("op_rate"), pl.len().alias("op_n"))
mm = mid.join(op_rate, on="op")
for s in ["sig_ls", "sig_il", "sig_el", "sig_adef", "sig_static", "sig_ls_moved"]:
    t = mm.group_by(s).agg(pl.len().alias("n"), C("copy").mean().round(4).alias("obs"), C("op_rate").mean().round(4).alias("op_expected"))
    print(s, t.sort(s).to_dicts())
# gap-band copy table for the rome-types cells (1.8-3.6k, 3.6-7.2k) by sig_ls x sig_il
for lo_, hi_ in [(1800, 3600), (3600, 7200), (7200, 14400)]:
    b = t25.filter((C("gs") >= lo_) & (C("gs") < hi_)).with_columns((C("type") == "copy").alias("copy"), (C("ls").abs() <= 60).alias("sig_ls"), (C("il").abs() <= 60).alias("sig_il"))
    print(f"band {lo_}-{hi_}:", b.group_by("reg", "sig_ls", "sig_il").agg(pl.len().alias("n"), C("copy").mean().round(3).alias("copy_rate")).sort("reg", "sig_ls", "sig_il").to_dicts())
mid.select("MVT_ID_mvt", "mo", "reg", "op", "gs", "gl", "ls", "es", "is_", "il", "el", "al", "aobt_def", "copy", "type", "y", "ship12", "seen").write_parquet(OUT + "item1_mid25.parquet")
print("done")
