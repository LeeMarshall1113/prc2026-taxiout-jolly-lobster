"""romenear 02: Item 1 evaluation. (a) the hard-binding family rule 'serve chain + w x (take-off minus LOBT - chain)' on Rome plan rows with
LOBT = SCHED (within 60 s) and lower window edge > thr: held-out SSE with w fitted leave-one-operator-out, temporal (lit -> dark, dark -> lit),
against a shuffled-signal control (which hard-binding rows carry LOBT = SCHED is permuted), body/tail/top-k splits; expected gain and
loss-if-wrong on the scored rows. (b) does the shipped with-plan mixture (v143 piece agent_rome1c, type probability P_c) already carry the
row-level signals (LOBT = SCHED, plan never moved, EOBT = LOBT, AOBT_3 default)? Logistic of copy on logit(P_c) + signal, 2025 plan rows gap
1,800-14,400 s, by regime. Inputs only for 2026."""
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
C = pl.col
R = ""; I = R + "data/interim/"; OUT = R + "data/interim/local/romenear/"
N26, RMSE = 344841.0, 217.28
W = 3606.0
f = pl.read_parquet(I + "agent_rome1_frame.parquet")
v146 = pl.read_parquet(I + "agent_bundle_v146_final.parquet").select(C("MVT_ID_mvt"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("v146"))
tp = pl.read_parquet(I + "agent_rome1c_typeP.parquet").select("MVT_ID_mvt", "P_c", "P_loao")
r26 = pl.read_parquet(I + "agent_rome1c_rows.parquet").select("MVT_ID_mvt", C("P").alias("P26"), C("new").alias("rome1c_new"))
f = f.join(v146, on="MVT_ID_mvt", how="left").join(tp, on="MVT_ID_mvt", how="left").join(r26, on="MVT_ID_mvt", how="left")
f = f.filter(~C("np") & C("gl").is_not_null() & C("aobt").is_not_null()).with_columns(
    (C("gl") - W).alias("lo"), (C("lobt") - C("sched")).alias("ls"), (C("eobt") - C("lobt")).alias("el"), (C("iobt") - C("lobt")).alias("il"),
    pl.when(C("mo") <= 4).then(pl.lit("dark")).otherwise(pl.lit("lit")).alias("reg"))
g = f.with_columns((C("ga") // 60).alias("gam"))
md = g.group_by("STAND_mvt", "gam").agg(pl.len().alias("k")).sort("k", descending=True).group_by("STAND_mvt").agg(C("gam").first().alias("gmode"), C("k").first().alias("kmode"), C("k").sum().alias("kall"))
f = g.join(md, on="STAND_mvt", how="left").with_columns(((C("gam") == C("gmode")) & (C("kmode") / C("kall") > 0.15)).fill_null(False).alias("aobt_def"))
bs = lambda d: d["ship12"].to_numpy().astype(float)  # noqa: E731


def fitw(y, a, t):
    """least-squares w in [0, 1.2] for y ~ a + w (t - a)"""
    d = t - a; r = y - a
    if (d ** 2).sum() == 0: return 0.0
    return float(np.clip((d * r).sum() / (d ** 2).sum(), 0, 1.2))


def sse(y, v): return float(((y - v) ** 2).sum())


print("=== (a) hard-binding family rule ===")
rng = np.random.default_rng(0)
summary = []
for thr in (1800, 2800, 3600):
    hb = f.filter((C("yr") == 2025) & (C("lo") > thr) & C("ship12").is_not_null() & C("y").is_not_null())
    sig = (hb["ls"].abs() <= 60).to_numpy(); y = hb["y"].to_numpy().astype(float); a = bs(hb); gl = hb["gl"].to_numpy().astype(float); lo = hb["lo"].to_numpy().astype(float)
    op = hb["op"].fill_null("NA").to_numpy().astype(str); reg = hb["reg"].to_numpy().astype(str); mo = hb["mo"].to_numpy()
    fam = sig; n = int(fam.sum())
    w_in = fitw(y[fam], a[fam], gl[fam])
    g_in = sse(y[fam], a[fam]) - sse(y[fam], a[fam] + w_in * (gl[fam] - a[fam]))
    # leave-one-operator-out
    pred = a.copy()
    for o in np.unique(op[fam]):
        tr = fam & (op != o); te = fam & (op == o)
        w = fitw(y[tr], a[tr], gl[tr]) if tr.sum() >= 5 else 0.0
        pred[te] = a[te] + w * (gl[te] - a[te])
    g_loao = sse(y[fam], a[fam]) - sse(y[fam], pred[fam]); gains_loao = (y[fam] - a[fam]) ** 2 - (y[fam] - pred[fam]) ** 2
    # temporal
    pred_t = a.copy(); wt = {}
    for trr, ter in (("lit", "dark"), ("dark", "lit")):
        tr = fam & (reg == trr); te = fam & (reg == ter); w = fitw(y[tr], a[tr], gl[tr]) if tr.sum() >= 3 else 0.0; wt[ter] = (w, int(tr.sum()), int(te.sum()))
        pred_t[te] = a[te] + w * (gl[te] - a[te])
    g_temp = {k: sse(y[fam & (reg == k)], a[fam & (reg == k)]) - sse(y[fam & (reg == k)], pred_t[fam & (reg == k)]) for k in ("dark", "lit")}
    # month-held-out
    pred_m = a.copy()
    for m in np.unique(mo[fam]):
        tr = fam & (mo != m); te = fam & (mo == m); w = fitw(y[tr], a[tr], gl[tr]); pred_m[te] = a[te] + w * (gl[te] - a[te])
    g_lomo = sse(y[fam], a[fam]) - sse(y[fam], pred_m[fam]); gm = (y[fam] - a[fam]) ** 2 - (y[fam] - pred_m[fam]) ** 2
    # control 1: the hard-binding rows WITHOUT the signal, same rule, w fitted on them
    oth = (~sig); w_o = fitw(y[oth], a[oth], gl[oth]); g_oth = sse(y[oth], a[oth]) - sse(y[oth], a[oth] + w_o * (gl[oth] - a[oth]))
    # control 2: shuffled signal (same count), LOMO gain
    ctl = []
    for _ in range(300):
        s2 = np.zeros(len(sig), bool); s2[rng.choice(len(sig), n, replace=False)] = True
        p2 = a.copy()
        for m in np.unique(mo[s2]):
            tr = s2 & (mo != m); te = s2 & (mo == m); w = fitw(y[tr], a[tr], gl[tr]) if tr.sum() >= 3 else 0.0; p2[te] = a[te] + w * (gl[te] - a[te])
        ctl.append(sse(y[s2], a[s2]) - sse(y[s2], p2[s2]))
    ctl = np.array(ctl)
    srt = np.sort(gm)[::-1]
    pos = (y[fam] - lo[fam]) / W
    body = pos < 1.5  # all family rows are 'tail' rows by gap; split by |y - a|
    big = np.abs(y[fam] - a[fam]) > 1800
    print(f"thr {thr}: family n {n} (other hard-binding rows {int(oth.sum())}); at LOBT {(np.abs(y[fam]-gl[fam])<=60).sum()}; label pos mean {pos.mean():.2f}; chain pos mean {((a[fam]-lo[fam])/W).mean():.2f}")
    print(f"   w in-sample {w_in:.2f} gain {g_in/1e6:.1f}M | LOMO gain {g_lomo/1e6:.1f}M (months better {sum(gm[mo[fam]==m].sum()>0 for m in np.unique(mo[fam]))}/{len(np.unique(mo[fam]))}) | LOAO gain {g_loao/1e6:.1f}M (ops better {sum(gains_loao[op[fam]==o].sum()>0 for o in np.unique(op[fam]))}/{len(np.unique(op[fam]))})")
    print(f"   temporal: lit->dark w {wt['dark'][0]:.2f} (fit {wt['dark'][1]}, judge {wt['dark'][2]}) gain {g_temp['dark']/1e6:.1f}M ; dark->lit w {wt['lit'][0]:.2f} (fit {wt['lit'][1]}, judge {wt['lit'][2]}) gain {g_temp['lit']/1e6:.1f}M")
    print(f"   control: no-signal hard-binding rows w {w_o:.2f} gain {g_oth/1e6:.1f}M on {int(oth.sum())} rows (per row {g_oth/max(oth.sum(),1)/1e3:.0f}k vs family per row {g_lomo/n/1e3:.0f}k); shuffled-signal LOMO gain mean {ctl.mean()/1e6:.1f}M p95 {np.percentile(ctl,95)/1e6:.1f}M (family {g_lomo/1e6:.1f}M, P(ctl>=fam) {(ctl>=g_lomo).mean():.3f})")
    print(f"   LOMO minus top-5 {(g_lomo-srt[:5].sum())/1e6:.1f}M, minus top-20 {(g_lomo-srt[:20].sum())/1e6:.1f}M; rows |y-chain|>1800: {big.sum()} carry {gm[big].sum()/max(g_lomo,1):.2f} of gain; rows worse: {(gm<0).sum()}")
    # scored rows
    s26 = f.filter((C("yr") == 2026) & (C("lo") > thr) & (C("ls").abs() <= 60))
    a26 = s26["v146"].to_numpy().astype(float); gl26 = s26["gl"].to_numpy().astype(float); lo26 = s26["lo"].to_numpy().astype(float)
    w_use = fitw(y[fam], a[fam], gl[fam])
    new26 = a26 + w_use * (gl26 - a26)
    # expected gain under the 2025 family label-position distribution (empirical), loss if labels sit at the chain's 2025 position distribution
    ysim = lo26[:, None] + pos[None, :] * W
    eg = (((ysim - a26[:, None]) ** 2) - ((ysim - new26[:, None]) ** 2)).mean(1)
    cpos = (a[fam] - lo[fam]) / W
    ysim2 = lo26[:, None] + cpos[None, :] * W
    lw = (((ysim2 - a26[:, None]) ** 2) - ((ysim2 - new26[:, None]) ** 2)).mean(1)
    print(f"   scored rows {s26.height} (Jan {(s26['mo']==1).sum()}, Jul {(s26['mo']==7).sum()}); served pos mean {((a26-lo26)/W).mean():.2f}; rome1c P mean {s26['P26'].mean():.3f}; w {w_use:.2f}; new pos {((new26-lo26)/W).mean():.2f}; mean |move| {np.abs(new26-a26).mean():.0f} s")
    print(f"   expected gain {eg.sum()/(2*N26*RMSE):+.3f} s (minus top 5 rows {(eg.sum()-np.sort(eg)[::-1][:5].sum())/(2*N26*RMSE):+.3f}); loss if labels follow the chain's 2025 positions {lw.sum()/(2*N26*RMSE):+.3f} s; if every label at the late edge {(((lo26-a26)**2)-((lo26-new26)**2)).sum()/(2*N26*RMSE):+.3f} s")
    summary.append(dict(thr=thr, n=n, w=w_use, lomo=g_lomo, loao=g_loao, temp_dark=g_temp["dark"], temp_lit=g_temp["lit"], ctl_mean=ctl.mean(), ctl_p95=np.percentile(ctl, 95), n26=s26.height, eg=eg.sum() / (2 * N26 * RMSE), lw=lw.sum() / (2 * N26 * RMSE)))
    s26.select("MVT_ID_mvt", "mo", "op", "FLIGHT_mvt", "gs", "gl", "lo", "ga", "v146", "P26").with_columns(pl.Series("new", np.rint(new26)), pl.Series("eg_s2", eg)).write_csv(OUT + f"item1_scored_thr{thr}.csv")
    # typeP on the 2025 family: what the shipped classifier believes
    print(f"   2025 family: P_c mean {hb.filter(pl.Series(fam))['P_c'].mean():.3f} (copy rate {(np.abs(y[fam]-gl[fam])<=60).mean():.3f}); P_loao mean {hb.filter(pl.Series(fam))['P_loao'].mean():.3f}; other hard-binding rows P_c {hb.filter(pl.Series(oth))['P_c'].mean():.3f} (copy rate {(np.abs(y[oth]-gl[oth])<=60).mean():.3f})")
pl.DataFrame(summary).write_csv(OUT + "item1_rule_summary.csv")

print("\n=== (b) incremental signal given the shipped type probability (2025 plan rows, gap 1,800-14,400 s) ===")
m = f.filter((C("yr") == 2025) & (C("gs") >= 1800) & (C("gs") < 14400) & C("P_c").is_not_null() & C("type").is_not_null())
cpy = (m["type"] == "copy").to_numpy().astype(int)
sigs = {"ls_eq": (m["ls"].abs() <= 60).to_numpy(), "static": ((m["ls"].abs() <= 60) & (m["el"].abs() <= 60)).to_numpy(), "el_eq": (m["el"].abs() <= 60).to_numpy(),
        "il_eq": (m["il"].abs() <= 60).to_numpy(), "aobt_def": m["aobt_def"].to_numpy(), "hb2800": ((m["lo"] > 2800) & (m["ls"].abs() <= 60)).to_numpy()}
reg = m["reg"].to_numpy().astype(str); mo = m["mo"].to_numpy()
for pn in ("P_c", "P_loao"):
    P = np.clip(m[pn].to_numpy().astype(float), 1e-4, 1 - 1e-4); lg = np.log(P / (1 - P))
    for nm, s in sigs.items():
        out = []
        for rg in ("dark", "lit", "all"):
            k = np.ones(len(cpy), bool) if rg == "all" else reg == rg
            X0 = lg[k, None]; X1 = np.column_stack([lg[k], s[k].astype(float)])
            # held-out by month log-loss: fit on other months
            ll0 = ll1 = 0.0
            for mm in np.unique(mo[k]):
                tr = mo[k] != mm; te = ~tr
                if te.sum() == 0 or tr.sum() == 0: continue
                a0 = LogisticRegression(C=10.0).fit(X0[tr], cpy[k][tr]); a1 = LogisticRegression(C=10.0).fit(X1[tr], cpy[k][tr])
                p0 = np.clip(a0.predict_proba(X0[te])[:, 1], 1e-6, 1 - 1e-6); p1 = np.clip(a1.predict_proba(X1[te])[:, 1], 1e-6, 1 - 1e-6)
                yy = cpy[k][te]; ll0 += -(yy * np.log(p0) + (1 - yy) * np.log(1 - p0)).sum(); ll1 += -(yy * np.log(p1) + (1 - yy) * np.log(1 - p1)).sum()
            coef = LogisticRegression(C=10.0).fit(X1, cpy[k]).coef_[0][1]
            out.append(f"{rg}: n {k.sum()} on {int(s[k].sum())} coef {coef:+.2f} dLL {ll0-ll1:+.1f}")
        print(f"{pn} + {nm:9s}: " + " | ".join(out))
print("(coef = log-odds added by the signal on top of the shipped probability; dLL = month-held-out log-likelihood improvement, nats)")
print("done")
