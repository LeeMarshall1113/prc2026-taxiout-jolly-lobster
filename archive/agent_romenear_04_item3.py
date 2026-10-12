"""romenear 04: Item 3. Regime-specific copy shares. (1) What copy share each scored Rome no-plan cell is served with (implied by v146:
p = (served - 1,025) / (gap - 1,025)) against the 2025 share of the cell by regime (dark 1-4 / lit 5-12) and by season. (2) Does the
shipped no-plan probability (P2_12, leave-one-month-out, pooled 12 months, no month input) track the month level of the copy share?
(3) Regime or season: the ground feed switched on 2025-05-24; copy share of no-plan mid-gap rows in the weeks either side of the cut,
plan rows as the control group; winter lit months (Nov, Dec) against winter dark months (Jan-Mar). (4) Expected-squared-error-optimal
served value per cell under pooled / regime / season shares, gain against served, and the held-out dark-month judgement."""
import numpy as np, polars as pl
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(250)
C = pl.col
R = ""; I = R + "data/interim/"; OUT = R + "data/interim/local/romenear/"
N26, RMSE, M0 = 344841.0, 217.28, 1025.0
f = pl.read_parquet(I + "agent_rome1_frame.parquet")
v146 = pl.read_parquet(I + "agent_bundle_v146_final.parquet").select(C("MVT_ID_mvt"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("v146"))
f = f.join(v146, on="MVT_ID_mvt", how="left").with_columns(
    pl.when(C("mo") <= 4).then(pl.lit("dark")).otherwise(pl.lit("lit")).alias("reg"),
    pl.when(C("mo").is_in([11, 12, 1, 2, 3])).then(pl.lit("winter")).when(C("mo").is_in([5, 6, 7, 8, 9])).then(pl.lit("summer")).otherwise(pl.lit("shoulder")).alias("season"),
    pl.when(C("gs") < 3600).then(pl.lit("a<3.6k")).when(C("gs") < 7200).then(pl.lit("b3.6-7.2k")).when(C("gs") < 14400).then(pl.lit("c7.2-14.4k")).when(C("gs") < 50000).then(pl.lit("d14.4-50k")).otherwise(pl.lit("e50k+")).alias("band"),
    pl.from_epoch(C("S"), time_unit="s").alias("S_utc"), C("hgl2").is_not_null().alias("recovered"))
NP = f.filter(C("np") & (C("gs") >= 1800))
t25 = NP.filter((C("yr") == 2025) & C("type").is_in(["copy", "real", "day-early"])).with_columns((C("type") == "copy").alias("copy"))
t26 = NP.filter(C("yr") == 2026)
print("=== (1) served share per scored no-plan cell (v146) against 2025 shares ===")
s = t26.filter(~C("recovered") & (C("gs") >= 3600) & (C("gs") < 50000)).with_columns(((C("v146") - M0) / (C("gs") - M0)).alias("p_served"))
print(s.group_by("mo", "band").agg(pl.len().alias("n"), C("p_served").mean().round(3).alias("p_served_mean"), C("p_served").median().round(3).alias("p_served_med")).sort("mo", "band"))
print("2025 copy share by band x regime (no-plan, plan not recovered):")
u = t25.filter(~C("recovered"))
print(u.group_by("band", "reg").agg(pl.len().alias("n"), C("copy").sum().alias("copies"), C("copy").mean().round(3).alias("share")).sort("band", "reg"))
print("2025 copy share by band x season:")
print(u.group_by("band", "season").agg(pl.len().alias("n"), C("copy").sum().alias("copies"), C("copy").mean().round(3).alias("share")).sort("band", "season"))
print("2025 copy share by band x month (mid gap 3.6-14.4k pooled):")
print(u.filter((C("gs") >= 3600) & (C("gs") < 14400)).group_by("mo").agg(pl.len().alias("n"), C("copy").sum().alias("copies"), C("copy").mean().round(3).alias("share"), C("P2_12").mean().round(3).alias("P2_12_mean"), ((C("ship12") - M0) / (C("gs") - M0)).mean().round(3).alias("chain_implied_p")).sort("mo"))
print("\n=== (2) is the shipped probability calibrated by month? (mid-gap no-plan, not recovered; P2_12 is the LOMO classifier) ===")
m = u.filter((C("gs") >= 3600) & (C("gs") < 14400) & C("P2_12").is_not_null())
cpy = m["copy"].to_numpy().astype(float); P = m["P2_12"].to_numpy(); mo = m["mo"].to_numpy(); gs = m["gs"].to_numpy().astype(float); y = m["y"].to_numpy().astype(float)
a = m["norome12"].to_numpy().astype(float); ship = m["ship12"].to_numpy().astype(float)
print("rows", len(cpy), "overall copy", cpy.mean().round(3), "mean P", P.mean().round(3))
for rg, k in (("dark 1-4", mo <= 4), ("lit 5-12", mo >= 5), ("winter", np.isin(mo, [11, 12, 1, 2, 3])), ("summer", np.isin(mo, [5, 6, 7, 8, 9])), ("Jan", mo == 1), ("Jul", mo == 7)):
    print(f"  {rg:9s}: n {k.sum():4d} copy share {cpy[k].mean():.3f}  mean P {P[k].mean():.3f}  diff {cpy[k].mean()-P[k].mean():+.3f}  (se {np.sqrt(cpy[k].mean()*(1-cpy[k].mean())/k.sum()):.3f})")
# the mixture value with a month-level logit shift: does a regime / season shift of the probability reduce held-out SSE?
def mixv(p, k=None): return p * (gs if k is None else gs[k]) + (1 - p) * M0
def shift(p, d):
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4))) + d; return 1 / (1 + np.exp(-z))
print("\n=== (3) regime or season: the 2025-05-24 cut, plan rows as control ===")
cut = np.datetime64("2025-05-24T18:29")
for wk in (4, 8):
    lo_ = cut - np.timedelta64(wk * 7, "D"); hi_ = cut + np.timedelta64(wk * 7, "D")
    for nm, d in (("no-plan 3.6-14.4k", u.filter((C("gs") >= 3600) & (C("gs") < 14400))), ("no-plan 1.8k+", u), ("plan rows 1.8k+ (control)", f.filter(~C("np") & (C("yr") == 2025) & (C("gs") >= 1800) & C("type").is_not_null()).with_columns((C("type") == "copy").alias("copy")))):
        b = d.filter((C("S_utc") >= lo_) & (C("S_utc") < cut)); aft = d.filter((C("S_utc") >= cut) & (C("S_utc") < hi_))
        print(f"  +-{wk} weeks {nm:26s}: before {int(b['copy'].sum())}/{b.height} = {b['copy'].mean():.3f}  after {int(aft['copy'].sum())}/{aft.height} = {aft['copy'].mean():.3f}")
print("winter lit (Nov-Dec) vs winter dark (Jan-Mar) vs summer lit (May-Sep), no-plan mid-gap:")
mm = u.filter((C("gs") >= 3600) & (C("gs") < 14400)).with_columns(pl.when(C("mo").is_in([11, 12])).then(pl.lit("winter-lit")).when(C("mo").is_in([1, 2, 3])).then(pl.lit("winter-dark")).when(C("mo") == 4).then(pl.lit("spring-dark")).when(C("mo").is_in([5, 6, 7, 8, 9])).then(pl.lit("summer-lit")).otherwise(pl.lit("autumn-lit")).alias("cell"))
print(mm.group_by("cell").agg(pl.len().alias("n"), C("copy").sum().alias("copies"), (pl.len() - C("copy").sum()).alias("non_copies"), C("copy").mean().round(3).alias("share"), C("P2_12").mean().round(3).alias("P_mean")).sort("cell"))
print("counts of copies and of non-copies per month (is the share driven by the denominator?):")
print(u.filter((C("gs") >= 3600) & (C("gs") < 14400)).group_by("mo").agg(C("copy").sum().alias("copies"), (pl.len() - C("copy").sum()).alias("non_copies")).sort("mo").to_dicts())
print("\n=== (4) expected-SSE-optimal served value per scored cell and the held-out dark-month judgement ===")
# shares for the scored cells: pooled, regime-matched (Jan 2026 lit, Jul 2026 dark), season-matched (Jan winter, Jul summer), same-month 2025
def share(d, cond):
    x = d.filter(cond); return (x["copy"].sum() + 0.5) / (x.height + 1.0), x.height
rows = []
for mo26, regm, seas in ((1, "lit", "winter"), (7, "dark", "summer")):
    for band in ("b3.6-7.2k", "c7.2-14.4k", "d14.4-50k"):
        sc = s.filter((C("mo") == mo26) & (C("band") == band))
        if sc.height == 0: continue
        ub = u.filter(C("band") == band)
        pp, np_ = share(ub, pl.lit(True)); pr, nr = share(ub, C("reg") == regm); ps, ns = share(ub, C("season") == seas); pm, nm_ = share(ub, C("mo") == mo26)
        gs26 = sc["gs"].to_numpy().astype(float); v = sc["v146"].to_numpy().astype(float); pserved = sc["p_served"].mean()
        def gain_if(ptrue, vnew):  # expected SSE saved per row set if the true share is ptrue, real rows at M0 (spread ignored, same in both arms)
            e_old = ptrue * (gs26 - v) ** 2 + (1 - ptrue) * (M0 - v) ** 2; e_new = ptrue * (gs26 - vnew) ** 2 + (1 - ptrue) * (M0 - vnew) ** 2
            return (e_old - e_new).sum() / (2 * N26 * RMSE)
        vr = pr * gs26 + (1 - pr) * M0; vs_ = ps * gs26 + (1 - ps) * M0
        rows.append(dict(mo=mo26, band=band, n26=sc.height, p_served=round(pserved, 3), p_pooled=round(pp, 3), p_regime=round(pr, 3), n_reg=nr, p_season=round(ps, 3), n_seas=ns, p_samemonth=round(pm, 3), n_month=nm_,
                         gain_regime_if_regime=round(gain_if(pr, vr), 3), loss_regime_if_season=round(gain_if(ps, vr), 3), gain_season_if_season=round(gain_if(ps, vs_), 3), loss_season_if_regime=round(gain_if(pr, vs_), 3)))
print(pl.DataFrame(rows))
# held-out dark months: serve the mixture at a logit shift fitted on lit months (regime arm) or on same-season months (season arm) and judge on dark months, and the reverse
print("\nheld-out judgement on 2025 mid-gap no-plan rows (chain value ship12 is the served-style base; candidate = ship12 moved to the mixture at the shifted probability with the shipped weights 0.5 / 0.85):")
w = np.where(gs < 7200, 0.5, 0.85)
def cand(p, k=None):
    kk = np.ones(len(a), bool) if k is None else k
    return a[kk] + w[kk] * (mixv(p, kk) - a[kk])
base_sse = ((y - ship) ** 2)
print("  chain SSE per row (M s^2): dark %.2f lit %.2f" % (base_sse[mo <= 4].mean() / 1e6, base_sse[mo >= 5].mean() / 1e6), "| rebuilt-from-norome at d=0 differs from ship12 by rms %.0f s" % np.sqrt(((cand(P) - ship) ** 2).mean()))
grid = np.linspace(-1.5, 1.5, 31)
def fit_shift(k):
    return float(min(grid, key=lambda d: ((y[k] - cand(shift(P[k], d), k)) ** 2).sum()))
for nm, fitk, judk in (("regime: fit lit, judge dark", mo >= 5, mo <= 4), ("regime: fit dark, judge lit", mo <= 4, mo >= 5),
                       ("season: fit winter, judge Jan", np.isin(mo, [11, 12, 2, 3]), mo == 1), ("season: fit summer, judge Jul", np.isin(mo, [5, 6, 8, 9]), mo == 7),
                       ("same-regime for Jul: fit dark 1-4, judge Jul", mo <= 4, mo == 7), ("same-regime for Jan: fit lit 5-12 minus Jan, judge Jan", mo >= 5, mo == 1)):
    d = fit_shift(fitk); g = (((y[judk] - cand(P[judk], judk)) ** 2) - ((y[judk] - cand(shift(P[judk], d), judk)) ** 2))
    print(f"  {nm:48s}: shift {d:+.2f} (share {P[fitk].mean():.2f} -> {shift(P[fitk], d).mean():.2f}); judged rows {judk.sum()}; gain {g.sum()/1e6:+.1f}M s^2, rows better {(g>0).sum()}, worse {(g<0).sum()}, minus top-5 {(g.sum()-np.sort(g)[::-1][:5].sum())/1e6:+.1f}M")
# month-held-out optimal shift (what the frame says in general)
print("  leave-one-month-out shift:", {int(k): fit_shift(mo != k) for k in np.unique(mo)})
s.select("MVT_ID_mvt", "mo", "band", "gs", "v146", "p_served", "op", "FLIGHT_mvt").write_csv(OUT + "item3_scored_cells.csv")
print("done")
