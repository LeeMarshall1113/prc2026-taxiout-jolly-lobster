"""day-equation 2, section 13: Amsterdam scored big-discrepancy days, UNVALIDATED candidate with input diagnostics.
Usage: python agent_dayeq2_08_ehamsnow.py   (writes data/interim/agent_dayeq2_eham_snow_final.parquet on v156)"""
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa
from agent_eusnow_feat import snow_features_apts

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(120)
APT = "EHAM"; SIG_F = 5.0; K = 3.0; LAM = 0.5; CAP = 300.0; MIN_N = 100
AMB_GAP, AMB_PRED = 5400.0, 1800.0
BASE = I + "agent_bundle_v156_final.parquet"


def tracks(yr):
    dw = pl.read_parquet(I + f"agent_multipush_dwells_{yr}.parquet").filter(C("apt") == APT).select(C("MVT_ID_mvt").alias("id"), C("est_t"))
    return dw


def prep(f, year):
    f = f.filter(C("rule") == "I")
    try:
        sn = snow_features_apts(f.select(C("id").alias("MVT_ID_mvt"), pl.lit(APT).alias("ADEP_mvt"), C("T").alias("MVT_TIME_UTC_mvt")), Path(R + "data/raw"), [APT]).select(C("MVT_ID_mvt").alias("id"), (C("lt_snow_3h") > 0).alias("snow"))
        f = f.join(sn, on="id", how="left")
    except IndexError:
        f = f.with_columns(pl.lit(False).alias("snow"))
    f = f.join(tracks(year), on="id", how="left").with_columns(C("snow").fill_null(False), C("day_pred").alias("day"), C("T").dt.hour().alias("hour"),
                                                                 (C("T").dt.epoch("s").cast(pl.Float64) - C("est_t")).alias("W"))
    f = f.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"), ((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("p_copy"))
    f = f.with_columns(pl.when(C("amb")).then(((1 - C("p_copy")) * (C("gs") - C("taxi")) / 60.0).clip(lower_bound=0)).otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm_pred"),
                       pl.when(C("amb")).then(C("p_copy") * (1 - C("p_copy")) * ((C("gs") - C("taxi")) / 60.0) ** 2).otherwise(0.0).alias("var_amb"),
                       (~C("amb") & (C("gs") > C("pred")) & ~C("fs").is_in(HEDGED)).fill_null(False).alias("shift"))
    pub = load_pub(year, [APT])
    days = (f.group_by("day").agg(n=pl.len(), O=C("dm_pred").sum(), Vamb=C("var_amb").sum(), snowday=C("snow").any(), nsnow=C("snow").sum(), nshift=C("shift").sum())
            .join(pub, on="day", how="inner").with_columns((C("n_pub") - C("n")).alias("dn")))
    days = days.filter((C("dn").abs() <= DN_MAX) & (C("n") >= MIN_N) & (C("n_pub") >= MIN_N)).with_columns((C("pub") - C("O")).alias("D"), C("day").dt.month().alias("mon"))
    return f, days


def solve(g, P):
    s = g["shift"].to_numpy(); gs = g["gs"].to_numpy()[s]; yh = g["pred"].to_numpy()[s]
    amb_c = float(g.filter(C("amb"))["dm_pred"].sum()); rest = float(g.filter(~C("amb") & ~C("shift"))["dm_pred"].sum())
    return solve_delta(gs, yh, P - amb_c - rest), s, gs, yh


# ---- 2025 parameters and track reference
f25, d25 = prep(frame_2025([APT]), 2025)
Dv = d25["D"]; bias = float(Dv.median()); rsd = 1.4826 * float((Dv - bias).abs().median()); sigL2 = max(rsd ** 2 - SIG_F ** 2 - float(d25["Vamb"].median()), 100.0); O_med = float(d25["O"].median())
deltas = [solve(g, float(d25.filter(C("day") == d)["pub"][0]))[0] for (d,), g in f25.partition_by("day", as_dict=True).items() if d25.filter(C("day") == d).height]
bias_delta = float(np.median(deltas))
print(f"2025 EHAM IFR: days {d25.height}, bias D {bias:+.0f} min, sigma_L {np.sqrt(sigL2):.0f} min, O_med {O_med:.0f}, bias delta {bias_delta:+.1f} s")
ref25 = f25.filter((C("gs") > 0) & C("W").is_not_null() & (C("W") < 20000)).with_columns((C("pred") - C("W")).alias("pw"), (C("y") - C("W")).alias("yw"))
snow_days25 = d25.filter(C("snowday"))["day"].to_list()
jan_ord = d25.filter((C("mon") == 1) & ~C("snowday"))["day"].to_list()


def qs(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [round(float(np.quantile(x, q))) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] if len(x) else []


print("2025 reference, delayed flights (gs > 0) with a track: served - W quantiles (10/25/50/75/90) | label - W | n")
for nm, dl in [("snow days 2025", snow_days25), ("ordinary January 2025 days", jan_ord)]:
    s = ref25.filter(C("day").is_in(dl)); print(f"  {nm} ({len(dl)} days): {qs(s['pw'])} | {qs(s['yw'])} | {s.height}; seen share {s.height / max(1, f25.filter(C('day').is_in(dl) & (C('gs') > 0)).height):.2f}")

# ---- 2026 selected days
f6, d6 = prep(frame_2026([APT], BASE), 2026)
sig = np.sqrt(SIG_F ** 2 + sigL2 * np.maximum(1.0, d6["O"].to_numpy() / O_med) ** 2); z = (d6["D"].to_numpy() - bias) / sig
d6 = d6.with_columns(pl.Series("z", z), pl.Series("sig", sig)).with_columns((C("z").abs() > K).alias("sel"))
sel = d6.filter(C("sel")).sort("day")
print(f"\n2026 EHAM IFR: gated days {d6.height}; selected (|z| > 3): {sel.height}")
print(sel.select("day", "n", "nshift", "nsnow", "D", "z", "sig").with_columns(pl.col(pl.Float64).round(1)))
out_rows = []; summary = []
for r in sel.iter_rows(named=True):
    d = r["day"]; g = f6.filter(C("day") == d); P = r["pub"]
    dl, s, gs, yh = solve(g, P); n = int(s.sum()); leave = int((gs - yh - dl < 0).sum())
    s_row = dl - bias_delta
    # (b) long-gap candidates
    lg = g.filter(C("gs") > 3600).with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60).alias("impl"), ((C("gs") - C("pred") - C("taxi")) / 60).clip(lower_bound=0).alias("hold_c"))
    share = float(lg["impl"].sum() / r["O"]) if r["O"] else 0.0; maxexp = float(lg["hold_c"].sum()); need = abs(r["D"] - bias) / float(lg["hold_c"].mean()) if lg.height and lg["hold_c"].mean() > 0 else float("nan")
    # (c) tracks
    dd = g.filter((C("gs") > 0) & C("W").is_not_null() & (C("W") < 20000)).with_columns((C("pred") - C("W")).alias("pw"))
    seen = dd.height / max(1, g.filter(C("gs") > 0).height)
    # (d) hour profile: share of the day's implied delay (gs clipped) in the top 3 hours, and W by hour
    hp = g.with_columns(C("gs").clip(lower_bound=0).alias("gsp")).group_by("hour").agg(C("gsp").sum().alias("dsum"), C("W").mean().alias("Wm"), pl.len().alias("n")).sort("dsum", descending=True)
    top3 = float(hp["dsum"].head(3).sum() / hp["dsum"].sum()) if hp["dsum"].sum() else 0.0
    print(f"\n== {d}: D {r['D']:+.0f} min ({r['z']:+.1f} sigma; sigma_day {r['sig']:.0f}), n {r['n']}, shifted rows {n}, snow rows {r['nsnow']}")
    print(f"  (a) exact solve shrink 1 no cap: delta {dl:+.0f} s (minus 2025 bias delta: {s_row:+.0f}); rows leaving the max(.,0) sum {leave} of {n} ({leave / max(n, 1):.2f})")
    print(f"  (b) long-gap rows (gs > 3,600): {lg.height} (plan {lg.filter(C('hasplan')).height}, no-plan {lg.filter(~C('hasplan')).height}); served / gap median {float((lg['pred'] / lg['gs']).median()) if lg.height else float('nan'):.2f}; "
          f"share of the day's implied delay {share:.2f}; deficit explainable if ALL were holds {maxexp:.0f} min (|D - bias| {abs(r['D'] - bias):.0f}); holds needed {need:.1f} of {lg.height}")
    print(f"  (c) tracks: delayed flights seen {seen:.2f}; served - W quantiles {qs(dd['pw'])} (2025 snow days {qs(ref25.filter(C('day').is_in(snow_days25))['pw'])}, ordinary Jan 2025 {qs(ref25.filter(C('day').is_in(jan_ord))['pw'])}); "
          f"W quantiles {qs(dd['W'])} (2025 snow {qs(ref25.filter(C('day').is_in(snow_days25))['W'])})")
    print(f"  (d) hour profile: top-3 hours hold {top3:.2f} of the day's delay against schedule (ordinary Jan 2025 days: {np.mean([float(f25.filter(C('day') == dj).with_columns(C('gs').clip(lower_bound=0).alias('gsp')).group_by('hour').agg(C('gsp').sum().alias('x')).sort('x', descending=True)['x'].head(3).sum() / max(1e-9, f25.filter(C('day') == dj)['gs'].clip(lower_bound=0).sum())) for dj in jan_ord[:8]]):.2f}); W by hour (top delay hours): {hp.head(5).select('hour', 'n', (C('dsum') / 60).round(0).alias('delay_min'), C('Wm').round(0)).rows()}")
    # the candidate shift
    a = np.clip(LAM * s_row, -CAP, CAP)
    new = g["pred"].to_numpy().copy(); new[s] = np.maximum(new[s] + a, 0)
    gain_spread = n * (2 * a * s_row - a * a) / SE_SCALE; gain_075 = 0.75 * n * s_row ** 2 / SE_SCALE; loss_hold = n * a * a / SE_SCALE
    summary.append(dict(day=str(d), D=round(r["D"]), z=round(r["z"], 1), n_shift=n, shift=round(a), s=round(s_row), gain_if_spread=round(gain_spread, 3), gain_075=round(gain_075, 3), loss_if_holds=round(loss_hold, 3), leave=leave, long_gap=lg.height, holds_needed=round(need, 1) if np.isfinite(need) else None))
    out_rows.append(g.select("id", "hasplan", "day", "pred").with_columns(pl.Series("new", new)))
print("\nsummary per day (shift applied = 0.5 x (delta - bias delta), capped +-300):")
print(pl.DataFrame(summary))
rows = pl.concat(out_rows).with_columns((C("new") - C("pred")).alias("move"))
np_sw = rows.filter(~C("hasplan") & (C("day") >= pl.date(2026, 1, 2)) & (C("day") <= pl.date(2026, 1, 7)))
print(f"no-plan rows of 2-7 January on the selected days: {np_sw.height}, moved {np_sw.filter(C('move') != 0).height}, mean move {float(np_sw['move'].mean()) if np_sw.height else 0:+.1f} s, max {float(np_sw['move'].abs().max()) if np_sw.height else 0:.0f}")
print(np_sw.group_by("day").agg(n=pl.len(), moved=(C("move") != 0).sum(), mean_move=C("move").mean().round(1)).sort("day"))
# build on v156, window last
b = pl.read_parquet(BASE)
rk = (pl.scan_parquet(R + "data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP").select(C("MVT_ID_mvt").alias("id"), (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("glr")).collect())
j = b.rename({"MVT_ID_mvt": "id"}).join(rows.select("id", "new"), on="id", how="left").join(rk, on="id", how="left").with_columns(pl.coalesce(C("new"), C("TAXITIME_SEC_mvt").cast(pl.Float64)).alias("new"))
lo = pl.max_horizontal(C("glr") - 3606, pl.lit(0)); hi = C("glr") + 3606
j = j.with_columns(pl.when(C("glr").is_null()).then(C("new")).otherwise(pl.min_horizontal(pl.max_horizontal(C("new"), lo), hi)).alias("new"))
out = pl.DataFrame({"MVT_ID_mvt": b["MVT_ID_mvt"], "TAXITIME_SEC_mvt": j["new"].round().cast(b.schema["TAXITIME_SEC_mvt"])}); assert out["TAXITIME_SEC_mvt"].null_count() == 0
out.write_parquet(I + "agent_dayeq2_eham_snow_final.parquet")
dmv = (out["TAXITIME_SEC_mvt"] - b["TAXITIME_SEC_mvt"]).cast(pl.Float64).to_numpy()
print(f"built agent_dayeq2_eham_snow_final.parquet (UNVALIDATED): rows moved {(dmv != 0).sum()}, rms {np.sqrt((dmv[dmv != 0] ** 2).mean()) if (dmv != 0).any() else 0:.1f}, max {np.abs(dmv).max():.0f}, sum dp^2/row {(dmv ** 2).sum() / len(dmv):.1f}, mean {dmv.mean():+.3f}")
rows.write_parquet(I + "agent_dayeq2_rows_EHAM_snow.parquet")
