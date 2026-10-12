"""day-equation 2, section 9: Istanbul piece A' = round one's day-level shift of delayed flights at LTFM with the stop-day exclusion
(snow day AND D - bias < -3 sigma_day); A'' = the same with the positive-discrepancy shift allocated by served excess over the airport-hour median.
Usage: python agent_dayeq2_04_ltfm_level.py 2025 | 2026"""
import json
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa
from agent_ltreg_feat import snow_features

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(120)
MODE = sys.argv[1] if len(sys.argv) > 1 else "2025"
APT = "LTFM"; SIG_F = 5.0; GATE_Z = 3.0; MIN_N = 100; LAMS = [round(0.1 * i, 1) for i in range(11)]
AMB_GAP, AMB_PRED = 5400.0, 1800.0
BASE = I + "agent_bundle_v156_final.parquet"


def prep(f, year):
    sn = snow_features(f.select(C("id").alias("MVT_ID_mvt"), C("T").alias("MVT_TIME_UTC_mvt")), Path(R + "data/raw")).select(C("MVT_ID_mvt").alias("id"), (C("lt_snow_3h") > 0).alias("snow"))
    f = f.join(sn, on="id", how="left").with_columns(C("snow").fill_null(False), C("day_pred").alias("day"), C("T").dt.hour().alias("hour"))
    f = f.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"), ((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("p_copy"))
    f = f.with_columns(pl.when(C("amb")).then(((1 - C("p_copy")) * (C("gs") - C("taxi")) / 60.0).clip(lower_bound=0)).otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm_pred"),
                       pl.when(C("amb")).then(C("p_copy") * (1 - C("p_copy")) * ((C("gs") - C("taxi")) / 60.0) ** 2).otherwise(0.0).alias("var_amb"),
                       (~C("amb") & (C("gs") > C("pred")) & ~C("fs").is_in(HEDGED)).fill_null(False).alias("shift"))
    hm = f.group_by("mon", "hour").agg(C("pred").median().alias("hmed"))
    f = f.join(hm, on=["mon", "hour"], how="left").with_columns((C("pred") - C("hmed")).clip(lower_bound=0).alias("excess"))
    pub = load_pub(year, [APT])
    days = (f.group_by("day").agg(n=pl.len(), O=C("dm_pred").sum(), Vamb=C("var_amb").sum(), snowday=C("snow").any(), nsnow=C("snow").sum())
            .join(pub, on="day", how="inner").with_columns((C("n_pub") - C("n")).alias("dn")))
    days = days.filter((C("dn").abs() <= DN_MAX) & (C("n") >= MIN_N) & (C("n_pub") >= MIN_N)).with_columns((C("pub") - C("O")).alias("D"), (C("pub7") - C("O")).alias("D7"), C("day").dt.month().alias("mon"))
    return f, days


def wsolve(gs, yh, w, target_min):
    """delta with sum max(gs - yh - delta * w, 0)/60 = target; bisection on [-7200, 7200] (monotone)."""
    def S(d):
        return np.clip(gs - yh - d * w, 0, None).sum() / 60.0
    lo, hi = -7200.0, 7200.0
    if S(lo) < target_min:
        return lo
    if S(hi) > target_min:
        return hi
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if S(m) > target_min:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def day_solves(f, days):
    """per day: flat delta (A') and excess-weighted delta (A'') for the real and the shifted series, at lam 1 and no bias."""
    recs = []
    for (d,), g in f.partition_by("day", as_dict=True, maintain_order=True).items():
        r = days.filter(C("day") == d)
        if r.height == 0:
            continue
        s = g["shift"].to_numpy(); gs = g["gs"].to_numpy()[s]; yh = g["pred"].to_numpy()[s]; ex = g["excess"].to_numpy()[s]
        amb_c = float(g.filter(C("amb"))["dm_pred"].sum()); rest = float(g.filter(~C("amb") & ~C("shift"))["dm_pred"].sum())
        w = ex / ex.mean() if ex.mean() > 0 else np.ones_like(ex)
        rec = dict(day=d, nshift=int(s.sum()))
        for pc, P in [("pub", r["pub"][0]), ("pub7", r["pub7"][0])]:
            if P is None or not np.isfinite(P) or not s.any():
                rec[f"delta_{pc}"] = np.nan; rec[f"deltaw_{pc}"] = np.nan; continue
            tgt = P - amb_c - rest
            rec[f"delta_{pc}"] = wsolve(gs, yh, np.ones_like(gs), tgt)
            rec[f"deltaw_{pc}"] = wsolve(gs, yh, w, tgt)
        recs.append(rec)
    return days.join(pl.DataFrame(recs), on="day", how="left")


def gate_params(do, col):
    """from OTHER months' days: bias (median D) and sigma_L^2; returns (bias, sigL2, O_med)."""
    Dv = do[col].drop_nulls(); bias = float(Dv.median()); rsd = 1.4826 * float((Dv - bias).abs().median())
    sigL2 = max(rsd ** 2 - SIG_F ** 2 - float(do["Vamb"].median()), 100.0)
    return bias, sigL2, float(do["O"].median())


def excluded(dm, col, bias, sigL2, O_med):
    sig = np.sqrt(SIG_F ** 2 + sigL2 * np.maximum(1.0, dm["O"].to_numpy() / O_med) ** 2)
    return dm["snowday"].to_numpy() & ((dm[col].to_numpy() - bias) < -GATE_Z * sig), sig


def apply(f, dm, pc, which, lam, bias_delta, excl):
    """new values for the rows of dm's days: pred + lam * (delta - bias) [* w for A''] on shifted rows, excluded days unchanged."""
    col = f"delta_{pc}" if which == "flat" else f"deltaw_{pc}"
    dsel = dm.with_columns(pl.Series("excl", excl)).select("day", C(col).alias("dl"), C(f"delta_{pc}").alias("dflat"), "excl", (C("D" if pc == "pub" else "D7")).alias("Dx"))
    g = f.join(dsel, on="day", how="inner")
    sh = lam * (C("dl") - bias_delta)
    if which == "flat":
        shift = sh
    else:  # positive discrepancy after bias (file too high) -> weighted by excess; negative -> flat
        shift = pl.when((C("dflat") - bias_delta) < 0).then(sh * C("w")).otherwise(lam * (C("dflat") - bias_delta))
    g = g.with_columns(pl.when(C("shift") & ~C("excl") & C("dl").is_finite()).then(pl.max_horizontal(C("pred") + shift, pl.lit(0.0))).otherwise(C("pred")).alias("new"))
    return g


f0 = frame_2025([APT]); f, days = prep(f0, 2025)
wmean = f.filter(C("shift")).group_by("day").agg(C("excess").mean().alias("exm"))
f = f.join(wmean, on="day", how="left").with_columns(pl.when(C("exm") > 0).then(C("excess") / C("exm")).otherwise(1.0).alias("w"))
dd = day_solves(f, days)
months = sorted(dd["mon"].unique().to_list())
print("2025 LTFM rows", f.height, "gated days", dd.height, "snow rows", int(f["snow"].sum()), "snow rows on shift set", int(f.filter(C("shift") & C("snow")).height))

if MODE == "2025":
    outs = {}
    for pc in ["pub", "pub7"]:
        dcol = "D" if pc == "pub" else "D7"
        parts = []; lam_rec = {}; excl_days = []
        for m in months:
            do = dd.filter((C("mon") != m) & C(dcol).is_not_null()); dm = dd.filter(C("mon") == m)
            bias, sigL2, O_med = gate_params(do, dcol)
            ex_o, _ = excluded(do, dcol, bias, sigL2, O_med); ex_m, sig_m = excluded(dm, dcol, bias, sigL2, O_med)
            excl_days += [str(d) for d in dm.filter(pl.Series(ex_m))["day"].to_list()]
            do_k = do.filter(pl.Series(~ex_o))
            for which in ["flat", "wexc"]:
                col = f"delta_{pc}" if which == "flat" else f"deltaw_{pc}"
                bias_delta = float(do_k[col].drop_nulls().median())
                # nested lam: SSE over the other months' non-excluded days
                fo = f.filter(C("mon") != m)
                best = None
                for lam in LAMS:
                    go = apply(fo, do, pc, which, lam, bias_delta, ex_o)
                    sse = float(((go["new"] - go["y"]) ** 2).sum())
                    if best is None or sse < best[0]:
                        best = (sse, lam)
                lam = best[1]; lam_rec[(which, m)] = lam
                gm = apply(f.filter(C("mon") == m), dm, pc, which, lam, bias_delta, ex_m).with_columns(pl.lit(which).alias("which"), pl.lit(pc).alias("pc"), pl.lit(lam).alias("lam"), pl.lit(bias_delta).alias("bias_delta"))
                parts.append(gm)
        outs[pc] = pl.concat(parts)
        print(f"series {pc}: excluded days {sorted(set(excl_days))}; nested lam flat {[lam_rec[('flat', m)] for m in months]}, wexc {[lam_rec[('wexc', m)] for m in months]}")

    def score(t, label):
        t = t.with_columns(((C("pred") - C("y")) ** 2 - (C("new") - C("y")) ** 2).alias("g"))
        tot = t["g"].sum() / SE_SCALE
        byday = t.group_by("day").agg(C("g").sum(), C("Dx").first(), C("bias_delta").first(), C("excl").first()).sort("day"); gd = byday["g"].to_numpy()
        lo, hi, p = day_bootstrap(gd); top5 = np.sort(gd)[-5:].sum() / SE_SCALE
        bym = t.group_by("mon").agg((C("g").sum() / SE_SCALE).round(3).alias("g")).sort("mon")
        sgn = np.sign(byday["Dx"].to_numpy() - 0)  # sign of the raw discrepancy D (positive = published above the file)
        pos = gd[sgn > 0].sum() / SE_SCALE; neg = gd[sgn < 0].sum() / SE_SCALE
        print(f"{label}: gain {tot:+.3f} board-s; bootstrap [{lo:+.3f}, {hi:+.3f}] P>0 {p:.2f}; without top-5 days {tot - top5:+.3f}; by month {dict(zip(bym['mon'].to_list(), bym['g'].to_list()))}; "
              f"positive-D days {pos:+.3f} ({int((sgn > 0).sum())}), negative-D days {neg:+.3f} ({int((sgn < 0).sum())}); share of days improved {float((gd > 0).mean()):.2f}; "
              f"snow rows {t.filter(C('snow'))['g'].sum() / SE_SCALE:+.3f} ({t.filter(C('snow') & (C('new') != C('pred'))).height} shifted), other rows {t.filter(~C('snow'))['g'].sum() / SE_SCALE:+.3f}; "
              f"mean |shift| {float((t['new'] - t['pred']).abs().filter(t['new'] != t['pred']).mean()) if (t['new'] != t['pred']).any() else 0:.0f} s, max {float((t['new'] - t['pred']).abs().max()):.0f}")
        return tot

    for which in ["flat", "wexc"]:
        print(f"\n===== {'A-prime (flat, CANDIDATE)' if which == 'flat' else 'A-double-prime (excess-weighted, comparison)'}")
        score(outs["pub"].filter(C("which") == which), "REAL   ")
        score(outs["pub7"].filter(C("which") == which), "CONTROL")
    sel = outs["pub"].filter(C("which") == "flat")
    print("\nexcluded 2025 days (real series):", sorted(set(sel.filter(C("excl"))["day"].cast(pl.Utf8).to_list())))
    print("identity residual p90 2025 (|D| on non-excluded days, min):", float(dd.filter(C("D").is_not_null())["D"].abs().quantile(0.9)))
    lams = {m: float(sel.filter(C("mon") == m)["lam"][0]) for m in months}
    json.dump({"lam": float(np.median(list(lams.values()))), "lams": lams}, open(I + "agent_dayeq2_ltfm_level_cfg.json", "w"))
    sel.write_parquet(I + "agent_dayeq2_val_LTFM_level.parquet")
else:
    cfg = json.load(open(I + "agent_dayeq2_ltfm_level_cfg.json")); lam = cfg["lam"]
    bias, sigL2, O_med = gate_params(dd.filter(C("D").is_not_null()), "D")
    ex_all, _ = excluded(dd, "D", bias, sigL2, O_med); bias_delta = float(dd.filter(pl.Series(~ex_all))["delta_pub"].drop_nulls().median())
    print(f"2025 parameters: lam {lam}, bias_delta {bias_delta:+.1f} s, bias D {bias:+.0f} min, sigma_L {np.sqrt(sigL2):.0f} min")
    f6 = frame_2026([APT], BASE); f6, days6 = prep(f6, 2026)
    f6 = f6.with_columns(pl.lit(1.0).alias("w")); d6 = day_solves(f6, days6)
    ex6, sig6 = excluded(d6, "D", bias, sigL2, O_med)
    d6 = d6.with_columns(pl.Series("excl", ex6), pl.Series("sig", sig6), ((C("D") - bias) / pl.Series("sig", sig6)).alias("z"), (C("D") / C("n") * 60).alias("D_s_per_flt"))
    print("2026 gated days", d6.height, "of", f6["day"].n_unique(), "; excluded by the stop-day gate:", d6.filter(C("excl")).select("day", "D", "z", "nsnow").to_dicts())
    g6 = apply(f6, d6, "pub", "flat", lam, bias_delta, ex6).with_columns((C("new") - C("pred")).alias("move"))
    mv = g6.filter(C("move") != 0)
    print(f"rows moved {mv.height}, rms {float((mv['move'] ** 2).mean()) ** 0.5:.1f} s, max |move| {float(mv['move'].abs().max()):.0f} s, mean move Jan {float(g6.filter(C('mon') == 1)['move'].mean()):+.2f} s, Jul {float(g6.filter(C('mon') == 7)['move'].mean()):+.2f} s (all rows)")
    ds = g6.group_by("day").agg(n=pl.len(), nshift=C("shift").sum(), shift_s=C("move").filter(C("move") != 0).mean(), D=C("Dx").first(), delta=C("dl").first()).join(d6.select("day", "D_s_per_flt", "z"), on="day", how="left")
    print("ten largest day shifts:"); print(ds.sort(C("shift_s").abs(), descending=True).head(10).with_columns(pl.col(pl.Float64).round(1)))
    # arithmetic: n x shift^2 under the 2025 shrink (stated as arithmetic): expected SSE reduction if the shift were exactly right would be sum shift^2 ... reported as sum move^2 / 1.5e8
    print(f"arithmetic (not a measured gain): sum of squared shifts on the scored rows / 1.5e8 = {float((g6['move'] ** 2).sum()) / SE_SCALE:.3f} board-s (the ceiling if every shift were exactly the error; the 2025 ratio of gain to sum shift^2 is printed by the 2025 run)")
    # loss if the identity were off by the 2025 p90 residual on every day: shift error = lam * p90 / n_shift * 60 s per row
    p90 = 4.0  # min, from the 2025 identity (docs/LAB-NOTES.md: LTFM p90 3.5)
    err = lam * p90 * 60 / np.maximum(ds["nshift"].to_numpy(), 1)
    print(f"loss if the identity were off by its 2025 p90 residual ({p90} min) every day: {float((ds['nshift'].to_numpy() * err ** 2).sum()) / SE_SCALE:.4f} board-s")
    # build on v156
    b = pl.read_parquet(BASE)
    r = (pl.scan_parquet(R + "data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP").select(C("MVT_ID_mvt").alias("id"), C("FLIGHT_ID_mvt").is_not_null().alias("plan"), (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("glr"),
                                                                                              (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("gsr")).collect())
    j = b.rename({"MVT_ID_mvt": "id"}).join(g6.select("id", "new"), on="id", how="left").join(r, on="id", how="left")
    j = j.with_columns(pl.coalesce(C("new"), C("TAXITIME_SEC_mvt").cast(pl.Float64)).alias("new"))
    j = j.with_columns(pl.when(C("new").is_not_null() & ~C("plan") & (C("gsr") >= 0) & C("id").is_in(g6["id"])).then(pl.min_horizontal(C("new"), C("gsr") + 1200)).otherwise(C("new")).alias("new"))
    lo = pl.max_horizontal(C("glr") - 3606, pl.lit(0)); hi = C("glr") + 3606
    j = j.with_columns(pl.when(C("glr").is_null()).then(C("new")).otherwise(pl.min_horizontal(pl.max_horizontal(C("new"), lo), hi)).alias("new"))
    out = pl.DataFrame({"MVT_ID_mvt": b["MVT_ID_mvt"], "TAXITIME_SEC_mvt": j["new"].round().cast(b.schema["TAXITIME_SEC_mvt"])})
    assert out["TAXITIME_SEC_mvt"].null_count() == 0
    out.write_parquet(I + "agent_dayeq2_ltfm_level_final.parquet")
    dmv = (out["TAXITIME_SEC_mvt"] - b["TAXITIME_SEC_mvt"]).cast(pl.Float64).to_numpy()
    print(f"built agent_dayeq2_ltfm_level_final.parquet: rows moved {(dmv != 0).sum()}, rms {np.sqrt((dmv[dmv != 0] ** 2).mean()):.1f}, max {np.abs(dmv).max():.0f}, sum dp^2/row {(dmv ** 2).sum() / len(dmv):.1f}, mean {dmv.mean():+.3f}")
    g6.write_parquet(I + "agent_dayeq2_rows_LTFM_level.parquet")
