"""day-equation 2, section 10: A-big = round one's day-level shift of delayed flights applied only on days with |D - bias| > k sigma_day
(IFR rows at EHAM / LSZH); k = 3 candidate, 2 and 4 for information. Usage:
  python agent_dayeq2_05_big.py APT 2025 | 2026 | diag
diag: the stop-day mechanism table (LTFM February stop days / the airport's snow days)."""
import json
import sys
from pathlib import Path
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa
from agent_eusnow_feat import snow_features_apts

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(120)
APT = sys.argv[1]; MODE = sys.argv[2]
SIG_F = 5.0 if APT != "LTFM" else 5.0; GATE_Z = 3.0; MIN_N = 100; LAMS = [round(0.1 * i, 1) for i in range(11)]
AMB_GAP, AMB_PRED = 5400.0, 1800.0
KS = [2.0, 3.0, 4.0]; KCAND = 3.0
BASE = I + "agent_bundle_v156_final.parquet"
IFR = APT in ("EHAM", "LSZH")


def prep(f, year):
    if IFR:
        f = f.filter(C("rule") == "I")
    sn = snow_features_apts(f.select(C("id").alias("MVT_ID_mvt"), pl.lit(APT).alias("ADEP_mvt"), C("T").alias("MVT_TIME_UTC_mvt")), Path(R + "data/raw"), [APT]).select(C("MVT_ID_mvt").alias("id"), (C("lt_snow_3h") > 0).alias("snow"))
    f = f.join(sn, on="id", how="left").with_columns(C("snow").fill_null(False), C("day_pred").alias("day"))
    f = f.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"), ((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("p_copy"))
    f = f.with_columns(pl.when(C("amb")).then(((1 - C("p_copy")) * (C("gs") - C("taxi")) / 60.0).clip(lower_bound=0)).otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm_pred"),
                       pl.when(C("amb")).then(C("p_copy") * (1 - C("p_copy")) * ((C("gs") - C("taxi")) / 60.0) ** 2).otherwise(0.0).alias("var_amb"),
                       (~C("amb") & (C("gs") > C("pred")) & ~C("fs").is_in(HEDGED)).fill_null(False).alias("shift"))
    pub = load_pub(year, [APT])
    days = (f.group_by("day").agg(n=pl.len(), O=C("dm_pred").sum(), Vamb=C("var_amb").sum(), snowday=C("snow").any(), nsnow=C("snow").sum(), nshift=C("shift").sum())
            .join(pub, on="day", how="inner").with_columns((C("n_pub") - C("n")).alias("dn")))
    days = days.filter((C("dn").abs() <= DN_MAX) & (C("n") >= MIN_N) & (C("n_pub") >= MIN_N)).with_columns((C("pub") - C("O")).alias("D"), (C("pub7") - C("O")).alias("D7"), C("day").dt.month().alias("mon"))
    return f, days


def day_solves(f, days):
    recs = []
    for (d,), g in f.partition_by("day", as_dict=True, maintain_order=True).items():
        r = days.filter(C("day") == d)
        if r.height == 0:
            continue
        s = g["shift"].to_numpy(); gs = g["gs"].to_numpy()[s]; yh = g["pred"].to_numpy()[s]
        amb_c = float(g.filter(C("amb"))["dm_pred"].sum()); rest = float(g.filter(~C("amb") & ~C("shift"))["dm_pred"].sum())
        rec = dict(day=d)
        for pc, P in [("pub", r["pub"][0]), ("pub7", r["pub7"][0])]:
            rec[f"delta_{pc}"] = solve_delta(gs, yh, P - amb_c - rest) if (P is not None and np.isfinite(P) and s.any()) else np.nan
        recs.append(rec)
    return days.join(pl.DataFrame(recs), on="day", how="left")


def params(do, col):
    Dv = do[col].drop_nulls(); bias = float(Dv.median()); rsd = 1.4826 * float((Dv - bias).abs().median())
    return bias, max(rsd ** 2 - SIG_F ** 2 - float(do["Vamb"].median()), 100.0), float(do["O"].median())


def sig_day(dm, sigL2, O_med):
    return np.sqrt(SIG_F ** 2 + sigL2 * np.maximum(1.0, dm["O"].to_numpy() / O_med) ** 2)


def apply(f, dm, pc, lam, bias_delta, big):
    col = f"delta_{pc}"
    dsel = dm.with_columns(pl.Series("big", big)).select("day", C(col).alias("dl"), "big", C("D" if pc == "pub" else "D7").alias("Dx"))
    g = f.join(dsel, on="day", how="inner")
    return g.with_columns(pl.when(C("shift") & C("big") & C("dl").is_finite()).then(pl.max_horizontal(C("pred") + lam * (C("dl") - bias_delta), pl.lit(0.0))).otherwise(C("pred")).alias("new"))


f0 = frame_2025([APT]); f, days = prep(f0, 2025); dd = day_solves(f, days)
months = sorted(dd["mon"].unique().to_list())
print(f"{APT} 2025 ({'IFR rows' if IFR else 'all rows'}): rows {f.height}, gated days {dd.height}, snow rows {int(f['snow'].sum())}, snow days {int(dd['snowday'].sum())}")

if MODE == "diag":
    # the stop-day mechanism: LTFM February stop days or the airport's snow days with |D| > 300 min
    DMIN = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    sel = dd.filter(C("snowday") & (C("D").abs() > DMIN)).sort("day")
    print("day | n_shift | D min | s = true mean error of shifted rows (s) | n s^2 / 1.5e8 | exact delta (s) | gain at delta (board-s) | rows leaving the sum at delta | range hit | share of true residual outside the shifted set | share of shifted rows with error > day mean shift")
    for r in sel.iter_rows(named=True):
        g = f.filter(C("day") == r["day"]); s_ = g["shift"].to_numpy(); yv = g["y"].to_numpy(); yh = g["pred"].to_numpy(); gs = g["gs"].to_numpy()
        err = yv - yh; s = float(err[s_].mean()); n = int(s_.sum()); dl = r["delta_pub"]
        gain = float(((yh[s_] - yv[s_]) ** 2 - (yh[s_] + dl - yv[s_]) ** 2).sum()) / SE_SCALE
        leave = int((gs[s_] - yh[s_] - dl < 0).sum())
        Rtrue = (np.clip(gs - yv, 0, None) - np.clip(gs - yh, 0, None)) / 60.0
        share_out = float(Rtrue[~s_].sum() / Rtrue.sum()) if Rtrue.sum() != 0 else float("nan")
        print(f"{r['day']} | {n} | {r['D']:+.0f} | {s:+.0f} | {n * s * s / SE_SCALE:.3f} | {dl:+.0f} | {gain:+.3f} | {leave} | {'yes' if abs(dl) >= 7199 else 'no'} | {share_out:.2f} | {float((err[s_] > s).mean()):.2f}")
    sys.exit(0)

if MODE == "2025":
    res = {}
    for k in KS:
        for pc in ["pub", "pub7"]:
            dcol = "D" if pc == "pub" else "D7"; parts = []; lams = []
            for m in months:
                do = dd.filter((C("mon") != m) & C(dcol).is_not_null()); dm = dd.filter(C("mon") == m)
                bias, sigL2, O_med = params(do, dcol)
                big_o = np.abs(do[dcol].to_numpy() - bias) > k * sig_day(do, sigL2, O_med); big_m = np.abs(dm[dcol].to_numpy() - bias) > k * sig_day(dm, sigL2, O_med)
                bias_delta = float(do[f"delta_{pc}"].drop_nulls().median())
                fo = f.filter(C("mon") != m); best = None
                for lam in LAMS:
                    go = apply(fo, do, pc, lam, bias_delta, big_o); sse = float(((go["new"] - go["y"]) ** 2).sum())
                    if best is None or sse < best[0]:
                        best = (sse, lam)
                lam = best[1]; lams.append(lam)
                gm = apply(f.filter(C("mon") == m), dm, pc, lam, bias_delta, big_m).with_columns(pl.lit(lam).alias("lam"), pl.lit(bias_delta).alias("bias_delta"))
                stop = dm["snowday"].to_numpy() & ((dm[dcol].to_numpy() - bias) < -GATE_Z * sig_day(dm, sigL2, O_med))
                zz = (dm[dcol].to_numpy() - bias) / sig_day(dm, sigL2, O_med)
                gm = gm.join(dm.with_columns(pl.Series("stop", stop), pl.Series("z", zz)).select("day", "stop", "z", "snowday"), on="day", how="left")
                parts.append(gm)
            res[(k, pc)] = (pl.concat(parts), lams)
    for k in KS:
        print(f"\n===== k = {k}{' (CANDIDATE)' if k == KCAND else ' (information only)'}")
        for pc in ["pub", "pub7"]:
            t, lams = res[(k, pc)]
            t = t.with_columns(((C("pred") - C("y")) ** 2 - (C("new") - C("y")) ** 2).alias("g"))
            bd = t.filter(C("big")).group_by("day").agg(C("g").sum(), C("Dx").first(), C("stop").first(), C("snowday").first(), C("z").first(), (C("new") - C("pred")).filter(C("new") != C("pred")).mean().alias("shift")).sort("day")
            gd = bd["g"].to_numpy(); tot = gd.sum() / SE_SCALE
            if len(gd) == 0:
                print(f"{pc}: no day passes the threshold"); continue
            lo, hi, p = day_bootstrap(gd); top3 = np.sort(gd)[-3:].sum() / SE_SCALE
            Dx = bd["Dx"].to_numpy(); pos = gd[Dx > 0].sum() / SE_SCALE; neg = gd[Dx < 0].sum() / SE_SCALE
            print(f"{'REAL   ' if pc == 'pub' else 'CONTROL'}: days passing {len(gd)}; gain {tot:+.3f} board-s; bootstrap [{lo:+.3f}, {hi:+.3f}] P>0 {p:.2f}; without best 3 days {tot - top3:+.3f}; days improved {int((gd > 0).sum())}/{len(gd)}; "
                  f"positive-D days {pos:+.3f} ({int((Dx > 0).sum())}), negative-D {neg:+.3f} ({int((Dx < 0).sum())}); nested lam {lams}; rows shifted {int((t['new'] != t['pred']).sum())}, mean |shift| {float((t['new'] - t['pred']).abs().filter(t['new'] != t['pred']).mean()) if (t['new'] != t['pred']).any() else 0:.0f} s")
            if pc == "pub":
                print("  per day:", [(str(r["day"]), round(r["Dx"]), round(r["z"], 1), round(r["g"] / SE_SCALE, 3), "snow" if r["snowday"] else "", "STOP" if r["stop"] else "") for r in bd.iter_rows(named=True)])
                sn = bd.filter(C("snowday")); st = bd.filter(C("stop"))
                print(f"  snow-flag days passing: {sn.height}, all gain: {bool((sn['g'] > 0).all()) if sn.height else 'n/a'}, their gain {sn['g'].sum() / SE_SCALE:+.3f}; stop-gate days: {[str(d) for d in st['day'].to_list()]} gain {st['g'].sum() / SE_SCALE:+.3f}")
                if k == KCAND:
                    ok = tot > 0 and p >= 0.9 and (tot - top3) > 0 and len(gd) >= 10 and (sn.height == 0 or bool((sn["g"] > 0).all()))
                    res["gate"] = (ok, tot); t.write_parquet(I + f"agent_dayeq2_val_{APT}_big.parquet"); json.dump({"lam": float(np.median(lams)), "lams": lams}, open(I + f"agent_dayeq2_{APT}_big_cfg.json", "w"))
            elif k == KCAND:
                ok, tot_r = res["gate"]; ok = ok and tot < 0.5 * tot_r
                print(f"GATE k = 3: {'PASS' if ok else 'FAIL'} (control {tot:+.3f} vs real {tot_r:+.3f})")
else:
    cfg = json.load(open(I + f"agent_dayeq2_{APT}_big_cfg.json")); lam = cfg["lam"]
    bias, sigL2, O_med = params(dd.filter(C("D").is_not_null()), "D"); bias_delta = float(dd["delta_pub"].drop_nulls().median())
    print(f"2025 parameters: lam {lam}, bias_delta {bias_delta:+.1f} s, bias D {bias:+.0f} min, sigma_L {np.sqrt(sigL2):.0f} min, O_med {O_med:.0f}")
    f6 = frame_2026([APT], BASE); f6, days6 = prep(f6, 2026); d6 = day_solves(f6, days6)
    sg = sig_day(d6, sigL2, O_med); z = (d6["D"].to_numpy() - bias) / sg; big = np.abs(z) > KCAND
    d6 = d6.with_columns(pl.Series("z", z), pl.Series("big", big), (C("D") / C("nshift") * 60).alias("D_s_per_delayed"))
    print("2026 gated days", d6.height, "of", f6["day"].n_unique(), "; days passing k = 3:")
    print(d6.filter(C("big")).select("day", "n", "nshift", "nsnow", "D", "z", "D_s_per_delayed", "delta_pub").sort("day").with_columns(pl.col(pl.Float64).round(1)))
    g6 = apply(f6, d6, "pub", lam, bias_delta, big).with_columns((C("new") - C("pred")).alias("move"))
    mv = g6.filter(C("move") != 0)
    print(f"rows moved {mv.height} (plan {mv.filter(C('hasplan')).height}, no-plan {mv.filter(~C('hasplan')).height}), rms {float((mv['move'] ** 2).mean()) ** 0.5 if mv.height else 0:.1f} s, max |move| {float(mv['move'].abs().max()) if mv.height else 0:.0f}, mean move Jan {float(g6.filter(C('mon') == 1)['move'].mean()):+.2f}, Jul {float(g6.filter(C('mon') == 7)['move'].mean()):+.2f}")
    if APT == "EHAM":
        sw = g6.filter(~C("hasplan") & (C("day") >= pl.date(2026, 1, 2)) & (C("day") <= pl.date(2026, 1, 7)))
        print(f"no-plan snow-week rows (2-7 Jan): {sw.height}, moved {sw.filter(C('move') != 0).height}, mean move {float(sw['move'].mean()):+.1f} s, min {float(sw['move'].min()):+.0f}, max {float(sw['move'].max()):+.0f}")
        print(sw.group_by("day").agg(n=pl.len(), moved=(C("move") != 0).sum(), mean_move=C("move").mean(), pred_mean=C("pred").mean()).sort("day").with_columns(pl.col(pl.Float64).round(1)))
    print(f"arithmetic (not a measured gain): sum shift^2 / 1.5e8 = {float((g6['move'] ** 2).sum()) / SE_SCALE:.3f} board-s")
    ds = g6.filter(C("big")).group_by("day").agg(nshift=C("shift").sum(), shift_s=C("move").filter(C("move") != 0).mean())
    err = lam * 5.0 * 60 / np.maximum(ds["nshift"].to_numpy(), 1)
    print(f"loss if the identity were off by its 2025 p90 residual (5 min) on every passing day: {float((ds['nshift'].to_numpy() * err ** 2).sum()) / SE_SCALE:.4f} board-s")
    b = pl.read_parquet(BASE)
    r = (pl.scan_parquet(R + "data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP").select(C("MVT_ID_mvt").alias("id"), (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("glr")).collect())
    j = b.rename({"MVT_ID_mvt": "id"}).join(g6.select("id", "new"), on="id", how="left").join(r, on="id", how="left").with_columns(pl.coalesce(C("new"), C("TAXITIME_SEC_mvt").cast(pl.Float64)).alias("new"))
    lo = pl.max_horizontal(C("glr") - 3606, pl.lit(0)); hi = C("glr") + 3606
    j = j.with_columns(pl.when(C("glr").is_null()).then(C("new")).otherwise(pl.min_horizontal(pl.max_horizontal(C("new"), lo), hi)).alias("new"))
    out = pl.DataFrame({"MVT_ID_mvt": b["MVT_ID_mvt"], "TAXITIME_SEC_mvt": j["new"].round().cast(b.schema["TAXITIME_SEC_mvt"])}); assert out["TAXITIME_SEC_mvt"].null_count() == 0
    name = f"agent_dayeq2_{APT.lower()}_big_final.parquet"; out.write_parquet(I + name)
    dmv = (out["TAXITIME_SEC_mvt"] - b["TAXITIME_SEC_mvt"]).cast(pl.Float64).to_numpy()
    print(f"built {name}: rows moved {(dmv != 0).sum()}, rms {np.sqrt((dmv[dmv != 0] ** 2).mean()) if (dmv != 0).any() else 0:.1f}, max {np.abs(dmv).max():.0f}, sum dp^2/row {(dmv ** 2).sum() / len(dmv):.1f}, mean {dmv.mean():+.3f}")
    g6.write_parquet(I + f"agent_dayeq2_rows_{APT}_big.parquet")
