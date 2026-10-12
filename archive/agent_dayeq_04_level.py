"""day-equation 04 (use 2): day-level calibration of ordinary flights from the identity published = sum_i max(gs_i - y_i, 0) (minutes).
D = published - sum_i max(gs_i - yhat_i, 0). Correction: solve delta so that sum_i max(gs_i - yhat_i - delta, 0) = published - bias (bisection, exact
non-linearity), then apply lam * delta to (a) flights with margin gs - yhat > 0, (b) all flights of the day, (c) all flights proportional to yhat
(delta_i = delta * yhat_i / mean yhat over the margin set). lam chosen leave-one-month-out among the frame months on 2025; control = published shifted 7 days.
Ambiguous rows (Rome mixture etc.) enter with their expected contribution under the shipped hedge and are never shifted.
Usage: python agent_dayeq_04_level.py 2025 | 2026"""
import glob
import sys
import numpy as np
import polars as pl

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(200)
C = pl.col
R = ""
E = R + "data/external/eurocontrol/"
I = R + "data/interim/"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
EXACT = ["EGLL", "EDDM", "EDDF", "LIRF", "LSZH"]  # rounding-level identity (02); LSZH next
MODE = sys.argv[1] if len(sys.argv) > 1 else "2025"
FRAME_MONTHS = [1, 2, 3, 7, 8, 9, 10, 11]
AMB_GAP, AMB_PRED = 5400.0, 1800.0
DN_MAX = 15
LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SE_SCALE = 1.5e8


def load_pub(year):
    d = pl.read_csv(f"{E}all_pre_departure_delays_{year}.csv", infer_schema_length=0).filter(C("APT_ICAO").is_in(APTS))
    return d.select(C("APT_ICAO").alias("apt"), C("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"), C("DLY_ALL_PRE_2").cast(pl.Float64, strict=False).alias("pub"),
                    C("FLT_DEP_1").cast(pl.Float64, strict=False).alias("n_pub"))


if MODE == "2025":
    files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
    raw = (pl.scan_parquet(files).filter(C("PHASE_mvt") == "DEP")
           .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"), C("FLIGHT_mvt").alias("fs")).collect())
    eb = pl.read_parquet(I + "agent_eb136_tbl.parquet").select("id", "mon", "pred").filter(C("mon").is_in(FRAME_MONTHS))
    f = raw.join(eb, on="id", how="inner")
    pub = load_pub(2025)
else:
    raw = (pl.scan_parquet(R + "data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
           .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"), C("FLIGHT_mvt").alias("fs")).collect())
    v = pl.read_parquet(I + "agent_bundle_v150_final.parquet").select(C("MVT_ID_mvt").alias("id"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred"))
    f = raw.join(v, on="id", how="left").with_columns(C("T").dt.month().alias("mon"), pl.lit(None, pl.Float64).alias("y"))
    pub = load_pub(2026)
f = f.with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs"))
taxi = f.filter(C("gs").abs() < 3600).group_by("apt", "mon").agg(taxi=C("pred").median())
f = f.join(taxi, on=["apt", "mon"], how="left")
f = f.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"), (C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day"))
# ambiguous rows: expected contribution under the shipped hedge p (copy: 0; real: gs - taxi); rows with pred > gs (day-early hedges) contribute 0
f = f.with_columns(((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("p_copy"))
f = f.with_columns(pl.when(C("amb")).then(((1 - C("p_copy")) * (C("gs") - C("taxi")) / 60.0).clip(lower_bound=0)).otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm_pred"),
                   pl.when(C("amb")).then(C("p_copy") * (1 - C("p_copy")) * ((C("gs") - C("taxi")) / 60.0) ** 2).otherwise(0.0).alias("var_amb"))
f = f.filter(C("gs").is_not_null() & C("pred").is_not_null())
days = f.group_by("apt", "day").agg(n=pl.len(), O=C("dm_pred").sum(), k=C("amb").sum(), n_pos=((C("gs") > C("pred")) & ~C("amb")).sum(), sd_amb=C("var_amb").sum().sqrt(),
                                    **({"true_sum": ((C("gs") - C("y")) / 60.0).clip(lower_bound=0).sum()} if MODE == "2025" else {}))
days = days.join(pub, on=["apt", "day"], how="inner").with_columns((C("pub") - C("O")).alias("D"), (C("n_pub") - C("n")).alias("dn"))
pub7 = pub.select("apt", (C("day") - pl.duration(days=7)).alias("day"), C("pub").alias("pub7"))
days = days.join(pub7, on=["apt", "day"], how="left")

print(MODE, "rows", f.height, "airport-days", days.height)
print("\nD = published - predicted sum, per airport (minutes): bias, robust sd, sd, and per flight")
print(days.group_by("apt").agg(days=pl.len(), n=C("n").mean(), D_med=C("D").median(), D_mad=(C("D") - C("D").median()).abs().median() * 1.4826, D_sd=C("D").std(),
                               D_per_flt_s=(C("D") / C("n") * 60).median(), abs_D_per_flt_p90=(C("D") / C("n") * 60).abs().quantile(0.9), sd_amb=C("sd_amb").mean(),
                               dn_med=C("dn").median()).sort("apt").with_columns(pl.col(pl.Float64).round(1)))
if MODE == "2026":
    print("\nTop 20 airport-days by |D| (2026, v150):")
    print(days.sort(C("D").abs(), descending=True).head(20).select("apt", "day", "n", "pub", "O", "D", (C("D") / C("n") * 60).alias("D_s_per_flt"), "k", "sd_amb", "dn").with_columns(pl.col(pl.Float64).round(1)))
    print("\nTop 20 airport-days by |D| among the exact airports:")
    print(days.filter(C("apt").is_in(EXACT)).sort(C("D").abs(), descending=True).head(20).select("apt", "day", "n", "pub", "O", "D", (C("D") / C("n") * 60).alias("D_s_per_flt"), "k", "sd_amb", "dn").with_columns(pl.col(pl.Float64).round(1)))

# ---- per-day solve
f = f.sort("apt", "day")
dayidx = {(r["apt"], r["day"]): r for r in days.iter_rows(named=True)}


def solve_delta(gs, yh, target):
    """delta with sum max(gs - yh - delta, 0) = target (minutes); gs, yh seconds. Monotone decreasing; bisection."""
    def S(d):
        return np.clip(gs - yh - d, 0, None).sum() / 60.0
    lo, hi = -7200.0, 7200.0
    if S(lo) < target:
        return lo
    if S(hi) > target:
        return hi
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if S(m) > target:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


# bias per airport: median D (2025: computed on the same frame months, nested below per held-out month; 2026: from 2025 run, stored)
recs = []
groups = f.partition_by("apt", "day", as_dict=True, maintain_order=True)
for (a, d), g in groups.items():
    r = dayidx.get((a, d))
    if r is None:
        continue
    ordm = ~g["amb"].to_numpy()
    gs = g["gs"].to_numpy(); yh = g["pred"].to_numpy()
    amb_contrib = g.filter(C("amb"))["dm_pred"].sum()
    rec = dict(apt=a, day=d, n=g.height, n_ord=int(ordm.sum()), D=r["D"], k=r["k"], sd_amb=r["sd_amb"], dn=r["dn"])
    for pc in ["pub", "pub7"]:
        P = r[pc]
        # count gate: the published day must describe our set of flights (month edges, missing days): |FLT_DEP_1 - our n| <= DN_MAX
        if P is None or not np.isfinite(P) or r["dn"] is None or abs(r["dn"]) > DN_MAX:
            rec[f"delta_{pc}"] = np.nan; continue
        rec[f"delta_{pc}"] = solve_delta(gs[ordm], yh[ordm], P - amb_contrib)  # bias handled at scoring (subtracted from P via per-airport median D)
    recs.append(rec)
dd = pl.DataFrame(recs)
print("\ndelta (seconds, lam = 1, no bias) per airport:")
print(dd.group_by("apt").agg(days=pl.len(), med=C("delta_pub").median(), mad=(C("delta_pub") - C("delta_pub").median()).abs().median() * 1.4826, p90abs=C("delta_pub").abs().quantile(0.9)).sort("apt").with_columns(pl.col(pl.Float64).round(1)))
dd.write_parquet(I + f"agent_dayeq_level_days_{MODE}.parquet")

if MODE == "2025":
    # scoring with nested lam and nested bias: for held-out month m, bias = median D over other months; lam = argmin SSE over other months
    f = f.join(dd.select("apt", "day", "delta_pub", "delta_pub7"), on=["apt", "day"], how="left").with_columns(C("day").dt.month().alias("dmon"))
    # bias in delta units: solve with P - bias is costly; approximate: delta_biased = delta - bias_delta where bias_delta = median delta over other months (equivalent to removing the airport's systematic offset)
    res = []
    for a in APTS:
        fa = f.filter((C("apt") == a) & ~C("amb") & C("y").is_not_null())
        if fa.height == 0:
            continue
        yv = fa["y"].to_numpy(); yh = fa["pred"].to_numpy(); gs = fa["gs"].to_numpy(); mon = fa["dmon"].to_numpy(); dl = fa["delta_pub"].to_numpy(); dl7 = fa["delta_pub7"].to_numpy()
        pos = gs > yh
        base = ((yh - yv) ** 2)
        out = dict(apt=a, rows=fa.height, base_rmse=np.sqrt(base.mean()))
        for pc, dlx in [("pub", dl), ("pub7", dl7)]:
            for variant in ["a_pos", "b_all", "c_prop"]:
                tot_gain = 0.0; lams = []; gains_day = {}
                for m in FRAME_MONTHS:
                    te = mon == m; trn = ~te
                    bias = np.nanmedian(dlx[trn]) if np.isfinite(dlx[trn]).any() else 0.0
                    dcorr = np.nan_to_num(dlx - bias)
                    if variant == "a_pos":
                        shift = np.where(pos, dcorr, 0.0)
                    elif variant == "b_all":
                        shift = dcorr
                    else:
                        shift = dcorr * yh / np.maximum(np.nanmedian(yh), 1)
                    best = None
                    for lam in LAMS:
                        sse = ((yh[trn] + lam * shift[trn] - yv[trn]) ** 2).sum()
                        if best is None or sse < best[0]:
                            best = (sse, lam)
                    lam = best[1]; lams.append(lam)
                    g = base[te].sum() - ((yh[te] + lam * shift[te] - yv[te]) ** 2).sum()
                    tot_gain += g
                    # per-day gains for the top-5 removal
                    dayv = fa["day"].to_numpy()[te]; gd = base[te] - (yh[te] + lam * shift[te] - yv[te]) ** 2
                    for dv, gv in zip(dayv, gd):
                        gains_day[dv] = gains_day.get(dv, 0.0) + gv
                gd = np.array(sorted(gains_day.values()))
                out[f"{pc}_{variant}_gain_s"] = tot_gain / SE_SCALE
                out[f"{pc}_{variant}_lam"] = str(lams)
                out[f"{pc}_{variant}_no_top5"] = (tot_gain - gd[-5:].sum()) / SE_SCALE if len(gd) > 5 else np.nan
                out[f"{pc}_{variant}_no_bot5"] = (tot_gain - gd[:5].sum()) / SE_SCALE if len(gd) > 5 else np.nan
                out[f"{pc}_{variant}_days_pos"] = float((gd > 0).mean())
        res.append(out)
    rs = pl.DataFrame(res)
    for variant in ["a_pos", "b_all", "c_prop"]:
        print(f"\n===== variant {variant}: gain in board seconds (positive = better), all 8 frame months scaled as-is (sum SSE / 1.5e8)")
        print(rs.select("apt", "rows", "base_rmse", f"pub_{variant}_gain_s", f"pub_{variant}_no_top5", f"pub_{variant}_no_bot5", f"pub_{variant}_days_pos", f"pub_{variant}_lam",
                        f"pub7_{variant}_gain_s", f"pub7_{variant}_lam").with_columns(pl.col(pl.Float64).round(3)))
    rs.write_parquet(I + "agent_dayeq_level_score_2025.parquet")
    # disruption split: top 10% days by |D| per airport vs the rest, variant a at lam chosen = median of nested lams
    print("\ndisruption split (variant a_pos, lam = nested): days in the top 10% of |D| vs the rest; gain in board s")
    for a in APTS:
        fa = f.filter((C("apt") == a) & ~C("amb") & C("y").is_not_null())
        if fa.height == 0:
            continue
        yv = fa["y"].to_numpy(); yh = fa["pred"].to_numpy(); gs = fa["gs"].to_numpy(); dl = fa["delta_pub"].to_numpy(); mon = fa["dmon"].to_numpy()
        pos = gs > yh
        lam_s = rs.filter(C("apt") == a)[f"pub_a_pos_lam"][0]; lam = float(np.median(eval(lam_s)))
        shift = np.where(pos, np.nan_to_num(dl - np.nanmedian(dl)), 0.0)
        gain = (yh - yv) ** 2 - (yh + lam * shift - yv) ** 2
        Dabs = fa.join(dd.select("apt", "day", C("D").alias("Dd")), on=["apt", "day"], how="left")["Dd"].abs().to_numpy()
        thr = np.nanquantile(np.unique(Dabs), 0.9)
        hi = Dabs >= thr
        print(f"{a}: lam {lam:.1f}; disruption days gain {gain[hi].sum()/SE_SCALE:+.3f} ({hi.sum()} rows), ordinary days {gain[~hi].sum()/SE_SCALE:+.3f}; "
              f"mean |shift| applied {np.abs(lam*shift).mean():.0f} s; rows shifted {(shift != 0).mean():.2f}")
else:
    # build agent_dayeq_level_final.parquet on v150 for the exact airports with lam and bias from the 2025 run (stored by hand below after reading the 2025 log)
    import json
    cfg = json.load(open(I + "agent_dayeq_level_cfg.json"))
    f = f.join(dd.select("apt", "day", "delta_pub"), on=["apt", "day"], how="left")
    new = f["pred"].to_numpy().copy(); moved = np.zeros(f.height, bool)
    apt = f["apt"].to_numpy(); amb = f["amb"].to_numpy(); gs = f["gs"].to_numpy(); yh = f["pred"].to_numpy(); dl = f["delta_pub"].to_numpy()
    for a, c in cfg.items():
        s = (apt == a) & ~amb & np.isfinite(dl)
        shift = np.where(gs > yh, dl - c["bias"], 0.0) if c["variant"] == "a_pos" else (dl - c["bias"])
        new[s] = np.maximum(yh[s] + c["lam"] * shift[s], 0); moved |= s
    outp = pl.DataFrame({"id": f["id"], "new": new, "moved": moved})
    v = pl.read_parquet(I + "agent_bundle_v150_final.parquet")
    v2 = v.join(outp.rename({"id": "MVT_ID_mvt"}), on="MVT_ID_mvt", how="left").with_columns(
        pl.when(C("moved")).then(C("new").round(0)).otherwise(C("TAXITIME_SEC_mvt").cast(pl.Float64)).cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select("MVT_ID_mvt", "TAXITIME_SEC_mvt")
    v2.write_parquet(I + "agent_dayeq_level_final.parquet")
    mv = (v2["TAXITIME_SEC_mvt"] - v["TAXITIME_SEC_mvt"]).to_numpy()
    print("built agent_dayeq_level_final.parquet: rows moved", int((mv != 0).sum()), "mean move", mv[mv != 0].mean() if (mv != 0).any() else 0, "rms", np.sqrt((mv ** 2).mean()), "max |move|", np.abs(mv).max())
    j = f.with_columns(pl.Series("new", new)).with_columns((C("new") - C("pred")).alias("move")).filter(C("move").abs() > 1000)
    print("rows moved by more than 1000 s:", j.height)
    print(j.select("apt", "day", "fs", "gs", "pred", "new", "move", "delta_pub").sort(C("move").abs(), descending=True).head(40))
