"""day-equation 2, section 11: the stop-day exclusion at the six piece-A airports. Usage: python agent_dayeq2_06_excl.py APT 2025 | 2026
2025: piece A reproduced (variant a, nested lam / bias delta) with and without the stop-gate days; the difference with a day bootstrap.
2026: stop-gate and weather-free-gate days on the pre-A base (v152) with what piece A (v156 - v152) applied there; build if adopted."""
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
SIG_F = 33.0 if APT == "LIRF" else 5.0; GATE_Z = 3.0; MIN_N = 100; LAMS = [round(0.1 * i, 1) for i in range(11)]
AMB_GAP, AMB_PRED = 5400.0, 1800.0


def prep(f, year):
    try:
        sn = snow_features_apts(f.select(C("id").alias("MVT_ID_mvt"), pl.lit(APT).alias("ADEP_mvt"), C("T").alias("MVT_TIME_UTC_mvt")), Path(R + "data/raw"), [APT]).select(C("MVT_ID_mvt").alias("id"), (C("lt_snow_3h") > 0).alias("snow"))
        f = f.join(sn, on="id", how="left")
    except IndexError:  # the airport's METAR carries no snow / freezing observation in the file: no snow row anywhere
        print("no snow observation in the METAR file for", APT); f = f.with_columns(pl.lit(False).alias("snow"))
    f = f.with_columns(C("snow").fill_null(False), C("day_pred").alias("day"))
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


def apply(f, dm, pc, lam, bias_delta, excl):
    col = f"delta_{pc}"
    dsel = dm.with_columns(pl.Series("excl", excl)).select("day", C(col).alias("dl"), "excl", C("D" if pc == "pub" else "D7").alias("Dx"))
    g = f.join(dsel, on="day", how="inner")
    return g.with_columns(pl.when(C("shift") & ~C("excl") & C("dl").is_finite()).then(pl.max_horizontal(C("pred") + lam * (C("dl") - bias_delta), pl.lit(0.0))).otherwise(C("pred")).alias("new"))


f0 = frame_2025([APT]); f, days = prep(f0, 2025); dd = day_solves(f, days)
months = sorted(dd["mon"].unique().to_list())
print(f"{APT} 2025: rows {f.height}, gated days {dd.height}, snow rows {int(f['snow'].sum())}, snow days {int(dd['snowday'].sum())}")

if MODE == "2025":
    parts = []; lams = []; stop_all = []; wf_all = []
    for m in months:
        do = dd.filter((C("mon") != m) & C("D").is_not_null()); dm = dd.filter(C("mon") == m)
        bias, sigL2, O_med = params(do, "D"); zz = (dm["D"].to_numpy() - bias) / sig_day(dm, sigL2, O_med)
        stop = dm["snowday"].to_numpy() & (zz < -GATE_Z); wf = zz < -GATE_Z
        bias_delta = float(do["delta_pub"].drop_nulls().median()); fo = f.filter(C("mon") != m); best = None
        for lam in LAMS:  # piece A's nested lam (no exclusion, as A was built)
            go = apply(fo, do, "pub", lam, bias_delta, np.zeros(do.height, bool)); sse = float(((go["new"] - go["y"]) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, lam)
        lam = best[1]; lams.append(lam)
        gA = apply(f.filter(C("mon") == m), dm, "pub", lam, bias_delta, np.zeros(dm.height, bool)).rename({"new": "newA"})
        gX = apply(f.filter(C("mon") == m), dm, "pub", lam, bias_delta, stop).select("id", C("new").alias("newX"))
        g = gA.join(gX, on="id", how="left").join(dm.with_columns(pl.Series("stop", stop), pl.Series("wf", wf), pl.Series("z", zz)).select("day", "stop", "wf", "z"), on="day", how="left").with_columns(pl.lit(lam).alias("lam"), pl.lit(bias_delta).alias("bias_delta"))
        parts.append(g)
    t = pl.concat(parts).with_columns(((C("pred") - C("y")) ** 2 - (C("newA") - C("y")) ** 2).alias("gA"), ((C("pred") - C("y")) ** 2 - (C("newX") - C("y")) ** 2).alias("gX"))
    bd = t.group_by("day").agg(C("gA").sum(), C("gX").sum(), C("stop").first(), C("wf").first(), C("z").first(), C("Dx").first(), C("snowday").first() if "snowday" in t.columns else pl.lit(None).alias("snowday"),
                               (C("newA") - C("pred")).filter(C("shift")).mean().alias("shiftA"), ((C("y") - C("pred")).filter(C("shift")).mean()).alias("s_true"), C("shift").sum().alias("nshift"),
                               ((C("gs") - C("newA")) < 0).filter(C("shift")).sum().alias("left_sum"), C("lam").first(), C("dl").first()).sort("day")
    totA = bd["gA"].sum() / SE_SCALE; totX = bd["gX"].sum() / SE_SCALE
    diff = (bd["gX"] - bd["gA"]).to_numpy(); lo, hi, p = day_bootstrap(diff)
    print(f"piece A reproduced: nested lam {lams}; total A {totA:+.3f} board-s; A-excl {totX:+.3f}; difference {totX - totA:+.3f} [{lo:+.3f}, {hi:+.3f}] P(diff > 0) {p:.2f}")
    st = bd.filter(C("stop")); wfd = bd.filter(C("wf") & ~C("stop"))
    print(f"stop-gate days {st.height}: " + "; ".join(f"{r['day']} D {r['Dx']:+.0f} z {r['z']:+.1f} A gain {r['gA'] / SE_SCALE:+.3f} applied shift {r['shiftA']:+.0f} s (solved delta {r['dl']:+.0f}) vs true mean error {r['s_true']:+.0f} s, rows {r['nshift']}, left the sum {r['left_sum']}" for r in st.iter_rows(named=True)))
    print(f"weather-free gate, other days {wfd.height}: " + "; ".join(f"{r['day']} D {r['Dx']:+.0f} z {r['z']:+.1f} A gain {r['gA'] / SE_SCALE:+.3f} shift {r['shiftA']:+.0f} vs true {r['s_true']:+.0f}, left {r['left_sum']}" for r in wfd.iter_rows(named=True)))
    print(f"A's gain on stop-gate days {st['gA'].sum() / SE_SCALE:+.3f}; on weather-free-gate days (all) {bd.filter(C('wf'))['gA'].sum() / SE_SCALE:+.3f}; A without the stop days {totX:+.3f}; A without all weather-free days {(bd.filter(~C('wf'))['gA'].sum()) / SE_SCALE:+.3f}")
    adopt = bool(p >= 0.9 and st.height >= 3 and totX > totA)
    print(f"ADOPT A-excl at {APT}: {adopt}")
    json.dump({"adopt": adopt, "lam": float(np.median(lams)), "stop_days": [str(d) for d in st["day"].to_list()]}, open(I + f"agent_dayeq2_excl_{APT}.json", "w"))
else:
    bias, sigL2, O_med = params(dd.filter(C("D").is_not_null()), "D")
    base = I + "agent_bundle_v152_final.parquet"
    f6 = frame_2026([APT], base); f6, d6 = prep(f6, 2026); d6 = day_solves(f6, d6)
    z = (d6["D"].to_numpy() - bias) / sig_day(d6, sigL2, O_med)
    d6 = d6.with_columns(pl.Series("z", z), (pl.Series("z", z) < -GATE_Z).alias("wf")).with_columns((C("wf") & C("snowday")).alias("stop"))
    v6 = pl.read_parquet(I + "agent_bundle_v156_final.parquet").select(C("MVT_ID_mvt").alias("id"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("v156"))
    lv = pl.read_parquet(I + "agent_dayeq_level_final.parquet").select(C("MVT_ID_mvt").alias("id"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("lv"))
    v0 = pl.read_parquet(I + "agent_bundle_v150_final.parquet").select(C("MVT_ID_mvt").alias("id"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("v150"))
    g = f6.join(v6, on="id").join(lv, on="id").join(v0, on="id").with_columns((C("lv") - C("v150")).alias("dA")).join(d6.select("day", "D", "z", "wf", "stop", "nshift"), on="day", how="inner")
    sel = g.filter(C("wf")).group_by("day").agg(C("D").first(), C("z").first(), C("stop").first(), n=pl.len(), rowsA=(C("dA") != 0).sum(), meanA=C("dA").filter(C("dA") != 0).mean(), nsnow=C("snow").sum()).sort("day")
    print(f"{APT} scored months on the pre-A base v152 (bias {bias:+.0f} min, sigma_L {np.sqrt(sigL2):.0f}): days passing the weather-free gate (stop = also a snow day):")
    print(sel.with_columns(pl.col(pl.Float64).round(1)))
    cfg = json.load(open(I + f"agent_dayeq2_excl_{APT}.json"))
    if cfg["adopt"] and g.filter(C("stop")).height:
        g.filter(C("stop")).select("id", (C("v156") - C("dA")).alias("new")).write_parquet(I + f"agent_dayeq2_rows_{APT}_excl.parquet")
        print(f"rows to restore to their pre-A value at {APT}: {g.filter(C('stop') & (C('dA') != 0)).height}")
    else:
        print(f"nothing to build at {APT} (adopt {cfg['adopt']}, scored stop days {int(g.filter(C('stop'))['day'].n_unique())})")
