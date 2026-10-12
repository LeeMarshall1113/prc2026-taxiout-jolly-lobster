"""day-equation 01: is any published DAILY pre-departure-delay series an exact function of our off-block stamps?
For every airport-day of 2025 compute candidate statistics of d_i = (take-off - label) - schedule = B_i - S_i on our rows,
compare with DLY_ALL_PRE_2 / FLT_DEP_IFR_2 (all-cause), DLY_ATC_PRE_2 / FLT_DEP_3 / DLY_ATC_PRE_3 (ATC), slot adherence counts.
Then the Rome copy-day diagnostic: does the published figure follow the STAMP (copy = delay 0) or the TRUE delay?"""
import sys
import numpy as np
import polars as pl

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(120)
C = pl.col
R = ""
E = R + "data/external/eurocontrol/"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
HELI = {"EC35", "EC45", "EC55", "A109", "A139", "AS32", "AS50", "AS55", "AS65", "B06", "B407", "B412", "B429", "EC30", "EC20", "H60", "R44", "R22", "S76", "NH90", "EC75", "A169", "A189", "B105", "BK17", "EC25", "EC35", "H145", "H135", "AW09"}

tr = pl.read_parquet(R + "data/interim/agent_eccount_train25.parquet").filter(C("y").is_not_null() & C("gs").is_not_null())
tr = tr.with_columns((C("gs") - C("y")).alias("d"),  # recorded pre-departure delay in seconds, B - S
                     C("typ").is_in(HELI).alias("heli"), (C("ades") == C("apt")).alias("local"),
                     ((C("y") - C("gs")).abs() <= 90).alias("copy"))
tr = tr.select("apt", "day", "y", "gs", "d", "hasplan", "heli", "local", "copy")
print("rows", tr.height, "d null", tr["d"].null_count())
# sanity: distribution of d per airport
print(tr.group_by("apt").agg(n=pl.len(), d_med=C("d").median(), d_p90=C("d").quantile(0.9), d_neg=(C("d") < -300).mean(), copies_gap2h=(C("copy") & (C("gs") > 7200)).sum(),
                             noplan=(~C("hasplan")).mean(), heli=C("heli").mean(), local=C("local").mean()).sort("apt"))

dm = C("d") / 60.0
def stats(df):
    return df.group_by("apt", "day").agg(
        n=pl.len(),
        s_pos=dm.clip(lower_bound=0).sum(),
        s_gt5=dm.filter(C("d") > 300).sum(), s_gt15=dm.filter(C("d") > 900).sum(), s_gt0_m5=(dm - 5).filter(C("d") > 300).sum(),
        s_all=dm.sum(), s_gt15_m15=(dm - 15).filter(C("d") > 900).sum(),
        n_gt5=(C("d") > 300).sum(), n_gt15=(C("d") > 900).sum(), n_gt0=(C("d") > 0).sum(), n_gt30=(C("d") > 1800).sum(), n_gt60=(C("d") > 3600).sum(),
        n_early=(C("d") < -300).sum(), s_pos_cap=dm.clip(lower_bound=0, upper_bound=180).sum(),
        s_pos_le2h=dm.clip(lower_bound=0).filter(C("gs") <= 7200).sum(),
        ncopy=(C("copy") & (C("gs") > 7200)).sum(), copy_gap=(dm.filter(C("copy") & (C("gs") > 7200))).sum(),  # copies: d = gs - y ~ 0 by construction
        copy_truegap=((C("gs") - 1025) / 60.0).filter(C("copy") & (C("gs") > 7200)).sum(),
    ).with_columns(C("day").cast(pl.Utf8))

variants = {"all": tr, "plan": tr.filter(C("hasplan")), "ifrlike": tr.filter(~C("heli") & ~C("local")), "plan_ifrlike": tr.filter(C("hasplan") & ~C("heli") & ~C("local"))}
ours = {k: stats(v) for k, v in variants.items()}
del variants
import gc; gc.collect()


def load(name, year=2025):
    d = pl.read_csv(f"{E}{name}_{year}.csv", infer_schema_length=0).filter(C("APT_ICAO").is_in(APTS))
    d = d.with_columns(C("FLT_DATE").str.slice(0, 10).alias("day"), C("APT_ICAO").alias("apt"))
    num = [c for c in d.columns if c.startswith(("FLT_", "DLY_")) and c != "FLT_DATE"]
    return d.with_columns([C(c).cast(pl.Float64, strict=False) for c in num]).select(["apt", "day"] + num)


pub = (load("all_pre_departure_delays").join(load("atc_pre_departure_delays"), on=["apt", "day"], how="full", coalesce=True, suffix="_atc")
       .join(load("atfm_slot_adherence"), on=["apt", "day"], how="full", coalesce=True, suffix="_slot"))
print("published columns", pub.columns)
print(pub.filter(C("apt") == "LIRF").head(5))
# which published columns are non-empty per airport
print(pub.group_by("apt").agg([C(c).is_not_null().sum().alias(c) for c in pub.columns if c not in ("apt", "day")]).sort("apt"))

PUBCOLS = ["DLY_ALL_PRE_2", "DLY_ATC_PRE_2", "FLT_DEP_3", "DLY_ATC_PRE_3", "FLT_DEP_REG_1", "FLT_DEP_OUT_LATE_1", "FLT_DEP_OUT_EARLY_1"]
OURCOLS = ["s_pos", "s_gt5", "s_gt15", "s_gt0_m5", "s_all", "s_gt15_m15", "n_gt5", "n_gt15", "n_gt0", "n_gt30", "n_gt60", "n_early", "s_pos_cap", "s_pos_le2h"]


def fit(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if ok.sum() < 20 or x.std() == 0 or y.std() == 0:
        return None
    r = np.corrcoef(x, y)[0, 1]
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    res1 = y - x  # identity residual, slope 1 intercept 0
    return dict(n=int(ok.sum()), r=r, slope=b, icpt=a, res_med=np.median(np.abs(res)), res_p90=np.quantile(np.abs(res), 0.9), res_sd=res.std(),
                id_med=np.median(np.abs(res1)), id_p90=np.quantile(np.abs(res1), 0.9), y_sd=y.std(), y_mean=y.mean())


best = {}
for a in APTS:
    print(f"\n=============== {a}")
    rows = []
    for vn, o in ours.items():
        j = o.filter(C("apt") == a).join(pub.filter(C("apt") == a), on=["apt", "day"], how="inner")
        for pc in PUBCOLS:
            y = j[pc].to_numpy().astype(float)
            if np.isfinite(y).sum() < 20 or np.nanstd(y) == 0:
                continue
            for oc in OURCOLS:
                f = fit(j[oc].to_numpy().astype(float), y)
                if f:
                    rows.append(dict(variant=vn, pub=pc, ours=oc, **f))
    if not rows:
        print("no published series"); continue
    t = pl.DataFrame(rows)
    for pc in t["pub"].unique().sort():
        s = t.filter(C("pub") == pc).sort("res_sd").head(4)
        print(f"-- {pc}: y mean {s['y_mean'][0]:.0f} sd {s['y_sd'][0]:.0f}; best definitions by residual sd:")
        print(s.select("variant", "ours", "n", "r", "slope", "icpt", "res_med", "res_p90", "res_sd", "id_med", "id_p90").with_columns(pl.col(pl.Float64).round(3)))
        best[(a, pc)] = s.row(0, named=True)

# ------------- Rome copy-day diagnostic (and all airports with copies)
print("\n\n=============== copy-day diagnostic: published minus fit on NON-copy days, evaluated on copy days")
for a in APTS:
    for pc in ["DLY_ALL_PRE_2", "DLY_ATC_PRE_2"]:
        if (a, pc) not in best:
            continue
        b = best[(a, pc)]
        o = ours[b["variant"]].filter(C("apt") == a).join(pub.filter(C("apt") == a).select("apt", "day", pc), on=["apt", "day"], how="inner").filter(C(pc).is_not_null())
        if o.filter(C("ncopy") > 0).height == 0:
            continue
        x = o[b["ours"]].to_numpy().astype(float); y = o[pc].to_numpy().astype(float); nc = o["ncopy"].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        m0 = ok & (nc == 0); m1 = ok & (nc > 0)
        bb, aa = np.polyfit(x[m0], y[m0], 1)
        res0 = y[m0] - (aa + bb * x[m0])
        # on copy days: residual under the stamp hypothesis (copies contribute 0, as in x), and under the true-delay hypothesis
        res_stamp = y[m1] - (aa + bb * x[m1])
        tg = o["copy_truegap"].to_numpy()[m1]
        res_true = y[m1] - (aa + bb * (x[m1] + tg))
        sd0 = res0.std()
        print(f"{a} {pc} [{b['variant']}/{b['ours']}, slope {bb:.3f}, icpt {aa:.0f}]: non-copy days {m0.sum()} residual sd {sd0:.0f} min; copy days {m1.sum()}, "
              f"copies {nc[m1].sum()}, mean true gap per day {tg.mean():.0f} min; residual under STAMP: mean {res_stamp.mean():+.0f} (sd {res_stamp.std():.0f}), "
              f"under TRUE: mean {res_true.mean():+.0f} (sd {res_true.std():.0f}); z-score per day: stamp {np.mean(res_stamp)/sd0*np.sqrt(m1.sum()):+.1f}, true {np.mean(res_true)/sd0*np.sqrt(m1.sum()):+.1f}")
        if a == "LIRF":
            d = o.filter(C("ncopy") > 0).select("day", "n", "ncopy", "copy_truegap", b["ours"], pc).with_columns(
                (C(pc) - (aa + bb * C(b["ours"]))).round(0).alias("res_stamp"), (C(pc) - (aa + bb * (C(b["ours"]) + C("copy_truegap")))).round(0).alias("res_true")).sort("copy_truegap", descending=True)
            print(d.head(30))
