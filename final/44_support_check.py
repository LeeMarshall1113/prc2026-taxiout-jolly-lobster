"""Stage 44: label-free support checks for the final-phase corrector inputs.

  snow     Would the Istanbul snow path and the Istanbul day-calibration snow gate fire on the snow hours of the served
           months, and what do the other airports see?  Uses the METAR files in data/raw/weather and the departure times of
           the ranking file only.  The rule being tested is the one shipped for January/July (archive/agent_ltreg_feat.py:
           a snow observation = present weather with SN, FZRA or FZDZ; the Istanbul snow features are lt_snow_3h, _12h, _n24h,
           _since, _now; archive/agent_dayeq2_04_ltfm_level.py: a day is a snow day when any Istanbul departure has
           lt_snow_3h = 1, and the stop-day gate fires on a snow day whose published delay total is more than 3 sigma
           BELOW what the file implies).  The second half of the gate needs a served file and is only reported as the
           snow-day list that the gate will look at.
  support  For the corrector serve matrix of final/40_serve_inputs.py: per feature and airport-month, the share of served
           rows outside the 1st to 99th percentile range of the 2025 fold rows of the same airport (the fold being the
           stored data/interim/agent_bigc13_fold.npz and the stored plan-clock columns), next to the same statistic
           for the January/July 2026 serve file, and the columns that shift most.

    python final/44_support_check.py snow --ranking data/raw/final_ranking.parquet --months 1,2,6,7
    python final/44_support_check.py support --serve-dir data/final/J9_serve [--ref-serve data/interim/agent_bigc13_serve_in.npz]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "archive"))
from fcommon import RAW, REPO, parse_months  # noqa: E402

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def metar(apt: str) -> pl.DataFrame:
    r = pl.read_csv(RAW / "weather" / f"{apt}.csv", null_values=["M", "", "None"], infer_schema_length=0)
    r = r.select(pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False).dt.replace_time_zone("UTC").alias("t"),
                 pl.col("wxcodes").fill_null("").alias("wx")).drop_nulls("t").unique(subset=["t"], keep="last").sort("t")
    return r.with_columns(sn=pl.col("wx").str.contains("SN") | pl.col("wx").str.contains("FZRA") | pl.col("wx").str.contains("FZDZ"))


def snow_table():
    rows = []
    for a in APTS:
        w = metar(a).with_columns(pl.col("t").dt.year().alias("y"), pl.col("t").dt.month().alias("m"), pl.col("t").dt.date().alias("d"), pl.col("t").dt.truncate("1h").alias("h"))
        for y, m in ((2025, 1), (2025, 2), (2026, 1), (2026, 2), (2025, 6), (2026, 6), (2025, 7), (2026, 7)):
            s = w.filter((pl.col("y") == y) & (pl.col("m") == m))
            sn = s.filter(pl.col("sn"))
            rows.append(dict(apt=a, year=y, month=m, obs=s.height, snow_obs=sn.height, snow_hours=sn["h"].n_unique(), snow_days=sn["d"].n_unique()))
    d = pl.DataFrame(rows)
    print("METAR snow (SN, FZRA, FZDZ) per airport-month: observations, snow observations, distinct snow hours, days with snow")
    print(d.filter(pl.col("month").is_in([1, 2])).sort("apt", "year", "month").to_pandas().to_string(index=False))
    print("June and July: snow hours", d.filter(pl.col("month").is_in([6, 7]) & (pl.col("snow_hours") > 0)).height, "airport-months")
    f25, f26 = d.filter((pl.col("year") == 2025) & (pl.col("month") == 2)), d.filter((pl.col("year") == 2026) & (pl.col("month") == 2))
    j = f25.join(f26, on="apt", suffix="_26").with_columns((pl.col("snow_hours_26") / pl.col("snow_hours").clip(lower_bound=1)).round(2).alias("ratio_hours_26_over_25"))
    print("\nFebruary snow hours 2026 / 2025:")
    print(j.select("apt", "snow_hours", "snow_hours_26", "ratio_hours_26_over_25", "snow_days", "snow_days_26").to_pandas().to_string(index=False))
    return d


def ltfm_days(ranking: Path, months: list[int]):
    """Istanbul snow flags per day (departures of the ranking file), February 2026 next to the seven February 2025 stop days."""
    from agent_ltreg_feat import snow_features
    dep = (pl.scan_parquet(ranking).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM"))
           .select("MVT_ID_mvt", "MVT_TIME_UTC_mvt").collect())
    dep = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
    sf = snow_features(dep, RAW)
    d = dep.join(sf, on="MVT_ID_mvt").with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    g = d.group_by("day").agg(n=pl.len(), share_snow3h=(pl.col("lt_snow_3h") > 0).mean(), max_n24h=pl.col("lt_snow_n24h").max(),
                              snow_rows=(pl.col("lt_snow_3h") > 0).sum()).sort("day")
    return g


def snow(a):
    months = parse_months(a.months)
    snow_table()
    print("\nIstanbul: days on which the snow path (lt_snow_3h) fires for at least one departure, served months")
    g = ltfm_days(Path(a.ranking), months)
    snowd = g.filter(pl.col("snow_rows") > 0)
    for m in months:
        s = snowd.filter(pl.col("day").dt.month() == m)
        tot = g.filter(pl.col("day").dt.month() == m)
        print(f"  month {m:02d}: {s.height} snow days of {tot.height}; snow rows {int(s['snow_rows'].sum())} of {int(tot['n'].sum())}")
    print(snowd.filter(pl.col("day").dt.month().is_in([2])).to_pandas().to_string(index=False))
    # the seven 2025 stop days, same statistics from the 2025 training departures
    tr = (pl.scan_parquet(str(RAW / "training_2025-*.parquet")).filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM"))
          .select("MVT_ID_mvt", "MVT_TIME_UTC_mvt").collect())
    from agent_ltreg_feat import snow_features
    d25 = tr.join(snow_features(tr, RAW), on="MVT_ID_mvt").with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day"))
    g25 = d25.group_by("day").agg(n=pl.len(), snow_rows=(pl.col("lt_snow_3h") > 0).sum(), max_n24h=pl.col("lt_snow_n24h").max()).sort("day")
    stop = [f"2025-02-{x:02d}" for x in (6, 7, 11, 19, 20, 22, 23)]
    print("\nFebruary 2025 days the stop-day gate excluded (log of archive/agent_dayeq2_04_ltfm_level.py), same statistics:")
    print(g25.filter(pl.col("day").cast(pl.Utf8).is_in(stop)).to_pandas().to_string(index=False))
    f25 = g25.filter(pl.col("day").dt.month() == 2)
    print(f"February 2025: {int((f25['snow_rows'] > 0).sum())} snow days of {f25.height}; 2025 snow days that the gate did NOT exclude: "
          f"{sorted(f25.filter((pl.col('snow_rows') > 0) & ~pl.col('day').cast(pl.Utf8).is_in(stop))['day'].cast(pl.Utf8).to_list())}")
    print("\nThe gate's second half (published total more than 3 sigma below the implied sum) needs the served file: not evaluated here.")


def support(a):
    """Share of served rows outside the 1st to 99th percentile of the airport's 2025 fold rows, per feature, airport and month.
    The fold matrix is read one block at a time (base, extra, variant differences, plan-clock columns, views) to stay small."""
    I = REPO / "data" / "interim"
    nine = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
    fold = np.load(I / "agent_bigc13_fold.npz", allow_pickle=True)
    apt_f = fold["apt"].astype(str)
    cols = [str(c) for c in fold["cols"]]
    ex = np.load(I / "agent_bigc12_extra.npz", allow_pickle=True)
    pcd = np.load(I / "agent_pcd_cols_app.npz", allow_pickle=True)
    vf = np.load(I / "agent_bigc12_vfac_clean5.npz", allow_pickle=True)
    sets = {}
    for tag, sd, pcdp, vfp in (("new", Path(a.serve_dir), Path(a.serve_dir) / "agent_pcd_cols_app.npz", Path(a.serve_dir) / "serve_views.npz"),
                               ("ref", None, I / "agent_pcd_cols_app.npz", I / "agent_bigc12_vfac_clean5.npz")):
        z = np.load(Path(a.ref_serve) if sd is None else sd / "agent_bigc13_serve_in.npz", allow_pickle=True)
        sets[tag] = dict(Xs=z["Xs_pre"], apt=z["apt"].astype(str), mon=z["mon_s"], Ds=np.load(pcdp, allow_pickle=True)["Ds"], V=np.load(vfp, allow_pickle=True)["Xs"])
    ncore = len(cols)
    blocks = [("base", [c for c in cols], lambda: fold["X"], lambda d: d["Xs"][:, :ncore]),
              ("extra", [str(c) for c in ex["cols"]], lambda: ex["Xf"], lambda d: d["Xs"][:, ncore:ncore + 52]),
              ("variant", ["d_NEW", "d_ET"], lambda: np.load(I / "agent_bigc12_extra2.npz", allow_pickle=True)["Xf"], lambda d: d["Xs"][:, ncore + 52:ncore + 54]),
              ("pcd", [str(c) for c in pcd["names"]], lambda: pcd["D"], lambda d: d["Ds"]),
              ("views", [str(c) for c in vf["cols"]], lambda: vf["Xf"], lambda d: d["V"])]
    rows = []
    for bname, bcols, getf, gets in blocks:
        Xf = getf()
        lo, hi = {}, {}
        for ap in nine:
            m = apt_f == ap
            q = np.nanpercentile(Xf[m].astype(np.float64), [1, 99], axis=0)
            lo[ap], hi[ap] = q[0], q[1]
        del Xf
        for tag, d in sets.items():
            Xs = gets(d)
            for ap in nine:
                for mo in sorted(set(d["mon"].tolist())):
                    s_ = (d["apt"] == ap) & (d["mon"] == mo)
                    if s_.sum() < 200:
                        continue
                    v = Xs[s_].astype(np.float64)
                    fin = np.isfinite(v)
                    outside = ((v < lo[ap]) | (v > hi[ap])) & fin
                    for k, c in enumerate(bcols):
                        rows.append((tag, ap, int(mo), bname, c, int(s_.sum()), float(outside[:, k].sum() / max(fin[:, k].sum(), 1)), float((~fin[:, k]).mean())))
        print("block", bname, "done", flush=True)
    d = pl.DataFrame(rows, schema=["set", "apt", "month", "block", "col", "rows", "out_share", "nan_share"], orient="row")
    outp = Path(a.outdir) / "support_check.csv"
    d.write_csv(outp)
    print("written", outp, "(share outside the 1st to 99th percentile of the airport's 2025 fold rows; about 0.02 is expected for a column that did not move)")
    summ = (d.group_by("set", "apt", "month").agg(pl.col("out_share").mean().round(4).alias("mean_out"), (pl.col("out_share") > 0.10).sum().alias("cols_over_10pct"),
                                                  (pl.col("nan_share") > 0.5).sum().alias("cols_mostly_nan")).sort("set", "apt", "month"))
    print(summ.to_pandas().to_string(index=False))
    print("\nlargest out-of-range shares in the new set:")
    print(d.filter(pl.col("set") == "new").sort("out_share", descending=True).head(40).to_pandas().to_string(index=False))
    # epoch-size columns: any served value above 1e8 in absolute size
    big = []
    for bname, bcols, _, gets in blocks:
        X = gets(sets["new"])
        for k, c in enumerate(bcols):
            col = X[:, k]
            if np.isfinite(col).any() and float(np.nanmax(np.abs(col))) > 1e8:
                big.append((c, float(np.nanmax(np.abs(col)))))
    print("\ncolumns with served values above 1e8 (epoch-timestamp check):", big or "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snow", "support"])
    ap.add_argument("--ranking", default=str(RAW / "final_ranking.parquet"))
    ap.add_argument("--months", default="1,2,6,7")
    ap.add_argument("--serve-dir", default=str(REPO / "data/final/J9_serve"))
    ap.add_argument("--ref-serve", default=str(REPO / "data/interim/agent_bigc13_serve_in.npz"))
    ap.add_argument("--outdir", default=str(REPO / "data/final"))
    a = ap.parse_args()
    {"snow": snow, "support": support}[a.cmd](a)


if __name__ == "__main__":
    main()
