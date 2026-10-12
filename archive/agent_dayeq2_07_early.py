"""day-equation 2, section 12, step 1: can the monthly taxi-out figure minus the daily-equation total of delayed flights recover the chain's
mean error on NOT-delayed flights? 2025 frame months, labels known. Usage: python agent_dayeq2_07_early.py"""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(200)
APTS = ["EGLL", "LEBL", "LEMD", "EHAM", "LIRF", "EDDF", "EDDM"]
BAND = 300.0
mo = pl.read_csv(E + "taxi_out_additional_time_2025.csv", infer_schema_length=0).filter(C("APT_ICAO").is_in(APTS)).select(
    C("APT_ICAO").alias("apt"), C("MONTH_NUM").cast(pl.Int64).alias("mon"), C("TOTAL_REF_NB_FL").cast(pl.Float64).alias("n_ref"),
    ((C("TOTAL_REF_TIME_MIN").cast(pl.Float64) + C("TOTAL_ADD_TIME_MIN").cast(pl.Float64)) * 60 / C("TOTAL_REF_NB_FL").cast(pl.Float64)).alias("mu_pub"))
pub = load_pub(2025, APTS)
f = frame_2025(APTS)
f = f.with_columns(pl.when(C("apt") == "EHAM").then(C("rule") == "I").otherwise(True).alias("keep")).filter(C("keep") & (C("y") <= 7200))
f = f.with_columns((C("gs") - C("pred")).alias("margin"), (C("gs") - C("y") > 0).alias("del_true"), (C("y") - C("pred")).alias("err"),
                   ((C("gs") - C("pred")) / 60).clip(lower_bound=0).alias("dm_pred"))
f = f.with_columns((C("margin") > 0).alias("del_file"), (C("margin").abs() <= BAND).alias("band"))
# monthly published daily delay sum (minutes) by the off-block date of the served value's month
pubm = pub.with_columns(C("day").dt.month().alias("mon")).group_by("apt", "mon").agg(C("pub").sum().alias("delay_pub_min"))
rows = []
for (a, m), g in f.partition_by("apt", "mon", as_dict=True).items():
    r = mo.filter((C("apt") == a) & (C("mon") == m)); pm = pubm.filter((C("apt") == a) & (C("mon") == m))
    if r.height == 0 or pm.height == 0:
        continue
    n = g.height; mu_lab = float(g["y"].mean()); mu_pub = float(r["mu_pub"][0])
    nd_t = g.filter(~C("del_true")); nd_f = g.filter(~C("del_file")); nd_fb = g.filter(~C("del_file") & ~C("band"))
    b_t = float(nd_t["err"].mean()); b_f = float(nd_f["err"].mean()); b_fb = float(nd_fb["err"].mean())
    # recovery: not-delayed total = month total - delayed total; delayed total (file-classified) = sum gs - 60 x published daily delay
    gs_del = float(g.filter(C("del_file"))["gs"].sum()); delay_pub = 60 * float(pm["delay_pub_min"][0])
    rows.append(dict(apt=a, mon=m, n=n, mu_pub=mu_pub, mu_lab=mu_lab, n_nd_file=nd_f.height, b_true=b_t, b_file=b_f, b_file_noband=b_fb, prize_file=nd_f.height * b_f ** 2 / SE_SCALE,
                     truth_nd=float(nd_f["y"].mean()), gs_del=gs_del, delay_pub=delay_pub, del_true_in_file_nd=float(nd_f["del_true"].mean()), band_n=int(g["band"].sum()),
                     delay_pub_vs_true=float(delay_pub - 60 * g.filter(C("del_true")).with_columns(((C("gs") - C("y")) / 60).alias("d"))["d"].sum())))
t = pl.DataFrame(rows).sort("apt", "mon")
t = t.with_columns((C("mu_pub") - C("mu_lab")).alias("off_m"))
off = t.group_by("apt").agg(C("off_m").mean().alias("offset"), C("off_m").std().alias("offset_sd"))
t = t.join(off, on="apt", how="left").with_columns((((C("mu_pub") - C("offset")) * C("n") - (C("gs_del") - C("delay_pub"))) / C("n_nd_file")).alias("rec_nd"))
t = t.with_columns((C("rec_nd") - C("truth_nd")).alias("rec_err"), (C("rec_nd") - (C("truth_nd") - C("b_file"))).alias("rec_level"))
print("per airport-month: b = chain mean error on not-delayed rows (true stamp / file margin / file margin outside the 300 s band), prize n b^2, published-vs-label monthly offset, recovered level error (s per flight)")
print(t.select("apt", "mon", "n", "n_nd_file", "band_n", "b_true", "b_file", "b_file_noband", "prize_file", "off_m", "offset", "rec_err", "rec_level", "del_true_in_file_nd", "delay_pub_vs_true").with_columns(pl.col(pl.Float64).round(2)))
print("\nper airport: rms b (file), prize sum (board-s, 8 months), rms recovered-level error, offset sd; verdict step 1 = rms error < 0.5 rms b")
s = t.group_by("apt").agg(rms_b=(C("b_file") ** 2).mean().sqrt(), prize=C("prize_file").sum(), rms_rec_err=(C("rec_err") ** 2).mean().sqrt(), mean_rec_err=C("rec_err").mean(), offset_sd=C("offset_sd").first(),
                          band_share=(C("band_n") / C("n")).mean(), del_in_nd=C("del_true_in_file_nd").mean()).sort("apt")
print(s.with_columns(pl.col(pl.Float64).round(3), (C("rms_rec_err") < 0.5 * C("rms_b")).alias("step1_pass")))
t.write_parquet(I + "agent_dayeq2_early_step1.parquet")
