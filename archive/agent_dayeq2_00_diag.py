"""day-equation 2, step 0: diagnostics on 2025 (held-out chain values) before any design is frozen.
Per airport-day: D = published - sum max(gs - pred, 0); true residual R = sum [max(gs - y, 0) - max(gs - pred, 0)] split by row group
(long-gap ambiguous rows, hold candidates gs > 3600, ordinary) and plan state; Istanbul by month with February; the spread of the day level;
hold-row anatomy (stamp - schedule); Barcelona schedule copies."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(100)
APTS = ["LIRF", "LTFM", "EGLL", "EDDM", "EDDF", "LEBL", "LEMD", "EHAM", "LFPG", "LSZH"]
f = frame_2025(APTS)
pub = load_pub(2025, APTS)
print("kinds in the typed table:", f.group_by("kind").len().sort("len", descending=True).head(20))
f = f.with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"), (C("gs") > 3600).alias("cand"),
                   ((C("gs") - C("pred")) / 60).clip(lower_bound=0).alias("dm_pred"), ((C("gs") - C("y")) / 60).clip(lower_bound=0).alias("dm_true"),
                   (C("T") - pl.duration(seconds=C("y"))).dt.date().alias("day_true"))
f = f.with_columns((C("dm_true") - C("dm_pred")).alias("r"),
                   pl.when(C("amb")).then(pl.lit("amb")).when(C("cand")).then(pl.lit("cand")).otherwise(pl.lit("ord")).alias("grp"))
days = (f.group_by("apt", "day_pred").agg(n=pl.len(), O=C("dm_pred").sum(), R=C("r").sum(), R_amb=C("r").filter(C("grp") == "amb").sum(), R_cand=C("r").filter(C("grp") == "cand").sum(),
                                           R_ord=C("r").filter(C("grp") == "ord").sum(), R_noplan=C("r").filter(~C("hasplan")).sum(), k_amb=C("amb").sum(), k_cand=(C("cand") & ~C("amb")).sum())
        .rename({"day_pred": "day"}).join(pub, on=["apt", "day"], how="inner").with_columns((C("pub") - C("O")).alias("D"), (C("n_pub") - C("n")).alias("dn")))
days = days.filter(C("dn").abs() <= DN_MAX).with_columns(C("day").dt.month().alias("mon"))
print("\nper airport: D and the true residual R split by group (minutes/day; sd across days)")
print(days.group_by("apt").agg(days=pl.len(), D_sd=C("D").std(), R_sd=C("R").std(), R_amb_sd=C("R_amb").std(), R_cand_sd=C("R_cand").std(), R_ord_sd=C("R_ord").std(),
                               ident=(C("D") - C("R")).abs().median(), share_amb=(C("R_amb") ** 2).sum() / (C("R") ** 2).sum(), share_cand=(C("R_cand") ** 2).sum() / (C("R") ** 2).sum(),
                               share_ord=(C("R_ord") ** 2).sum() / (C("R") ** 2).sum()).sort("apt").with_columns(pl.col(pl.Float64).round(2)))
print("\nLTFM by month:")
print(days.filter(C("apt") == "LTFM").group_by("mon").agg(days=pl.len(), D_med=C("D").median(), D_sd=C("D").std(), R_amb=C("R_amb").abs().mean(), R_cand=C("R_cand").abs().mean(), R_ord=C("R_ord").abs().mean(),
                                                           share_amb=(C("R_amb") ** 2).sum() / (C("R") ** 2).sum(), share_ord=(C("R_ord") ** 2).sum() / (C("R") ** 2).sum()).sort("mon").with_columns(pl.col(pl.Float64).round(1)))
print("\nLTFM worst 15 days by |D|:")
print(days.filter(C("apt") == "LTFM").sort(C("D").abs(), descending=True).head(15).select("day", "n", "pub", "O", "D", "R", "R_amb", "R_cand", "R_ord", "R_noplan", "k_amb", "k_cand").with_columns(pl.col(pl.Float64).round(0)))
for a in ["EGLL", "EDDF", "EDDM", "LEBL", "LEMD", "LIRF"]:
    print(f"\n{a} worst 8 days by |D|:")
    print(days.filter(C("apt") == a).sort(C("D").abs(), descending=True).head(8).select("day", "n", "pub", "O", "D", "R", "R_amb", "R_cand", "R_ord", "R_noplan", "k_amb", "k_cand").with_columns(pl.col(pl.Float64).round(0)))
# level spread: delta solved on ordinary+cand rows (not amb) with exact target
rows = []
for (a, d), g in f.partition_by("apt", "day_pred", as_dict=True).items():
    r = days.filter((C("apt") == a) & (C("day") == d))
    if r.height == 0:
        continue
    m = ~g["amb"].to_numpy(); amb_c = g.filter(C("amb"))["dm_pred"].sum()
    delta = solve_delta(g["gs"].to_numpy()[m], g["pred"].to_numpy()[m], float(r["pub"][0]) - amb_c)
    rows.append(dict(apt=a, day=d, delta=delta, n_pos=int(((g["gs"] > g["pred"]) & ~g["amb"]).sum())))
dd = pl.DataFrame(rows)
print("\nday level delta (s) spread per airport (lam = 1): median, robust sd, sd; n_pos mean")
print(dd.group_by("apt").agg(days=pl.len(), med=C("delta").median(), rsd=(C("delta") - C("delta").median()).abs().median() * 1.4826, sd=C("delta").std(), n_pos=C("n_pos").mean()).sort("apt").with_columns(pl.col(pl.Float64).round(1)))
dd.write_parquet(I + "agent_dayeq2_diag_delta_2025.parquet")
# hold anatomy: rows with y > 3600 and gs > 3600 at the exact airports: stamp - schedule = gs - y
h = f.filter(C("apt").is_in(["EGLL", "EDDM", "EDDF", "LEBL", "LEMD", "LTFM"]) & (C("y") > 3600) & (C("gs") > 3600)).with_columns(((C("gs") - C("y")) / 60).alias("stamp_minus_sched_min"))
print("\nhold anatomy (y > 3600, gs > 3600): stamp - schedule in minutes, by airport and plan")
print(h.group_by("apt", "hasplan").agg(n=pl.len(), q10=C("stamp_minus_sched_min").quantile(0.1), med=C("stamp_minus_sched_min").median(), q90=C("stamp_minus_sched_min").quantile(0.9), pred_med=C("pred").median(), y_med=C("y").median(), gs_med=C("gs").median()).sort("apt", "hasplan").with_columns(pl.col(pl.Float64).round(1)))
print("candidate counts per day (gs > 3600, not amb) and hold rate among them, by airport:")
cc = f.filter(C("cand") & ~C("amb")).group_by("apt").agg(n=pl.len(), hold=(C("y") > 3600).mean(), hold_n=(C("y") > 3600).sum(), days=C("day_pred").n_unique())
print(cc.with_columns((C("n") / C("days")).alias("per_day")).sort("apt"))
# Barcelona copies: plan rows, gs >= 1800, |y - gs| <= 90
b = f.filter((C("apt") == "LEBL") & C("hasplan") & (C("gs") >= 1800)).with_columns(((C("y") - C("gs")).abs() <= 90).alias("copy"))
print("\nLEBL plan rows gs >= 1800:", b.height, "copies", int(b["copy"].sum()), "copy_ok share", b["copy_ok"].mean(), "copies with copy_ok", int(b.filter(C("copy"))["copy_ok"].sum()))
print(b.group_by("mon").agg(n=pl.len(), copies=C("copy").sum(), rate=C("copy").mean(), p_served=((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0, 1).mean()).sort("mon").with_columns(pl.col(pl.Float64).round(4)))
print("LEBL copies per day distribution:", b.filter(C("copy")).group_by("day_pred").len()["len"].describe())
print("LEBL ident residual on days with >= 3 copies vs others:")
bc = b.group_by("day_pred").agg(copies=C("copy").sum(), copy_mass=(C("copy").cast(pl.Float64) * (C("gs") - C("taxi")) / 60).sum()).rename({"day_pred": "day"})
dl = days.filter(C("apt") == "LEBL").join(bc, on="day", how="left").with_columns(C("copies").fill_null(0), C("copy_mass").fill_null(0.0))
print(dl.with_columns((C("copies") >= 3).alias("many")).group_by("many").agg(days=pl.len(), D_mean=C("D").mean(), D_sd=C("D").std(), copy_mass=C("copy_mass").mean(), R_mean=C("R").mean()).with_columns(pl.col(pl.Float64).round(1)))
print("corr(D, -copy_mass) at LEBL:", np.corrcoef(dl["D"].to_numpy(), -dl["copy_mass"].to_numpy())[0, 1])
