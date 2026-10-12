"""can the 2025 frame reproduce production's bias/lam (agent_dayeq_level_cfg.json) for the six piece-A airports? frames: faithful served (as rule C) and eb136 pred (as dayeq_04)."""
import sys, json
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa
APTS = ["EDDF", "EDDM", "EGLL", "LEBL", "LEMD", "LIRF"]
ROWRULES = ["R1", "R2", "R3a", "R3b"]
LAMS = [round(0.1 * i, 1) for i in range(11)]
cfg = json.load(open(I + "agent_dayeq_level_cfg.json"))
sv = pl.read_parquet(I + "agent_waithedge_tbl.parquet", columns=["id", "served"]).with_columns(C("id").cast(pl.Float64))
pub = load_pub(2025, APTS)
for name in ["eb136", "served"]:
    f = frame_2025(APTS).with_columns(C("id").cast(pl.Float64))
    if name == "served":
        f = f.join(sv, on="id", how="inner").with_columns(C("served").alias("pred"))
    f = f.filter(C("y").is_not_null() & C("gs").is_not_null()).with_columns(C("day_pred").dt.month().alias("dmon"))
    f = f.filter(C("dmon").is_in(FRAME_MONTHS)).with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"))
    f = f.with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60.0).alias("dm"), (~C("amb")).alias("sh"))
    days = f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum()).rename({"day_pred": "day"}).join(pub, on=["apt", "day"], how="inner").filter((C("n_pub") - C("n")).abs() <= DN_MAX)
    parts = f.partition_by("apt", "day_pred", as_dict=True); rows = []
    for r in days.iter_rows(named=True):
        g = parts[(r["apt"], r["day"])]; m = g["sh"].to_numpy()
        rows.append({"apt": r["apt"], "day": r["day"], "d": solve_delta(g["gs"].to_numpy()[m], g["pred"].to_numpy()[m], r["pub"] - r["fixed"]) if np.isfinite(r["pub"]) else np.nan})
    f = f.join(pl.DataFrame(rows), left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
    for a in APTS:
        fa = f.filter(C("apt") == a); d = fa["d"].to_numpy(); yh = fa["pred"].to_numpy(); y = fa["y"].to_numpy(); pos = (fa["gs"].to_numpy() > yh) & np.isfinite(d) & fa["sh"].to_numpy()
        bias = float(np.median(d[np.isfinite(d)])); best = min((((np.maximum(yh[pos] + l * (d[pos] - bias), 0) - y[pos]) ** 2).sum(), l) for l in LAMS)
        print(name, a, "bias", round(bias, 3), "lam", best[1], "| production bias", round(cfg[a]["bias"], 3), "lam", cfg[a]["lam"])
