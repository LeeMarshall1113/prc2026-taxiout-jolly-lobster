"""generalisation audit step 5: day-level calibration (rule A, nested lam/bias as in agent_nextpools_04_score) by month, incl. Apr/May/Jun/Dec as unseen months
(parameters from the 8 frame months). Base = faithful served chain for months 1,2,3,7-11; for 4,5,6,12 = pv + dose*12-month corrector (no rules/Rome/blend)."""
import sys, glob
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_float_precision(3)
APTS = ["LFPG", "EGLL", "LEBL", "LEMD", "EDDF", "EDDM", "LIRF"]
LAMS = [round(0.1 * i, 1) for i in range(11)]
files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
raw = (pl.scan_parquet(files).filter((C("PHASE_mvt") == "DEP") & C("ADEP_mvt").is_in(APTS))
       .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
               C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y")).collect())
sv = pl.read_parquet(I + "agent_waithedge_tbl.parquet", columns=["id", "served", "rule"]).with_columns(C("id").cast(pl.Float64))
g12 = pl.read_parquet(I + "agent_genaudit_12m.parquet", columns=["id", "pk"]).with_columns(C("id").cast(pl.Float64))
base = g12.join(sv, on="id", how="left").with_columns(pl.coalesce(C("served"), C("pk")).alias("pred"), C("rule").fill_null(""))
f = raw.with_columns(C("id").cast(pl.Float64)).join(base.select("id", "pred", "rule"), on="id", how="inner")
f = f.with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs"), (C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day_pred"))
f = f.filter(C("gs").is_not_null() & C("y").is_not_null())
f = f.with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"), C("rule").is_in(["R1", "R2", "R3a", "R3b"]).alias("rr"), C("day_pred").dt.month().alias("dmon"))
f = f.with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60.0).alias("dm"), (~C("amb") & ~C("rr")).alias("sh"))
print("rows", f.height, f.group_by("dmon").len().sort("dmon").to_dicts())
pub = load_pub(2025, APTS)
days = f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum()).rename({"day_pred": "day"}).join(pub, on=["apt", "day"], how="inner")
days = days.filter(((C("n_pub") - C("n")).abs() <= DN_MAX))
print("gated days", days.height)
parts = {k: g for k, g in f.partition_by("apt", "day_pred", as_dict=True).items()}
rows = []
for r in days.iter_rows(named=True):
    g = parts.get((r["apt"], r["day"]))
    if g is None:
        continue
    m = g["sh"].to_numpy(); gs = g["gs"].to_numpy()[m]; yh = g["pred"].to_numpy()[m]
    rows.append({"apt": r["apt"], "day": r["day"], "d": solve_delta(gs, yh, r["pub"] - r["fixed"]) if (r["pub"] is not None and np.isfinite(r["pub"])) else np.nan})
dd = pl.DataFrame(rows)
f = f.join(dd, left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
FM = FRAME_MONTHS
out = []
for a in APTS:
    fa = f.filter(C("apt") == a)
    yh = fa["pred"].to_numpy(); y = fa["y"].to_numpy(); d = fa["d"].to_numpy(); gs = fa["gs"].to_numpy(); mon = fa["dmon"].to_numpy()
    pos = (gs > yh) & fa["sh"].to_numpy()

    def fit(tr):
        t = tr & pos & np.isfinite(d)
        bias = float(np.median(d[tr & np.isfinite(d)]))
        best = None
        for lam in LAMS:
            sse = ((np.maximum(yh[t] + lam * (d[t] - bias), 0) - y[t]) ** 2).sum()
            if best is None or sse < best[0]:
                best = (sse, lam)
        return bias, best[1]
    new = yh.copy()
    new2 = yh.copy()
    medd = {}
    par = {}
    for m in range(1, 13):
        te = mon == m
        if not te.any():
            continue
        tr = np.isin(mon, FM) & (mon != m)
        bias, lam = fit(tr)
        par[m] = (round(bias, 1), lam)
        s = te & pos & np.isfinite(d)
        new[s] = np.maximum(yh[s] + lam * (d[s] - bias), 0)
        mb = float(np.median(d[te & np.isfinite(d)])) if (te & np.isfinite(d)).any() else bias
        medd[m] = mb
        new2[s] = np.maximum(yh[s] + lam * (d[s] - mb), 0)
    base_e = (yh - y) ** 2; new_e = (new - y) ** 2; new_e2 = (new2 - y) ** 2
    for m in range(1, 13):
        te = mon == m
        if te.any():
            out.append(dict(apt=a, mon=m, n=int(te.sum()), moved=int(((new != yh) & te).sum()), gain=float((base_e - new_e)[te].sum() / SE_SCALE), gain_ob=float((base_e - new_e2)[te].sum() / SE_SCALE), medd=round(medd.get(m, 0.0), 1), lam=par[m][1], bias=par[m][0]))
res = pl.DataFrame(out)
print(res.pivot(on="mon", index="apt", values="gain").sort("apt"))
print(res.pivot(on="mon", index="apt", values="gain_ob").sort("apt")); print(res.pivot(on="mon", index="apt", values="medd").sort("apt"))
print("TOTAL oracle-bias by month", res.group_by("mon").agg(g=C("gain_ob").sum()).sort("mon").to_dicts())
print("TOTAL by month", res.group_by("mon").agg(g=C("gain").sum(), moved=C("moved").sum()).sort("mon").to_dicts())
print(res.pivot(on="mon", index="apt", values="lam").sort("apt"))
res.write_parquet("data/interim/agent_genaudit_dayeq.parquet")
