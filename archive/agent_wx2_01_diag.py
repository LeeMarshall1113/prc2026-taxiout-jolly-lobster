"""wx2 diag: univariate signal of network-disruption (A) and runway-config anomaly (C) features against fold residuals.
Run: ./.venv/Scripts/python.exe archive/capped_run.py --cap-gb 2 -- ./.venv/Scripts/python.exe archive/agent_wx2_01_diag.py
"""
import glob, json, sys
import numpy as np, polars as pl
from scipy.stats import spearmanr
sys.path.insert(0, ".")
from prc import weather

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt"]
def _load(f):
    return (pl.scan_parquet(f).select("MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
            pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t"),
            pl.col("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("ts"))
            .filter(pl.col("ADEP_mvt").is_in(APTS) | pl.col("ADES_mvt").is_in(APTS)).collect())


raw = pl.concat([_load(f) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
dep_all = raw.filter(pl.col("PHASE_mvt") == "DEP").filter(pl.col("ADEP_mvt").is_in(APTS))
arr = raw.filter(pl.col("PHASE_mvt") == "ARR").filter(pl.col("ADES_mvt").is_in(APTS)).select("ADES_mvt", "ADEP_mvt", "t", "ts")

del raw; import gc; gc.collect()
z = np.load("data/interim/agent_bulkr_fold_SA_s0.npz", allow_pickle=True)
fold = pl.DataFrame({"MVT_ID_mvt": z["ids"].astype(float), "y": z["y"].astype(float), "p": z["both_guard"].astype(float), "day": pl.Series(z["days"]).str.slice(5, 2)})
dep_all = dep_all.select("MVT_ID_mvt", "ADEP_mvt", "t", "RUNWAY_mvt")
dep = dep_all.join(fold, on="MVT_ID_mvt", how="inner").sort(["ADEP_mvt", "t"])
del z, fold
import gc; gc.collect()
print("dep rows joined", dep.height, flush=True)

w = weather.load().with_columns(pl.col("wx_time").dt.epoch("s").alias("wt")).sort(["wx_station", "wt"])
W = {}
for s in APTS:
    g = w.filter(pl.col("wx_station") == s)
    W[s] = dict(t=g["wt"].to_numpy(),
                snow=(g["wx_snow"].fill_null(0).to_numpy().astype(int) | g["wx_freezing"].fill_null(0).to_numpy().astype(int)),
                lv=g["wx_lowvis"].fill_null(0).to_numpy().astype(int), ts=g["wx_thunder"].fill_null(0).to_numpy().astype(int),
                wind=g["wx_wind_kt"].fill_null(0).to_numpy())


def flag_at(s, tt, key, maxage=5400):
    d = W[s]
    i = np.searchsorted(d["t"], tt, side="right") - 1
    ok = (i >= 0) & (tt - d["t"][np.clip(i, 0, None)] <= maxage)
    return np.where(ok, d[key][np.clip(i, 0, None)], 0)


# config keys from ALL departures (30-min bins)
keys = {}
for a in APTS:
    g = dep_all.filter(pl.col("ADEP_mvt") == a).with_columns((pl.col("t") // 1800).alias("b"), pl.col("RUNWAY_mvt").fill_null("NA"))
    c = g.group_by("b", "RUNWAY_mvt").len()
    tot = c.group_by("b").agg(pl.col("len").sum().alias("n"))
    c = c.join(tot, on="b").filter((pl.col("n") >= 3) & (pl.col("len") / pl.col("n") >= 0.15))
    k = c.group_by("b").agg(pl.col("RUNWAY_mvt").sort().str.join("+").alias("key"))
    keys[a] = dict(zip(k["b"].to_list(), k["key"].to_list()))

out = []
for a in APTS:
    D = dep.filter(pl.col("ADEP_mvt") == a)
    T = D["t"].to_numpy()
    F = {"id": D["MVT_ID_mvt"].to_numpy(), "apt": np.full(len(T), a), "y": D["y"].to_numpy(), "p": D["p"].to_numpy(),
         "day": D["day"].to_numpy(), "T": T}
    for nm, key in (("net_snow", "snow"), ("net_lvp", "lv"), ("net_ts", "ts")):
        F[nm] = sum(flag_at(o, T, key) for o in APTS if o != a).astype(float)
    A = arr.filter(pl.col("ADES_mvt") == a).sort("t")
    at = A["t"].to_numpy()
    late = np.clip((A["t"] - A["ts"]).to_numpy().astype(float), -3600, 10800)
    cs_l = np.concatenate([[0], np.cumsum(late)])
    cs_s = np.concatenate([[0], np.cumsum((late > 900).astype(float))])
    orig = A["ADEP_mvt"].to_numpy()
    fo = np.zeros(len(at))
    for o in APTS:
        m = orig == o
        if m.any():
            fo[m] = np.maximum.reduce([flag_at(o, at[m] - 7200, "snow"), flag_at(o, at[m] - 7200, "lv"), flag_at(o, at[m] - 7200, "ts")])
    cs_f = np.concatenate([[0], np.cumsum(fo)])
    for wn, wl in (("60", 3600), ("180", 10800)):
        hi = np.searchsorted(at, T, side="right")
        lo = np.searchsorted(at, T - wl, side="right")
        n = hi - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            F["arr_late_share" + wn] = np.where(n >= 3, (cs_s[hi] - cs_s[lo]) / np.maximum(n, 1), np.nan)
            F["arr_late_mean" + wn] = np.where(n >= 3, (cs_l[hi] - cs_l[lo]) / np.maximum(n, 1), np.nan)
        if wn == "180":
            F["arr_from_flagged180"] = cs_f[hi] - cs_f[lo]
    F["wind"] = flag_at(a, T, "wind").astype(float)
    kd = keys[a]
    b = (T // 1800).astype(np.int64)
    key = np.array([kd.get(int(x), "") for x in b])
    prevk = np.array([kd.get(int(x) - 1, "") for x in b])
    mon = F["day"].astype(str)
    mi = mon.astype(int)
    bins_all = np.array(sorted(kd))
    bm = (bins_all * 1800).astype("datetime64[s]").astype("datetime64[M]").astype(int) % 12 + 1
    bk = np.array([kd[x] for x in bins_all])
    cnt, tot_k, tot_m = {}, {}, {}
    for k_, m_ in zip(bk, bm):
        cnt[(k_, int(m_))] = cnt.get((k_, int(m_)), 0) + 1
        tot_k[k_] = tot_k.get(k_, 0) + 1
        tot_m[int(m_)] = tot_m.get(int(m_), 0) + 1
    kf = np.array([(tot_k.get(k_, 0) - cnt.get((k_, m_), 0)) / max(1, len(bk) - tot_m.get(m_, 0)) for k_, m_ in zip(key, mi)])
    F["cfg_rare"] = np.where(key != "", -np.log(np.clip(kf, 1e-4, 1)), np.nan)
    F["cfg_chg"] = np.where((key != "") & (prevk != ""), (key != prevk).astype(float), np.nan)
    cb = np.array(sorted(x for x in kd if (x - 1) in kd and kd[x] != kd[x - 1]))
    j = np.searchsorted(cb, b, side="right") - 1
    F["cfg_since"] = np.where(j >= 0, np.minimum(b - cb[np.clip(j, 0, None)], 6), 6).astype(float)
    F["cfg_rare_x_wind"] = F["cfg_rare"] * F["wind"]
    F["n_act"] = np.array([k_.count("+") + 1 if k_ else 0 for k_ in key])
    F["mon"] = mon
    F["hour"] = ((T % 86400) // 3600).astype(int)
    out.append(pl.DataFrame(F))
    print(a, len(T), flush=True)
df = pl.concat(out).with_columns((pl.col("y") - pl.col("p")).alias("r"))
df.write_parquet("data/interim/agent_wx2_diag.parquet")
print("saved", df.height, flush=True)

FEATS = ["net_snow", "net_lvp", "net_ts", "arr_late_share60", "arr_late_share180", "arr_late_mean60", "arr_late_mean180",
         "arr_from_flagged180", "cfg_rare", "cfg_chg", "cfg_since", "cfg_rare_x_wind"]


def rho(x, r):
    m = np.isfinite(x) & np.isfinite(r)
    if m.sum() < 500 or np.ptp(x[m]) == 0:
        return np.nan
    return spearmanr(x[m], r[m])[0]


def demean(x, cells):
    s = pl.DataFrame({"x": x, "c": cells}).with_columns((pl.col("x") - pl.col("x").mean().over("c")).alias("d"))
    return s["d"].to_numpy()


r = df["r"].to_numpy(); apt = df["apt"].to_numpy().astype(str); mon = df["mon"].to_numpy().astype(str)
cell = np.char.add(np.char.add(np.char.add(apt, mon), df["hour"].to_numpy().astype(str)), df["n_act"].to_numpy().astype(str))
lim = np.sort(np.abs(r))[-50]
rr = demean(r, apt)
rd = demean(r, cell)
res = {}
for f in FEATS:
    x = df[f].to_numpy().astype(float)
    xf = np.where(np.isfinite(x), x, np.nanmean(x))
    xa = np.where(np.isfinite(x), demean(xf, apt), np.nan)
    p = rho(xa, rr)
    trim = rho(xa, np.where(np.abs(rr) > lim, np.nan, rr))
    xd = np.where(np.isfinite(x), demean(xf, cell), np.nan)
    pd_ = rho(xd, rd)
    per_apt = {a: rho(x[apt == a], r[apt == a]) for a in APTS}
    pc = np.array([rho(x[(apt == a) & (mon == m)], r[(apt == a) & (mon == m)]) for a in APTS for m in sorted(set(mon))])
    mm = [rho(xa[mon == m], rr[mon == m]) for m in sorted(set(mon))]
    sgn = int(np.sum(np.sign(mm) == np.sign(p))) if np.isfinite(p) else 0
    res[f] = dict(pooled=p, trim50=trim, cell_demeaned=pd_, months_same_sign=sgn, month_rhos=[round(float(v), 4) for v in mm],
                  per_apt={a: (None if not np.isfinite(v) else round(float(v), 3)) for a, v in per_apt.items()},
                  cell_mean=float(np.nanmean(pc)))
    print(f"{f:20s} pooled {p:+.4f} trim50 {trim:+.4f} cellDM {pd_:+.4f} months-same-sign {sgn}/12 cell-mean {np.nanmean(pc):+.4f}", flush=True)
    print("   apt:", res[f]["per_apt"], flush=True)
json.dump(res, open("logs/agent_wx2_01_diag.json", "w"), indent=1, default=float)
