"""wx2 03: wind-component features vs runway heading (angle C3) on the SA_s0 residual. Gate + LOMO bound."""
import glob, re, json
import numpy as np, polars as pl
from scipy.stats import spearmanr

APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
df = pl.read_parquet("data/interim/agent_wx2_diag.parquet")
rw = pl.concat([pl.scan_parquet(f).filter(pl.col("PHASE_mvt") == "DEP").select("MVT_ID_mvt", "RUNWAY_mvt").collect()
                for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
df = df.join(rw, left_on="id", right_on="MVT_ID_mvt", how="left")
df = df.with_columns(pl.col("RUNWAY_mvt").fill_null("").str.extract(r"^(\d+)", 1).cast(pl.Float64, strict=False).mul(10).alias("hdg"))
out = []
for a in APTS:
    D = df.filter(pl.col("apt") == a).sort("T")
    w = pl.read_csv(f"data/raw/weather_wind/{a}.csv", null_values=["M", ""], infer_schema_length=0).with_columns(
        pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M").dt.replace_time_zone("UTC").dt.epoch("s").alias("wt"),
        pl.col("drct").cast(pl.Float64), pl.col("sknt").cast(pl.Float64)).sort("wt")
    wt = w["wt"].to_numpy(); T = D["T"].to_numpy()
    i = np.searchsorted(wt, T, side="right") - 1
    ok = (i >= 0) & (T - wt[np.clip(i, 0, None)] <= 5400)
    i = np.clip(i, 0, None)
    drct = np.where(ok, w["drct"].to_numpy()[i], np.nan); spd = np.where(ok, w["sknt"].to_numpy()[i], np.nan)
    hdg = D["hdg"].to_numpy()
    ang = np.deg2rad(drct - hdg)
    head = spd * np.cos(ang); cross = np.abs(spd * np.sin(ang))
    tail = np.clip(-head, 0, None)
    # preferred flow = modal runway heading of the airport (by departures)
    hv = hdg[np.isfinite(hdg)]
    vals, cnt = np.unique(hv, return_counts=True)
    pref = vals[np.argmax(cnt)]
    pref_head = spd * np.cos(np.deg2rad(drct - pref))
    # opposite-direction modal: headings >90 deg from pref count as non-preferred flow
    nonpref_rwy = np.abs(((hdg - pref + 180) % 360) - 180) > 90
    wind_nonpref = pref_head < 0
    cr = D["cfg_rare"].to_numpy()
    F = dict(head=head, cross=cross, tail=tail, tail5=(tail >= 5).astype(float) + 0 * tail, tail_flag_nan=None)
    F.pop("tail_flag_nan")
    F["tail5"] = np.where(np.isfinite(tail), (tail >= 5).astype(float), np.nan)
    F["wind_nonpref"] = np.where(np.isfinite(pref_head) & (spd >= 5), wind_nonpref.astype(float), np.nan)
    F["rwy_nonpref"] = np.where(np.isfinite(hdg), nonpref_rwy.astype(float), np.nan)
    F["wind_vs_rwy_mismatch"] = np.where(np.isfinite(F["wind_nonpref"]) & np.isfinite(F["rwy_nonpref"]), (F["wind_nonpref"] != F["rwy_nonpref"]).astype(float), np.nan)
    F["rare_x_tail"] = cr * tail; F["rare_x_cross"] = cr * cross; F["rare_x_nonpref"] = cr * F["wind_nonpref"]
    F["tail_x_spd"] = tail * spd
    out.append(D.select("apt", "mon", "hour", "n_act", "r").with_columns([pl.Series(k, v) for k, v in F.items()]))
d = pl.concat(out)
FE = ["head", "cross", "tail", "tail5", "wind_nonpref", "rwy_nonpref", "wind_vs_rwy_mismatch", "rare_x_tail", "rare_x_cross", "rare_x_nonpref", "tail_x_spd"]
apt = d["apt"].to_numpy().astype(str); mon = d["mon"].to_numpy().astype(str); r = d["r"].to_numpy()
cell = np.char.add(np.char.add(np.char.add(apt, mon), d["hour"].to_numpy().astype(str)), d["n_act"].to_numpy().astype(str))


def demean(x, c):
    return pl.DataFrame({"x": x, "c": c}).with_columns((pl.col("x") - pl.col("x").mean().over("c")).alias("d"))["d"].to_numpy()


def rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return np.nan if m.sum() < 500 or np.ptp(x[m]) == 0 else spearmanr(x[m], y[m])[0]


rr = demean(r, apt); rd = demean(r, cell)
for f in FE:
    x = d[f].to_numpy().astype(float); xf = np.where(np.isfinite(x), x, np.nanmean(x))
    xa = np.where(np.isfinite(x), demean(xf, apt), np.nan); xd = np.where(np.isfinite(x), demean(xf, cell), np.nan)
    p = rho(xa, rr); pd_ = rho(xd, rd)
    mm = [rho(xa[mon == m], rr[mon == m]) for m in sorted(set(mon))]
    sg = int(np.sum(np.sign(mm) == np.sign(p)))
    pa = {a: round(float(rho(x[apt == a], r[apt == a])), 3) for a in APTS}
    print(f"{f:22s} pooled {p:+.4f} cellDM {pd_:+.4f} months {sg}/12  {pa}", flush=True)
# LOMO ridge bound
base = float(np.sqrt((r ** 2).mean()))
for nm, fs in {"wind3": ["head", "cross", "tail"], "flags": ["tail5", "wind_nonpref", "rwy_nonpref", "wind_vs_rwy_mismatch"],
               "inter": ["rare_x_tail", "rare_x_cross", "rare_x_nonpref", "tail_x_spd"], "all": FE}.items():
    X = np.column_stack([np.nan_to_num(d[f].to_numpy().astype(float), nan=np.nanmean(d[f].to_numpy().astype(float))) for f in fs])
    corr = np.zeros(len(r))
    for a in APTS:
        for m in sorted(set(mon)):
            te = (apt == a) & (mon == m); tr = (apt == a) & (mon != m)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            A = np.column_stack([np.ones(tr.sum()), (X[tr] - mu) / sd])
            c = np.linalg.solve(A.T @ A + 1e3 * np.eye(A.shape[1]), A.T @ r[tr])
            corr[te] = np.column_stack([np.ones(te.sum()), (X[te] - mu) / sd]) @ c
    corr = np.clip(corr, -60, 60)
    pa = {a: round(float(np.sqrt(((r - corr)[apt == a] ** 2).mean()) - np.sqrt((r[apt == a] ** 2).mean())), 3) for a in APTS}
    print(f"LOMO {nm:7s} dRMSE all {np.sqrt(((r - corr) ** 2).mean()) - base:+.4f}  {pa}", flush=True)
