"""next-pools RULE C build on v161: cell slopes (coverage class x lateness bin) replace the uniform lam of Rule A at LFPG, EGLL, LEBL, LEMD, EDDF, EDDM, LIRF.
New value = v161 + beta_cell*(d - bias) - lam_prod*(d - bias_prod) on the rows where production's uniform shift applied (gs > v161 - lam_prod*(d - bias_prod), unambiguous, row-rule rows and
hedged strings excluded). Parameters (bias, lam_A, beta) fitted on all eight 2025 frame months exactly as in agent_nextpools_06_cd.py."""
import sys, json
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa
APTS = ["LFPG", "EGLL", "EDDF", "EDDM", "LIRF"]
ROWRULES = ["R1", "R2", "R3a", "R3b"]
LAMS = [round(0.1 * i, 1) for i in range(11)]
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def cellof(cat, gs):
    cls = np.where(cat == "matched", 0, np.where(cat == "points_no_liftoff", 1, 2))
    lb = np.where(gs < 900, 0, np.where(gs < 1800, 1, np.where(gs < 3600, 2, 3)))
    return cls * 4 + lb


def solve_days(f, pub, col):
    days = f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum()).rename({"day_pred": "day"}).join(pub, on=["apt", "day"], how="inner").filter((C("n_pub") - C("n")).abs() <= DN_MAX)
    parts = f.partition_by("apt", "day_pred", as_dict=True); rows = []
    for r in days.iter_rows(named=True):
        g = parts[(r["apt"], r["day"])]; m = g["sh"].to_numpy()
        rows.append({"apt": r["apt"], "day": r["day"], col: solve_delta(g["gs"].to_numpy()[m], g["pred"].to_numpy()[m], r["pub"] - r["fixed"]) if r["pub"] is not None and np.isfinite(r["pub"]) else np.nan})
    return pl.DataFrame(rows)


def prep(f):
    f = f.with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"), C("rule").fill_null("").is_in(ROWRULES).alias("rr"))
    return f.with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60.0).alias("dm"), (~C("amb") & ~C("rr")).alias("sh"))


# ---- 2025 fit
f = frame_2025(APTS)
sv = pl.read_parquet(I + "agent_waithedge_tbl.parquet", columns=["id", "served", "cov"]).with_columns(C("id").cast(pl.Float64))
f = f.with_columns(C("id").cast(pl.Float64)).join(sv, on="id", how="left").with_columns(C("cov").fill_null("other")).filter(C("y").is_not_null() & C("gs").is_not_null())
f = f.with_columns(C("day_pred").dt.month().alias("dmon")).filter(C("dmon").is_in(FRAME_MONTHS))
f = prep(f)
f = f.join(solve_days(f, load_pub(2025, APTS), "d"), left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
f = f.with_columns(pl.when(C("cov") == "matched").then(pl.lit("matched")).when(C("cov") == "points_no_liftoff").then(pl.lit("points_no_liftoff")).otherwise(pl.lit("other")).alias("cat"))
cfg = json.load(open(I + "agent_dayeq_level_cfg.json")); cfg["LFPG"] = dict(lam=0.8, bias=-11.229803192691179)
par = {}
for a in APTS:
    fa = f.filter(C("apt") == a)
    yh = fa["pred"].to_numpy(); y = fa["y"].to_numpy(); d = fa["d"].to_numpy(); gs = fa["gs"].to_numpy()
    pos = (gs > yh) & fa["sh"].to_numpy() & np.isfinite(d); cell = cellof(fa["cat"].to_numpy(), gs)
    bias = cfg[a]["bias"]; lam = cfg[a]["lam"]; xs = np.where(pos, d - bias, 0.0)
    sxx = np.array([(xs[pos & (cell == c)] ** 2).sum() for c in range(12)]); sxy = np.array([(xs[pos & (cell == c)] * (y - yh)[pos & (cell == c)]).sum() for c in range(12)])
    tau = 0.25 * sxx.mean(); beta = (sxy + tau * lam) / (sxx + tau)
    par[a] = dict(bias=bias, lam=lam, beta=beta)
    print(a, "bias", round(bias, 2), "lam", lam, "beta cells (trk: <15m,15-30,30-60,>1h | est | unt):", np.round(beta, 2).tolist())
del f
# ---- 2026 on v161
base = "submissions/jolly-lobster_v161.parquet"
g6 = frame_2026(APTS, base)
cen = pl.read_parquet(I + "agent_coverage_census_2026.parquet", columns=[ID, "category"]).rename({ID: "id", "category": "cat"}).with_columns(C("id").cast(g6["id"].dtype))
g6 = g6.join(cen, on="id", how="left").with_columns(C("cat").fill_null("other"))
# d: the six from the stored 2026 solve (on v150, as production), LFPG solved on v160
dd = pl.read_parquet(I + "agent_dayeq_level_days_2026.parquet").select("apt", "day", C("delta_pub").alias("d"))
v160 = pl.read_parquet(I + "agent_bundle_v160_final.parquet").select(C(ID).alias("id"), C(T).cast(pl.Float64).alias("p160")).with_columns(C("id").cast(g6["id"].dtype))
gl = g6.filter(C("apt") == "LFPG").join(v160, on="id", how="left").with_columns(C("p160").alias("pred"))
gl = prep(gl.with_columns(C("rule")))
dl = solve_days(gl, load_pub(2026, ["LFPG"]), "d")
dd = pl.concat([dd.filter(C("apt") != "LFPG").select("apt", "day", "d"), dl.select("apt", "day", "d")])
print("d table", dd.group_by("apt").len().to_dicts())
g6 = g6.join(dd, left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
v = g6["pred"].to_numpy(); gs = g6["gs"].to_numpy(); d = g6["d"].to_numpy(); apt = g6["apt"].to_numpy(); cat = g6["cat"].to_numpy()
rr = g6["rule"].fill_null("").is_in(ROWRULES).fill_null(False).to_numpy().astype(bool); hz = g6["fs"].is_in(HEDGED).fill_null(False).to_numpy().astype(bool)
inc = np.zeros(len(v))
for a in APTS:
    c = cfg[a]; p = par[a]; s = (apt == a) & np.isfinite(d)
    prod = c["lam"] * (d - c["bias"]); ypre = v - np.where(s, prod, 0.0)
    amb = (gs >= 5400) & (ypre >= 1800)
    m = s & (gs > ypre) & ~amb & ~rr & ~hz
    cell = cellof(cat, gs)
    new = p["beta"][cell] * (d - c["bias"])
    inc = np.where(m, new - prod, inc)
    print(a, "rows with increment", int(m.sum()), "of", int((apt == a).sum()), "rms inc", float(np.sqrt((inc[m] ** 2).mean())) if m.any() else 0, "mean", float(inc[m].mean()) if m.any() else 0)
print("overall rms inc (moved rows)", float(np.sqrt((inc[inc != 0] ** 2).mean())), "max |inc|", float(np.abs(inc).max()), "sum inc^2 per scored row", float((inc ** 2).sum() / 344841))
b = pl.read_parquet(base)
r = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP").select(ID, ((C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds()).alias("gl")).collect())
new = pl.DataFrame({ID: g6["id"].cast(b.schema[ID]), "inc": inc})
j = b.join(new, on=ID, how="left").join(r, on=ID, how="left").with_columns(C("inc").fill_null(0.0))
j = j.with_columns(pl.when(C("inc") != 0).then(pl.max_horizontal(C(T).cast(pl.Float64) + C("inc"), pl.lit(0.0))).otherwise(C(T).cast(pl.Float64)).alias("n"))
lo = pl.max_horizontal(C("gl") - 3606, pl.lit(0)); hi = C("gl") + 3606
j = j.with_columns(pl.when(C("gl").is_null() | (C("inc") == 0)).then(C("n")).otherwise(pl.min_horizontal(pl.max_horizontal(C("n"), lo), hi)).alias("n"))
out = b.select(ID).join(j.select(ID, C("n").round().cast(b.schema[T]).alias(T)), on=ID, how="left"); assert (out[ID] == b[ID]).all()
out.write_parquet(I + "agent_nextpools_c5_final.parquet")
ch = j.with_columns((C("n") - C(T)).alias("dd")).filter(C("dd") != 0)
print("rows changed", ch.height, "rms", float((ch["dd"] ** 2).mean() ** 0.5), "max", int(ch["dd"].abs().max()), "mean", float(ch["dd"].mean()), "sum d^2 per scored row", float((ch["dd"] ** 2).sum() / b.height))
