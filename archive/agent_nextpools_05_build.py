"""next-pools step 3: RULE A at LFPG. (i) 2025 parameters on all eight frame months (bias, lam as frozen); (ii) bias-stability check on the 2026 scored months from
the v160 file and the published 2026 series (no labels); (iii) if stable, build data/interim/agent_nextpools_lfpg_final.parquet on v160 (LFPG rows only, window last)."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa
ROWRULES = ["R1", "R2", "R3a", "R3b"]
LAMS = [round(0.1 * i, 1) for i in range(11)]
A = ["LFPG"]


def prep(f, pub):
    f = f.with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"), C("rule").fill_null("").is_in(ROWRULES).alias("rr"))
    f = f.with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60.0).alias("dm"), (~C("amb") & ~C("rr")).alias("sh"))
    days = f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum()).rename({"day_pred": "day"}).join(pub, on=["apt", "day"], how="inner")
    days = days.filter(((C("n_pub") - C("n")).abs() <= DN_MAX))
    parts = f.partition_by("apt", "day_pred", as_dict=True)
    rows = []
    for r in days.iter_rows(named=True):
        g = parts[(r["apt"], r["day"])]; m = g["sh"].to_numpy()
        P = r["pub"]
        rows.append(dict(apt=r["apt"], day=r["day"], d=solve_delta(g["gs"].to_numpy()[m], g["pred"].to_numpy()[m], P - r["fixed"]) if P is not None and np.isfinite(P) else np.nan))
    return f.join(pl.DataFrame(rows), left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left"), days.height


# (i) 2025
f5 = frame_2025(A)
sv = pl.read_parquet(I + "agent_waithedge_tbl.parquet", columns=["id", "served"]).with_columns(C("id").cast(pl.Float64))
f5 = f5.with_columns(C("id").cast(pl.Float64)).join(sv, on="id", how="inner").with_columns(C("served").alias("pred")).filter(C("y").is_not_null() & C("gs").is_not_null())
f5 = f5.with_columns(C("day_pred").dt.month().alias("dmon")).filter(C("dmon").is_in(FRAME_MONTHS))
f5, nd = prep(f5, load_pub(2025, A))
d5 = f5["d"].to_numpy(); yh = f5["pred"].to_numpy(); y = f5["y"].to_numpy(); pos = (f5["gs"].to_numpy() > yh) & f5["sh"].to_numpy()
ok = np.isfinite(d5); bias = float(np.median(d5[ok]))
best = None
for lam in LAMS:
    t = pos & ok
    sse = ((np.maximum(yh[t] + lam * (d5[t] - bias), 0) - y[t]) ** 2).sum()
    if best is None or sse < best[0]:
        best = (sse, lam)
lam = best[1]
print("2025 LFPG days", nd, "bias", bias, "lam", lam, " per-month median delta", f5.group_by("dmon").agg(m=C("d").median()).sort("dmon").to_dicts())
# (ii) 2026
base = I + "agent_bundle_v160_final.parquet"
f6 = frame_2026(A, base)
f6, nd6 = prep(f6, load_pub(2026, A))
print("2026 LFPG scored rows", f6.height, "days with delta", nd6, "months", f6.group_by("mon").len().sort("mon").to_dicts())
m6 = f6.group_by("mon").agg(med=C("d").median(), n=C("d").count()).sort("mon")
print(m6)
med26 = float(np.nanmedian(f6["d"].to_numpy()))
stable = abs(med26 - bias) <= 25
print(f"2026 median delta {med26:.1f} vs 2025 bias {bias:.1f}: |diff| {abs(med26-bias):.1f} -> stability {'PASS' if stable else 'FAIL'}")
d6 = f6["d"].to_numpy(); yh6 = f6["pred"].to_numpy(); gs6 = f6["gs"].to_numpy()
s6 = (gs6 > yh6) & f6["sh"].to_numpy() & np.isfinite(d6)
mv = np.where(s6, lam * (d6 - bias), 0.0)
print("rows moved", int(s6.sum()), "of", f6.height, "rms move", float(np.sqrt((mv[s6] ** 2).mean())), "max", float(np.abs(mv).max()), "mean", float(mv[s6].mean()), "sum dp^2 per scored row", float((mv ** 2).sum() / f6.height))
if not stable:
    print("NOT BUILT"); sys.exit(0)
hz = np.isin(f6["fs"].to_numpy(), HEDGED); print("hedged strings among shifted LFPG rows (set to unmoved):", int((hz & s6).sum()), f6["fs"].to_numpy()[hz & s6].tolist()); mv = np.where(hz, 0.0, mv)
new = pl.DataFrame({"id": f6["id"], "mv": mv})
b = pl.read_parquet(base)
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
r = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP").select(ID, "ADEP_mvt", ((C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds()).alias("gl")).collect())
j = b.join(new.rename({"id": ID}), on=ID, how="left").join(r, on=ID, how="left").with_columns(C("mv").fill_null(0.0))
j = j.with_columns(pl.when(C("mv") != 0).then(pl.max_horizontal(C(T).cast(pl.Float64) + C("mv"), pl.lit(0.0))).otherwise(C(T).cast(pl.Float64)).alias("n"))
lo = pl.max_horizontal(C("gl") - 3606, pl.lit(0)); hi = C("gl") + 3606
j = j.with_columns(pl.when(C("gl").is_null() | (C("mv") == 0)).then(C("n")).otherwise(pl.min_horizontal(pl.max_horizontal(C("n"), lo), hi)).alias("n"))
out = b.select(ID).join(j.select(ID, C("n").round().cast(b.schema[T]).alias(T)), on=ID, how="left"); assert (out[ID] == b[ID]).all()
out.write_parquet(I + "agent_nextpools_lfpg_final.parquet")
ch = j.with_columns((C("n") - C(T)).alias("d")).filter(C("d") != 0)
print("rows changed", ch.height, dict(ch.group_by("ADEP_mvt").len().iter_rows()), "rms", float((ch["d"] ** 2).mean() ** 0.5), "max", int(ch["d"].abs().max()), "mean", float(ch["d"].mean()), "sum d^2 per scored row", float((ch["d"] ** 2).sum() / b.height))
