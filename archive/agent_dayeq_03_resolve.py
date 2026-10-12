"""day-equation 03: resolving power of the daily identity sum_i max(B_i - S_i, 0) = published (minutes), on 2025 as it would be in 2026.
Ordinary rows: stamp = take-off - held-out chain prediction. Ambiguous rows (gap >= AMB_GAP and served value >= AMB_PRED): two hypotheses,
COPY (stamp = schedule, delay 0, label = gap) and REAL (ordinary taxi, delay = gap - taxi). Posterior over assignments from the chain's implied
mixture prior and the day equation with Gaussian noise at the measured floor. Control: published series shifted by 7 days.
Usage: python agent_dayeq_03_resolve.py [2025|2026]"""
import glob
import itertools
import sys
import numpy as np
import polars as pl

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(200)
C = pl.col
R = ""
E = R + "data/external/eurocontrol/"
I = R + "data/interim/"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
MODE = sys.argv[1] if len(sys.argv) > 1 else "2025"
AMB_GAP, AMB_PRED, MAXK = 5400.0, 1800.0, 14
FRAME_MONTHS = [1, 2, 3, 7, 8, 9, 10, 11]
HEDGED = ["WMTT3508", "SVII3006", "ITYTY672", "NOSOS431", "QFAA7006", "TAY9LA"]


def load_pub(year):
    d = pl.read_csv(f"{E}all_pre_departure_delays_{year}.csv", infer_schema_length=0).filter(C("APT_ICAO").is_in(APTS))
    return d.select(C("APT_ICAO").alias("apt"), C("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"), C("DLY_ALL_PRE_2").cast(pl.Float64, strict=False).alias("pub"))


if MODE == "2025":
    files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
    raw = (pl.scan_parquet(files).filter(C("PHASE_mvt") == "DEP").filter(C("SCHED_TIME_UTC_mvt").is_not_null())
           .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"), C("FLIGHT_ID_mvt").is_not_null().alias("hasplan"), C("FLIGHT_mvt").alias("fs")).collect())
    eb = pl.read_parquet(I + "agent_eb136_tbl.parquet").select("id", "mon", "pred", "P2", "mid", "rule").filter(C("mon").is_in(FRAME_MONTHS))
    f = raw.join(eb, on="id", how="inner")
    pub = load_pub(2025)
else:
    raw = (pl.scan_parquet(R + "data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
           .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   C("FLIGHT_ID_mvt").is_not_null().alias("hasplan"), C("FLIGHT_mvt").alias("fs")).collect())
    v = pl.read_parquet(I + "agent_bundle_v150_final.parquet").select(C("MVT_ID_mvt").alias("id"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred"))
    f = raw.join(v, on="id", how="left").with_columns(C("T").dt.month().alias("mon"), pl.lit(None, pl.Float64).alias("y"), pl.lit(None, pl.Float64).alias("P2"))
    pub = load_pub(2026)
f = f.with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs"))
print(MODE, "rows", f.height, "pred null", f["pred"].null_count(), "S null", f["S"].null_count())
# ordinary taxi estimate for the REAL hypothesis: airport-month median prediction on short-gap rows
taxi = f.filter(C("gs").abs() < 3600).group_by("apt", "mon").agg(taxi=C("pred").median())
f = f.join(taxi, on=["apt", "mon"], how="left")
f = f.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"))
# prior P(copy) from the served hedge: pred = p*gs + (1-p)*taxi
f = f.with_columns(((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.02, 0.98).alias("p_copy"))
# day assignment: ordinary rows by predicted off-block date; ambiguous rows by take-off date minus taxi (REAL) -- the COPY case contributes 0 anywhere
f = f.with_columns(pl.when(C("amb")).then(C("T") - pl.duration(seconds=C("taxi"))).otherwise(C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day"),
                   pl.when(C("amb")).then(0.0).otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm_ord"),
                   ((C("gs") - C("taxi")) / 60.0).alias("dm_real"))
if MODE == "2025":
    f = f.with_columns(((C("y") - C("gs")).abs() <= 90).alias("is_copy"), ((C("gs") - C("y")) / 60.0).clip(lower_bound=0).alias("dm_true"))
print("ambiguous rows per airport:")
print(f.filter(C("amb")).group_by("apt").agg(n=pl.len(), noplan=(~C("hasplan")).sum(), gs=C("gs").mean(), pred=C("pred").mean(), p_copy=C("p_copy").mean(),
                                            **({"copy_rate": C("is_copy").mean()} if MODE == "2025" else {})).sort("apt"))
days = f.group_by("apt", "day").agg(n=pl.len(), O=C("dm_ord").sum(), k=C("amb").sum(), **({"true_sum": C("dm_true").sum()} if MODE == "2025" else {}))
days = days.join(pub, on=["apt", "day"], how="inner")
# control: published shifted by 7 days
pub7 = pub.with_columns((C("day") - pl.duration(days=7)).alias("day")).rename({"pub": "pub7"})
days = days.join(pub7, on=["apt", "day"], how="left")

# ---- noise floor on days with no ambiguous row
floor = {}
print("\nnoise floor: published - predicted ordinary sum, days with k = 0 (minutes)")
for a in APTS:
    d = days.filter((C("apt") == a) & (C("k") == 0))
    r = (d["pub"] - d["O"]).to_numpy()
    if len(r) < 10:
        print(a, "too few k=0 days", len(r)); continue
    med = np.median(r); mad = 1.4826 * np.median(np.abs(r - med))
    floor[a] = (med, max(mad, 1.0), r.std())
    print(f"{a}: k=0 days {len(r)}, bias {med:+.0f}, robust sd {mad:.0f}, sd {r.std():.0f}, p90 |res-bias| {np.quantile(np.abs(r-med),0.9):.0f}; "
          + (f"(with true stamps: sd {(d['pub']-d['true_sum']).to_numpy().std():.0f})" if MODE == "2025" else ""))
    if MODE == "2025":
        # the within-day prediction noise as a function of n: compare sd to sqrt(n) * per-flight rms
        pass

# ---- posterior per day
amb = f.filter(C("amb")).with_columns(C("day").alias("aday"))
out = []
for a in APTS:
    if a not in floor:
        continue
    bias, sd, _ = floor[a]
    da = days.filter((C("apt") == a) & (C("k") > 0))
    for row in da.iter_rows(named=True):
        g = amb.filter((C("apt") == a) & (C("aday") == row["day"]))
        k = g.height
        if k == 0 or k > MAXK:
            continue
        p = g["p_copy"].to_numpy(); e = g["dm_real"].to_numpy()
        for pubcol in ["pub", "pub7"]:
            P = row[pubcol]
            if P is None or not np.isfinite(P):
                continue
            target = P - row["O"] - bias
            # enumerate assignments: z=1 copy (0 delay), z=0 real (delay e)
            Z = np.array(list(itertools.product([0, 1], repeat=k)), dtype=float)  # 2^k x k
            extra = Z @ np.zeros(k) + (1 - Z) @ e
            logprior = (Z * np.log(p) + (1 - Z) * np.log(1 - p)).sum(axis=1)
            loglik = -0.5 * ((target - extra) / sd) ** 2
            w = logprior + loglik; w = np.exp(w - w.max()); w /= w.sum()
            q = (w[:, None] * Z).sum(axis=0)  # posterior P(copy)
            for i, rr in enumerate(g.iter_rows(named=True)):
                out.append(dict(apt=a, day=row["day"], k=k, pubcol=pubcol, id=rr["id"], fs=rr["fs"], gs=rr["gs"], taxi=rr["taxi"], pred=rr["pred"], p=p[i], q=q[i],
                                post=q[i] * rr["gs"] + (1 - q[i]) * rr["taxi"], y=rr["y"], is_copy=rr.get("is_copy"), target=target, e=e[i], sd=sd, noplan=not rr["hasplan"]))
po = pl.DataFrame(out)
po.write_parquet(I + f"agent_dayeq_post_{MODE}.parquet")
print("\nposterior rows", po.height, "days skipped (k > MAXK):", days.filter(C("k") > MAXK).height)

if MODE == "2025":
    for pubcol in ["pub", "pub7"]:
        s = po.filter(C("pubcol") == pubcol)
        print(f"\n===== {pubcol} {'(CONTROL, shifted 7 days)' if pubcol == 'pub7' else ''}")
        rows = []
        for a in APTS:
            t = s.filter(C("apt") == a)
            if t.height == 0:
                continue
            yv = t["y"].to_numpy(); c = t["is_copy"].to_numpy().astype(float); p = t["p"].to_numpy(); q = t["q"].to_numpy()
            se_ship = ((t["pred"].to_numpy() - yv) ** 2).sum(); se_post = ((t["post"].to_numpy() - yv) ** 2).sum()
            ll_p = -(c * np.log(p) + (1 - c) * np.log(1 - p)).mean(); ll_q = -(c * np.log(np.clip(q, 1e-6, 1 - 1e-6)) + (1 - c) * np.log(np.clip(1 - q, 1e-6, 1 - 1e-6))).mean()
            k1 = t.filter(C("k") == 1)
            se1 = ((k1["pred"].to_numpy() - k1["y"].to_numpy()) ** 2).sum() - ((k1["post"].to_numpy() - k1["y"].to_numpy()) ** 2).sum()
            # day bootstrap of the gain
            dd = t.with_columns(((C("pred") - C("y")) ** 2 - (C("post") - C("y")) ** 2).alias("g")).group_by("day").agg(C("g").sum())["g"].to_numpy()
            rng = np.random.default_rng(0); bs = np.array([rng.choice(dd, len(dd)).sum() for _ in range(2000)]) / 1.5e8
            # calibration of q in bins
            bins = np.digitize(q, [0.2, 0.4, 0.6, 0.8]); cal = [(round(float(q[bins == b].mean()), 2), round(float(c[bins == b].mean()), 2), int((bins == b).sum())) for b in range(5) if (bins == b).sum() > 0]
            rows.append(dict(apt=a, rows=t.height, days=t["day"].n_unique(), copy_rate=c.mean(), logloss_prior=ll_p, logloss_post=ll_q,
                             gain_s=(se_ship - se_post) / 1.5e8, gain_k1_s=se1 / 1.5e8, bs_lo=np.quantile(bs, 0.05), bs_hi=np.quantile(bs, 0.95), P_pos=(bs > 0).mean(), cal=str(cal)))
        print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(3)))
    # the Rome copy-vs-real reading per row for the largest stakes
    s = po.filter((C("pubcol") == "pub") & (C("apt") == "LIRF")).with_columns(((C("pred") - C("y")) ** 2 - (C("post") - C("y")) ** 2).alias("g"))
    print("\nLIRF rows with the largest |gain| (2025):")
    print(s.sort(C("g").abs(), descending=True).head(20).select("day", "k", "fs", "gs", "pred", "y", "is_copy", "p", "q", "post", "target", "e", "sd", "g").with_columns(pl.col(pl.Float64).round(2)))
else:
    s = po.filter(C("pubcol") == "pub").with_columns((C("post") - C("pred")).alias("move"))
    print("\n2026: rows by airport with |move| > 1000 s")
    print(s.filter(C("move").abs() > 1000).group_by("apt").agg(n=pl.len(), days=C("day").n_unique(), up=(C("move") > 0).sum(), sum_move=C("move").sum(), rms=((C("move") ** 2).mean()).sqrt()).sort("apt"))
    print("\nhedged rows:")
    print(s.filter(C("fs").is_in(HEDGED)).select("apt", "day", "k", "fs", "gs", "taxi", "pred", "p", "q", "post", "target", "e", "sd").with_columns(pl.col(pl.Float64).round(2)))
    print("\nall rows moved by more than 1000 s (with day numbers):")
    print(s.filter(C("move").abs() > 1000).sort("apt", "day").select("apt", "day", "k", "fs", "noplan", "gs", "taxi", "pred", "p", "q", "post", "move", "target", "e", "sd").with_columns(pl.col(pl.Float64).round(1)))
    # day table for the scored months: published vs predicted ordinary sum at Rome
    print("\nLIRF scored days with ambiguous rows:")
    print(days.filter((C("apt") == "LIRF") & (C("k") > 0)).sort("day").with_columns(pl.col(pl.Float64).round(0)))
