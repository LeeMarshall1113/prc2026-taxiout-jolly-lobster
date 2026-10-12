"""day-equation 05 (use 1, Rome only, after 03): copy-versus-real posterior for Rome long-gap rows with the day's REAL noise:
sd_day^2 = floor^2 + sum over the day's OTHER hedged rows (mid-gap 1,800-5,400 s and ambiguous rows outside the enumeration) of p(1-p)(gs - taxi)^2.
Rows with served value above the gap (day-early hedges) are not decidable by a positive-part sum and are left out. Count gate |FLT_DEP_1 - n| <= 15.
Usage: python agent_dayeq_05_rome_types.py 2025 | 2026"""
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
MODE = sys.argv[1] if len(sys.argv) > 1 else "2025"
AMB_GAP, AMB_PRED, MAXK, DN_MAX = 5400.0, 1800.0, 20, 15
FRAME_MONTHS = [1, 2, 3, 7, 8, 9, 10, 11]
HEDGED = ["WMTT3508", "SVII3006", "ITYTY672", "NOSOS431", "QFAA7006", "TAY9LA"]
FLOOR = 33.0  # minutes: 2025 k=0 days, identity residual with TRUE stamps at LIRF (03); the ordinary-prediction part enters through var below
ORD_FLOOR = 121.0  # minutes: robust sd of published - predicted ordinary sum on 2025 k=0 days (03)


def load_pub(year):
    d = pl.read_csv(f"{E}all_pre_departure_delays_{year}.csv", infer_schema_length=0).filter(C("APT_ICAO") == "LIRF")
    return d.select(C("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"), C("DLY_ALL_PRE_2").cast(pl.Float64, strict=False).alias("pub"), C("FLT_DEP_1").cast(pl.Float64, strict=False).alias("n_pub"))


if MODE == "2025":
    files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
    raw = (pl.scan_parquet(files).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF"))
           .select(C("MVT_ID_mvt").alias("id"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"), C("FLIGHT_ID_mvt").is_not_null().alias("hasplan"), C("FLIGHT_mvt").alias("fs")).collect())
    eb = pl.read_parquet(I + "agent_eb136_tbl.parquet").select("id", "mon", "pred").filter(C("mon").is_in(FRAME_MONTHS))
    f = raw.join(eb, on="id", how="inner"); pub = load_pub(2025)
else:
    raw = (pl.scan_parquet(R + "data/raw/ranking.parquet").filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF"))
           .select(C("MVT_ID_mvt").alias("id"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"), C("FLIGHT_ID_mvt").is_not_null().alias("hasplan"), C("FLIGHT_mvt").alias("fs")).collect())
    v = pl.read_parquet(I + "agent_bundle_v150_final.parquet").select(C("MVT_ID_mvt").alias("id"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("pred"))
    f = raw.join(v, on="id", how="left").with_columns(C("T").dt.month().alias("mon"), pl.lit(None, pl.Float64).alias("y")); pub = load_pub(2026)
f = f.with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs")).filter(C("gs").is_not_null())
taxi = f.filter(C("gs").abs() < 3600).group_by("mon").agg(taxi=C("pred").median())
f = f.join(taxi, on="mon", how="left")
f = f.with_columns(((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("p_copy"))
f = f.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED) & (C("pred") <= C("gs"))).alias("enum"),   # enumerated rows
                   ((C("gs") >= 1800) & (C("pred") >= 0.5 * C("taxi") + 300) & (C("pred") <= C("gs"))).alias("hedged"))  # every hedged row (noise)
f = f.with_columns(C("hedged") & ~C("enum").alias("noise_row"))
f = f.with_columns(pl.when(C("enum")).then(C("T") - pl.duration(seconds=C("taxi"))).otherwise(C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day"),
                   ((C("gs") - C("taxi")) / 60.0).alias("e"),
                   pl.when(C("enum")).then(0.0).otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm_ord"),
                   pl.when(C("hedged") & ~C("enum")).then(C("p_copy") * (1 - C("p_copy")) * ((C("gs") - C("taxi")) / 60.0) ** 2).otherwise(0.0).alias("var_noise"))
if MODE == "2025":
    f = f.with_columns(((C("y") - C("gs")).abs() <= 90).alias("is_copy"))
days = f.group_by("day").agg(n=pl.len(), O=C("dm_ord").sum(), k=C("enum").sum(), nh=(C("hedged") & ~C("enum")).sum(), sd_noise=C("var_noise").sum().sqrt()).join(pub, on="day", how="inner")
days = days.with_columns((C("n_pub") - C("n")).alias("dn"), (C("pub") - C("O")).alias("D"))
pub7 = pub.select((C("day") - pl.duration(days=7)).alias("day"), C("pub").alias("pub7"))
days = days.join(pub7, on="day", how="left")
print(MODE, "LIRF rows", f.height, "enumerated", int(f["enum"].sum()), "hedged noise rows", int((f["hedged"] & ~f["enum"]).sum()))
print("per-day sd of the hedged (non-enumerated) rows' total gap under the shipped probabilities (minutes):")
print(days.group_by(C("day").dt.month().alias("mon")).agg(days=pl.len(), sd_noise_med=C("sd_noise").median(), sd_noise_p90=C("sd_noise").quantile(0.9), nh=C("nh").mean(), k=C("k").mean()).sort("mon").with_columns(pl.col(pl.Float64).round(1)))
k0 = days.filter((C("k") == 0) & (C("dn").abs() <= DN_MAX))
if k0.height > 5:
    r = (k0["pub"] - k0["O"]).to_numpy(); print(f"k=0 gated days {len(r)}: bias {np.median(r):+.0f}, robust sd {1.4826*np.median(np.abs(r-np.median(r))):.0f}, mean sd_noise {k0['sd_noise'].mean():.0f}")
bias = float(np.median((k0["pub"] - k0["O"]).to_numpy())) if k0.height > 5 else 0.0
if MODE == "2025":
    # measured noise on ALL gated days using the true types of the enumerated rows: published - O - sum over enum of true contribution
    tt = f.filter(C("enum")).with_columns(pl.when(C("is_copy")).then(0.0).otherwise(C("e")).alias("tc")).group_by("day").agg(tc=C("tc").sum())
    dd = days.join(tt, on="day", how="left").with_columns(C("tc").fill_null(0.0)).filter(C("dn").abs() <= DN_MAX).with_columns((C("pub") - C("O") - C("tc")).alias("r"))
    print("all gated days, residual with true enumerated types: by sd_noise tercile (does the Bernoulli noise model fit?)")
    dd = dd.with_columns(((C("sd_noise") ** 2 + ORD_FLOOR ** 2).sqrt()).alias("sd_model"))
    for lo, hi in [(0, 100), (100, 200), (200, 400), (400, 1e9)]:
        s = dd.filter((C("sd_noise") >= lo) & (C("sd_noise") < hi))
        if s.height:
            rr = s["r"].to_numpy(); print(f"  sd_noise {lo}-{hi}: days {s.height}, residual robust sd {1.4826*np.median(np.abs(rr-np.median(rr))):.0f}, sd {rr.std():.0f}, model sd {s['sd_model'].mean():.0f}")
en = f.filter(C("enum"))
out = []
for row in days.filter((C("k") > 0) & (C("k") <= MAXK) & (C("dn").abs() <= DN_MAX)).iter_rows(named=True):
    g = en.filter(C("day") == row["day"]); k = g.height
    p = np.clip(g["p_copy"].to_numpy(), 0.02, 0.98); e = g["e"].to_numpy()
    sd = float(np.sqrt(ORD_FLOOR ** 2 + row["sd_noise"] ** 2))
    Z = np.array(list(itertools.product([0, 1], repeat=k)), dtype=float); extra = (1 - Z) @ e
    logprior = (Z * np.log(p) + (1 - Z) * np.log(1 - p)).sum(axis=1)
    for pc in ["pub", "pub7"]:
        P = row[pc]
        if P is None or not np.isfinite(P):
            continue
        target = P - row["O"] - bias
        w = logprior - 0.5 * ((target - extra) / sd) ** 2; w = np.exp(w - w.max()); w /= w.sum(); q = (w[:, None] * Z).sum(axis=0)
        for i, rr in enumerate(g.iter_rows(named=True)):
            out.append(dict(day=row["day"], k=k, pubcol=pc, id=rr["id"], fs=rr["fs"], noplan=not rr["hasplan"], gs=rr["gs"], taxi=rr["taxi"], pred=rr["pred"], p=p[i], q=q[i],
                            post=q[i] * rr["gs"] + (1 - q[i]) * rr["taxi"], y=rr["y"], is_copy=rr.get("is_copy"), target=target, e=e[i], sd=sd, pub=row["pub"], O=row["O"]))
po = pl.DataFrame(out); po.write_parquet(I + f"agent_dayeq_rome_post_{MODE}.parquet")
if MODE == "2025":
    for pc in ["pub", "pub7"]:
        t = po.filter(C("pubcol") == pc)
        yv = t["y"].to_numpy(); c = t["is_copy"].to_numpy().astype(float); p = t["p"].to_numpy(); q = np.clip(t["q"].to_numpy(), 1e-6, 1 - 1e-6)
        g = (t["pred"].to_numpy() - yv) ** 2 - (t["post"].to_numpy() - yv) ** 2
        dd = t.with_columns(pl.Series("g", g)).group_by("day").agg(C("g").sum())["g"].to_numpy()
        rng = np.random.default_rng(0); bs = np.array([rng.choice(dd, len(dd)).sum() for _ in range(2000)]) / 1.5e8
        k1 = t["k"].to_numpy() == 1
        bins = np.digitize(q, [0.2, 0.4, 0.6, 0.8]); cal = [(round(float(q[bins == b].mean()), 2), round(float(c[bins == b].mean()), 2), int((bins == b).sum())) for b in range(5) if (bins == b).sum() > 0]
        print(f"\n== {pc}: rows {t.height}, days {t['day'].n_unique()}, copy rate {c.mean():.2f}; log-loss prior {-(c*np.log(p)+(1-c)*np.log(1-p)).mean():.3f} post {-(c*np.log(q)+(1-c)*np.log(1-q)).mean():.3f}; "
              f"gain {g.sum()/1.5e8:+.2f} board-s (8 months; k=1 days {g[k1].sum()/1.5e8:+.2f}), day-bootstrap 5-95% [{np.quantile(bs,0.05):+.2f}, {np.quantile(bs,0.95):+.2f}], P>0 {(bs>0).mean():.2f}; calibration (q, truth, n) {cal}")
        # by gap band and plan
        tb = t.with_columns(pl.Series("g", g), pl.when(C("gs") < 10800).then(pl.lit("1.5-3h")).when(C("gs") < 21600).then(pl.lit("3-6h")).otherwise(pl.lit(">6h")).alias("band"))
        print(tb.group_by("band", "noplan").agg(n=pl.len(), copy=C("is_copy").mean(), p=C("p").mean(), q=C("q").mean(), gain_s=C("g").sum() / 1.5e8).sort("band", "noplan").with_columns(pl.col(pl.Float64).round(3)))
    t = po.filter(C("pubcol") == "pub").with_columns(((C("pred") - C("y")) ** 2 - (C("post") - C("y")) ** 2).alias("g"), C("day").dt.month().alias("mon"))
    print("\ngain by month and plan state (board-s, that month's rows only):")
    print(t.group_by("mon", "noplan").agg(n=pl.len(), gain_s=C("g").sum() / 1.5e8, moved1000=((C("post") - C("pred")).abs() > 1000).sum()).sort("mon", "noplan").with_columns(pl.col(pl.Float64).round(3)))
    print("\nlargest |gain| rows:"); print(t.sort(C("g").abs(), descending=True).head(15).select("day", "k", "fs", "gs", "pred", "y", "is_copy", "p", "q", "post", "target", "e", "sd", "g").with_columns(pl.col(pl.Float64).round(1)))
else:
    t = po.filter(C("pubcol") == "pub").with_columns((C("post") - C("pred")).alias("move"))
    print(f"\n2026 enumerated rows {t.height} on {t['day'].n_unique()} days; |move| > 1000 s: {t.filter(C('move').abs() > 1000).height}; sum of moves {t['move'].sum():.0f} s")
    print("hedged rows (any in the enumeration):"); print(t.filter(C("fs").is_in(HEDGED)).with_columns(pl.col(pl.Float64).round(1)))
    hd = f.filter(C("fs").is_in(HEDGED)).select("fs", "T", "S", "gs", "pred", "taxi", "p_copy", "enum", "hedged", "day")
    print("hedged rows in the frame:"); print(hd)
    print("their days:"); print(days.filter(C("day").is_in(hd["day"].to_list() + [d for d in hd["day"].to_list()])).with_columns(pl.col(pl.Float64).round(0)))
    print("\nrows moved by more than 1000 s:")
    print(t.filter(C("move").abs() > 1000).sort("day").select("day", "k", "fs", "noplan", "gs", "taxi", "pred", "p", "q", "post", "move", "target", "e", "sd", "pub", "O").with_columns(pl.col(pl.Float64).round(1)))
    t.write_parquet(I + "agent_dayeq_rome_moves_2026.parquet")
    # NOSOS431: copy (delay 0) against day-early (stamp = true off-block - 24 h, delay = gs - 86,400 - taxi > 0 on the stamp's day)
    for fs in ["NOSOS431", "QFAA7006"]:
        h = f.filter(C("fs") == fs)
        for r in h.iter_rows(named=True):
            d_de = (r["gs"] - 86400 - r["taxi"]) / 60.0
            bday = (r["T"] - pl.duration(seconds=86400 + r["taxi"])) if False else None
            import datetime as _dt
            sday = (r["T"] - _dt.timedelta(seconds=86400 + r["taxi"])).date()
            dd = days.filter(C("day") == sday)
            print(f"\n{fs}: gap {r['gs']/3600:.1f} h, served {r['pred']:.0f}; day-early delay on {sday} would be {d_de:+.0f} min; that day:")
            print(dd.with_columns(pl.col(pl.Float64).round(0)))
            if dd.height:
                D = float(dd["D"][0]); sdn = float(np.sqrt(ORD_FLOOR ** 2 + float(dd["sd_noise"][0]) ** 2))
                print(f"  D - bias = {D - bias:+.0f} min, noise sd {sdn:.0f}: z under COPY {(D - bias)/sdn:+.1f}, z under DAY-EARLY {(D - bias - d_de)/sdn:+.1f}")
    # build: posterior means for NO-PLAN enumerated rows only (the 2025-validated subset), hedged flights untouched
    mv = t.filter(C("noplan") & ~C("fs").is_in(HEDGED)).select(C("id").alias("MVT_ID_mvt"), C("post").round(0).alias("new"))
    v = pl.read_parquet(I + "agent_bundle_v150_final.parquet")
    v2 = v.join(mv, on="MVT_ID_mvt", how="left").with_columns(pl.coalesce(C("new"), C("TAXITIME_SEC_mvt").cast(pl.Float64)).cast(pl.Int32).alias("TAXITIME_SEC_mvt")).select("MVT_ID_mvt", "TAXITIME_SEC_mvt")
    v2.write_parquet(I + "agent_dayeq_types_final.parquet")
    dmv = (v2["TAXITIME_SEC_mvt"] - v["TAXITIME_SEC_mvt"]).to_numpy()
    print(f"\nbuilt agent_dayeq_types_final.parquet: rows moved {(dmv != 0).sum()}, sum dp^2 per scored row {(dmv.astype(float) ** 2).sum() / len(dmv):.1f}, sum of moves {dmv.sum():+.0f} s")
