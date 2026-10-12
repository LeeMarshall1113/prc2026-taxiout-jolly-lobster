"""next-pools round two: RULE C (coverage x lateness cell slopes) and RULE D (slot-adherence day shares), increments over RULE A. Frozen in docs/LAB-NOTES.md."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa
import json
FRAME = sys.argv[1]
APTS = ["LFPG", "EGLL", "EDDF", "EDDM", "LIRF"]
CFG = json.load(open(I + "agent_dayeq_level_cfg.json")); CFG["LFPG"] = dict(lam=0.8, bias=-11.229803192691179)
ROWRULES = ["R1", "R2", "R3a", "R3b"]
LAMS = [round(0.1 * i, 1) for i in range(11)]
f = frame_2025(APTS)
sv = pl.read_parquet(I + "agent_waithedge_tbl.parquet", columns=["id", "served", "cov"]).with_columns(C("id").cast(pl.Float64))
f = f.with_columns(C("id").cast(pl.Float64)).join(sv, on="id", how="left")
if FRAME == "served":
    f = f.filter(C("served").is_not_null()).with_columns(C("served").alias("pred"))
f = f.with_columns(C("cov").fill_null("other")).filter(C("y").is_not_null() & C("gs").is_not_null())
f = f.with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"), C("rule").fill_null("").is_in(ROWRULES).alias("rr"), C("day_pred").dt.month().alias("dmon"),
                   C("T").dt.date().alias("tday"),
                   pl.when(C("cov") == "matched").then(0).when(C("cov") == "points_no_liftoff").then(1).otherwise(2).alias("cls"),
                   pl.when(C("gs") < 900).then(0).when(C("gs") < 1800).then(1).when(C("gs") < 3600).then(2).otherwise(3).alias("lb"))
f = f.filter(C("dmon").is_in(FRAME_MONTHS)).with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60.0).alias("dm"), (~C("amb") & ~C("rr")).alias("sh"),
                                                         (C("cls") * 4 + C("lb")).alias("cell"), C("fs").str.slice(0, 3).fill_null("NA").alias("air"))
pub = load_pub(2025, APTS)
days = f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum()).rename({"day_pred": "day"}).join(pub, on=["apt", "day"], how="inner").filter((C("n_pub") - C("n")).abs() <= DN_MAX)
parts = f.partition_by("apt", "day_pred", as_dict=True)
rows = []
for r in days.iter_rows(named=True):
    g = parts[(r["apt"], r["day"])]; m = g["sh"].to_numpy()
    rec = dict(apt=r["apt"], day=r["day"])
    for nm, P in [("d", r["pub"]), ("d7", r["pub7"])]:
        rec[nm] = solve_delta(g["gs"].to_numpy()[m], g["pred"].to_numpy()[m], P - r["fixed"]) if P is not None and np.isfinite(P) else np.nan
    rows.append(rec)
f = f.join(pl.DataFrame(rows), left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
del parts
sl = pl.read_csv(E + "atfm_slot_adherence_2025.csv", infer_schema_length=0).filter(C("APT_ICAO").is_in(APTS)).select(
    C("APT_ICAO").alias("apt"), C("FLT_DATE").str.slice(0, 10).str.to_date().alias("tday"), C("FLT_DEP_1").cast(pl.Float64).alias("n1"), C("FLT_DEP_REG_1").cast(pl.Float64).alias("reg"),
    C("FLT_DEP_OUT_LATE_1").cast(pl.Float64).alias("late"))
sl = sl.with_columns((C("reg") / C("n1")).alias("x1"), (C("late") / pl.max_horizontal(C("reg"), pl.lit(1.0))).alias("x2")).select("apt", "tday", "x1", "x2")
sl7 = sl.with_columns((C("tday") + pl.duration(days=7)).alias("tday")).rename({"x1": "x1c", "x2": "x2c"})
f = f.join(sl, on=["apt", "tday"], how="left").join(sl7, on=["apt", "tday"], how="left")
print("rows", f.height, "x1 present", f["x1"].is_not_null().mean())
SEs = SE_SCALE


def fitA(yh, y, d, pos, tr):
    ok = tr & np.isfinite(d); bias = float(np.median(d[ok])); t = tr & pos; best = None
    for lam in LAMS:
        sse = ((np.maximum(yh[t] + lam * (d[t] - bias), 0) - y[t]) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, lam)
    return bias, best[1]


def run(a, dcol, xcols, frame):
    fa = f.filter(C("apt") == a)
    yh = fa["pred"].to_numpy(); y = fa["y"].to_numpy(); d = fa[dcol].to_numpy(); gs = fa["gs"].to_numpy()
    pos = (gs > yh) & fa["sh"].to_numpy() & np.isfinite(d); cell = fa["cell"].to_numpy(); mon = fa["dmon"].to_numpy(); air = fa["air"].to_numpy()
    X = np.column_stack([fa[c].to_numpy() for c in xcols]); base = (yh - y) ** 2
    if frame == "mh":
        splits = [(mon != m, mon == m) for m in FRAME_MONTHS]
    elif frame == "temporal":
        splits = [(mon <= 3, mon >= 7), (mon >= 7, mon <= 3)]
    else:
        vc = fa.group_by("air").len(); keep = vc.filter(C("len") >= 0.01 * fa.height)["air"].to_list()
        ag = np.where(np.isin(air, keep), air, "OTHER"); splits = [(ag != k, ag == k) for k in np.unique(ag)]
    nA = yh.copy(); nC = yh.copy(); nD = yh.copy()
    for tr, te in splits:
        bias, lam = CFG[a]["bias"], CFG[a]["lam"]
        xs = np.where(pos, d - bias, 0.0); t = tr & pos
        A_ = np.where(pos, np.maximum(yh + lam * xs, 0), yh)
        sxx = np.array([(xs[t & (cell == c)] ** 2).sum() for c in range(12)]); sxy = np.array([(xs[t & (cell == c)] * (y - yh)[t & (cell == c)]).sum() for c in range(12)])
        tau = 0.25 * sxx.mean(); beta = (sxy + tau * lam) / (sxx + tau)
        C_ = np.where(pos, np.maximum(yh + beta[cell] * xs, 0), yh)
        Xt = X[t]; okx = np.isfinite(Xt).all(axis=1)
        if okx.sum() > 50:
            mu = Xt[okx].mean(0); sd = Xt[okx].std(0) + 1e-9
            Z = np.where(np.isfinite(X), (X - mu) / sd, 0.0)
            res = (y - A_)[t][okx]; Zt = Z[t][okx]
            G = Zt.T @ Zt; lamr = 0.25 * np.trace(G) / G.shape[0]
            k = np.linalg.solve(G + lamr * np.eye(G.shape[0]), Zt.T @ res)
            D_ = np.where(pos, np.maximum(A_ + Z @ k, 0), A_)
        else:
            D_ = A_
        nA[te] = A_[te]; nC[te] = C_[te]; nD[te] = D_[te]
    gA = base - (nA - y) ** 2
    slim = fa.select("apt", "day_pred", "dmon")
    return slim, gA, (base - (nC - y) ** 2) - gA, (base - (nD - y) ** 2) - gA


def bootg(g, key, n=2000):
    u, inv = np.unique(key, return_inverse=True); s = np.bincount(inv, weights=g); rng = np.random.default_rng(0)
    bs = s[rng.integers(0, len(s), size=(n, len(s)))].sum(1) / SEs
    return np.quantile(bs, .05), np.quantile(bs, .95), (bs > 0).mean(), s


R_ = {}
for frame in ["mh", "loao", "temporal"]:
    for a in APTS:
        R_[(frame, a)] = run(a, "d", ["x1", "x2"], frame)
for a in APTS:
    R_[("ctl", a)] = run(a, "d7", ["x1c", "x2c"], "mh")
for rule, k in [("C", 2)]:
    print(f"\n######## RULE {rule}: increment over A, board-s")
    print("month-held-out per airport:", {a: round(R_[("mh", a)][k].sum() / SEs, 3) for a in APTS})
    print("loao per airport:", {a: round(R_[("loao", a)][k].sum() / SEs, 3) for a in APTS})
    print("temporal per airport:", {a: round(R_[("temporal", a)][k].sum() / SEs, 3) for a in APTS})
    for lab, aps in [("all7", APTS), ("LFPG", ["LFPG"]), ("EGLL", ["EGLL"]), ("LIRF", ["LIRF"])]:
        g = np.concatenate([R_[("mh", a)][k] for a in aps]); key = np.concatenate([(R_[("mh", a)][0]["apt"] + R_[("mh", a)][0]["day_pred"].cast(pl.Utf8)).to_numpy() for a in aps])
        mon = np.concatenate([R_[("mh", a)][0]["dmon"].to_numpy() for a in aps]); lo, hi, P, s = bootg(g, key); ds = np.sort(s)[::-1]
        print(f"  {lab}: {g.sum()/SEs:+.3f} [{lo:+.3f},{hi:+.3f}] P {P:.2f} Jan {g[mon==1].sum()/SEs:+.3f} Jul {g[mon==7].sum()/SEs:+.3f} no-top5-days {(g.sum()-ds[:5].sum())/SEs:+.3f} no-top20-rows {(g.sum()-np.sort(g)[::-1][:20].sum())/SEs:+.3f}")
    print("  loao all7", sum(R_[("loao", a)][k].sum() for a in APTS) / SEs)
    for lab, sel in [("test 7-11 after fit 1-3", lambda m: m >= 7), ("test 1-3 after fit 7-11", lambda m: m <= 3)]:
        print("  temporal", lab, sum(R_[("temporal", a)][k][sel(R_[("temporal", a)][0]["dmon"].to_numpy())].sum() for a in APTS) / SEs)
    print("  control all7 (inputs shifted 7 d)", sum(R_[("ctl", a)][k].sum() for a in APTS) / SEs)
