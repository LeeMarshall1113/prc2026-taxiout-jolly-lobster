"""Rome no-plan copy-or-real mixture as a general function: footprint, features and weights come from inputs, not from a stored row list.

Derived from archive/agent_rome2_18_frames.py and agent_rome2_20_sensor_feats.py (arrival-side and input-side state features of a Rome
departure), agent_rome2_54_build.py (round-2 classifier: operator and destination priors, 3 seeds x 250 rounds) and its weights
(0.5 for gaps under 7,200 s, 0.85 from 7,200 to 14,400 s, constant 1,025.5 s for a real taxi).

Footprint: LIRF departure without a flight plan (LOBT missing), gap 3,600 to under 14,400 s, no hidden plan recovered from the
destination's arrival record. Value = max(round(a + w x s x (P x gap + (1 - P) x 1,025.5 - a)), 0), a = the input prediction, P = copy
probability of the classifier fitted on all twelve 2025 months, s = 1 (--form flat, as shipped) or min(1, m_d / m_ref) (--form scaled), m_d =
mean P of the day's informative no-plan Rome rows (gap of 1,800 s or more), m_ref = 2025 copy rate of such rows. --form off leaves
the input unchanged. Validation (b) of docs/LAB-NOTES.md: neither form passes the full rule (March loses in both);
the Rome with-plan mixture of v143 is NOT ported.

Default (--form default): the shipped flat form for every month except February, where it is OFF. Both forms pass on every warm month that could be tested
(months 7 to 9) and the flat form is the one already built, so nothing is re-selected; both forms lose in March and the flat form also in February, hence February off.
June itself could not be tested on 2025 (no base predictions exist for those rows in months 4 to 6). The scaled form stays behind --form scaled.

  python final/34_rome_mixtures.py --pred P.parquet --ranking R.parquet --out Q.parquet [--form default|flat|scaled|off] [--months 2,6]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from fcommon import ID_COL, RAW, TARGET_COL  # noqa: E402

C = pl.col
I = REPO / "data" / "interim"
M0 = 1025.5
HC_DEST = ("LTFJ", "LMML", "LTFM", "LLBG")
OP = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().str.extract(r"^([A-Z]{2,3})", 1)
BASE = ["gs", "round5", "round15", "sh", "dow", "npf", "apron", "stand_num", "gl", "ga", "ge", "gi"]
ARRF = ["arr_n_w1h", "arr_copy_w1h", "arr_np_w1h", "arr_n_w3h", "arr_copy_w3h", "arr_np_w3h", "arr_copy_w12h", "arr_np_w12h", "op6_n", "op6_copy", "op6_np", "st6_n", "st6_copy"]
LINK = ["L1_exists", "L1_copy", "L1_np", "L1_age", "L1_sameop", "L2_exists", "L2_copy", "L2_np", "L2_age", "L2_fdiff", "Nx_exists", "Nx_gap", "Nx_copy"]
COLS = BASE + ARRF + LINK
PO = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
          lambda_l2=5, verbose=-1, num_threads=4)


def garble(x):
    if x is None:
        return 0.0
    x = x.strip().upper()
    if len(x) < 6 or not x[:3].isalpha():
        return 0.0
    p = x[:3]
    for k in (2, 1):
        if len(x) > 3 + k and x[3:3 + k] == p[3 - k:3] and x[3 + k].isdigit():
            return 1.0
    return 0.0


def _win(times, vals, tq, w):
    lo = np.searchsorted(times, tq - w, "left")
    hi = np.searchsorted(times, tq + w, "right")
    cs = np.r_[0, np.cumsum(vals)] if vals.ndim == 1 else np.vstack([np.zeros((1, vals.shape[1])), np.cumsum(vals, axis=0)])
    return hi - lo, cs[hi] - cs[lo]


def sensor_features(d: pl.DataFrame, a: pl.DataFrame) -> pl.DataFrame:
    """Arrival-side state features of each Rome departure (label-free). d: id, op, stand, T, S (epoch s), fnum; a: tin, op, stand, fnum, copy, np.

    Port of agent_rome2_20_sensor_feats.py for one frame (all rows of the frame are one 'year').
    """
    a = a.sort("tin")
    n = d.height
    T = d["T"].to_numpy().astype(np.int64)
    tin = a["tin"].to_numpy().astype(np.int64)
    acopy, anp = a["copy"].to_numpy().astype(float), a["np"].to_numpy().astype(float)
    V = np.column_stack([acopy, anp])
    out = {}
    for w, nm in [(3600, "w1h"), (10800, "w3h"), (43200, "w12h")]:
        cnt, s = _win(tin, V, T, w)
        out[f"arr_n_{nm}"] = cnt.astype(float)
        out[f"arr_copy_{nm}"] = np.where(cnt > 0, s[:, 0] / np.maximum(cnt, 1), np.nan)
        out[f"arr_np_{nm}"] = np.where(cnt > 0, s[:, 1] / np.maximum(cnt, 1), np.nan)
    op, st, fnum = d["op"].to_numpy(), d["stand"].to_numpy(), d["fnum"].to_numpy()
    aop, ast, afn = a["op"].to_numpy(), a["stand"].to_numpy(), a["fnum"].to_numpy()

    def grp(keys):
        ix = {}
        for i, k in enumerate(keys):
            if k is not None:
                ix.setdefault(k, []).append(i)
        return {k: np.array(v) for k, v in ix.items()}

    F = {k: np.full(n, np.nan) for k in ["op6_n", "op6_copy", "op6_np", "st6_n", "st6_copy", "L1_exists", "L1_copy", "L1_np", "L1_age", "L1_sameop",
                                         "L2_exists", "L2_copy", "L2_np", "L2_age", "L2_fdiff", "Nx_exists", "Nx_gap", "Nx_copy"]}
    for k, ix in grp(aop).items():
        q = np.where(op == k)[0]
        if len(q) == 0:
            continue
        cnt, s = _win(tin[ix], V[ix], T[q], 21600)
        F["op6_n"][q] = cnt
        F["op6_copy"][q] = np.where(cnt > 0, s[:, 0] / np.maximum(cnt, 1), np.nan)
        F["op6_np"][q] = np.where(cnt > 0, s[:, 1] / np.maximum(cnt, 1), np.nan)
    for k, ix in grp(ast).items():
        q = np.where(st == k)[0]
        if len(q) == 0:
            continue
        tt = tin[ix]
        cnt, s = _win(tt, V[ix], T[q], 21600)
        F["st6_n"][q] = cnt
        F["st6_copy"][q] = np.where(cnt > 0, s[:, 0] / np.maximum(cnt, 1), np.nan)
        pos = np.searchsorted(tt, T[q] - 600, "right") - 1
        ok = pos >= 0
        a_i = ix[np.where(ok, pos, 0)]
        age = T[q] - tin[a_i]
        good = ok & (age <= 86400)
        F["L1_exists"][q] = good.astype(float)
        F["L1_copy"][q] = np.where(good, acopy[a_i], np.nan)
        F["L1_np"][q] = np.where(good, anp[a_i], np.nan)
        F["L1_age"][q] = np.where(good, age, np.nan)
        F["L1_sameop"][q] = np.where(good, (aop[a_i] == op[q]).astype(float), np.nan)
        for qi in q:
            if op[qi] is None:
                F["L2_exists"][qi] = 0
                continue
            lo = np.searchsorted(tt, T[qi] - 28800, "left")
            hi = np.searchsorted(tt, T[qi] - 600, "right")
            best = -1
            for j in range(hi - 1, lo - 1, -1):
                if aop[ix[j]] == op[qi]:
                    best = ix[j]
                    break
            if best < 0:
                F["L2_exists"][qi] = 0
            else:
                F["L2_exists"][qi] = 1
                F["L2_copy"][qi], F["L2_np"][qi], F["L2_age"][qi] = acopy[best], anp[best], T[qi] - tin[best]
                F["L2_fdiff"][qi] = (fnum[qi] - afn[best]) if not (np.isnan(fnum[qi]) or np.isnan(afn[best])) else np.nan
        pos2 = np.searchsorted(tt, T[q], "right")
        ok2 = pos2 < len(tt)
        a2 = ix[np.where(ok2, pos2, 0)]
        g2 = ok2 & ((tin[a2] - T[q]) <= 86400)
        F["Nx_exists"][q] = g2.astype(float)
        F["Nx_gap"][q] = np.where(g2, tin[a2] - T[q], np.nan)
        F["Nx_copy"][q] = np.where(g2, acopy[a2], np.nan)
    out.update(F)
    return d.with_columns([pl.Series(k, v) for k, v in out.items()])


def rome_frames(ranking: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rome departure and arrival frames of a ranking file, input columns only."""
    num = lambda c: C(c).str.extract(r"(\d+)", 1).cast(pl.Float64)  # noqa: E731
    dep = (pl.scan_parquet(ranking).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF"))
           .select(C(ID_COL).alias("id"), C("FLIGHT_mvt"), OP.alias("op"), C("STAND_mvt").alias("stand"), C("ADES_mvt").alias("dest"),
                   C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("T"), C("SCHED_TIME_UTC_mvt").dt.epoch("s").alias("S"),
                   C("LOBT_flt").is_null().alias("np"),
                   (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"), (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
                   (C("MVT_TIME_UTC_mvt") - C("EOBT_1_flt")).dt.total_seconds().alias("ge"), (C("MVT_TIME_UTC_mvt") - C("IOBT_flt")).dt.total_seconds().alias("gi"),
                   C("MVT_TIME_UTC_mvt").dt.month().alias("mon"))
           .collect().with_columns(num("FLIGHT_mvt").alias("fnum"), (C("T") - C("S")).cast(pl.Float64).alias("gs")))
    arr = (pl.scan_parquet(ranking).filter((C("PHASE_mvt") == "ARR") & (C("ADES_mvt") == "LIRF") & C("BLOCK_TIME_UTC_mvt").is_not_null())
           .select(OP.alias("op"), C("FLIGHT_mvt").alias("fl"), C("STAND_mvt").alias("stand"), C("BLOCK_TIME_UTC_mvt").dt.epoch("s").alias("tin"),
                   C("SCHED_TIME_UTC_mvt").dt.epoch("s").alias("ts"), C("LOBT_flt").is_null().alias("np")).collect()
           .with_columns(((C("tin") - C("ts")).abs() <= 60).alias("copy")))
    arr = arr.with_columns(num("fl").alias("fnum"))
    return dep, arr


def prep(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.Series("garb", [garble(x) for x in df["FLIGHT_mvt"].to_list()]),
                           pl.Series("dest_hc", [1.0 if x in HC_DEST else 0.0 for x in df["dest"].to_list()]),
                           ((C("S") % 300) == 0).cast(pl.Float64).alias("round5"), ((C("S") % 900) == 0).cast(pl.Float64).alias("round15"),
                           (C("S") // 3600 % 24).alias("sh"), ((C("S") // 86400 + 3) % 7).alias("dow"),
                           C("stand").str.slice(0, 1).cast(pl.Float64, strict=False).alias("apron"), C("stand").cast(pl.Float64, strict=False).alias("stand_num"),
                           C("np").cast(pl.Float64).alias("npf"))


def fit_and_predict(te: pl.DataFrame, seeds=range(3)) -> np.ndarray:
    """Round-2 classifier fitted on all twelve 2025 months (stored sensor table); P(copy) for the rows of `te`."""
    s = pl.read_parquet(I / "agent_rome2_sensor.parquet").filter((C("yr") == 2025) & (C("gs") >= 1800))
    dp = pl.read_parquet(I / "agent_rome2_depall.parquet").filter((C("ADEP_mvt") == "LIRF") & (C("yr") == 2025)).select("MVT_ID_mvt", C("ADES_mvt").alias("dest"))
    s = s.join(dp, on="MVT_ID_mvt", how="left").rename({"STAND_mvt": "stand"})
    s = prep(s.with_columns(C("copy_raw").cast(pl.Int32).alias("c")))
    n = s.height
    mo = s["mo"].to_numpy()
    c = s["c"].to_numpy()
    p0 = c.mean()

    def enc(col):
        v = s[col].to_list()
        u = sorted({x for x in v if x is not None})
        ix = {k: i for i, k in enumerate(u)}
        ii = np.array([ix.get(x, -1) for x in v])
        K, N = np.zeros((len(u), 13)), np.zeros((len(u), 13))
        for i in range(n):
            if ii[i] >= 0:
                K[ii[i], mo[i]] += c[i]
                N[ii[i], mo[i]] += 1
        return ix, ii, K, N

    def prior(K, N, ex):
        keep = [m for m in range(1, 13) if m not in ex]
        return (K[:, keep].sum(1) + 5 * p0) / (N[:, keep].sum(1) + 5), N[:, keep].sum(1)

    ixo, iio, Ko, No = enc("op")
    ixd, iid, Kd, Nd = enc("dest")
    pr, nn, pd_, nd = (np.zeros(n) for _ in range(4))
    for k in range(1, 13):
        sel = mo == k
        po, no = prior(Ko, No, {k})
        pde, nde = prior(Kd, Nd, {k})
        io, idd = iio[sel], iid[sel]
        pr[sel] = np.where(io >= 0, po[np.maximum(io, 0)], p0)
        nn[sel] = np.where(io >= 0, no[np.maximum(io, 0)], 0)
        pd_[sel] = np.where(idd >= 0, pde[np.maximum(idd, 0)], p0)
        nd[sel] = np.where(idd >= 0, nde[np.maximum(idd, 0)], 0)
    Xtr = np.column_stack([s[x].cast(pl.Float64).to_numpy() for x in COLS] + [pr, np.log1p(nn)] + [s[x].cast(pl.Float64).to_numpy() for x in ["garb", "dest_hc"]] + [pd_, np.log1p(nd)])
    po, no = prior(Ko, No, set())
    pde, nde = prior(Kd, Nd, set())
    io = np.array([ixo.get(x, -1) for x in te["op"].to_list()])
    idd = np.array([ixd.get(x, -1) for x in te["dest"].to_list()])
    Xte = np.column_stack([te[x].cast(pl.Float64).to_numpy() for x in COLS] + [np.where(io >= 0, po[np.maximum(io, 0)], p0), np.log1p(np.where(io >= 0, no[np.maximum(io, 0)], 0))]
                          + [te[x].cast(pl.Float64).to_numpy() for x in ["garb", "dest_hc"]] + [np.where(idd >= 0, pde[np.maximum(idd, 0)], p0), np.log1p(np.where(idd >= 0, nde[np.maximum(idd, 0)], 0))])
    p = np.zeros(te.height)
    for sd in seeds:
        q = {**PO, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd}
        p += lgb.train(q, lgb.Dataset(Xtr, c), 250).predict(Xte) / len(list(seeds))
    return p, float(c[s["np"].to_numpy()].mean())   # m_ref: copy rate of informative no-plan rows, as in validation (b)


def rome_noplan_mixture(ranking: Path, pred: pl.DataFrame, form: str = "scaled", months: list[int] | None = None, seeds=range(3)) -> tuple[pl.DataFrame, dict]:
    """Apply the mixture; returns the new prediction table and a small report."""
    if form == "off":
        return pred, dict(rows=0)
    default = form == "default"
    if default:
        form = "flat"
    spec = importlib.util.spec_from_file_location("mod_30", HERE / "30_general_rules.py")
    g30 = importlib.util.module_from_spec(spec)
    sys.modules["mod_30"] = g30
    spec.loader.exec_module(g30)
    dep, arr = rome_frames(ranking)
    if months:
        dep = dep.filter(C("mon").is_in(months))
    d = prep(sensor_features(dep, arr))
    inf = d.filter(C("np") & (C("gs") >= 1800))
    P, m_ref = fit_and_predict(inf, seeds)
    inf = inf.with_columns(pl.Series("P", P), (C("T") // 86400).alias("day"))
    md = inf.group_by("day").agg(C("P").mean().alias("m_d"))
    inf = inf.join(md, on="day", how="left")
    ctx = g30.Ctx(ranking, months, None, with_hidden=True)
    hid = set(ctx.hidden["id"].to_list())
    fp = inf.filter((C("gs") >= 3600) & (C("gs") < 14400) & ~C("id").is_in(list(hid)))
    if default:
        fp = fp.filter(C("mon") != 2)   # February off
    cur = pred.select(C(ID_COL).alias("id"), C(TARGET_COL).cast(pl.Float64).alias("a"))
    fp = fp.join(cur, on="id", how="inner")
    gs, Pf, a = fp["gs"].to_numpy(), fp["P"].to_numpy(), fp["a"].to_numpy()
    w = np.where(gs < 7200, 0.5, 0.85)
    sc = np.minimum(1.0, fp["m_d"].to_numpy() / m_ref) if form == "scaled" else 1.0
    new = np.maximum(np.rint(a + w * sc * (Pf * gs + (1 - Pf) * M0 - a)), 0)
    upd = pl.DataFrame({ID_COL: fp["id"], "_n": new})
    out = pred.join(upd, on=ID_COL, how="left").with_columns(pl.coalesce(C("_n"), C(TARGET_COL).cast(pl.Float64)).round(0).cast(pred.schema[TARGET_COL]).alias(TARGET_COL)).drop("_n")
    by = fp.with_columns(pl.Series("moved", new != a)).group_by("mon").agg(pl.len().alias("footprint"), C("moved").sum().alias("moved")).sort("mon")
    return out, dict(rows=int((new != a).sum()), footprint=fp.height, by_month=by.rows(), m_ref=m_ref, mean_P=float(Pf.mean()) if fp.height else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--form", choices=["default", "flat", "scaled", "off"], default="default")
    ap.add_argument("--months", default="")
    ap.add_argument("--sample", type=float, default=None, help="plumbing: a random fraction of the predictions")
    a = ap.parse_args()
    months = [int(t) for t in a.months.split(",") if t.strip()] or None
    pred = pl.read_parquet(a.pred)
    if a.sample:
        pred = pred.sample(fraction=a.sample, seed=0)
    out, rep = rome_noplan_mixture(a.ranking, pred, a.form, months, seeds=range(1) if a.sample else range(3))
    print("rome no-plan mixture:", rep)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(a.out)


if __name__ == "__main__":
    main()
