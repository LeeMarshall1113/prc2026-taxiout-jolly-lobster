"""arith3 step 5 (experiment arith3-serve, 2026-10-06): the SERVE side of the 45 round-three columns (docs/LAB-NOTES.md section 1).
The serve rows are appended to the fold rows with a month code of their own (SERVE_MON) that is not in the pool, and the unchanged agent_arith3_02_cols.build3 is called on the joint arrays:
fold rows get every cell statistic over all pool months with their own month left out, serve rows get it over ALL pool months looked up by their own key, exactly as a held-out month's rows got it.
Group Y (label-free day / window block) is computed over the fold rows and over the serve rows separately. No 2026 label is read anywhere (serve rows carry a constant 0 label that no statistic reads).
usage: python agent_arith3_05_servecols.py time          -> data/interim/agent_arith3_time_serve.npz (take-off time of the scored rows; id and time columns of ranking.parquet only)
       python agent_arith3_05_servecols.py selftest H    -> month H of 2025 treated as the serve set; compared column by column with build3 holding H out (the fit's way). exit 1 on failure
       python agent_arith3_05_servecols.py cols          -> the 45 columns of the 2026 rows (data/interim/agent_arith3_serve_cols.npz), direct recomputation of sample columns, 2026-vs-2025 table
Library for agent_arith3_06_fit_serve.py: load_serve(), build_pair()."""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, "archive")
import agent_arith3_02_cols as C3   # noqa: E402
import agent_bigc_te as TE   # noqa: E402

I = os.environ.get("ARITH_I", "data/interim/")
C3.I = I
SERVE_MON = 99
ALL = list("KYQGSR")
TSERVE = "agent_arith3_time_serve.npz"
SCOLS = "agent_arith3_serve_cols.npz"
t0 = time.time()


def P(*a):
    print("[%5.0fs]" % (time.time() - t0), *a, flush=True)


def _times(files, ids):
    """take-off time (MVT_TIME_UTC_mvt, epoch seconds) of the departure rows of the raw files, in the order of ids. Only the id, phase and time columns are read."""
    import polars as pl
    d = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]) for f in files]).filter(pl.col("PHASE_mvt") == "DEP")
    assert d["MVT_ID_mvt"].n_unique() == d.height, "duplicate departure ids"
    d = d.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"))
    m = dict(zip(d["MVT_ID_mvt"].to_list(), d["t"].to_list()))
    return np.array([m.get(v, -1) if m.get(v, -1) is not None else -1 for v in ids.tolist()], np.int64)


def make_time():
    import glob
    si = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True)
    ids = si["ids"].astype(float)
    t = _times(["data/raw/ranking.parquet"], ids)
    assert (t >= 0).all(), "scored rows without a take-off time: %d" % int((t < 0).sum())
    mo = (t.astype("datetime64[s]").astype("datetime64[M]").astype(int) % 12) + 1
    print("2026 rows", len(ids), "min", t.min(), "max", t.max(), "calendar month of t equals mon_s on share %.5f" % float((mo == si["mon_s"]).mean()))
    # the same code on the 2025 raw files must give the stored fit-side time array (agent_arith3_00_time.py)
    f = np.load(I + "agent_arith3_time.npz")
    t25 = _times(sorted(glob.glob("data/raw/training_2025-*.parquet")), f["ids"])
    same = np.array_equal(t25, f["t"])
    print("2025 time rebuilt with the serve-side code equals agent_arith3_time.npz: %s (rows %d, differing %d)" % (same, len(t25), int((t25 != f["t"]).sum())))
    print("precision: share of rows with a non-zero seconds part: 2025 %.4f, 2026 %.4f; distinct seconds values 2025 %d, 2026 %d" % (
        float((f["t"] % 60 != 0).mean()), float((t % 60 != 0).mean()), len(np.unique(f["t"] % 60)), len(np.unique(t % 60))))
    assert same, "serve-side time code does not reproduce the fit-side time array"
    np.savez(I + TSERVE, t=t, ids=ids)


def serve_dict(pv, apt, hour, rwy, adep, wkend, K, Dp, vf, t):
    """serve-side arrays in the types load_inputs() gives the fold arrays"""
    f32 = lambda a: np.asarray(a, np.float32).copy()  # noqa: E731
    D = dict(pv=f32(pv), apt=np.array([str(a) for a in apt]), hour=f32(hour), RUNWAY_mvt=f32(rwy), ADEP_mvt=f32(adep), is_weekend=f32(wkend),
             Kf=np.asarray(K, np.int64), Dpcd=f32(Dp), vf={n: f32(vf[n]) for n in C3.VPOINT + C3.VQ}, t=np.asarray(t, np.int64))
    n = len(D["pv"])
    assert all(len(D[k]) == n for k in ("apt", "hour", "RUNWAY_mvt", "ADEP_mvt", "is_weekend", "Kf", "Dpcd", "t")) and all(len(v) == n for v in D["vf"].values())
    assert D["Dpcd"].shape[1] == 9 and D["Kf"].shape[1] == 16
    return D


def load_serve():
    """the 2026 scored rows (serve_in row order): inputs only"""
    names = json.load(open(I + "agent_yearfp_names.json"))["names"]
    NI = {n: i for i, n in enumerate(names)}
    si = np.load(I + os.environ.get("ARITH_SERVE", "agent_bigc13_serve_in.npz"), allow_pickle=True)
    ids = si["ids"].astype(float)
    colsS = [str(c) for c in si["cols_pre"]]
    Xs = si["Xs_pre"]
    kk = np.load(I + "agent_bigc13_keys2.npz", allow_pickle=True)
    C = np.load(I + os.environ.get("ARITH_PCD", "agent_pcd_cols_app.npz"), allow_pickle=True)
    assert np.array_equal(C["ids_s"].astype(float), ids), "pcd serve columns are not in serve_in row order"
    vs = np.load(I + "agent_bigc12_vfac_clean5.npz", allow_pickle=True)["Xs"]
    tz = np.load(I + TSERVE)
    assert np.array_equal(tz["ids"], ids), "serve time file is not in serve_in row order"
    D = serve_dict(si["pv26"], si["apt"], Xs[:, colsS.index("hour")], Xs[:, colsS.index("RUNWAY_mvt")], Xs[:, colsS.index("ADEP_mvt")], Xs[:, colsS.index("is_weekend")],
                   kk["Ks"], C["Ds"], {n: vs[:, NI["vf_d_" + n] - 318] for n in C3.VPOINT + C3.VQ}, tz["t"])
    D["ids"] = ids
    D["mon_s"] = si["mon_s"]
    D["is9"] = si["is9"].astype(bool)
    assert len(ids) == 344841
    return D


class _SdOnly:
    """stand-in for the (N,112) key statistics: build3 reads one column only (sd of key st_rwy)"""
    def __init__(self, sd, col):
        self.sd, self.col = sd, col

    def __getitem__(self, key):
        rows, c = key
        assert c == self.col, (c, self.col)
        return self.sd[rows]


def bounds(blk, nfit):
    """winsorising bounds as agent_arith3_02_fit.winsor: 0.5 / 99.5 percentiles of every seventh FIT row"""
    sub = np.arange(nfit)[::7]
    return np.nanpercentile(blk[sub], [0.5, 99.5], axis=0)


def build_pair(Df, Ds, sdf, sds, dayf, days, groups=ALL):
    """Df: fold dict (load_inputs layout, with y and mon); Ds: serve dict (serve_dict layout, no label). sdf / sds: sd of key st_rwy from the corrector's key statistics (fit rows leave-own-month-out, serve rows from all fit months).
    dayf / days: day blocks computed over each row set separately. Returns raw (blkF, blkS), names, (lo, hi) of the fit rows."""
    nf, ns = len(Df["pv"]), len(Ds["pv"])
    assert SERVE_MON not in set(Df["mon"].tolist())
    D = {k: np.concatenate([Df[k], Ds[k]]) for k in ("hour", "RUNWAY_mvt", "ADEP_mvt", "is_weekend", "pv", "apt", "Kf", "Dpcd", "t")}
    D["vf"] = {n: np.concatenate([Df["vf"][n], Ds["vf"][n]]) for n in C3.VPOINT + C3.VQ}
    D["y"] = np.concatenate([np.asarray(Df["y"], np.float64), np.zeros(ns)])     # serve rows: constant placeholder, never read (not pool rows)
    D["mon"] = np.concatenate([np.asarray(Df["mon"], np.int64), np.full(ns, SERVE_MON, np.int64)])
    D["KN"] = Df["KN"]
    KI = {k: i for i, k in enumerate(D["KN"])}
    use = np.arange(nf + ns) < nf
    months = sorted(set(D["mon"].tolist()))
    pool_mask = np.array([m != SERVE_MON for m in months])
    col = KI["st_rwy"] * 7 + 2
    blk, names = C3.build3(D, groups, use, pool_mask, _SdOnly(np.concatenate([sdf, sds]), col), KI, day=np.concatenate([dayf, days]))
    assert blk.shape == (nf + ns, sum(len(C3.GROUPS3[g]) for g in groups))
    lo, hi = bounds(blk, nf)
    return blk[:nf], blk[nf:], names, (lo, hi)


def fold_extras(D3):
    """residual target, airport label means and key-statistics object inputs, as agent_arith_02_fit builds them"""
    y, pv = D3["y"], D3["pv"]
    r = np.clip(y - pv, -900, 900)
    k0 = np.load(I + "agent_bigc12_keys.npz", allow_pickle=True)
    apt_f = k0["apt_f"]
    ymu_apt = {a: y[apt_f == a].mean() for a in set(apt_f.tolist())}
    ymu_f = np.array([ymu_apt[a] for a in apt_f])
    ymu_s = np.array([ymu_apt.get(v, y.mean()) for v in k0["apt_s"]])
    return r, ymu_f, ymu_s


def cmp(a, b):
    """max abs difference over rows finite in both, count of rows where exactly one is NaN"""
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    return (float(np.abs(a[both].astype(np.float64) - b[both]).max()) if both.any() else 0.0), int((fa != fb).sum())


def selftest(h):
    D3 = C3.load_inputs()
    DAY = C3.day_block(D3)
    mon = D3["mon"]
    months = sorted(set(mon.tolist()))
    KI = {k: i for i, k in enumerate(D3["KN"])}
    col = KI["st_rwy"] * 7 + 2
    r, ymu_f, _ = fold_extras(D3)
    y, Kf = D3["y"], D3["Kf"]
    # --- reference: the held-out fit's own construction (agent_arith3_02_fit.lomo3)
    use = mon != h
    pool_mask = np.array([m != h for m in months])
    s112 = TE.Te2(Kf).fold(r, y, ymu_f, mon, h, months)
    ref, names = C3.build3(D3, ALL, use, pool_mask, s112, KI, day=DAY)
    sd_ref = s112[:, col].copy()
    del s112
    rlo, rhi = np.nanpercentile(ref[np.where(use)[0][::7]], [0.5, 99.5], axis=0)
    P("reference built", ref.shape)
    # --- serve path: month h as if it were the scored set, everything else the fold
    fi, hi_ = np.where(mon != h)[0], np.where(mon == h)[0]
    Df = {k: D3[k][fi] for k in ("hour", "RUNWAY_mvt", "ADEP_mvt", "is_weekend", "pv", "apt", "Kf", "Dpcd", "t", "y", "mon")}
    Df["vf"] = {n: v[fi] for n, v in D3["vf"].items()}
    Df["KN"] = D3["KN"]
    Ds = serve_dict(D3["pv"][hi_], D3["apt"][hi_], D3["hour"][hi_], D3["RUNWAY_mvt"][hi_], D3["ADEP_mvt"][hi_], D3["is_weekend"][hi_], Kf[hi_], D3["Dpcd"][hi_],
                    {n: v[hi_] for n, v in D3["vf"].items()}, D3["t"][hi_])
    fit_m = [m for m in months if m != h]
    T2 = TE.Te2(Kf[fi])
    sdf = T2.fold(r[fi], y[fi], ymu_f[fi], mon[fi], None, fit_m)[:, col].copy()
    sds = T2.serve(r[fi], y[fi], mon[fi], fit_m, Kf[hi_], ymu_f[hi_])[:, col].copy()
    d1, d2 = cmp(sdf, sd_ref[fi])[0], cmp(sds, sd_ref[hi_])[0]
    P("key statistic (SR label sd) serve path vs held-month path: max abs diff fit rows %.3g, held rows %.3g" % (d1, d2))
    days = C3.day_block(Ds)                       # the serve analogue: the scored rows alone
    bF, bS, nm2, (lo, hi) = build_pair(Df, Ds, sdf, sds, DAY[fi], days)
    ok = nm2 == names and bF.shape == (len(fi), 45) and bS.shape == (len(hi_), 45) and d1 < 1e-3 and d2 < 1e-3
    # edge rows for group Y: rows whose UTC day or +-1 h window holds a row of another month at the same airport
    t, apt = D3["t"], D3["apt"]
    ac = np.unique(apt, return_inverse=True)[1].astype(np.int64)
    dk = ac * 100000 + t // 86400
    edge = np.isin(dk[hi_], np.unique(dk[fi]))
    key = ac.astype(np.float64) * 1e10 + t
    ko = np.sort(key[fi])
    p = np.searchsorted(ko, key[hi_])
    near = np.minimum(np.abs(ko[np.clip(p, 0, len(ko) - 1)] - key[hi_]), np.abs(ko[np.clip(p - 1, 0, len(ko) - 1)] - key[hi_])) <= 3600
    edge |= near
    P("held month %d: %d rows, %d fit rows; Y edge rows (share a day or a +-1 h window with another month): %d" % (h, len(hi_), len(fi), int(edge.sum())))
    print("column | held rows: max abs diff, NaN mismatches | fit rows: max abs diff, NaN mismatches | winsor bound diff (lo, hi) | after winsorising, held rows")
    TOL = 1e-3
    for j, n in enumerate(names):
        isY = n.startswith("Y_")
        rows = ~edge if isY else np.ones(len(hi_), bool)
        dh, nh = cmp(bS[rows, j], ref[hi_[rows], j])
        df_, nf_ = cmp(bF[:, j], ref[fi, j])
        bl, bh = abs(float(lo[j] - rlo[j])), abs(float(hi[j] - rhi[j]))
        dw, _ = cmp(np.clip(bS[rows, j], lo[j], hi[j]), np.clip(ref[hi_[rows], j], rlo[j], rhi[j]))
        good = dh < TOL and nh == 0 and df_ < TOL and nf_ == 0 and bl < TOL and bh < TOL and dw < TOL
        ok &= good
        extra = ""
        if isY:
            de, ne = cmp(bS[edge, j], ref[hi_[edge], j])
            extra = " | edge rows (%d): max diff %.4g, NaN mismatches %d, rows differing by > 1e-3: %d" % (int(edge.sum()), de, ne, int((np.abs(bS[edge, j] - ref[hi_[edge], j]) > TOL).sum()))
        print("%-18s | %.3g, %d | %.3g, %d | %.3g, %.3g | %.3g%s%s" % (n, dh, nh, df_, nf_, bl, bh, dw, extra, "" if good else "   <-- FAIL"))
    print("SELFTEST month %d: %s" % (h, "PASS" if ok else "FAIL"), flush=True)
    return ok


def direct_checks(D3, Ds, bS, names):
    """independent recomputation (pandas group-bys on the raw keys) of sample columns for the real 2026 rows"""
    import pandas as pd
    KN = D3["KN"]
    nf = len(D3["pv"])
    res = []
    pvf, pvs = D3["pv"].astype(np.float64), Ds["pv"].astype(np.float64)

    def look(keyf, keys, val):
        g = pd.DataFrame({"k": keyf, "v": val}).dropna().groupby("k")["v"].agg(["mean", "count"])
        m = pd.Series(keys).map(g["mean"]).to_numpy()
        c = pd.Series(keys).map(g["count"]).fillna(0).to_numpy()
        return np.where(c > 20, m, np.nan), c
    oa_f, oa_s = D3["Kf"][:, KN.index("op_apt")], Ds["Kf"][:, KN.index("op_apt")]
    m, c = look(oa_f, oa_s, pvf)
    res.append(("K_OA_dev", pvs - m))
    res.append(("R_OA_logn", np.log1p(c)))
    m, _ = look(oa_f, oa_s, D3["Dpcd"][:, 1].astype(np.float64))
    res.append(("S_dLOBT_OA", Ds["Dpcd"][:, 1] - m))
    tr_f, tr_s = D3["Kf"][:, KN.index("ty_rwy")], Ds["Kf"][:, KN.index("ty_rwy")]
    m, c = look(tr_f, tr_s, pvf)
    res.append(("K_TR_dev", pvs - m))
    res.append(("R_TR_logn", np.log1p(c)))
    kf_ = ["%s|%s|%s|%s" % q for q in zip(D3["apt"], np.nan_to_num(D3["RUNWAY_mvt"], nan=-1).astype(int), np.nan_to_num(D3["hour"], nan=-1).astype(int), np.nan_to_num(D3["is_weekend"], nan=-1).astype(int))]
    ks_ = ["%s|%s|%s|%s" % q for q in zip(Ds["apt"], np.nan_to_num(Ds["RUNWAY_mvt"], nan=-1).astype(int), np.nan_to_num(Ds["hour"], nan=-1).astype(int), np.nan_to_num(Ds["is_weekend"], nan=-1).astype(int))]
    m, _ = look(np.array(kf_), np.array(ks_), pvf)
    res.append(("K_RHW_dev", pvs - m))
    d = pd.DataFrame({"a": Ds["apt"], "d": Ds["t"] // 86400, "pv": pvs})
    g = d.groupby(["a", "d"])["pv"]
    n1 = g.transform("count").to_numpy() - 1
    res.append(("Y_day_dev", np.where(n1 >= 5, pvs - (g.transform("sum").to_numpy() - pvs) / np.maximum(n1, 1), np.nan)))
    # label median of the stand x runway cell, 300 sampled rows, histogram rule written out again
    sr_f, sr_s = D3["Kf"][:, KN.index("st_rwy")], Ds["Kf"][:, KN.index("st_rwy")]
    o = np.argsort(sr_f, kind="stable")
    srs = sr_f[o]
    rng = np.random.default_rng(0)
    samp = rng.choice(np.where(Ds["is9"])[0], 300, replace=False)
    q50 = np.full(len(pvs), np.nan)
    for i in samp:
        a, b = np.searchsorted(srs, sr_s[i], "left"), np.searchsorted(srs, sr_s[i], "right")
        yy = D3["y"][o[a:b]]
        if len(yy) <= 20:
            continue
        H = np.bincount(np.minimum((np.clip(yy, 0, None) / 20.0).astype(int), 75), minlength=76).astype(float)
        cs = np.cumsum(H)
        tg = 0.5 * len(yy)
        j = min(int((cs < tg).sum()), 75)
        q50[i] = (j + min(max((tg - (cs[j - 1] if j else 0.0)) / max(H[j], 1e-9), 0), 1)) * 20.0
    v = np.full(len(pvs), np.nan)
    v[samp] = pvs[samp] - q50[samp]
    res.append(("Q_SR_p50", v))
    ok = True
    print("direct recomputation on the 2026 rows (independent group-bys): column | max abs diff | NaN mismatches | rows compared")
    for n, v in res:
        j = names.index(n)
        sel = np.arange(len(pvs)) if n != "Q_SR_p50" else samp
        dmax, nn = cmp(bS[sel, j], np.asarray(v, np.float64)[sel])
        good = dmax < 2e-2 and nn == 0     # float32 storage of values up to a few thousand seconds
        ok &= good
        print("  %-12s | %.4g | %d | %d%s" % (n, dmax, nn, int(np.isfinite(bS[sel, j]).sum()), "" if good else "   <-- FAIL"))
    assert nf == len(D3["mon"])
    return ok


def auc(a, b):
    import pandas as pd
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return np.nan
    rk = pd.Series(np.concatenate([a, b])).rank().values
    v = (rk[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return max(v, 1 - v)


def table(D3, Ds, bF, bS, names, lo, hi):
    """2026 against 2025, per column: missing rate and 1 / 50 / 99 percentiles per month; flags; per airport for the worst five"""
    import pandas as pd
    mon, apt = D3["mon"], D3["apt"]
    ms, as_, is9 = Ds["mon_s"], Ds["apt"], Ds["is9"]
    months = sorted(set(mon.tolist()))
    rows, per_apt = [], []
    for j, n in enumerate(names):
        f, s = bF[:, j].astype(np.float64), bS[:, j].astype(np.float64)
        st = {m: (float(np.isnan(f[mon == m]).mean()),) + tuple(np.nanpercentile(f[mon == m], (1, 50, 99))) for m in months}
        A = np.array([st[m] for m in months])
        mn, mx = A.min(0), A.max(0)
        scale = max(float(np.nanpercentile(f[::5], 99) - np.nanpercentile(f[::5], 1)), 1e-9)
        out = {"col": n}
        flag = []
        for m in (1, 7):
            ss = s[(ms == m) & is9]
            v = (float(np.isnan(ss).mean()),) + tuple(np.nanpercentile(ss, (1, 50, 99)))
            num = den = 0.0
            for a in sorted(set(apt.tolist())):
                q = auc(f[(mon == m) & (apt == a)], s[(ms == m) & is9 & (as_ == a)])
                w = int(((ms == m) & is9 & (as_ == a)).sum())
                if np.isfinite(q):
                    num += q * w
                    den += w
                    per_apt.append(dict(col=n, apt=a, m=m, auc=q, nan25=float(np.isnan(f[(mon == m) & (apt == a)]).mean()), nan26=float(np.isnan(s[(ms == m) & is9 & (as_ == a)]).mean()),
                                        **{"p%d_25" % k: float(x) for k, x in zip((1, 50, 99), np.nanpercentile(f[(mon == m) & (apt == a)], (1, 50, 99)))},
                                        **{"p%d_26" % k: float(x) for k, x in zip((1, 50, 99), np.nanpercentile(s[(ms == m) & is9 & (as_ == a)], (1, 50, 99)))}))
            out["auc_m%d" % m] = num / den if den else np.nan
            for k, nm_ in enumerate(("nan", "p1", "p50", "p99")):
                out["%s_26m%d" % (nm_, m)] = v[k]
                out["%s_25m%d" % (nm_, m)] = st[m][k]
            if v[0] < mn[0] - 0.03 or v[0] > mx[0] + 0.03:
                flag.append("m%d missing rate outside every 2025 month" % m)
            for k, nm_ in ((1, "p1"), (2, "p50"), (3, "p99")):
                if v[k] < mn[k] - 0.15 * scale or v[k] > mx[k] + 0.15 * scale:
                    flag.append("m%d %s outside the 2025 monthly range" % (m, nm_))
            if out["auc_m%d" % m] > 0.60:
                flag.append("m%d auc %.2f" % (m, out["auc_m%d" % m]))
        for k, nm_ in enumerate(("nan", "p1", "p50", "p99")):
            out["%s_25min" % nm_], out["%s_25max" % nm_] = mn[k], mx[k]
        sv = s[np.isfinite(s) & is9]
        out["out_range_26"] = float(((sv < lo[j]) | (sv > hi[j])).mean()) if len(sv) else np.nan
        if out["out_range_26"] > 0.03:
            flag.append("share outside the winsorising bounds %.3f" % out["out_range_26"])
        out["flag_text"] = "; ".join(flag)
        rows.append(out)
    d = pd.DataFrame(rows)
    d.to_csv(I + "agent_arith3_drift.csv", index=False)
    pa = pd.DataFrame(per_apt)
    pa.to_csv(I + "agent_arith3_drift_airport.csv", index=False)
    pd.set_option("display.width", 320, "display.max_rows", 500, "display.max_columns", 60)
    show = ["col", "auc_m1", "auc_m7", "nan_25min", "nan_25max", "nan_26m1", "nan_26m7", "p1_25m1", "p1_26m1", "p50_25m1", "p50_26m1", "p99_25m1", "p99_26m1", "p1_25m7", "p1_26m7", "p50_25m7", "p50_26m7", "p99_25m7", "p99_26m7", "out_range_26"]
    print(d[show].round(3).to_string())
    print("\nFLAGS (2026 outside every 2025 month by the margins in the note, or auc > 0.60, or > 3% outside the winsorising bounds):")
    fl = d[d.flag_text != ""]
    print(fl[["col", "flag_text"]].to_string() if len(fl) else "  none")
    d["worst"] = d[["auc_m1", "auc_m7"]].max(1)
    w5 = d.sort_values("worst", ascending=False).col.tolist()[:5]
    print("\nworst five columns by same-month within-airport auc, by airport:")
    print(pa[pa.col.isin(w5)].round(3).sort_values(["col", "m", "apt"]).to_string())


def cols():
    D3 = C3.load_inputs()
    Ds = load_serve()
    r, ymu_f, ymu_s = fold_extras(D3)
    mon = D3["mon"]
    months = sorted(set(mon.tolist()))
    KI = {k: i for i, k in enumerate(D3["KN"])}
    col = KI["st_rwy"] * 7 + 2
    T2 = TE.Te2(D3["Kf"])
    sdf = T2.fold(r, D3["y"], ymu_f, mon, None, months)[:, col].copy()
    sds = T2.serve(r, D3["y"], mon, months, Ds["Kf"], ymu_s)[:, col].copy()
    del T2
    dayf, days = C3.day_block(D3), C3.day_block(Ds)
    bF, bS, names, (lo, hi) = build_pair(D3, Ds, sdf, sds, dayf, days)
    P("built", bF.shape, bS.shape)
    np.savez(I + SCOLS, S=bS, Sw=np.clip(bS, lo, hi), names=np.array(names), ids=Ds["ids"], lo=lo, hi=hi, sds=sds)
    ok = direct_checks(D3, Ds, bS, names)
    print("DIRECT CHECK:", "PASS" if ok else "FAIL", flush=True)
    table(D3, Ds, bF, bS, names, lo, hi)
    return ok


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "time":
        make_time()
    elif cmd == "selftest":
        sys.exit(0 if all([selftest(int(h)) for h in sys.argv[2].split(",")]) else 1)
    elif cmd == "cols":
        sys.exit(0 if cols() else 1)
