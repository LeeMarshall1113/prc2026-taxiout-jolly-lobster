"""experiment sb2 calib (docs/LAB-NOTES.md, C2): (airport, hour, day-of-week) out-of-fold residual
calibration, two-level shrinkage ((a,h) toward 0, then (a,h,d) toward (a,h)), kappa = sigma_w^2 / tau^2 by method of
moments on the rows it is fitted on. Only the 9 non-LTFM airports are touched.
"""
import numpy as np
import polars as pl

C = pl.col
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]


def _level(df, keys, target):
    """per-cell mean of `target`, n, and the MoM shrink factor s = n / (n + kappa)."""
    g = df.group_by(keys).agg(C(target).mean().alias("m"), pl.len().alias("n"))
    j = df.join(g, on=keys, how="left")
    sw2 = float(((j[target] - j["m"]) ** 2).mean())
    m, n = g["m"].to_numpy(), g["n"].to_numpy().astype(float)
    tau2 = max(float(np.mean(m ** 2) - np.mean(sw2 / n)), 1e-6 * sw2)
    kappa = sw2 / tau2
    return g.with_columns(pl.Series("s", n / (n + kappa))), kappa


def fit_table(e, apt, hour, dow):
    """fit on residuals e (rows at the 9 airports only). Returns a frame (apt, hour, dow, c) and an (apt, hour) fallback."""
    df = pl.DataFrame({"apt": apt, "hour": hour, "dow": dow, "e": e})
    ah, k1 = _level(df, ["apt", "hour"], "e")
    ah = ah.with_columns((C("s") * C("m")).alias("m_ah")).select("apt", "hour", "m_ah", C("m").alias("raw_ah"))
    d2 = df.join(ah, on=["apt", "hour"], how="left").with_columns((C("e") - C("raw_ah")).alias("dev"))
    ahd, k2 = _level(d2, ["apt", "hour", "dow"], "dev")
    tab = (ahd.join(ah, on=["apt", "hour"], how="left")
           .with_columns((C("m_ah") + C("s") * (C("m") + C("raw_ah") - C("m_ah"))).alias("c"))
           .select("apt", "hour", "dow", "c"))
    return tab, ah.select("apt", "hour", "m_ah"), (k1, k2)


def apply_table(tab, ah, apt, hour, dow):
    q = pl.DataFrame({"apt": apt, "hour": hour, "dow": dow}).with_row_index("_i")
    q = (q.join(tab, on=["apt", "hour", "dow"], how="left").join(ah, on=["apt", "hour"], how="left").sort("_i")
         .select(pl.coalesce(C("c"), C("m_ah"), pl.lit(0.0))))
    return q.to_series().to_numpy().astype(float)


def oof(P, y, apts, days, hour, dow):
    """OOF-calibrated composite over the re-stack's 10 day-folds (sorted days, i % 10); LTFM / other rows untouched."""
    ud = sorted(set(days))
    fmap = {d: i % 10 for i, d in enumerate(ud)}
    fo = np.array([fmap[d] for d in days])
    at9 = np.isin(apts, AIRPORTS) & (hour >= 0)
    e = y - P
    c = np.zeros(len(P))
    kap = []
    for k in range(10):
        tr, te = at9 & (fo != k), at9 & (fo == k)
        tab, ah, kk = fit_table(e[tr], apts[tr], hour[tr], dow[tr])
        kap.append(kk)
        c[te] = apply_table(tab, ah, apts[te], hour[te], dow[te])
    return np.maximum(P + c, 0.0), c, kap
