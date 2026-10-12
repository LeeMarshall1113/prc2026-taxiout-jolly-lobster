"""lv 13: DAY-LEVEL arrival taxi-in state for the Istanbul corrector (docs/LAB-NOTES.md; idea: docs/LAB-NOTES.md, LTFM departure residual ICC 0.09, airport-day corr +0.38 with the day-mean arrival taxi-in residual).
Taxi-in model (archive/agent_lv_09_arrT.py design: runway, stand, operator, type, hour, dow -- no date features), refitted so that for every LOMO held month h the residual of an arrival in month m comes from a model trained on months NOT in {h, m} (m == h: not in {h}).
Months handled: HM = the shipped fold months 1,2,3,4,7,8,9,10 (held h and arrival months m both in HM).  36 fits.  2026 arrivals: model trained on all of 2025.
day_features(dep_t, dep_m, arr_t, arr_m, resid) -> (n, 9) float, names:
  y_dmean   shrunk airport-day mean resid  sum/(n+20) of the departure's local (UTC+3) day, ONLY arrivals of the same month
  y_dmed    day median
  y_dslow   share of the day's arrivals with resid > 180 s
  y_w3      mean resid of arrivals landing in (t-3h, t]
  y_w6      same for (t-6h, t]
  y_wc3     arrivals in [t-1.5h, t+1.5h]
  y_dn      number of arrivals that day
  y_prev    shrunk mean of the previous local day (NaN if that day is in another month)
  y_slowc3  share > 180 s in the centred 3 h window
Writes data/interim/agent_lv_dayarr.npz (arrival table, residual matrix res[h_idx, i], dep info of the keys8 fold rows and of the serve rows, serve residuals)."""
import glob
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "archive")
import agent_lv_09_arrT as A  # noqa: E402

I = "data/interim/"; ID = "MVT_ID_mvt"; C = pl.col
HM = [1, 2, 3, 4, 7, 8, 9, 10]
NAMES = ["y_dmean", "y_dmed", "y_dslow", "y_w3", "y_w6", "y_wc3", "y_dn", "y_prev", "y_slowc3"]


def day_features(dep_t, dep_m, arr_t, arr_m, res, K=20.0):
    n = len(dep_t); out = np.full((n, 9), np.nan)
    day_d = (dep_t + 3 * 3600) // 86400; day_a = (arr_t + 3 * 3600) // 86400
    for m in np.unique(dep_m):
        di = np.flatnonzero(dep_m == m); ai = np.flatnonzero(arr_m == m)
        if len(ai) == 0: continue
        o = np.argsort(arr_t[ai]); ai = ai[o]; ta, ra, da = arr_t[ai], res[ai], day_a[ai]
        cs = np.concatenate([[0.0], np.cumsum(ra)]); cslow = np.concatenate([[0.0], np.cumsum(ra > 180)])
        days = np.unique(da); stats = {}
        for d in days:
            s = da == d; stats[d] = (ra[s].sum(), s.sum(), np.median(ra[s]), (ra[s] > 180).mean())
        dd = day_d[di]; t = dep_t[di]
        for d in np.unique(dd):
            if d in stats:
                sm, c, md, sl = stats[d]; k = np.flatnonzero(dd == d); r = di[k]
                out[r, 0] = sm / (c + K); out[r, 1] = md; out[r, 2] = sl; out[r, 6] = c
            if d - 1 in stats:
                sm, c, _, _ = stats[d - 1]; out[di[np.flatnonzero(dd == d)], 7] = sm / (c + K)
        for col, lo_s, hi_s in ((3, -10800, 0), (4, -21600, 0), (5, -5400, 5400)):
            lo = np.searchsorted(ta, t + lo_s, side="right" if hi_s == 0 else "left"); hi = np.searchsorted(ta, t + hi_s, side="right"); c = hi - lo
            out[di, col] = np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)
            if col == 5: out[di, 8] = np.where(c > 0, (cslow[hi] - cslow[lo]) / np.maximum(c, 1), np.nan)
    return out


if __name__ == "__main__":
    d25 = A.load(sorted(glob.glob("data/raw/training_2025-*.parquet"))); a25 = A.arrivals(d25)
    am = a25["m"].to_numpy(); y = a25["y"].to_numpy(); at = a25["t"].to_numpy()
    nh = len(HM); res = np.full((nh, a25.height), np.nan, dtype=np.float32)
    cache = {}
    for hi, h in enumerate(HM):
        for m in HM:
            key = tuple(sorted({h, m}))
            if key not in cache:
                tr = ~np.isin(am, list(key)); g, lv = A.fit(a25.filter(pl.Series(tr))); cache[key] = (g, lv)
                print("fit excluding", key, flush=True)
            g, lv = cache[key]; sel = am == m
            res[hi, sel] = y[sel] - A.predict(g, lv, a25.filter(pl.Series(sel)))
    K8 = np.load(I + "agent_lv_keys8.npz", allow_pickle=True); fid = K8["fid"]
    dep = A.deps_cols(d25, fid)
    dm = pl.DataFrame({ID: fid}).join(d25.filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")).select(ID, C("MVT_TIME_UTC_mvt").dt.month().alias("m")), on=ID, how="left", maintain_order="left")["m"].to_numpy()
    dep_t = dep["t"].fill_null(0).to_numpy().astype(np.int64)
    # serve
    g, lv = A.fit(a25)
    r26 = pl.read_parquet("data/raw/ranking.parquet", columns=A.cols); r26 = r26.filter((C("ADEP_mvt") == "LTFM") | (C("ADES_mvt") == "LTFM")); a26 = A.arrivals(r26)
    res26 = (a26["y"].to_numpy() - A.predict(g, lv, a26)).astype(np.float32)
    rl = pl.read_parquet(I + "agent_bndl_rank_ltfm.parquet").select(ID)
    sdep = A.deps_cols(r26, rl[ID].to_numpy()); st = sdep["t"].fill_null(0).to_numpy().astype(np.int64)
    sdm = pl.DataFrame({ID: rl[ID]}).join(r26.filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")).select(ID, C("MVT_TIME_UTC_mvt").dt.month().alias("m")), on=ID, how="left", maintain_order="left")["m"].to_numpy()
    np.savez(I + "agent_lv_dayarr.npz", HM=np.array(HM), arr_t=at, arr_m=am, res=res, fold_ids=fid, dep_t=dep_t, dep_m=dm, a26_t=a26["t"].to_numpy(), a26_m=a26["m"].to_numpy(), res26=res26,
             serve_ids=rl[ID].to_numpy(), s_t=st, s_m=sdm)
    # sanity: day-level correlation of day-mean resid with the departure residual (own-month residual model)
    K = np.load(I + "agent_lv_keys8.npz", allow_pickle=True); ref, yy = K["ref"], K["yf"]; mon = K["monf"]
    print("done; arrivals", a25.height, "serve arrivals", a26.height)
