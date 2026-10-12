"""lv 07: stand-handover columns for the Istanbul corrector (docs/LAB-NOTES.md; idea + clip from docs/LAB-NOTES.md).
Per departure row (t0 = take-off - REF = predicted off-block; nb = in-block time of ARRIVALS at LTFM, labelled records, 2025 raw for the fold rows and ranking.parquet for the 2026 serve rows):
  r_next   = clip(take-off - nb_next, -7200, 7200), nb_next = first in-block at the SAME stand >= t0 - 900; NaN if none within 2 h after take-off   (signed seconds)
  r_bind   = clip(REF - (take-off - nb_next + 300), -1800, 1800)   (>0 where the 'off-block must precede next in-block - 300 s' clip would bind)
  r_gap    = clip(nb_next - t0, -900, 7200)
  r_cnt    = arrivals in-block at the stand in [SCHED, take-off]
  a_c1,a_c2 = arrivals in-block at the stand AREA (leading letters) in [t0-1800, t0-600) and [t0-600, take-off]
Writes data/interim/agent_lv_handover.npz: fold (rows of agent_lv_keys12_full.npz), serve (rows of agent_bndl_rank_ltfm order) + names. Also prints where the plain clip's gain sits (snow flag / month).  No 2026 labels (arrival records in ranking.parquet are the labelled arrival fields, as the arrivals experiment)."""
import glob
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "archive")
I = "data/interim/"; ID = "MVT_ID_mvt"; C = pl.col
ep = lambda c: C(c).dt.epoch("s").cast(pl.Int64)
area = lambda c: C(c).str.extract(r"^([A-Za-z]+)", 1).fill_null(C(c).str.slice(0, 1))
cols = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "RUNWAY_mvt"]


def load(files):
    return pl.concat([pl.read_parquet(f, columns=cols).filter((C("ADEP_mvt") == "LTFM") | (C("ADES_mvt") == "LTFM")) for f in files])


def arrivals(d):
    a = (d.filter(C("PHASE_mvt") == "ARR").select(C("STAND_mvt").alias("stand"), ep("MVT_TIME_UTC_mvt").alias("land"), ep("BLOCK_TIME_UTC_mvt").alias("inb"),
         C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("taxi")).drop_nulls(["land", "inb", "taxi", "stand"])
         .filter((C("taxi") > 30) & (C("taxi") < 3600) & (C("inb") > C("land"))))
    return a.with_columns(area("stand").alias("area"))


def feats(dep, arr, ref):
    """dep: DataFrame rows aligned with ref (np): mvt, sched, stand (epoch ints / str). returns (n,6) float, names"""
    n = dep.height
    mvt = dep["mvt"].to_numpy().astype(float); sch = dep["sched"].to_numpy().astype(float); st = dep["stand"].to_numpy(); t0 = mvt - ref
    out = np.full((n, 6), np.nan)
    by_st = {}
    for (k,), g in arr.group_by(["stand"]):
        by_st[k] = np.sort(g["inb"].to_numpy())
    by_ar = {}
    for (k,), g in arr.group_by(["area"]):
        by_ar[k] = np.sort(g["inb"].to_numpy())
    areas = dep["area"].to_numpy()
    for k in np.unique(st[st != None]):                       # noqa: E711
        i = np.flatnonzero(st == k); nb = by_st.get(k)
        if nb is None or len(nb) == 0:
            out[i, 3] = 0; continue
        j = np.searchsorted(nb, t0[i] - 900, side="left"); has = j < len(nb); nbn = np.where(has, nb[np.minimum(j, len(nb) - 1)], np.nan)
        ok = has & (nbn <= mvt[i] + 7200)
        out[i, 0] = np.where(ok, np.clip(mvt[i] - nbn, -7200, 7200), np.nan)
        out[i, 1] = np.where(ok, np.clip(ref[i] - (mvt[i] - nbn + 300), -1800, 1800), np.nan)
        out[i, 2] = np.where(ok, np.clip(nbn - t0[i], -900, 7200), np.nan)
        out[i, 3] = np.searchsorted(nb, mvt[i], side="right") - np.searchsorted(nb, sch[i], side="left")
    for k in np.unique(areas[areas != None]):                 # noqa: E711
        i = np.flatnonzero(areas == k); nb = by_ar.get(k)
        if nb is None:
            out[i, 4] = 0; out[i, 5] = 0; continue
        out[i, 4] = np.searchsorted(nb, t0[i] - 600, side="left") - np.searchsorted(nb, t0[i] - 1800, side="left")
        out[i, 5] = np.searchsorted(nb, mvt[i], side="right") - np.searchsorted(nb, t0[i] - 600, side="left")
    return out, ["r_next", "r_bind", "r_gap", "r_cnt", "a_c1", "a_c2"]


if __name__ == "__main__":
    d25 = load(sorted(glob.glob("data/raw/training_2025-*.parquet")))
    arr25 = arrivals(d25)
    z = np.load(I + "agent_lv_keys12_full.npz", allow_pickle=True)
    f = pl.DataFrame({ID: z["fid"], "y": z["yf"], "ref": z["ref"], "mon": z["monf"].astype(np.int64)}).with_row_index("row")
    dep = d25.filter(C("PHASE_mvt") == "DEP").select(ID, ep("MVT_TIME_UTC_mvt").alias("mvt"), ep("SCHED_TIME_UTC_mvt").alias("sched"), C("STAND_mvt").alias("stand"))
    f = f.join(dep, on=ID, how="left").sort("row").with_columns(area("stand").alias("area"))
    assert f.height == len(z["fid"])
    miss = f["mvt"].is_null().to_numpy() | f["stand"].is_null().to_numpy() | f["sched"].is_null().to_numpy()
    print("fold rows", f.height, "missing mvt/stand/sched", int(miss.sum()))
    f = f.with_columns(C("mvt").fill_null(0), C("sched").fill_null(0))
    ref = f["ref"].to_numpy(); y = f["y"].to_numpy()
    X, names = feats(f, arr25, ref)
    X[miss] = np.nan
    for j, nme in enumerate(names):
        print(f"  {nme:8s} fold cov {np.isfinite(X[:, j]).mean():.3f} mean {np.nanmean(X[:, j]):9.1f}")
    # serve rows: ranking.parquet, same row order as agent_bndl_rank_ltfm
    rk = pl.read_parquet("data/raw/ranking.parquet", columns=cols)
    rk = rk.filter((C("ADEP_mvt") == "LTFM") | (C("ADES_mvt") == "LTFM"))
    arr26 = arrivals(rk)
    rl = pl.read_parquet(I + "agent_bndl_rank_ltfm.parquet").select(ID)
    sref = pl.read_parquet(I + "agent_combo_v129_noM4_final.parquet").select(ID, C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("ref"))
    sd = rl.join(rk.filter(C("PHASE_mvt") == "DEP").select(ID, ep("MVT_TIME_UTC_mvt").alias("mvt"), ep("SCHED_TIME_UTC_mvt").alias("sched"), C("STAND_mvt").alias("stand")), on=ID, how="left", maintain_order="left") \
        .join(sref, on=ID, how="left", maintain_order="left").with_columns(area("stand").alias("area"))
    smiss = sd["mvt"].is_null().to_numpy() | sd["stand"].is_null().to_numpy() | sd["sched"].is_null().to_numpy()
    sd = sd.with_columns(C("mvt").fill_null(0), C("sched").fill_null(0))
    XS, _ = feats(sd, arr26, sd["ref"].to_numpy()); XS[smiss] = np.nan
    print("serve rows", sd.height, "missing", int(smiss.sum()), "arrivals 2025", arr25.height, "2026", arr26.height)
    for j, nme in enumerate(names):
        print(f"  {nme:8s} serve cov {np.isfinite(XS[:, j]).mean():.3f} mean {np.nanmean(XS[:, j]):9.1f} | fold mean {np.nanmean(X[:, j]):9.1f} sd {np.nanstd(X[:, j]):8.1f} serve sd {np.nanstd(XS[:, j]):8.1f}")
    np.savez(I + "agent_lv_handover.npz", fold=X.astype(np.float32), serve=XS.astype(np.float32), names=np.array(names), fold_ids=z["fid"], serve_ids=sd[ID].to_numpy())
    # where does the plain clip (w 900, buf 300) gain sit? snow flag from train_features
    import polars as pl2
    T = pl.read_parquet(I + "train_features.parquet", columns=[ID, "wx_snow", "wx_freezing"])
    sn = pl.DataFrame({ID: z["fid"]}).join(T, on=ID, how="left", maintain_order="left")
    snow = ((sn["wx_snow"] == 1) | (sn["wx_freezing"] == 1)).fill_null(False).to_numpy()
    bind = np.isfinite(X[:, 1]) & (X[:, 1] < 0) & (X[:, 0] > 0)      # nb before take-off and lower bound above REF
    pc = np.where(bind, ref - X[:, 1], ref)                           # max(REF, take-off - nb + 300)
    gain = (y - pc) ** 2 - (y - ref) ** 2
    mon = z["monf"]
    ok = ~miss
    print("clip binds", int(bind[ok].sum()), "| dSSE total", float(gain[ok].sum()), "| on snow rows", float(gain[ok & snow].sum()), "(binds", int(bind[ok & snow].sum()), ") | non-snow", float(gain[ok & ~snow].sum()), "(binds", int(bind[ok & ~snow].sum()), ")")
    for m in sorted(set(mon.tolist())):
        s = ok & (mon == m); print(f"  month {m:2d} n {s.sum():6d} binds {int(bind[s].sum()):5d} snow rows {int(snow[s].sum()):5d} dSSE {float(gain[s].sum()):12.0f} of which on snow {float(gain[s & snow].sum()):12.0f}")
