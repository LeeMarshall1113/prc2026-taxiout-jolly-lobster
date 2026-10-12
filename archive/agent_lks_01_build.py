"""lks step 1: LTFM key table (fold rows = TX_S6 LTFM rows in months 1,2,3,7,8,9,10 with the v113 analogue delta; serve rows = 2026 LTFM DEP in ranking.parquet),
unseen-key audit on 2026, key-stat leak test. Writes data/interim/agent_lks_keys.npz. No 2026 labels are read (ranking.parquet has none)."""
import sys, glob, re, numpy as np, polars as pl
sys.path.insert(0, "archive"); import agent_lks_te as TE
C = pl.col; I = "data/interim/"; ID = "MVT_ID_mvt"; MONTHS = [1, 2, 3, 7, 8, 9, 10]
z = np.load(I + "modal/TX_S6.npz", allow_pickle=True); ids = z["ids"].astype("float64"); y = z["y"]; p0 = z["p0"]; apts = z["apts"]
T = pl.read_parquet(I + "train_features.parquet", columns=[ID, "month"])
mon = pl.DataFrame({ID: ids, "i": np.arange(len(ids))}).join(T, on=ID, how="left").sort("i")["month"].to_numpy()
dl = {}
for f in glob.glob("results/preds_ltfmseq_treat_f*.parquet"):
    a = pl.read_parquet(f).select(ID, pl.col("bagged").alias("t")); b = pl.read_parquet(f.replace("treat", "ctrl")).select(ID, pl.col("bagged").alias("c"))
    j = a.join(b, on=ID).with_columns((C("t") - C("c")).alias("d"))
    for k, v in zip(j[ID].to_numpy(), j["d"].to_numpy()): dl[k] = v
L = np.where((apts == "LTFM") & np.isin(mon, MONTHS))[0]
d = np.array([dl.get(k, np.nan) for k in ids[L]]); ok = np.isfinite(d); L = L[ok]; d = d[ok]
fid = ids[L]; yf = y[L]; ref = p0[L] + d; monf = mon[L]; print("fold rows", len(L), "months", np.unique(monf))
cols = [ID, "CALLSIGN_flt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "PHASE_mvt"]
s = lambda c: C(c).cast(pl.Utf8).fill_null("NA")
def prep(df):
    cs = s("CALLSIGN_flt"); st = s("STAND_mvt"); hb = (C("MVT_TIME_UTC_mvt").dt.hour() // 3).cast(pl.Utf8)
    pre = cs.str.extract(r"^([A-Z]{3})", 1).fill_null("NA"); num = cs.str.extract(r"^[A-Z]{3}(\d+)$", 1)
    csfn = pl.when(num.is_not_null()).then(pre + pl.lit("#") + (num.cast(pl.Int64, strict=False) // 100).cast(pl.Utf8)).otherwise(pre + pl.lit("@") + cs.str.len_chars().cast(pl.Utf8))
    area = st.str.extract(r"^([A-Z]+)", 1).fill_null(st.str.slice(0, 1) + pl.lit("n") + st.str.len_chars().cast(pl.Utf8))
    pier = st.str.extract(r"^([A-Z]?\d)", 1).fill_null(st.str.slice(0, 2))
    op, ty, rw, ad = s("AIRCRAFT_OPERATOR_flt"), s("AIRCRAFT_TYPE_mvt"), s("RUNWAY_mvt"), s("ADES_mvt")
    ex = {"op": [op], "csp": [pre], "csfn": [csfn], "ades": [ad], "cty": [ad.str.slice(0, 2)], "reg": [ad.str.slice(0, 1)], "stand": [st], "area": [area], "pier": [pier],
          "type": [ty], "wk": [s("WK_TBL_CAT_flt")], "mkt": [s("MARKET_SEGMENT_flt")], "rwy_stand": [rw, st], "rwy_area": [rw, area], "op_hb": [op, hb], "ty_stand": [ty, st],
          "op_stand": [op, st], "op_rwy": [op, rw], "rwy_hb": [rw, hb], "op_ty": [op, ty]}
    return df.select([C(ID)] + [pl.concat_str(v, separator="|").hash(seed=7).floordiv(2).cast(pl.Int64).alias(k) for k, v in ex.items()]), list(ex)
fr = []; idf = pl.DataFrame({ID: fid})
for f in sorted(glob.glob("data/raw/training_2025-*.parquet")):
    dd = pl.read_parquet(f, columns=cols).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")).join(idf, on=ID, how="semi"); fr.append(prep(dd)[0])
kf = pl.concat(fr).unique(ID); names = prep(pl.read_parquet("data/raw/ranking.parquet", columns=cols, n_rows=5))[1]
j = idf.with_row_index("_r").join(kf, on=ID, how="left").sort("_r"); Kf = j.select(names).to_numpy().astype(np.int64); assert j.height == len(fid) and not j[names[0]].is_null().any()
rk = pl.read_parquet("data/raw/ranking.parquet", columns=cols).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")); ks, _ = prep(rk); sid = ks[ID].to_numpy(); Ks = ks.select(names).to_numpy().astype(np.int64)
print("serve rows", len(sid))
seen = {n: float(np.isin(Ks[:, i], Kf[:, i]).mean()) for i, n in enumerate(names)}
unseen = {n: round(1 - v, 4) for n, v in seen.items()}; print("unseen share on 2026 LTFM rows:"); [print(" ", n, unseen[n], "nunique_fold", len(np.unique(Kf[:, i]))) for i, n in enumerate(names)]
keep = [i for i, n in enumerate(names) if unseen[n] <= 0.20]; kn = [names[i] for i in keep]; print("kept", kn, "dropped", [n for n in names if n not in kn])
Kall = np.vstack([Kf, Ks])[:, keep]; Nf = len(fid)
# leak test: perturbing the HELD month's labels must not change that month's (or any fold row's) features; perturbing a train month j must not change month j rows
ks_ = TE.KS(Kall); months = np.array(MONTHS); mi = np.searchsorted(months, monf); rf = np.arange(Nf)
rng = np.random.default_rng(0); res = yf - ref; worst = 0.0
for tgt, v in (("y", yf), ("r", res)):
    for h in (1, 8):
        hi = MONTHS.index(h); A = ks_.build(None, rf, tgt, v, mi, len(MONTHS), hi)
        v2 = v.copy(); v2[mi == hi] = rng.permutation(v[mi == hi]) * 3 + 500; B = ks_.build(None, rf, tgt, v2, mi, len(MONTHS), hi)
        dd_ = np.abs(A - B); print("rows changed by month", {int(m): int((dd_[mi==i].max(1)>0).sum()) for i,m in enumerate(MONTHS)}); print(tgt, h, "held-perturb", float(dd_.max()), "col", int(np.unravel_index(dd_.argmax(), dd_.shape)[1])); worst = max(worst, float(dd_.max()))              # held month perturbed: nothing at all may change
        for jm in (2, 9):
            ji = MONTHS.index(jm); v3 = v.copy(); v3[mi == ji] = rng.permutation(v[mi == ji]) * 3 + 500; Cc = ks_.build(None, rf, tgt, v3, mi, len(MONTHS), hi)
            dd_ = np.abs(A[mi == ji] - Cc[mi == ji]); print(tgt, h, jm, "train-perturb", float(dd_.max()), "col", int(np.unravel_index(dd_.argmax(), dd_.shape)[1])); worst = max(worst, float(dd_.max()))   # train month j perturbed: month-j rows unchanged
print("LEAK TEST max abs change (must be 0):", worst); assert worst == 0.0
np.savez(I + "agent_lks_keys.npz", fid=fid, sid=sid, yf=yf, ref=ref, monf=monf, Kall=Kall, kn=np.array(kn), unseen=np.array([unseen[n] for n in names]), allnames=np.array(names))
