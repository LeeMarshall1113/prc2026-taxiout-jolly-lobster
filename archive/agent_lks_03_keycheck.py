"""lks step 3: the Modal path's key/stat pipeline (raw-only part of agent_lks_crossval/final) checked locally: keys identical to the screen's, list-held leak test, serve unseen share vs the 12-month pool."""
import sys, glob, numpy as np, polars as pl
sys.path.insert(0, "archive"); import agent_lks_te as TE, agent_lks_keys as KY
C = pl.col; I = "data/interim/"
raw = pl.concat([pl.scan_parquet(f).select(["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt"] + [c for c in KY.RAWCOLS if c != "MVT_ID_mvt"]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")).collect() for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
DEPS = raw.with_columns(_m=C("MVT_TIME_UTC_mvt").dt.month(), y=C("TAXITIME_SEC_mvt").cast(pl.Float64)); print("LTFM DEP 2025", DEPS.height, "y null", DEPS["y"].null_count(), "months", sorted(DEPS["_m"].unique().to_list()))
KEEP = ["op", "csp", "csfn", "ades", "cty", "reg", "stand", "area", "pier", "type", "wk", "mkt", "rwy_stand", "rwy_area", "op_hb", "ty_stand", "op_stand", "op_rwy", "rwy_hb", "op_ty"]
kd, kn = KY.prep(DEPS); assert kn == KEEP
K = np.load(I + "agent_lks_keys.npz", allow_pickle=True); fid = K["fid"]
j = pl.DataFrame({"MVT_ID_mvt": fid}).join(kd, on="MVT_ID_mvt", how="left"); same = (j.select(KEEP).to_numpy().astype(np.int64) == K["Kall"][:len(fid)]).all(); print("keys identical to screen build:", bool(same))
rk = pl.read_parquet("data/raw/ranking.parquet", columns=KY.RAWCOLS + ["ADEP_mvt", "PHASE_mvt"]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")); ks, _ = KY.prep(rk)
Kf = kd.select(KEEP).to_numpy().astype(np.int64); Ks = ks.select(KEEP).to_numpy().astype(np.int64); Kc = np.vstack([Kf, Ks]); n = len(Kf)
y = DEPS["y"].to_numpy(); mi = DEPS["_m"].to_numpy() - 1; eng = TE.KS(Kc); rows = np.arange(n)
print("unseen share of 2026 LTFM rows vs 12-month 2025 pool:", {k: round(1 - float(np.isin(Ks[:, i], Kf[:, i]).mean()), 4) for i, k in enumerate(KEEP) if 1 - float(np.isin(Ks[:, i], Kf[:, i]).mean()) > 0.005})
rng = np.random.default_rng(1); worst = 0.0
for held in ([0, 6], [1, 7], [2, 8]):
    A = eng.build(None, rows, "y", y, mi, 12, held); y2 = y.copy(); h_ = np.isin(mi, held); y2[h_] = rng.permutation(y[h_]) * 3 + 500
    worst = max(worst, float(np.abs(A - eng.build(None, rows, "y", y2, mi, 12, held)).max()))
    for jm in (3, 10):
        y3 = y.copy(); y3[mi == jm] = rng.permutation(y[mi == jm]) * 3 + 500; B = eng.build(None, rows, "y", y3, mi, 12, held); worst = max(worst, float(np.abs(A[mi == jm] - B[mi == jm]).max()))
tr, sv = eng.build(None, rows, "y", y, mi, 12, None, serve_rows=n + np.arange(len(Ks))); y3 = y.copy(); y3[mi == 4] = y[mi == 4][::-1] * 2
worst = max(worst, float(np.abs(tr[mi == 4] - eng.build(None, rows, "y", y3, mi, 12, None)[mi == 4]).max()))
print("LEAK TEST (list-held, final path) max abs change:", worst, "| train feature NaN", int(np.isnan(tr).sum()), "serve NaN", int(np.isnan(sv).sum()))
print("serve vs train mean of ks_y_op_mean / ks_y_stand_mean / ks_y_op_stand_cnt:", [(float(tr[:, c].mean()), float(sv[:, c].mean())) for c in (0, 6 * 6, 16 * 6 + 3)]); assert worst == 0.0
