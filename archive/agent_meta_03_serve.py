"""experiment meta 03: serving twin of arm R k=0.25 (per apt x class NNLS on d_NEW, d_ET fitted on ALL fold months). Base = agent_bndl_board_final. NEAR-PASS board test, see docs/LAB-NOTES.md."""
import numpy as np, polars as pl
from scipy.optimize import nnls
C = pl.col; I = "data/interim/"; ID = "MVT_ID_mvt"; T = "TAXITIME_SEC_mvt"
AIR = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]; K = 0.25
CATS = ["matched","points_no_liftoff","no_adsb_points","other"]
L = lambda f: np.load(I + f, allow_pickle=True)
sa = L("agent_bulkr_fold_SA_s0.npz"); keep = np.isin(sa["apts"].astype(str), AIR); mb = L("agent_bndl_members.npz"); ids = sa["ids"][keep]
y = sa["y"][keep].astype(float); p = sa["both_guard"][keep].astype(float); a = sa["apts"][keep].astype(str); q = p + mb["M1"] + mb["M2"] + mb["M3"]; r = y - q
new = L("agent_bulkr_fold_NEW_s0.npz")["both_guard"][keep]; ctrl = L("agent_etrs_fold_CTRL_s0.npz")["both_guard"][keep]; et = L("agent_etrs_fold_ET_s0.npz")["both_guard"][keep]
M = np.column_stack([new - p, et - ctrl])
def cls(c):
    cat = np.where(np.isin(c["category"].to_numpy().astype(str), CATS[:3]), c["category"].to_numpy().astype(str), "other")
    return np.array([CATS.index(v) for v in cat]) * 2 + c["plan"].to_numpy().astype(int)
c25 = pl.read_parquet(I + "agent_coverage_census_2025.parquet").sort(ID); pos = np.searchsorted(c25[ID].to_numpy(), ids); cl = cls(c25)[pos]
cell = np.array([AIR.index(v) for v in a]) * 8 + cl; W = {}
for c in np.unique(cell):
    m = cell == c
    if m.sum() >= 300: W[c], _ = nnls(M[m], r[m])
print("cells fitted", len(W), "of", len(np.unique(cell)))
for ap_ in AIR:
    ws = np.array([W[c] for c in W if c // 8 == AIR.index(ap_)]); print(ap_, "mean w_NEW %.3f w_ET %.3f" % tuple(ws.mean(0)) if len(ws) else "none")
# serving
base = pl.read_parquet(I + "agent_bndl_board_final.parquet").sort(ID)
c26 = pl.read_parquet(I + "agent_coverage_census_2026.parquet").sort(ID); assert np.array_equal(c26[ID].to_numpy(), base[ID].to_numpy())
a26 = c26["apt"].to_numpy().astype(str); cell26 = np.array([AIR.index(v) if v in AIR else -1 for v in a26]) * 8 + cls(c26)
PARTS = {("EGLL",): "a", ("LEMD",): "b", ("EDDF","EDDM","EHAM","LEBL","LSZH"): "d", ("LFPG","LIRF"): "c"}
rd = lambda f: pl.read_parquet(I + f).sort(ID); dN = np.zeros(base.height); dE = np.zeros(base.height)
for aa, pc in PARTS.items():
    m = np.isin(a26, aa); bk, ref, lg, av = (rd(f) for f in (f"agent_bulkr_t_{pc}_final.parquet", f"agent_sb2_t_{pc}_final.parquet", f"agent_etrs_t_{pc}_final.parquet", f"agent_etrs_t_{pc}_avg_final.parquet"))
    for f in (bk, ref, lg, av): assert f[ID].equals(base[ID])
    dN[m] = (bk[T].to_numpy().astype(float) - ref[T].to_numpy().astype(float))[m]; dE[m] = (av[T].to_numpy().astype(float) - lg[T].to_numpy().astype(float))[m]
delta = np.zeros(base.height)
for c, w in W.items():
    m = cell26 == c; delta[m] = K * (w[0] * dN[m] + w[1] * dE[m])
out = np.maximum(base[T].to_numpy().astype(float) + delta, 0.0)
mv = delta != 0
print("rows moved %d (>0.5 s: %d); mean shift %+.3f (moved rows %+.3f); rms %.2f; max %.1f; sum dp2/row %.3f" % (mv.sum(), (np.abs(delta) > 0.5).sum(), delta.mean(), delta[mv].mean(), np.sqrt((delta ** 2).mean()), np.abs(delta).max(), (delta ** 2).sum() / len(delta)))
fd = K * (M * np.array([W.get(c, np.zeros(2)) for c in cell]).reshape(-1, 2)).sum(1); print("FOLD sum dp2/row %.3f rms %.2f -> serving/fold footprint ratio %.2f" % ((fd ** 2).sum() / len(fd), np.sqrt((fd ** 2).mean()), (delta ** 2).sum() / len(delta) / ((fd ** 2).sum() / len(fd))))
base.with_columns(pl.Series(T, np.round(out)).cast(base[T].dtype)).write_parquet(I + "agent_meta_v_cand.parquet"); print("-> agent_meta_v_cand.parquet")
