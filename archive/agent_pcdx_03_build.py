"""pcdx step 3: candidate = v143 with the pcd corrector at EHAM (dose 1.25, 12 serve seeds) instead of the old shipped term.
Assembly: float sum on v142 exactly as agent_bundle_v143.py (v142 + dE + dF), plus dX = (1.25 * mean pcd serve term - kF_old) on EHAM rows that no later rule set (nr = is9 & ~RR), rounded once.
Assertion first: with dX = 0 the assembly reproduces data/interim/agent_bundle_v143_final.parquet on every row. Writes data/interim/agent_pcdx_final.parquet. Existing files are read only."""
import glob
import numpy as np, polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
rd = lambda s: pl.read_parquet(I + s + ".parquet").sort(ID)  # noqa: E731
v142 = rd("agent_bundle_v142_final"); ids = v142[ID].to_numpy(); cur = v142[T].cast(pl.Float64).to_numpy()
v143 = rd("agent_bundle_v143_final"); assert (v143[ID].to_numpy() == ids).all(); t143 = v143[T].cast(pl.Float64).to_numpy()
def col(s):
    d = rd(s); assert (d[ID].to_numpy() == ids).all(), s; return d[T].cast(pl.Float64).to_numpy()
zp, zs = np.load(I + "agent_pcd_refit12_delta.npz"), np.load(I + "agent_mega_refit12_delta.npz")
assert (zp["ids"] == ids).all() and (zs["ids"] == ids).all(); dE = zp["dR"] - zs["dR"]
dF = col("agent_rome1c_final") - col("agent_bundle_v141_final")
rep = np.rint(np.maximum(cur + dE + dF, 0))
assert (rep == t143).all(), int((rep != t143).sum())
print("reproduction: v142 + dE + dF == v143 on all %d rows (assert passed)" % len(ids))
sb = np.load(I + "agent_bigc13_serve_base.npz", allow_pickle=True); si = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True)
assert (sb["ids"] == ids).all() and (si["ids"] == ids).all()
apt = si["apt"].astype(str); is9, RR, kold = sb["is9"], sb["RR"], sb["kF_old"]
fs = sorted(glob.glob(I + "agent_pcd_modal/serve/C4ff7V5_s*.npz")); assert len(fs) == 12
P = np.mean([np.load(f, allow_pickle=True)["pred"] for f in fs], 0); assert P.shape == ids.shape and np.isfinite(P).all()
nr = is9 & ~RR; m = nr & (apt == "EHAM")
# what v143 serves at EHAM: the old term (dE is zero there)
assert np.abs(dE[apt == "EHAM"]).max() == 0 and np.abs(dF[apt == "EHAM"]).max() == 0
knew = 1.25 * P
dX = np.where(m, knew - kold, 0.0)
print("EHAM rows with the term replaced: %d; corr(new term, old term) %.3f; mean shift %+.2f s; rms move %.2f; max |move| %.1f" % (
    int(m.sum()), np.corrcoef(knew[m], kold[m])[0, 1], dX[m].mean(), np.sqrt((dX[m] ** 2).mean()), np.abs(dX).max()))
assert np.corrcoef(knew[m], kold[m])[0, 1] >= 0.7 and abs(dX[m].mean()) <= 5 and np.abs(dX).max() <= 600
assert (dX[~m] == 0).all()
new = np.rint(np.maximum(cur + dE + dF + dX, 0))
print("rows changed vs v143: %d; mean shift %+.3f s; rms move %.2f; max |move| %.0f; non-EHAM rows changed %d" % (
    int((new != t143).sum()), (new - t143).mean(), np.sqrt(((new - t143) ** 2).mean()), np.abs(new - t143).max(), int(((new != t143) & (apt != "EHAM")).sum())))
out = v143.with_columns(pl.Series(T, new).cast(v143[T].dtype))
out.write_parquet(I + "agent_pcdx_final.parquet"); print("written", I + "agent_pcdx_final.parquet")
import collections
for a in sorted(set(apt.tolist())):
    mm = apt == a; print("  %s rows moved %d of %d" % (a, int((new != t143)[mm].sum()), int(mm.sum())))
