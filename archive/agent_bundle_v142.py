"""v142 = v141 + low-risk pieces (2026-10-04).
 A  passengers: certain bounds re-applied last, Rome plan-gate extra weight 0.1, Istanbul corrector dose 1.0
    (agent_mega_no4_final - agent_basefix_final; docs/LAB-NOTES.md)
 B  Rome plan-gate extra weight 0.2 on July unseen with-plan rows (agent_regime_gatejul_final - v141; docs/LAB-NOTES.md).
    Where B moves a row, A's delta on that row is dropped (no double plan-gate dose).
 C  hidden-plan upper bound at seven airports, 6 rows (agent_detrules_final - v136; docs/LAB-NOTES.md); only rows v141 left unchanged from v136.
 D  corrector serve term from 12 seeds instead of 6 (agent_mega_refit12_delta - agent_mega_refit_delta; docs/LAB-NOTES.md)
Float deltas are summed on v141, rounded once, clipped at zero."""
import numpy as np, polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
rd = lambda s: pl.read_parquet(I + s + ".parquet").sort(ID)  # noqa: E731
v141 = rd("agent_bundle_v141_final"); ids = v141[ID].to_numpy(); cur = v141[T].cast(pl.Float64).to_numpy()
def col(s):
    d = rd(s); assert (d[ID].to_numpy() == ids).all(), s; return d[T].cast(pl.Float64).to_numpy()
dA = col("agent_mega_no4_final") - col("agent_basefix_final")
dB = col("agent_regime_gatejul_final") - cur
v136 = col("agent_bundle_v136_final"); dC = col("agent_detrules_final") - v136; dC[cur != v136] = 0.0
z6, z12 = np.load(I + "agent_mega_refit_delta.npz"), np.load(I + "agent_mega_refit12_delta.npz")
assert (z6["ids"] == ids).all() and (z12["ids"] == ids).all(); dD = z12["dR"] - z6["dR"]
dA[dB != 0] = 0.0
for n, d in (("A passengers", dA), ("B gate July", dB), ("C hidden-plan bound", dC), ("D 12 seeds", dD)):
    m = d != 0; print(f"{n}: rows {int(m.sum())}, mean {d[m].mean() if m.any() else 0:+.2f}, rms {np.sqrt((d[m]**2).mean()) if m.any() else 0:.2f}, max |d| {np.abs(d).max():.1f}")
new = np.rint(np.maximum(cur + dA + dB + dC + dD, 0))
out = v141.with_columns(pl.Series(T, new).cast(v141[T].dtype))
print("rows changed vs v141:", int((new != cur).sum()), "mean shift %+.3f s" % (new - cur).mean(), "rms %.2f" % np.sqrt(((new - cur) ** 2).mean()), "max |move| %.0f" % np.abs(new - cur).max())
out.write_parquet(I + "agent_bundle_v142_final.parquet")
orig = pl.read_parquet("submissions/jolly-lobster_v141.parquet")
sub = orig.select(ID).join(out, on=ID, how="left"); assert sub[T].null_count() == 0 and (sub[ID] == orig[ID]).all()
sub.select(orig.columns).cast(dict(orig.schema)).write_parquet("submissions/jolly-lobster_v142.parquet"); print("written v142")
