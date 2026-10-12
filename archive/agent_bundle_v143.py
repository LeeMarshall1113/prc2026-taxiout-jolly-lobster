"""v143 = v142 + two pieces (2026-10-05 UTC).
 E  residual corrector refitted with nine plan-clock distance columns (prediction minus the taxi implied by each plan clock; window and clamp distances),
    12 serve seeds, same seven airports and doses as v141/v142: delta = agent_pcd_refit12_delta - agent_mega_refit12_delta (docs/LAB-NOTES.md).
 F  Rome with-plan two-sided copy mixture, clean build, dose 0.5 (agent_rome1c_final - v141; docs/LAB-NOTES.md section 7).
Float deltas are summed on v142, rounded once, clipped at zero."""
import numpy as np, polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
rd = lambda s: pl.read_parquet(I + s + ".parquet").sort(ID)  # noqa: E731
v142 = rd("agent_bundle_v142_final"); ids = v142[ID].to_numpy(); cur = v142[T].cast(pl.Float64).to_numpy()
def col(s):
    d = rd(s); assert (d[ID].to_numpy() == ids).all(), s; return d[T].cast(pl.Float64).to_numpy()
zp, zs = np.load(I + "agent_pcd_refit12_delta.npz"), np.load(I + "agent_mega_refit12_delta.npz")
assert (zp["ids"] == ids).all() and (zs["ids"] == ids).all(); dE = zp["dR"] - zs["dR"]
dF = col("agent_rome1c_final") - col("agent_bundle_v141_final")
for n, d in (("E plan-clock corrector", dE), ("F Rome with-plan mixture", dF)):
    m = d != 0; print(f"{n}: rows {int(m.sum())}, mean {d[m].mean():+.2f}, rms {np.sqrt((d[m]**2).mean()):.2f}, max |d| {np.abs(d).max():.1f}")
print("rows moved by both:", int(((dE != 0) & (dF != 0)).sum()), "corr of the two deltas there %.3f" % np.corrcoef(dE[(dE != 0) & (dF != 0)], dF[(dE != 0) & (dF != 0)])[0, 1])
new = np.rint(np.maximum(cur + dE + dF, 0))
out = v142.with_columns(pl.Series(T, new).cast(v142[T].dtype))
print("rows changed vs v142:", int((new != cur).sum()), "mean shift %+.3f s" % (new - cur).mean(), "rms %.2f" % np.sqrt(((new - cur) ** 2).mean()), "max |move| %.0f" % np.abs(new - cur).max())
out.write_parquet(I + "agent_bundle_v143_final.parquet")
orig = pl.read_parquet("submissions/jolly-lobster_v142.parquet")
sub = orig.select(ID).join(out, on=ID, how="left"); assert sub[T].null_count() == 0 and (sub[ID] == orig[ID]).all()
sub.select(orig.columns).cast(dict(orig.schema)).write_parquet("submissions/jolly-lobster_v143.parquet"); print("written v143")
