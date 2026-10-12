"""ablation audit step 7: serve-side build of the components that failed the audit (here: the neighbour blend), on the v138 file.

Reconstructions that must hold before anything is changed (asserted):
  (a) submissions v112 - v111 == the blend delta D = agent_blend_pre_ltfm - v112_backx at every non-LTFM row (the delta is exactly what was uploaded);
  (b) v132 == rint(max(lv4r + 0.5 x tail ehat, 0)) on all rows (the tail step is exactly additive);
  (c) v138 differs from v132 on 336 rows only (the Rome / duplicate / cap rules), so every other row of v138 is the additive v132 chain.
Change: rows not touched by the later rules: out = rint(max(v138 - D, 0)). The 336 rule rows keep their v138 value (their anchor shift would be at most
(1 - w) x D, and D is zero on most of them).
Writes data/interim/agent_ablate_final.parquet and prints the component table.
"""
import subprocess
import sys

import numpy as np
import polars as pl

I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ld = lambda p: pl.read_parquet(p).sort(ID)  # noqa: E731
g = lambda p: ld(p)[T].cast(pl.Float64).to_numpy()  # noqa: E731
v138f = ld(I + "agent_bundle_v138_final.parquet")
ids = v138f[ID].to_numpy()
v138, v132 = v138f[T].cast(pl.Float64).to_numpy(), g(I + "agent_bundle_v132_final.parquet")
apt = np.load(I + "agent_tailc_serve_ehat.npz", allow_pickle=True)["apt"].astype(str)
b0, b1 = g(I + "v112_backx_final.parquet"), g(I + "agent_blend_pre_ltfm.parquet")
D = b1 - b0
# (a)
v111, v112 = g("submissions/jolly-lobster_v111.parquet"), g("submissions/jolly-lobster_v112.parquet")
nl = apt != "LTFM"
assert np.array_equal((v112 - v111)[nl], D[nl]), "blend delta is not what was uploaded"
print("(a) v112 - v111 == D on all", int(nl.sum()), "non-LTFM rows: OK; rows moved", int((D != 0).sum()))
# (b)
lv4r = g(I + "agent_lv4r_ltfm_final.parquet"); ts = np.load(I + "agent_tailc_serve_ehat.npz", allow_pickle=True)
assert np.array_equal(np.rint(np.maximum(lv4r + 0.5 * ts["ehat"], 0)), v132)
print("(b) v132 == rint(max(lv4r + 0.5 ehat, 0)) on all rows: OK")
# (c)
rule = v138 != v132
print("(c) v138 vs v132 rows differing:", int(rule.sum()), " of which D != 0:", int((rule & (D != 0)).sum()))

out = v138.copy()
sel = (D != 0) & ~rule
out[sel] = np.rint(np.maximum(v138[sel] - D[sel], 0))
res = v138f.with_columns(pl.Series(T, out).cast(v138f.schema[T]))
res.write_parquet(I + "agent_ablate_final.parquet")
d = out - v138
print("\nTABLE  component changed | rows moved | rms move (all rows) | rms move (moved rows) | mean shift")
mv = d != 0
print(f"  remove neighbour blend (v112 dose blend) | {int(mv.sum()):,} | {float(np.sqrt((d**2).mean())):.2f} | {float(np.sqrt((d[mv]**2).mean())):.2f} | {float(d.mean()):+.3f}")
print("  by airport (rows moved, rms moved, mean):")
for a in sorted(set(apt)):
    m = mv & (apt == a)
    if m.any():
        print(f"    {a}: {int(m.sum()):,} rows, rms {float(np.sqrt((d[m]**2).mean())):.2f}, mean {float(d[m].mean()):+.2f}")
r = subprocess.run([sys.executable, "archive/agent_smallbundle_06_check.py", I + "agent_ablate_final.parquet", I + "agent_bundle_v138_final.parquet"], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
print(r.stdout[-2500:], r.stderr[-500:])
