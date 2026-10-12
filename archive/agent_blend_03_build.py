"""experiment blend 03: build the serving blend (docs/LAB-NOTES.md). Only run if s0.7's verdict PASSES.

blend = base_serving + sum_i w_i * (member_i_serving - base_serving), per airport, using data/interim/agent_blend_weights_final.json
(the ALL-DATA per-airport NNLS weights from 02). base_serving = data/interim/v112_backx_final.parquet (the serving
counterpart of the fold-side v109 analogue). Airports with no weights entry (i.e. LTFM, out of scope for every one of
these ADS-B arms) keep the base value untouched -- all five servable members already agree with base there.
Writes data/interim/agent_blend_pre_ltfm.parquet (pre LTFM post-step).
"""
import json

import numpy as np
import polars as pl

C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
I = "data/interim/"
BASE_FILE = I + "v112_backx_final.parquet"
MEMBER_SERVING = {
    "RS_v2": I + "combo_T1_R_final.parquet",
    "CAB": I + "combo_T1_CE_final.parquet",
    "ctrl4": I + "v111_seqsplice_final.parquet",
    "nbrB": I + "v110_neighbour_final.parquet",
    "BXnest_s0": I + "v112_backx_final.parquet",  # == base; correction is identically 0, see docs/LAB-NOTES.md's finding
}

wj = json.load(open(I + "agent_blend_weights_final.json"))
weights = wj["weights"]  # {airport: {member: w}}
print("final per-airport weights:", json.dumps(weights, indent=1))

base = pl.read_parquet(BASE_FILE).sort(ID)
n = base.height
apt = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
       .select(C(ID).cast(pl.Float64), C("ADEP_mvt").alias("apt")).collect())
j = base.select(C(ID).cast(pl.Float64)).join(apt, on=ID, how="left", maintain_order="left")
assert j.height == n and j["apt"].null_count() == 0
apts = j["apt"].to_numpy()
p_base = base[T].cast(pl.Float64).to_numpy()
new = p_base.copy()

members_loaded = {}
for m, f in MEMBER_SERVING.items():
    d = pl.read_parquet(f).sort(ID)
    assert d[ID].equals(base[ID]), f"{m} serving id mismatch"
    members_loaded[m] = d[T].cast(pl.Float64).to_numpy()

for A, wmap in weights.items():
    mA = apts == A
    if not mA.any():
        continue
    corr = np.zeros(int(mA.sum()))
    for m, w in wmap.items():
        if w <= 0:
            continue
        corr = corr + w * (members_loaded[m][mA] - p_base[mA])
    new[mA] = p_base[mA] + corr
    print(f"  {A}: n={int(mA.sum()):,} mean correction {float(corr.mean()):+.3f}s rms {float(np.sqrt((corr**2).mean())):.3f}s")

n_no_weight_airports = int((~np.isin(apts, list(weights.keys()))).sum())
print(f"rows with no weight entry (kept at base, e.g. LTFM): {n_no_weight_airports:,}")

new = np.clip(np.rint(new), 0, None)
res = base.with_columns(pl.Series(T, new).cast(base[T].dtype))
res.write_parquet(I + "agent_blend_pre_ltfm.parquet")
d = new - p_base
print(f"wrote {I}agent_blend_pre_ltfm.parquet: rows moved vs base {int((d != 0).sum()):,}; "
      f"mean shift {float(d.mean()):+.4f}s; rms (moved) {float(np.sqrt((d[d != 0] ** 2).mean())) if (d != 0).any() else 0:.2f}s; "
      f"min {float(new.min()):.0f}; negatives {int((new < 0).sum())}")
print("DONE")
