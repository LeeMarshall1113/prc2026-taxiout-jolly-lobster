"""rejudge step 61: serve-side build of the extra gate weight on the base agent_ablate_final.parquet.
out = rint(max(base + w (mix26 - base), 0)) on LIRF plan rows with schedule gap <= 10,800 s; mix26 = pre-gate stage input + (gate-bag output - input) / 0.6
from v105_s0ship / v105_gatebag_s0 (the 20-seed served gate, blend weight 0.6, so the delta is exactly 0.6 (mix - input)).
Known step reconstructed first: input + 0.6 (mix26 - input) must equal the gate-bag file on every row (clip at 0 aside).
Also prints the footprint scaled to the fold: expected total-RMSE gain = fold (dSSE / move^2) x 2026 sum move^2.
usage: agent_rejudge_61_gate_build.py W OUT
"""
import sys
import numpy as np
import polars as pl
C = pl.col
I = "data/interim/"
w = float(sys.argv[1]); out = sys.argv[2]
c = lambda n: pl.read_parquet(I + n).select(C("MVT_ID_mvt").cast(pl.Float64).alias("id"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias(n))  # noqa: E731
base_df = pl.read_parquet(I + "agent_ablate_final.parquet")
s0 = c("v105_s0ship.parquet").join(c("v105_gatebag_s0.parquet"), on="id")
d0 = s0["v105_gatebag_s0.parquet"].to_numpy() - s0["v105_s0ship.parquet"].to_numpy()
rec = np.maximum(s0["v105_s0ship.parquet"].to_numpy() + 0.6 * (s0["v105_s0ship.parquet"].to_numpy() + d0 / 0.6 - s0["v105_s0ship.parquet"].to_numpy()), 0)
print("reconstruction of the served gate step: max |diff|", float(np.abs(rec - s0["v105_gatebag_s0.parquet"].to_numpy()).max()), "rows", s0.height)
rk = (pl.scan_parquet("data/raw/ranking.parquet").filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_not_null())
      .select(C("MVT_ID_mvt").cast(pl.Float64).alias("id"), (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap")).filter(C("gap") <= 10800).collect())
sc = set(rk["id"].to_list())
mixd = pl.DataFrame({"id": s0["id"], "mix": s0["v105_s0ship.parquet"].to_numpy() + d0 / 0.6})
ids = base_df["MVT_ID_mvt"].cast(pl.Float64).to_numpy()
j = pl.DataFrame({"id": ids}).join(mixd, on="id", how="left")
mix = j["mix"].to_numpy()
inscope = np.isin(ids, np.array(sorted(sc)))
assert np.isfinite(mix[inscope]).all()
old = base_df["TAXITIME_SEC_mvt"].to_numpy().astype(float)
new = old.copy()
new[inscope] = np.rint(np.maximum(old[inscope] + w * (mix[inscope] - old[inscope]), 0))
print(f"w {w}: scope rows {int(inscope.sum())}, rows changed {int((new != old).sum())}, moved > 60 s {int((np.abs(new-old)>60).sum())}, mean {np.mean((new-old)[inscope]):+.2f}, rms {np.sqrt(np.mean((new-old)[inscope]**2)):.2f}, max |move| {np.abs(new-old).max():.0f}")
res = base_df.with_columns(pl.Series("TAXITIME_SEC_mvt", new).cast(base_df.schema["TAXITIME_SEC_mvt"]))
assert res["MVT_ID_mvt"].equals(base_df["MVT_ID_mvt"]) and (res["TAXITIME_SEC_mvt"] >= 0).all()
res.write_parquet(out)
print("wrote", out)
