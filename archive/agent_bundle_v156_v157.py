"""v156 = v155 + the Istanbul corrector with the nine-airport column lists built on Istanbul cells (piece agent_istnext_final minus v150; Istanbul non-snow rows only).
v157 = v156 + four long-gap Rome no-plan rows re-centred on the best-calibrated day-early probability from the 45 comparable 2025 records
(gap window, runway and operator habit, shrunk; values from the analysis: WMTT3508 share 0.50, SVII3006 share 0.57, NOSOS431 share 0.07, QFAA7006 midpoint of two values 21 s apart).
Out: data/interim/agent_bundle_v156_final.parquet, agent_bundle_v157_final.parquet"""
import polars as pl
C = pl.col; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; D = "data/interim/"
b = pl.read_parquet(D + "agent_bundle_v155_final.parquet"); r = pl.read_parquet(D + "agent_bundle_v150_final.parquet").rename({T: "r"}); p = pl.read_parquet(D + "agent_istnext_final.parquet").rename({T: "p"})
j = b.join(r, on=ID).join(p, on=ID).with_columns((C("p") - C("r")).alias("d"))
assert j.filter((C("d") != 0) & (C(T) != C("r"))).height == 0, "Istanbul piece overlaps rows changed since v150"
v156 = b.select(ID).join(j.select(ID, (C(T) + C("d")).cast(b.schema[T]).alias(T)), on=ID, how="left"); assert (v156[ID] == b[ID]).all()
v156.write_parquet(D + "agent_bundle_v156_final.parquet"); d = v156[T] - b[T]; print("v156 rows changed", int((d != 0).sum()), "mean", float(d.mean()), "max", int(d.max()), "min", int(d.min()))
NEW = {202680942: 72000, 213154026: 76200, 213145579: 110000, 213152353: 87495}
ids = v156[ID].cast(pl.Int64).to_list(); vals = v156[T].to_list(); n = 0
for i, k in enumerate(ids):
    if k in NEW: print("row", k, vals[i], "->", NEW[k]); vals[i] = NEW[k]; n += 1
assert n == 4
pl.DataFrame({ID: v156[ID], T: pl.Series(vals).cast(b.schema[T])}).write_parquet(D + "agent_bundle_v157_final.parquet")
