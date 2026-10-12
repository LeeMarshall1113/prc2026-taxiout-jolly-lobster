"""v150 = BASE + (corrector with new arithmetic columns at six airports: agent_arith_final minus v146).
v151 = BASE2 + (Rome re-filed plans, softly bound part, three rows: agent_second_romemoved_final minus v146).
usage: python agent_bundle_v150_v151.py <150|151> <base parquet>   -> data/interim/agent_bundle_v<V>_final.parquet"""
import sys, polars as pl
C = pl.col; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; D = "data/interim/"
V, base = sys.argv[1], sys.argv[2]
piece = {"150": "agent_arith_final", "151": "agent_second_romemoved_final"}[V]
a = pl.read_parquet(base); p = pl.read_parquet(D + piece + ".parquet").rename({T: "p"}); b = pl.read_parquet(D + "agent_bundle_v146_final.parquet").rename({T: "b"})
j = a.join(p, on=ID).join(b, on=ID).with_columns((C("p") - C("b")).alias("d"))
ov = j.filter((C("d") != 0) & (C(T) != C("b"))).height
assert ov == 0, f"{ov} rows of the piece were already moved in the base"
out = a.select(ID).join(j.select(ID, (C(T) + C("d")).cast(a.schema[T]).alias(T)), on=ID, how="left")
assert (out[ID] == a[ID]).all() and int((out[T] < 0).sum()) == 0
out.write_parquet(D + f"agent_bundle_v{V}_final.parquet"); d = out[T] - a[T]
print(f"v{V}: base {base}; rows changed {int((d != 0).sum())}; mean {float(d.mean()):+.3f}; max {int(d.max())}; min {int(d.min())}")
