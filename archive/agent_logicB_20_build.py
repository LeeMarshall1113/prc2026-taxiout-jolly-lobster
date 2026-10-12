"""logicB 20: two PROVEN clips on v145 (no fitted parameter, no label read).
 window : plan-clock window, label in [MVT-LOBT-3606, MVT-LOBT+3606] on 2,062,577 of 2,062,577 plan departures of 2025 (0 exceptions).
          Predictions of plan rows outside it move to the nearer edge.
 lower  : hidden-plan lower bound at LIRF and LTFM (flight-string key): no-plan departure with plan recovered from the destination's arrival
          record, gl = MVT-LOBT < 3606 -> label > gl + 3606 (LIRF 32/32, LTFM 25/25 in 2025). Predictions below it move up to it."""
import numpy as np, polars as pl
C = pl.col; W = 3606
R = ""
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ref = pl.read_parquet(R+"data/interim/agent_bundle_v145_final.parquet")
s = pl.read_parquet(R+"data/interim/local/logic_rows/dep2026.parquet").with_columns(gl=C("mvt")-C("lobt"), gap=C("mvt")-C("sch"), date=pl.from_epoch("mvt").dt.date())
p = s.filter(C("lobt").is_not_null()).with_columns(lo=C("gl")-W, hi=C("gl")+W)
w = p.filter((C("served")<C("lo"))|(C("served")>C("hi"))).with_columns(new=pl.when(C("served")<C("lo")).then(C("lo")).otherwise(C("hi")))
w = w.with_columns(d=C("new")-C("served"))
print("WINDOW rows outside exact window:", w.height, "sum d^2", float((w["d"].cast(pl.Float64)**2).sum()))
print(w.select("apt","date","fl","gl","gap","served","lo","hi","new","d").sort(C("d").abs(),descending=True))
H = pl.read_parquet(R+"data/interim/local/logic_rows/hidden26.parquet")
sv = s.select(C("id").cast(pl.Float64).alias("MVT_ID_mvt"), "served")
h = H.join(sv, on="MVT_ID_mvt").filter((C("cls")=="lowW")&C("adep").is_in(["LIRF","LTFM"])&(C("route")=="str")).with_columns(new=C("gl")+W).filter(C("served")<C("new")).with_columns(d=C("new")-C("served"), date=pl.from_epoch("mvt").dt.date())
print("LOWER rows:", h.height, "sum d^2", float((h["d"].cast(pl.Float64)**2).sum()))
print(h.select("adep","date","flight","gl","gs","served","new","d","link"))
def write(name, upd):
    mp = dict(upd); vals = [mp.get(float(i), x) for i, x in zip(ref[ID].to_list(), ref[T].to_list())]
    out = ref.with_columns(pl.Series(T, vals).cast(ref.schema[T])); assert out[ID].equals(ref[ID])
    out.write_parquet(R+f"data/interim/agent_logicB_{name}_final.parquet"); print("wrote", name, int((out[T]!=ref[T]).sum()), "rows changed")
write("window", zip(w["id"].cast(pl.Float64).to_list(), w["new"].to_list()))
write("hiddenlower", zip(h["MVT_ID_mvt"].to_list(), h["new"].to_list()))
w.write_parquet(R+"data/interim/local/logic_rows/logicB_window_rows.parquet"); h.write_parquet(R+"data/interim/local/logic_rows/logicB_lower_rows.parquet")
