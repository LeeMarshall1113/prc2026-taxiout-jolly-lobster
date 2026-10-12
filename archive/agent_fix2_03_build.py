import numpy as np, polars as pl
C=pl.col; R=""; ID,T="MVT_ID_mvt","TAXITIME_SEC_mvt"
v=pl.read_parquet(R+"data/interim/agent_bundle_v132_final.parquet")
r=pl.read_parquet(R+"data/interim/agent_fix2_c_rows.parquet")
tg=dict(zip(r[ID].to_list(),r["target"].to_list()))
new=np.array([tg.get(m,x) for m,x in zip(v[ID].to_list(),v[T].to_list())],float)
out=v.with_columns(pl.Series(T,np.rint(new)).cast(v.schema[T]))
assert out[ID].equals(v[ID])
d=out[T].cast(pl.Float64).to_numpy()-v[T].cast(pl.Float64).to_numpy()
print("rows changed",int((d!=0).sum()),"ids in target set",len(tg),"rms move",float(np.sqrt((d[d!=0]**2).mean())),"moves",np.sort(d[d!=0]).astype(int).tolist())
# prod-caught rows unchanged: ids caught by FLIGHT key are not in tg by construction
out.write_parquet(R+"data/interim/agent_fix2_c_final.parquet"); out.write_parquet(R+"data/interim/agent_fix2_both_final.parquet")
