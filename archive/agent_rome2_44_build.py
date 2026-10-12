import polars as pl, numpy as np
C=pl.col
R=""
ID,T="MVT_ID_mvt","TAXITIME_SEC_mvt"
base=pl.read_parquet(R+"data/interim/agent_bundle_v132_final.parquet")
F=pl.read_parquet(R+"data/interim/agent_rome2_footprint26b.parquet")
beta=float(np.load(R+"data/interim/agent_rome2_beta_all.npy")[0]); m=1025.5
for name,w in [("agent_rome2_A_final",0.5),("agent_rome2_A_full_final",beta)]:
    new=F["v132"].to_numpy().astype(float); a=new.copy()
    mix=F["P"].to_numpy()*F["gs"].to_numpy()+(1-F["P"].to_numpy())*m
    new=np.maximum(np.rint(a+w*(mix-a)),0)
    upd=pl.DataFrame({ID:F[ID],"_n":new})
    out=base.join(upd,on=ID,how="left").with_columns(pl.coalesce(C("_n"),C(T).cast(pl.Float64)).round(0).cast(base.schema[T]).alias(T)).drop("_n").select(base.columns)
    out.write_parquet(R+f"data/interim/{name}.parquet")
    d=out.sort(ID)[T].cast(pl.Float64).to_numpy()-base.sort(ID)[T].cast(pl.Float64).to_numpy()
    print(name,"w=%.3f"%w,"rows differing:",int((d!=0).sum()),"footprint rows",F.height)
