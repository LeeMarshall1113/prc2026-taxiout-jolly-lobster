"""error budget v136, step 1: eb130 fold table + fold equivalents of v133/v135/v136 (Rome no-plan) -> data/interim/agent_eb136_tbl.parquet
Leave-one-month-out copy probability P2 (clf52 outer), shipped bucket weights 0.5 / 0.85, hidden-plan readings keyed on callsign or flight (hidden2), lever B.
Rows < 1800 s gap and every other airport/class are untouched (eb130 values)."""
import polars as pl, numpy as np
C=pl.col; R=""; m0=1025.5; W=3606
eb=pl.read_parquet(R+"data/interim/agent_eb130_tbl.parquet")
z=np.load(R+"data/interim/agent_rome2_clf52.npz"); cid=z["ids"].tolist(); ix={i:k for k,i in enumerate(cid)}
d=(pl.read_parquet(R+"data/interim/agent_rome2_depall.parquet").filter((C("ADEP_mvt")=="LIRF")&C("np")&(C("yr")==2025))
   .select(C("MVT_ID_mvt").alias("id"),"gs","FLIGHT_mvt",C("ADES_mvt").alias("dest")))
H=pl.read_parquet(R+"data/interim/agent_rome2_hidden2.parquet").filter(C("yr")==2025).select(C("MVT_ID_mvt").alias("id"),"gl","a_l","t_aobt")
e=eb.filter((C("apt")=="LIRF")&(C("hasplan")==0)).join(d,on="id",how="inner").filter(C("gs")>=1800).join(H,on="id",how="left")
print("pocket rows with gap>=1800:",e.height,"in clf52:",sum(i in ix for i in e["id"].to_list()))
e=e.filter(C("id").is_in(cid))
rows=[ix[i] for i in e["id"].to_list()]
P=z["P2"][rows]
a=e["pred"].to_numpy(); gs=e["gs"].to_numpy().astype(float); gl=e["gl"].fill_null(0).to_numpy().astype(float); ta=np.clip(e["t_aobt"].fill_null(0).to_numpy().astype(float),0,None)
hid=e["gl"].is_not_null().to_numpy(); al=e["a_l"].fill_null(0).to_numpy().astype(float)
rule=np.where(~hid,"none",np.where((gl>=W)&(np.abs(gs-gl)<W),"R1",np.where(gl<W,"R2",np.where(np.abs(al)<W,"R3a","R3b"))))
new=a.copy(); mix=P*gs+(1-P)*m0
mid=(gs>=3600)&(gs<14400)&(rule=="none"); w=np.where(gs<7200,0.5,0.85)
new[mid]=a[mid]+w[mid]*(mix[mid]-a[mid])
r1=rule=="R1"; new[r1]=np.minimum(ta[r1],np.clip(gl[r1]-W,0,None))
r2=rule=="R2"; new[r2]=gs[r2]
r3=rule=="R3b"; new[r3]=ta[r3]
print({k:int((rule==k).sum()) for k in ["none","R1","R2","R3a","R3b"]},"mid",int(mid.sum()))
upd=pl.DataFrame({"id":e["id"],"pred_new":new,"P2":P,"gs_r":gs,"rule":rule,"mid":mid})
t=eb.rename({"pred":"pred130"}).join(upd,on="id",how="left").with_columns(pl.coalesce(C("pred_new"),C("pred130")).alias("pred")).drop("pred_new")
print("rows changed",int((t["pred"]!=t["pred130"]).sum()),"of",t.height)
t.write_parquet(R+"data/interim/agent_eb136_tbl.parquet")
