"""experiment bndl 01: small-gain bundle M1+M2+M3 LOMO on SA_s0 composite. docs/LAB-NOTES.md"""
import numpy as np, polars as pl, lightgbm as lgb, glob, itertools
C=pl.col; I="data/interim/"; ID="MVT_ID_mvt"
AIR=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
z=np.load(I+"agent_bulkr_fold_SA_s0.npz",allow_pickle=True); keep=np.isin(z["apts"].astype(str),AIR)
ids=z["ids"][keep]; y=z["y"][keep].astype(float); p=z["both_guard"][keep].astype(float)
a=z["apts"][keep].astype(str); mon=np.array([str(d)[5:7] for d in z["days"][keep]]); N=len(y)
months=sorted(set(mon)); H1=np.isin(mon,["01","02","03"]); H2=np.isin(mon,["07","08","09"])
r=y-p
# ---- M1
ctx=(pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt")=="DEP")
 .select(C(ID).cast(pl.Float64).alias("id"),
  (C("MVT_TIME_UTC_mvt")-C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
  (C("MVT_TIME_UTC_mvt")-C("LOBT_flt")).dt.total_seconds().alias("gl"),
  (C("MVT_TIME_UTC_mvt")-C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
  C("AOBT_3_flt").is_not_null().cast(pl.Float64).alias("plan"),
  C("MVT_TIME_UTC_mvt").dt.hour().cast(pl.Float64).alias("hr"),
  C("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Float64).alias("dow")).collect())
j=pl.DataFrame({"id":ids}).with_columns(pl.int_range(pl.len()).alias("_r")).join(ctx,on="id",how="left").sort("_r"); assert j.height==N
X=np.column_stack([p,np.array([AIR.index(v) for v in a]),j["gs"].to_numpy(),j["gl"].to_numpy(),j["ga"].to_numpy(),j["plan"].to_numpy(),j["hr"].to_numpy(),j["dow"].to_numpy()]).astype(float)
dB=np.zeros(N)
for seed in (0,1,2):
    for m in months:
        tr=mon!=m; te=mon==m
        mdl=lgb.LGBMRegressor(n_estimators=200,learning_rate=0.03,num_leaves=15,min_child_samples=2000,subsample=0.7,subsample_freq=1,colsample_bytree=0.8,random_state=seed,n_jobs=4,verbose=-1)
        mdl.fit(X[tr],r[tr],categorical_feature=[1]); dB[te]+=mdl.predict(X[te])/3
M1=0.5*dB; print("M1 done",flush=True)
# ---- M2
d=pl.read_parquet(I+"agent_wx2_diag.parquet").select(["id","apt","late_ph"] if False else ["id","arr_late_share60","arr_late_share180","arr_late_mean60","arr_late_mean180"])
jj=pl.DataFrame({"id":ids}).with_columns(pl.int_range(pl.len()).alias("_r")).join(d,on="id",how="left").sort("_r"); assert jj.height==N
F=np.column_stack([jj[c].to_numpy().astype(float) for c in ["arr_late_share60","arr_late_share180","arr_late_mean60","arr_late_mean180"]])
F=np.where(np.isnan(F),np.nanmean(F,0),F); M2=np.zeros(N)
for ap in AIR:
    ia=a==ap
    for m in months:
        te=ia&(mon==m); tr=ia&(mon!=m)
        mu,sd=F[tr].mean(0),F[tr].std(0)+1e-9
        A=np.column_stack([np.ones(tr.sum()),(F[tr]-mu)/sd])
        co=np.linalg.solve(A.T@A+1e3*np.eye(5),A.T@r[tr]); M2[te]=np.column_stack([np.ones(te.sum()),(F[te]-mu)/sd])@co
M2=np.clip(M2,-60,60); print("M2 done",flush=True)
# ---- M3
op=pl.concat([pl.read_parquet(f,columns=[ID,"AIRCRAFT_OPERATOR_flt"]) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))]).unique(ID)
o=pl.scan_parquet(I+"agent_offblock_rows_c238.parquet").select([ID,"apt","block_t","e_revany"]).filter(C("e_revany").is_not_nan()).collect()
o=o.join(op,on=ID,how="left").rename({"AIRCRAFT_OPERATOR_flt":"op"}).with_columns((C("e_revany")-C("block_t")).clip(-300,300).alias("ec"))
o=o.with_columns(C(ID).cast(pl.Float64).alias("id")).select(["id","apt","op","ec"]).unique("id")
jm=pl.DataFrame({"id":ids}).with_columns(pl.int_range(pl.len()).alias("_r")).join(o,on="id",how="left").sort("_r")
ec=jm["ec"].to_numpy(); ok=~np.isnan(ec); print("M3 matched rows",ok.sum())
def cm(keys):
    kk=np.array(["|".join(map(str,t)) for t in jm.select([C(k).cast(pl.Utf8).fill_null("NA") for k in keys]).rows()])
    out=np.full(N,np.nan)
    for m in months:
        tr=ok&(mon!=m); df=pl.DataFrame({"k":kk[tr],"e":ec[tr]}).group_by("k").agg(C("e").median().alias("md"),pl.len().alias("n")).filter(C("n")>=30)
        dd=dict(zip(df["k"].to_list(),df["md"].to_list())); idx=np.where(ok&(mon==m))[0]; out[idx]=[dd.get(k,np.nan) for k in kk[idx]]
    return out
x=cm(["apt","op"])-cm(["apt"]); okx=ok&~np.isnan(x); M3=np.zeros(N)
for m in months:
    tr=okx&(mon!=m); te=okx&(mon==m); b=(x[tr]*r[tr]).sum()/(x[tr]**2).sum(); M3[te]=b*x[te]
print("M3 done, rows touched",okx.sum(),flush=True)
np.savez(I+"agent_bndl_members.npz",ids=ids,M1=M1,M2=M2,M3=M3,p=p,y=y)
# ---- judge
rm=lambda q,k: np.sqrt(((y-q)[k]**2).mean()); ALL=np.ones(N,bool)
def judge(name,q):
    dall=rm(q,ALL)-rm(p,ALL); per={ap:rm(q,a==ap)-rm(p,a==ap) for ap in AIR}
    g=(y-p)**2-(y-q)**2; kk=np.ones(N,bool); kk[np.argsort(-g)[:50]]=False
    m50=rm(q,kk)-rm(p,kk); h1=rm(q,H1)-rm(p,H1); h2=rm(q,H2)-rm(p,H2)
    ok_=dall<=-0.10 and max(per.values())<=0.10 and m50<0 and h1<0 and h2<0
    print("%-14s all %+.4f worstApt %s %+.3f m50 %+.4f JanMar %+.4f JulSep %+.4f -> %s"%(name,dall,max(per,key=per.get),max(per.values()),m50,h1,h2,"PASS" if ok_ else "fail"),flush=True)
    return dall
print("control rmse %.3f n=%d"%(rm(p,ALL),N))
Ms={"M1":M1,"M2":M2,"M3":M3}
for k,v in Ms.items(): judge(k,p+v)
B=judge("BUNDLE",p+M1+M2+M3)
for k in Ms:
    rest=sum(v for kk,v in Ms.items() if kk!=k); print("  marginal",k,"= bundle - bundle_without = %+.4f"%(B-(rm(p+rest,ALL)-rm(p,ALL))))
judge("BUNDLE x0.5",p+0.5*(M1+M2+M3))
for ap in AIR: pass
