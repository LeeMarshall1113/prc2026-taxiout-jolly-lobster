"""experiment bigc 04: serving twin of the BIG LOMO corrector. docs/LAB-NOTES.md. Fit on all 8 fold months (R, k, rounds from the LOMO), apply to 2026 9-airport rows on top of v123.
python archive/agent_bigc_04_serve.py R1 0.75   -> data/interim/agent_bigc_board_final.parquet"""
import sys, json, numpy as np, polars as pl, lightgbm as lgb
C=pl.col; I="data/interim/"; ID="MVT_ID_mvt"; T="TAXITIME_SEC_mvt"
RG=sys.argv[1]; K=float(sys.argv[2]); SEEDS=(0,1,2)
AIR=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
z=np.load(I+"agent_bigc_fold.npz",allow_pickle=True); X=z["X"]; cols=list(z["cols"]); y=z["y"]; pv=z["pv"]; catidx=[int(i) for i in z["cat"]]
maps=json.load(open(I+"agent_bigc_maps.json"))
L0=np.load(I+"agent_bigc_lomo_s0.npz"); nround=int(round(float(np.mean(L0[RG+"_bi"])))); print("rounds",nround,flush=True)
R={"R1":dict(num_leaves=15,min_data_in_leaf=500,lambda_l2=10.0),"R2":dict(num_leaves=31,min_data_in_leaf=1000,lambda_l2=50.0),"R3":dict(num_leaves=7,min_data_in_leaf=2000,lambda_l2=100.0)}[RG]
# ---------------- serve frame
base=pl.read_parquet(I+"v137_ltfmsnow_final.parquet").sort(ID); D=np.load(I+"agent_bndl_serve_deltas.npz",allow_pickle=True); assert (D["ids"]==base[ID].to_numpy()).all()
rkf=pl.read_parquet(I+"agent_trans_rank_features.parquet"); S=base.select(ID,C(T).cast(pl.Float64).alias("p")).join(rkf,on=ID,how="left",maintain_order="left"); assert S.height==344841 and S["ADEP_mvt"].null_count()==0
p26=S["p"].to_numpy(); M=D["M1"]+D["M2"]+D["M3"]; pv26=p26+M; apt=S["ADEP_mvt"].to_numpy().astype(str); is9=np.isin(apt,AIR); n=S.height
LOW=["ADEP_mvt","MARKET_SEGMENT_flt","WK_TBL_CAT_flt","FLIGHT_TYPE_flt","FLIGHT_RULE_mvt","airport_plan","RUNWAY_mvt"]; HIGH=["STAND_mvt","ADES_mvt","AIRCRAFT_TYPE_mvt","AIRCRAFT_OPERATOR_flt","callsign_op"]
for c in LOW+HIGH: S=S.with_columns(C(c).replace_strict(maps[c],default=-1,return_dtype=pl.Int32).fill_null(-1).alias(c))
cen=pl.read_parquet(I+"agent_coverage_census_2026.parquet").select([ID,"category"]).with_columns(C("category").replace_strict(maps["category"],default=-1,return_dtype=pl.Int32).alias("cov_cat")).select([ID,"cov_cat"]).unique(ID)
ob=pl.read_parquet(I+"agent_offblock_rows_2026.parquet").unique(ID)
obc=["has_c","P_span","P_n","P_gap","P_dstand","P_long","seg_n","seg_rate","seg_start_gap","push","first_gs","first_dstand","revany_gap","rev_dist","rev_gs","rev_dstand"]
ecs=["e_stand","e_appear","e_ci20","e_dend","e_dnext","e_mv5","e_mv10","e_mv20","e_rev","e_start","e_revany","e_pushend"]
for c in ("src","prod_kind"): ob=ob.with_columns(C(c).cast(pl.Utf8).replace_strict(maps["ob_"+c],default=-1,return_dtype=pl.Int32).fill_null(-1).alias("ob_"+c))
ob=ob.select([ID,"ob_src","ob_prod_kind"]+obc+[(C("mvt_t")-C(e)).alias("m_"+e) for e in ecs])
bx=pl.read_parquet(I+"agent_backx_bx_2026.parquet").unique(ID)
# arrival lateness features: same definitions as agent_wx2_01_diag.py / agent_bndl_02_serve.py late_feats
rk=pl.read_parquet("data/raw/ranking.parquet",columns=["MVT_ID_mvt","PHASE_mvt","ADES_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt"])
arr=rk.filter((C("PHASE_mvt")=="ARR")&C("ADES_mvt").is_in(AIR)).select("ADES_mvt",C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t"),C("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("ts"))
dep=rk.filter(C("PHASE_mvt")=="DEP").select(ID,C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t")).unique(ID); del rk
Sx=S.select([ID]).join(dep,on=ID,how="left",maintain_order="left").with_columns(pl.Series("apt",apt))
late={}
for ap in AIR:
    A=arr.filter(C("ADES_mvt")==ap).sort("t"); at=A["t"].to_numpy(); lt=np.clip((A["t"]-A["ts"]).to_numpy().astype(float),-3600,10800)
    cs_l=np.concatenate([[0],np.cumsum(lt)]); cs_s=np.concatenate([[0],np.cumsum((lt>900).astype(float))]); Dd=Sx.filter(C("apt")==ap); Tt=Dd["t"].to_numpy(); F={}
    for wn,wl in (("60",3600),("180",10800)):
        hi=np.searchsorted(at,Tt,side="right"); lo=np.searchsorted(at,Tt-wl,side="right"); nn=hi-lo
        with np.errstate(invalid="ignore",divide="ignore"):
            F["s"+wn]=np.where(nn>=3,(cs_s[hi]-cs_s[lo])/np.maximum(nn,1),np.nan); F["m"+wn]=np.where(nn>=3,(cs_l[hi]-cs_l[lo])/np.maximum(nn,1),np.nan)
    late[ap]=pl.DataFrame({ID:Dd[ID].to_numpy(),"arr_late_share60":F["s60"],"arr_late_share180":F["s180"],"arr_late_mean60":F["m60"],"arr_late_mean180":F["m180"]})
wx=pl.concat(list(late.values()))
for o in (cen,ob,bx,wx): S=S.join(o,on=ID,how="left",maintain_order="left")
assert S.height==n
Xs=np.full((n,len(cols)),np.nan,np.float32)
for j,c in enumerate(cols):
    if c=="pv": Xs[:,j]=pv26
    else: Xs[:,j]=S[c].cast(pl.Float32).to_numpy()
# ---------------- feature shift audit (label-free), serve 9-apt rows vs fold rows
Xs9=Xs[is9]; sd=np.nanstd(X,0)+1e-9; shift=(np.nanmean(Xs9,0)-np.nanmean(X,0))/sd; nn_=np.abs(np.isnan(Xs9).mean(0)-np.isnan(X).mean(0))
print("top standardised mean shifts (serve-fold):",[(cols[i],round(float(shift[i]),2)) for i in np.argsort(-np.abs(shift))[:8]]); print("top NaN-rate shifts:",[(cols[i],round(float(nn_[i]),2)) for i in np.argsort(-nn_)[:6]],flush=True)
# ---------------- fit on all 8 months
r=np.clip(y-pv,-900,900).astype(np.float32); corr=np.zeros(n); imp=np.zeros(len(cols))
for s in SEEDS:
    P=dict(objective="l2",learning_rate=0.03,bagging_fraction=0.5,bagging_freq=1,feature_fraction=0.5,max_bin=63,num_threads=4,verbose=-1,seed=s,**R)
    b=lgb.train(P,lgb.Dataset(X,r,categorical_feature=catidx),nround); corr[is9]+=b.predict(Xs9)/len(SEEDS); imp+=b.feature_importance("gain"); print("seed",s,"fit",flush=True)
dp=K*corr; dp[~is9]=0
np.savez(I+"agent_bigc_serve_delta.npz",ids=S[ID].to_numpy(),corr=corr,dp=dp,imp=imp,cols=np.array(cols))
bf=pl.read_parquet(I+"agent_bndl_board_final.parquet").sort(ID); assert (bf[ID].to_numpy()==S[ID].to_numpy()).all()
cur=bf[T].cast(pl.Float64).to_numpy(); out=np.maximum(cur+dp,0); out=np.rint(out)
res=bf.with_columns(pl.Series(T,out).cast(bf.schema[T])); res.write_parquet(I+"agent_bigc_board_final.parquet")
mv=out!=cur; print("rows moved",int(mv.sum()),"of 9-apt",int(is9.sum()),"mean shift (all rows) %.3f"%float((out-cur).mean()),"(moved) %.3f"%float((out-cur)[mv].mean()),"rms dp %.2f"%float(np.sqrt((dp[is9]**2).mean())))
print("2026 sum dp^2/row (9-apt) %.2f ; all rows %.2f"%(float((dp[is9]**2).mean()),float((dp**2).sum()/n)))
Lf=np.load(I+"agent_bigc_lomo_s0.npz"); dpf=K*Lf[RG]; print("fold sum dp^2/row %.2f -> footprint ratio 2026/fold %.2f"%(float((dpf**2).mean()),float((dp[is9]**2).mean()/(dpf**2).mean())))
for ap in AIR: m_=apt==ap; print(ap,"mean dp %+.2f rms %.2f"%(dp[m_].mean(),np.sqrt((dp[m_]**2).mean())))
print("wrote",I+"agent_bigc_board_final.parquet")
