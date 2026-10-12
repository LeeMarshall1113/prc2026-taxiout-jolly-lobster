"""experiment bigc 04: serving twin of the BIG LOMO corrector. docs/LAB-NOTES.md. Fit on all 8 fold months (R, k, rounds from the LOMO), apply to 2026 9-airport rows on top of v123.
python archive/agent_bigc_04_serve.py R1 0.75   -> data/interim/agent_bigc_v2_board_final.parquet"""
import sys, json, numpy as np, polars as pl, lightgbm as lgb
C=pl.col; I="data/interim/"; ID="MVT_ID_mvt"; T="TAXITIME_SEC_mvt"
RG=sys.argv[1]; K=float(sys.argv[2]); SEEDS=(0,1,2)
AIR=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
z=np.load(I+"agent_bigc_fold.npz",allow_pickle=True); X=z["X"]; cols=list(z["cols"]); y=z["y"]; pv=z["pv"]; catidx=[int(i) for i in z["cat"]]
maps=json.load(open(I+"agent_bigc_maps.json"))
nround=int(round(float(np.mean([np.load(I+"agent_bigc_v2_r1x_s%d_%sV3.npz"%(q,RG))[RG+"V3_bi"].mean() for q in (0,1,2)])))); print("rounds",nround,flush=True); ex=np.load(I+"agent_bigc_extra.npz",allow_pickle=True); X=np.hstack([X,ex["Xf"]]); cols=cols+list(ex["cols"])
R={"R1":dict(num_leaves=15,min_data_in_leaf=500,lambda_l2=10.0),"R2":dict(num_leaves=31,min_data_in_leaf=1000,lambda_l2=50.0),"C1":dict(num_leaves=63,min_data_in_leaf=300,lambda_l2=20.0),"C2":dict(num_leaves=127,min_data_in_leaf=300,lambda_l2=50.0)}.get(RG)
R={"C3":dict(num_leaves=255,min_data_in_leaf=300,lambda_l2=50.0),"C4b":dict(num_leaves=511,min_data_in_leaf=150,lambda_l2=100.0),"C4ff7":dict(num_leaves=511,min_data_in_leaf=150,lambda_l2=100.0,feature_fraction=0.7)}[RG]
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
exX=ex["Xs"]; exc=list(ex["cols"]); Xs=np.full((n,len(cols)),np.nan,np.float32)
for j,c in enumerate(cols):
    if c=="pv": Xs[:,j]=pv26
    elif c in exc: Xs[:,j]=exX[:,exc.index(c)]
    else: Xs[:,j]=S[c].cast(pl.Float32).to_numpy()
# ---------------- dump the 2026 serve-side inputs for modal/agent_bigc_app.py::serve_one (everything that does not depend on fold LABELS)
sys.path.insert(0,"archive"); import agent_bigc_drift as DR
e2=np.load(I+"agent_bigc_extra2.npz",allow_pickle=True); kk2=np.load(I+"agent_bigc_keys2.npz",allow_pickle=True)
rkm=pl.read_parquet("data/raw/ranking.parquet",columns=[ID,"MVT_TIME_UTC_mvt","PHASE_mvt"]).filter(pl.col("PHASE_mvt")=="DEP").unique(ID)
mon_s=base.select(ID).join(rkm,on=ID,how="left")["MVT_TIME_UTC_mvt"].dt.month().to_numpy().astype(int); del rkm
span_f=X[:,cols.index("m_e_start")].astype(float); span_s=Xs[:,cols.index("m_e_start")].astype(float)
Dsv=DR.drift_serve(kk2["Kf_d"],pv.astype(float),span_f,kk2["Ks_d"],pv26.astype(float),span_s,mon_s)
Xs_pre=np.hstack([Xs,e2["Xs"]]).astype(np.float32); cols_pre=cols+["d_NEW","d_ET"]
np.savez(I+"agent_bigc_serve_in.npz",Xs_pre=Xs_pre,cols_pre=np.array(cols_pre),Dsv=Dsv,mon_s=mon_s,pv26=pv26,apt=apt,is9=is9,ids=S[ID].to_numpy())
print("wrote agent_bigc_serve_in.npz",Xs_pre.shape,Dsv.shape)
