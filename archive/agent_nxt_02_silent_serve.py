"""nxt 02: serving twin of silent-start fallback. Classifier fitted on ALL 2025 fold rows (agent_silent_fold.parquet, recipe of agent_silent_02_lomo.py),
3 seeds averaged; E[excess|apt] from all 2025 events; applied to 2026 EDDF/EHAM/LEBL rows with p_hat>0.1, k=0.3, on top of v123.
p feature for 2026 = v137 served prediction (analogue of both_guard)."""
import numpy as np, polars as pl, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
I="data/interim/"; ID="MVT_ID_mvt"; C=pl.col; THR=600; KEEP=["EDDF","EHAM","LEBL"]
d=pl.read_parquet(I+"agent_silent_fold.parquet").drop(["stand","type","src","STAND_mvt","e_ci20"],strict=False)
base=pl.read_parquet(I+"v137_ltfmsnow_final.parquet").select(ID,C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("p"))
rk=pl.scan_parquet("data/raw/ranking.parquet").filter((C("PHASE_mvt")=="DEP")&C("ADEP_mvt").is_in(KEEP)).select([ID,"ADEP_mvt","MVT_TIME_UTC_mvt","SCHED_TIME_UTC_mvt","LOBT_flt","IOBT_flt","EOBT_1_flt","AOBT_3_flt","AIRCRAFT_OPERATOR_flt","RUNWAY_mvt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt","AIRCRAFT_TYPE_mvt","STAND_mvt"]).collect()
ep=lambda n: C(n).dt.epoch("us")/1e6
raw=rk.with_columns(C("MVT_TIME_UTC_mvt").dt.epoch("us").alias("_m")/1e6).with_columns(
  [(C("_m")-ep(n)).alias("g_"+n.split("_")[0].lower()) for n in ["SCHED_TIME_UTC_mvt","LOBT_flt","IOBT_flt","EOBT_1_flt","AOBT_3_flt"]]+
  [(ep("EOBT_1_flt")-ep("LOBT_flt")).alias("eobt_m_lobt"),(ep("SCHED_TIME_UTC_mvt")-ep("LOBT_flt")).alias("sched_m_lobt"),
   C("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"),C("MVT_TIME_UTC_mvt").dt.minute().alias("mvt_min")]).select(
  [ID,"ADEP_mvt","g_sched","g_lobt","g_iobt","g_eobt","g_aobt","eobt_m_lobt","sched_m_lobt","dow","mvt_min","AIRCRAFT_OPERATOR_flt","RUNWAY_mvt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt","AIRCRAFT_TYPE_mvt","STAND_mvt"]).unique(ID).rename({"ADEP_mvt":"apt"})
keep=[ID,"mvt_t","e_start","e_appear","e_stand","e_revany","revany_gap","rev_dist","rev_dstand","push","first_gs","first_dstand","seg_n","seg_rate","seg_start_gap","P_span","P_n","P_gap","P_dstand","P_long","prod_kind","has_c","stand","type","src"]
o=pl.read_parquet(I+"agent_offblock_rows_2026.parquet",columns=keep).unique(ID)
cen=pl.read_parquet(I+"agent_coverage_census_2026.parquet").select([ID,"category"]).unique(ID)
s=raw.join(base,on=ID,how="inner").join(cen,on=ID,how="left").join(o,on=ID,how="left")
s=s.with_columns((C("mvt_t")-C("e_start")).alias("m_first"),(C("mvt_t")-C("e_appear")).alias("m_appear"),(C("mvt_t")-C("e_revany")).alias("m_rev"),(C("mvt_t")-C("e_stand")).alias("m_estand"),
  pl.lit(0).alias("month"),pl.lit(0.0).alias("y"),pl.lit(0.0).alias("r"))
print("2026 rows",s.height,"census null",s["category"].null_count(),"offblock null",s["mvt_t"].null_count(), "base match",s.height)
s=s.filter(C("apt").is_in(KEEP)); print("serve rows (3 apts)",s.height, s.group_by("apt").agg(C("mvt_t").is_not_null().mean()).to_dicts())
n25=d.height
d=d.with_columns(pl.lit(1).alias("_s"))
s=s.with_columns(pl.lit(2).alias("_s"))
common=[c for c in d.columns if c in s.columns or c=="_s"]
dd=pl.concat([d.select(common),s.select(common)],how="vertical_relaxed")
cats=["apt","category","prod_kind","AIRCRAFT_OPERATOR_flt","AIRCRAFT_TYPE_mvt","RUNWAY_mvt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt"]
for c in cats:
    if c in dd.columns: dd=dd.with_columns(C(c).cast(pl.Utf8).fill_null("NA").cast(pl.Categorical).to_physical().alias(c+"_c"))
num=["p","dow","mvt_min","m_first","m_appear","m_rev","m_estand","e_revany","revany_gap","rev_dist","rev_dstand","push","first_gs","first_dstand","seg_n","seg_rate","seg_start_gap","P_span","P_n","P_gap","P_dstand","P_long","has_c","g_sched","g_lobt","g_iobt","g_eobt","g_aobt","eobt_m_lobt","sched_m_lobt"]
dd=dd.with_columns([C(c).cast(pl.Float64) for c in num if c in dd.columns])
dd=dd.with_columns(((C("mvt_t")%86400)/3600).alias("hod")); num.append("hod"); num=[c for c in num if c in dd.columns]
feats=num+[c+"_c" for c in cats if c in dd.columns]
r=dd["r"].to_numpy(); apt=dd["apt"].to_numpy(); sflag=dd["_s"].to_numpy(); ids_all=dd[ID].to_numpy()
X=dd.select([C(c).cast(pl.Float32) for c in feats]).to_numpy(); del dd,d,s,raw,rk; import gc; gc.collect()
ev=(r>THR).astype(int)
tr=sflag==1; te=sflag==2
rng=np.random.default_rng(0); keepneg=rng.random(len(r))<0.25
catidx=[feats.index(c+"_c") for c in cats if c+"_c" in feats]
sub=tr&((ev==1)|keepneg); Xs=X[sub].copy(); Xt=X[te].copy(); del X; gc.collect(); w=np.where(ev[sub]==1,1.0,4.0)
ph=np.zeros(te.sum())
for seed in (0,1,2):
    prm=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10,num_threads=4,verbose=-1,cat_smooth=50,min_data_per_group=200,max_bin=127,seed=seed)
    b=lgb.train(prm,lgb.Dataset(Xs,ev[sub],weight=w,categorical_feature=catidx),num_boost_round=300)
    ph+=b.predict(Xt)/3; print("seed",seed,flush=True)
g=r[tr&(ev==1)].mean(); exc_t={}
for a in KEEP:
    k=tr&(ev==1)&(apt==a); exc_t[a]=r[k].mean() if k.sum()>=30 else g
print("excess",exc_t)
ids=ids_all[te]; ap=apt[te]
dl=np.where(ph>0.1,0.3*ph*np.array([exc_t[a] for a in ap]),0.0)
print("p_hat quantiles",np.quantile(ph,[.5,.9,.99,.999]),"rows p_hat>0.1",int((ph>0.1).sum()),"mean delta moved",dl[dl!=0].mean() if (dl!=0).any() else 0)
np.savez(I+"agent_nxt_silent_delta.npz",ids=ids,ph=ph,delta=dl,apt=ap)
v123=pl.read_parquet(I+"agent_bndl_board_final.parquet")
T="TAXITIME_SEC_mvt"; dm=dict(zip(ids.tolist(),dl.tolist()))
dv=np.array([dm.get(k,0.0) for k in v123[ID].to_numpy().tolist()])
out=np.maximum(v123[T].cast(pl.Float64).to_numpy()+dv,0)
res=v123.with_columns(pl.Series(T,np.rint(out)).cast(v123.schema[T]))
res.write_parquet(I+"agent_nxt_silent_final.parquet"); print("wrote; rows changed",int((res[T]!=v123[T]).sum()),"rows delta!=0",int((dv!=0).sum()))
