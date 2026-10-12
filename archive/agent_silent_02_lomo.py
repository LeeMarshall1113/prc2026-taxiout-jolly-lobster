import numpy as np, polars as pl, lightgbm as lgb, warnings, sys
warnings.filterwarnings("ignore")
THR=600
d=pl.read_parquet("data/interim/agent_silent_fold.parquet")
d=d.with_columns(pl.col("hour").cast(pl.Float64) if "hour" in d.columns else pl.lit(0.0))
cats=["apt","category","prod_kind","AIRCRAFT_OPERATOR_flt","AIRCRAFT_TYPE_mvt","RUNWAY_mvt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt"]
for c in cats:
    if c in d.columns: d=d.with_columns(pl.col(c).cast(pl.Utf8).fill_null("NA").cast(pl.Categorical).to_physical().alias(c+"_c"))
num=["p","dow","mvt_min","m_first","m_appear","m_rev","m_estand","e_revany","revany_gap","rev_dist","rev_dstand","push","first_gs","first_dstand","seg_n","seg_rate","seg_start_gap","P_span","P_n","P_gap","P_dstand","P_long","has_c","g_sched","g_lobt","g_iobt","g_eobt","g_aobt","eobt_m_lobt","sched_m_lobt"]
d=d.with_columns([pl.col(c).cast(pl.Float64) for c in num if c in d.columns])
hr=pl.col("mvt_t")  # mvt_t epoch seconds
d=d.with_columns(((pl.col("mvt_t")%86400)/3600).alias("hod"))
num.append("hod")
num=[c for c in num if c in d.columns]
feats=num+[c+"_c" for c in cats if c in d.columns]
X=d.select(feats).to_numpy().astype(np.float32)
r=d["r"].to_numpy(); ev=(r>THR).astype(int); mon=d["month"].to_numpy(); apt=d["apt"].to_numpy(); p=d["p"].to_numpy(); y=d["y"].to_numpy()
del d; import gc; gc.collect()
rng=np.random.default_rng(0); keepneg=rng.random(len(r))<0.25
catidx=[feats.index(c+"_c") for c in cats if c+"_c" in feats]
MODE=sys.argv[2] if len(sys.argv)>2 else 'lomo'
if MODE=='day':
    import zlib
    dd=pl.read_parquet('data/interim/agent_silent_fold.parquet',columns=['day'])['day'].to_numpy(); grp=np.array([zlib.crc32(x.encode())%8 for x in dd])
else: grp=mon
months=sorted(set(grp)); ph=np.zeros(len(r)); exc=np.zeros(len(r))
prm=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.7,bagging_freq=1,lambda_l2=10,num_threads=4,verbose=-1,cat_smooth=50,min_data_per_group=200,max_bin=127,seed=int(sys.argv[1]) if len(sys.argv)>1 else 0)
for m in months:
    tr=grp!=m; te=~tr; sub=tr&((ev==1)|keepneg); w=np.where(ev[sub]==1,1.0,4.0)
    ds=lgb.Dataset(X[sub],ev[sub],weight=w,categorical_feature=catidx,free_raw_data=True)
    b=lgb.train(prm,ds,num_boost_round=300)
    ph[te]=b.predict(X[te])
    # conditional excess by airport on train months
    g=np.mean(r[tr&(ev==1)])
    for a in np.unique(apt):
        k=tr&(ev==1)&(apt==a); v=r[k].mean() if k.sum()>=30 else g
        exc[te&(apt==a)]=v
    print("month",m,"n_ev_te",ev[te].sum(),"trees",b.num_trees(),flush=True)
np.savez("data/interim/agent_silent_oof_%s_s%s.npz"%(MODE,prm["seed"]),ph=ph,exc=exc)
