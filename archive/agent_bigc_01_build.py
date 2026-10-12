"""experiment bigc 01: fold feature matrix for the BIG month-grouped corrector. docs/LAB-NOTES.md. Label-free features only (y only as target)."""
import numpy as np, polars as pl, json
C=pl.col; I="data/interim/"; ID="MVT_ID_mvt"
AIR=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
z=np.load(I+"agent_bulkr_fold_SA_s0.npz",allow_pickle=True); keep=np.isin(z["apts"].astype(str),AIR)
m=np.load(I+"agent_bndl_members.npz"); ids=z["ids"][keep]; assert (ids==m["ids"]).all() and np.allclose(z["both_guard"][keep],m["p"])
days=z["days"][keep]; mon=np.array([int(str(d)[5:7]) for d in days])
pv=m["p"]+m["M1"]+m["M2"]+m["M3"]; y=m["y"]
N=len(ids); base=pl.DataFrame({ID:ids,"_r":np.arange(N)})
tfl=pl.scan_parquet(I+"train_features.parquet"); rk=pl.read_parquet(I+"agent_trans_rank_features.parquet",n_rows=5); tfc=tfl.collect_schema().names()
com=[c for c in rk.columns if c in tfc and c not in (ID,"month","doy")]
tf=tfl.select([ID]+[C(c).cast(pl.Float32) if tfl.collect_schema()[c] in (pl.Float64,pl.Int64,pl.Int32,pl.Int16,pl.Int8) else C(c) for c in com]).join(base.lazy().select(ID),on=ID,how="semi").collect().unique(ID); d=base.join(tf,on=ID,how="left").sort("_r"); assert d.height==N
print("bulk cols",len(com),"null ADEP",d["ADEP_mvt"].null_count(),flush=True)
# categoricals
LOW=["ADEP_mvt","MARKET_SEGMENT_flt","WK_TBL_CAT_flt","FLIGHT_TYPE_flt","FLIGHT_RULE_mvt","airport_plan","RUNWAY_mvt"]
HIGH=["STAND_mvt","ADES_mvt","AIRCRAFT_TYPE_mvt","AIRCRAFT_OPERATOR_flt","callsign_op"]
maps={}
for c in LOW+HIGH:
    vc=d[c].value_counts().sort("count",descending=True); vals=[v for v in vc[c].to_list() if v is not None]; maps[c]={v:i for i,v in enumerate(vals)}
    d=d.with_columns(C(c).replace_strict(maps[c],default=-1,return_dtype=pl.Int32).fill_null(-1).alias(c))
# extras
cen=pl.read_parquet(I+"agent_coverage_census_2025.parquet").select([ID,"category"]); cats=sorted(set(cen["category"].drop_nulls().to_list())); maps["category"]={v:i for i,v in enumerate(cats)}
cen=cen.with_columns(C("category").replace_strict(maps["category"],default=-1,return_dtype=pl.Int32).alias("cov_cat")).select([ID,"cov_cat"]).unique(ID)
ob=pl.read_parquet(I+"agent_offblock_rows_c238.parquet",columns=[ID,"src","prod_kind","mvt_t","has_c","P_span","P_n","P_gap","P_dstand","P_long","seg_n","seg_rate","seg_start_gap","push","first_gs","first_dstand","revany_gap","rev_dist","rev_gs","rev_dstand","e_stand","e_appear","e_ci20","e_dend","e_dnext","e_mv5","e_mv10","e_mv20","e_rev","e_start","e_revany","e_pushend"]).unique(ID)
obc=["has_c","P_span","P_n","P_gap","P_dstand","P_long","seg_n","seg_rate","seg_start_gap","push","first_gs","first_dstand","revany_gap","rev_dist","rev_gs","rev_dstand"]
ecs=["e_stand","e_appear","e_ci20","e_dend","e_dnext","e_mv5","e_mv10","e_mv20","e_rev","e_start","e_revany","e_pushend"]
for c in ("src","prod_kind"):
    vals=sorted(set(ob[c].drop_nulls().cast(pl.Utf8).to_list())); maps["ob_"+c]={v:i for i,v in enumerate(vals)}
    ob=ob.with_columns(C(c).cast(pl.Utf8).replace_strict(maps["ob_"+c],default=-1,return_dtype=pl.Int32).fill_null(-1).alias("ob_"+c))
ob=ob.select([ID,"ob_src","ob_prod_kind"]+obc+[(C("mvt_t")-C(e)).alias("m_"+e) for e in ecs])
bx=pl.read_parquet(I+"agent_backx_bx_2025.parquet").unique(ID)
wx=pl.read_parquet(I+"agent_wx2_diag.parquet").select(["id","arr_late_share60","arr_late_share180","arr_late_mean60","arr_late_mean180"]).rename({"id":ID}).unique(ID)
for o in (cen,ob,bx,wx): d=d.join(o,on=ID,how="left")
d=d.sort("_r"); assert d.height==N
fcols=[c for c in d.columns if c not in (ID,"_r")]
del tf,cen,ob,bx,wx,tfl
import gc; gc.collect()
X=np.empty((N,len(fcols)+1),np.float32)
for j,c in enumerate(fcols): X[:,j]=d[c].cast(pl.Float32).to_numpy()
X[:,-1]=pv; fcols=fcols+["pv"]; del d; gc.collect()
assert "month" not in fcols and "doy" not in fcols
aptc=np.array([AIR.index(a) for a in z["apts"][keep].astype(str)])
np.savez(I+"agent_bigc_fold.npz",X=X,cols=np.array(fcols),y=y,pv=pv,p=m["p"],apt=z["apts"][keep].astype(str),mon=mon,ids=ids,
         cat=np.array([fcols.index(c) for c in LOW+["cov_cat","ob_src","ob_prod_kind"]]))
json.dump(maps,open(I+"agent_bigc_maps.json","w"))
print("X",X.shape,"nan frac mean",np.isnan(X).mean(),flush=True)
print("cov_cat nan",np.isnan(X[:,fcols.index('cov_cat')]).mean(),"m_e_start nan",np.isnan(X[:,fcols.index('m_e_start')]).mean())
