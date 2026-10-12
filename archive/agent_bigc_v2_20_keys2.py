"""bigc Z: (a) pair keys, (b) LTFM-fold keys for the whole kept key set, (c) fold-side label-free drift features. docs/LAB-NOTES.md"""
import sys, glob, numpy as np, polars as pl
sys.path.insert(0,"archive"); import agent_bigc_drift as DR
C=pl.col; I="data/interim/"; ID="MVT_ID_mvt"; AIR=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
cols=[ID,"CALLSIGN_flt","ADES_mvt","AIRCRAFT_OPERATOR_flt","STAND_mvt","RUNWAY_mvt","AIRCRAFT_TYPE_mvt","MARKET_SEGMENT_flt","WK_TBL_CAT_flt","MVT_TIME_UTC_mvt","ADEP_mvt","PHASE_mvt"]
def prep(df):
    s=lambda c: C(c).cast(pl.Utf8).fill_null("NA"); hb=(C("MVT_TIME_UTC_mvt").dt.hour()//3).cast(pl.Utf8); A=s("ADEP_mvt"); op=s("AIRCRAFT_OPERATOR_flt"); st=s("STAND_mvt"); rw=s("RUNWAY_mvt"); ty=s("AIRCRAFT_TYPE_mvt"); ad=s("ADES_mvt")
    ex={"ades":[A,ad],"cty":[A,ad.str.slice(0,2)],"op_apt":[A,op],"op_stand":[A,op,st],"op_hb":[A,op,hb],"op_rwy":[A,op,rw],"st_rwy":[A,st,rw],"st_hb":[A,st,hb],"ty_st":[A,ty,st],"mkt":[A,s("MARKET_SEGMENT_flt")],"wk":[A,s("WK_TBL_CAT_flt")],
        "op_ades":[A,op,ad],"op_ty":[A,op,ty],"ty_rwy":[A,ty,rw],"ades_hb":[A,ad,hb],"st_op_rwy":[A,st,op,rw]}
    return df.select([C(ID)]+[pl.concat_str(v,separator="|").hash(seed=7).floordiv(2).cast(pl.Int64).alias(k) for k,v in ex.items()]), list(ex)
def keys_for(ids):
    idf=pl.DataFrame({ID:ids}); fr=[]
    for f in sorted(glob.glob("data/raw/training_2025-*.parquet")):
        d=pl.read_parquet(f,columns=cols).filter(C("PHASE_mvt")=="DEP").join(idf,on=ID,how="semi"); fr.append(prep(d)[0])
    names=prep(pl.read_parquet("data/raw/ranking.parquet",columns=cols,n_rows=5))[1]
    j=idf.with_row_index("_r").join(pl.concat(fr).unique(ID),on=ID,how="left").sort("_r"); return j.select(names).to_numpy().astype(np.int64),names
z=np.load(I+"agent_bigc_fold.npz",allow_pickle=True); zl=np.load(I+"agent_bigc_ltfm_fold.npz",allow_pickle=True); k0=np.load(I+"agent_bigc_keys.npz",allow_pickle=True)
Kf,names=keys_for(z["ids"]); Kl,_=keys_for(zl["ids"])
base=pl.read_parquet(I+"v137_ltfmsnow_final.parquet").sort(ID); rk=pl.read_parquet("data/raw/ranking.parquet",columns=cols).filter(C("PHASE_mvt")=="DEP"); ks,_=prep(rk)
Ks=base.select(ID).with_row_index("_r").join(ks.unique(ID),on=ID,how="left").sort("_r").select(names).to_numpy().astype(np.int64)
is9=np.isin(k0["apt_s"],AIR); mon=z["mon"]; new=["op_ades","op_ty","ty_rwy","ades_hb","st_op_rwy"]; keep=[]
for nm in new:
    k=names.index(nm); u=np.unique(Kf[:,k]); un_s=(~np.isin(Ks[is9,k],u)).mean(); un_f=np.mean([(~np.isin(Kf[mon==m,k],np.unique(Kf[mon!=m,k]))).mean() for m in sorted(set(mon.tolist()))])
    print("%-9s groups %7d | unseen serve(9apt) %.4f fold-LOMO %.4f -> %s"%(nm,len(u),un_s,un_f,"keep" if un_s<=0.2 else "DROP"),flush=True); 
    if un_s<=0.2: keep.append(nm)
kept_all=[str(n) for n in k0["names"]]+keep; idx=[names.index(n) for n in kept_all]
# LTFM unseen audit for the kept key set (LTFM serve rows vs LTFM fold rows)
isL=k0["apt_s"]=="LTFM"
for nm in kept_all:
    k=names.index(nm); print("LTFM %-9s unseen serve %.4f"%(nm,(~np.isin(Ks[isL,k],np.unique(Kl[:,k]))).mean()))
# fold drift (label-free): keys op_apt, op_stand, st_rwy, st_hb ; pv and ADS-B span (m_e_start)
cols_=list(z["cols"]); span=z["X"][:,cols_.index("m_e_start")].astype(float); dk=[names.index(n) for n in ["op_apt","op_stand","st_rwy","st_hb"]]
Df=DR.drift_fold(Kf[:,dk],z["pv"].astype(float),span,mon); print("drift fold",Df.shape,"sd per col",np.round(Df.std(0),2))
np.savez(I+"agent_bigc_keys2.npz",Kf=Kf[:,idx],Ks=Ks[:,idx],Kl=Kl[:,idx],names=np.array(kept_all),Kf_d=Kf[:,dk],Ks_d=Ks[:,dk],Kl_d=Kl[:,dk],Df=Df)
