"""bigc Y step: key table for the rich out-of-month target stats (fold rows + 2026 serve rows) with unseen-key audit; drops keys with > 20% unseen on 2026 9-apt rows. docs/LAB-NOTES.md"""
import sys, glob, numpy as np, polars as pl
sys.path.insert(0,"archive"); import agent_bigc_te as TE
C=pl.col; I="data/interim/"; ID="MVT_ID_mvt"; AIR=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
z=np.load(I+"agent_bigc_fold.npz",allow_pickle=True); ids=z["ids"]; apt_f=z["apt"].astype(str); mon=z["mon"]
cols=[ID,"CALLSIGN_flt","ADES_mvt","AIRCRAFT_OPERATOR_flt","STAND_mvt","RUNWAY_mvt","AIRCRAFT_TYPE_mvt","MARKET_SEGMENT_flt","WK_TBL_CAT_flt","MVT_TIME_UTC_mvt","ADEP_mvt","PHASE_mvt"]
def prep(df):
    s=lambda c: C(c).cast(pl.Utf8).fill_null("NA")
    hb=(C("MVT_TIME_UTC_mvt").dt.hour()//3).cast(pl.Utf8)
    A=s("ADEP_mvt")
    ex={"fn":[A,s("CALLSIGN_flt")],"ades":[A,s("ADES_mvt")],"cty":[A,s("ADES_mvt").str.slice(0,2)],"op_apt":[A,s("AIRCRAFT_OPERATOR_flt")],"op_stand":[A,s("AIRCRAFT_OPERATOR_flt"),s("STAND_mvt")],
        "op_hb":[A,s("AIRCRAFT_OPERATOR_flt"),hb],"op_rwy":[A,s("AIRCRAFT_OPERATOR_flt"),s("RUNWAY_mvt")],"st_rwy":[A,s("STAND_mvt"),s("RUNWAY_mvt")],"st_hb":[A,s("STAND_mvt"),hb],
        "ty_st":[A,s("AIRCRAFT_TYPE_mvt"),s("STAND_mvt")],"mkt":[A,s("MARKET_SEGMENT_flt")],"wk":[A,s("WK_TBL_CAT_flt")]}
    return df.select([C(ID)]+[pl.concat_str(v,separator="|").hash(seed=7).floordiv(2).cast(pl.Int64).alias(k) for k,v in ex.items()]), list(ex)
fr=[]; idf=pl.DataFrame({ID:ids})
for f in sorted(glob.glob("data/raw/training_2025-*.parquet")):
    d=pl.read_parquet(f,columns=cols).filter(C("PHASE_mvt")=="DEP").join(idf,on=ID,how="semi"); fr.append(prep(d)[0])
Kf_df=pl.concat(fr).unique(ID); names=prep(pl.read_parquet("data/raw/ranking.parquet",columns=cols,n_rows=5))[1]
j=idf.with_row_index("_r").join(Kf_df,on=ID,how="left").sort("_r"); Kf=j.select(names).to_numpy().astype(np.int64); assert j.height==len(ids)
base=pl.read_parquet(I+"v137_ltfmsnow_final.parquet").sort(ID); rk=pl.read_parquet("data/raw/ranking.parquet",columns=cols).filter(C("PHASE_mvt")=="DEP")
ks,_=prep(rk); js=base.select(ID).with_row_index("_r").join(ks.unique(ID),on=ID,how="left").sort("_r"); Ks=js.select(names).to_numpy().astype(np.int64)
apt_s=np.array(base.join(rk.select(ID,"ADEP_mvt").unique(ID),on=ID,how="left")["ADEP_mvt"].to_list()).astype(str); is9=np.isin(apt_s,AIR); nullk=(Kf==Kf[0,0]).all()
print("fold rows",len(ids),"serve rows",len(Ks),"9apt serve",int(is9.sum()))
keep=[]
for k,nm in enumerate(names):
    u=np.unique(Kf[:,k]); un_s=(~np.isin(Ks[is9,k],u)).mean()
    un_f=np.mean([(~np.isin(Kf[mon==m,k],np.unique(Kf[mon!=m,k]))).mean() for m in sorted(set(mon.tolist()))])
    ok=un_s<=0.20; print("%-9s groups %7d | unseen share serve(9apt) %.4f  fold-LOMO %.4f  -> %s"%(nm,len(u),un_s,un_f,"keep" if ok else "DROP (>20%)"),flush=True)
    if ok: keep.append(k)
np.savez(I+"agent_bigc_keys.npz",Kf=Kf[:,keep],Ks=Ks[:,keep],names=np.array([names[k] for k in keep]),apt_f=apt_f,apt_s=apt_s)
