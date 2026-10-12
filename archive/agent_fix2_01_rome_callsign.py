"""fix2 lever C: callsign-keyed reach of the shipped hidden-plan rules at Rome (LIRF). Rule text copied from
apply_upper_bound.py / apply_hidden_reading.py (W 3606, 0-8 h asof forward join, plan origin = our ADEP, gl >= W, |gs-gl| < W,
value = clip(rd, 0, gl-W); upper-bound cap only where the plan has no AOBT_3). Only the arrival-row KEY differs (CALLSIGN_flt).
No fitted parameter. Writes data/interim/agent_fix2_c_rows.parquet (2026 targets) and prints the 2025 evaluation."""
import numpy as np, polars as pl
C=pl.col; R=""
ID="MVT_ID_mvt"; W=3606.0
APTS=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH","LTFM"]
U=lambda c:C(c).str.strip_chars().str.to_uppercase()
def build(path, with_y):
    sc=pl.scan_parquet(path)
    ycol=[C("TAXITIME_SEC_mvt").alias("y")] if with_y else []
    dep=(sc.filter((C("PHASE_mvt")=="DEP")&C("AOBT_3_flt").is_null()&(C("ADEP_mvt")=="LIRF")&C("ADES_mvt").is_in(APTS)&C("FLIGHT_mvt").is_not_null())
         .select(C(ID),C("ADEP_mvt").alias("adep"),C("ADES_mvt").alias("ades"),U("FLIGHT_mvt").alias("flight"),C("MVT_TIME_UTC_mvt").alias("mvt"),
                 C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),(C("MVT_TIME_UTC_mvt")-C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),*ycol).collect().sort("mvt"))
    arrb=sc.filter((C("PHASE_mvt")=="ARR")&C("LOBT_flt").is_not_null())
    out={}
    for name,key in (("fl",U("FLIGHT_mvt")),("cs",U("CALLSIGN_flt"))):
        arr=(arrb.select(C("ADES_mvt").alias("ades"),key.alias("flight"),C("MVT_TIME_UTC_mvt").alias("arr_t"),C("LOBT_flt").alias("lobt"),
             C("ADEP_flt").alias("plan_adep"),C("AOBT_3_flt").alias("aobt")).filter(C("flight").is_not_null()).collect().sort("arr_t"))
        rec=(dep.join_asof(arr,left_on="mvt",right_on="arr_t",by=["ades","flight"],strategy="forward",tolerance="8h")
             .filter(C("lobt").is_not_null()&(C("plan_adep")==C("adep")))
             .with_columns((C("mvt")-C("lobt")).dt.total_seconds().alias("gl"),(C("mvt")-C("aobt")).dt.total_seconds().alias("rd"))
             .filter((C("gl")>=W)&((C("gs")-C("gl")).abs()<W)))
        out[name]=rec
    return dep,out
res={}
for yr,path,wy in ((2025,R+"data/raw/training_*.parquet",True),(2026,R+"data/raw/ranking.parquet",False)):
    dep,o=build(path,wy)
    fl,cs=o["fl"],o["cs"]
    new=cs.filter(~C(ID).is_in(fl[ID].to_list())).with_columns(pl.lit(yr).alias("yr"))
    print(yr,"LIRF no-plan dep rows",dep.height,"| in scope by FLIGHT key",fl.height,"(aobt null",fl["aobt"].is_null().sum(),") | by callsign key",cs.height,"| callsign-only",new.height,"(aobt null",new["aobt"].is_null().sum(),")")
    # agreement where both caught
    both=fl.join(cs,on=ID,suffix="_c")
    print("  both keys:",both.height,"same arrival",(both["arr_t"]==both["arr_t_c"]).sum())
    res[yr]=new
n25,n26=res[2025],res[2026]
def tgt(df,base):
    # value rule: reading where plan has AOBT_3, else the upper-bound cap
    cap=(df["gl"]-W).to_numpy(); rd=df["rd"].to_numpy().astype(float); has=df["aobt"].is_not_null().to_numpy()
    reading=np.clip(np.where(has,rd,np.nan),0,cap)
    return np.where(has,reading,np.clip(np.minimum(base,cap),0,None))
# 2025 eval against fold control (agent_rome2_dep.ctrl: production-like out-of-fold, months 1,2,3,7,8,9,10,11)
d=pl.read_parquet(R+"data/interim/agent_rome2_dep.parquet").filter(C("yr")==2025).select(C(ID),"ctrl","prod")
e=n25.join(d,on=ID,how="left").filter(C("ctrl").is_not_null())
base=e["ctrl"].to_numpy().astype(float); t=tgt(e,base); y=e["y"].to_numpy().astype(float)
g=(y-base)**2-(y-t)**2
e=e.with_columns(pl.Series("g",g),pl.Series("t",t),pl.Series("b",base))
N={1:130045,7:190713}; S={1:9538865220.97,7:15100920109.88}
print("2025 callsign-only rows",n25.height,"with fold ctrl",e.height)
print(e.group_by("mo").agg(pl.len().alias("n"),C("g").sum().alias("sse_gain"),(C("y")-C("b")).abs().mean().alias("mae_ctrl"),(C("y")-C("t")).abs().mean().alias("mae_new"),((C("y")-C("t")).abs()<=600).mean().alias("within600"),(C("y")<=C("gl")-W+1).mean().alias("below_cap")).sort("mo"))
for M in (1,7):
    gm=e.filter(C("mo")==M)["g"].sum(); print("month",M,"total-RMSE s",np.sqrt(S[M]/N[M])-np.sqrt((S[M]-gm)/N[M]))
gg=e.filter(C("mo").is_in([1,7]))["g"].to_numpy(); Ns=N[1]+N[7]; Ss=S[1]+S[7]
gs_=np.sort(gg)[::-1]
print("months1+7 pooled",np.sqrt(Ss/Ns)-np.sqrt((Ss-gg.sum())/Ns),"top10 share",gs_[:10].sum()/gg.sum(),"minus-10",np.sqrt(Ss/Ns)-np.sqrt((Ss-(gg.sum()-gs_[:10].sum()))/Ns))
ga=e["g"].to_numpy(); print("all fold months SSE gain",ga.sum(),"rows gaining",(ga>0).sum(),"losing",(ga<0).sum(),"top10 share",np.sort(ga)[::-1][:10].sum()/ga.sum())
print("violations of the upper bound (label above gl-W):",(y>(e['gl'].to_numpy()-W)+1).sum(),"of",e.height, "| all 2025 callsign-only rows (any month) label above cap:",(n25['y'].to_numpy()>(n25['gl'].to_numpy()-W)+1).sum(),"of",n25.height)
# 2026
v=pl.read_parquet(R+"data/interim/agent_bundle_v132_final.parquet")
f=n26.join(v.rename({"TAXITIME_SEC_mvt":"v132"}),on=ID,how="left")
b26=f["v132"].to_numpy().astype(float); t26=np.rint(tgt(f,b26))
f=f.with_columns(pl.Series("target",t26)).select(ID,"mo","flight","gs","gl","rd","aobt","v132","target")
f.write_parquet(R+"data/interim/agent_fix2_c_rows.parquet")
print(f.sort("gl"))
d26=t26-b26; print("2026: rows",f.height,"moved",(d26!=0).sum(),"rms move",np.sqrt((d26[d26!=0]**2).mean()) if (d26!=0).any() else 0,"sum dp^2",(d26**2).sum(),"-> total RMSE s if v132 were truth-equal to target:",(d26**2).sum()/(2*344841*221.65))
