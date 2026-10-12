"""Label-free arrival-side / input-side state features for LIRF departures (serve-safe), plus label-based neighbour oracles (2025 only)."""
import polars as pl, numpy as np
C=pl.col
R=""
d=pl.read_parquet(R+"data/interim/agent_rome2_depL.parquet")
a=pl.read_parquet(R+"data/interim/agent_rome2_arr.parquet")
ep=lambda col: col.dt.epoch("s")
d=d.with_columns(ep(C("MVT_TIME_UTC_mvt")).alias("T"),ep(C("SCHED_TIME_UTC_mvt")).alias("S")).with_row_index("ri")
a=a.with_columns(ep(C("t_in")).alias("tin"),ep(C("t_sched")).alias("ts")).sort("tin")
# fl numbers
num=lambda c: C(c).str.extract(r"(\d+)",1).cast(pl.Float64)
a=a.with_columns(num("fl").alias("fnum")); d=d.with_columns(num("FLIGHT_mvt").alias("fnum"))
n=d.height
out={}
T=d["T"].to_numpy().astype(np.int64); yr=d["yr"].to_numpy()
tin=a["tin"].to_numpy().astype(np.int64); acopy=a["copy"].to_numpy().astype(np.float64); anp=a["np"].to_numpy().astype(np.float64); ayr=a["yr"].to_numpy()
# window helper over sorted arrays: returns count, copy sum, np sum in [T-w, T+w]
def win(times,vals,Tq,w,self_excl=None):
    lo=np.searchsorted(times,Tq-w,"left"); hi=np.searchsorted(times,Tq+w,"right")
    cs=np.r_[0,np.cumsum(vals,axis=0)] if vals.ndim==1 else np.vstack([np.zeros((1,vals.shape[1])),np.cumsum(vals,axis=0)])
    return hi-lo,(cs[hi]-cs[lo])
V=np.column_stack([acopy,anp])
for w,nm in [(3600,"w1h"),(10800,"w3h"),(43200,"w12h")]:
    cnt,s=win(tin,V,T,w); out[f"arr_n_{nm}"]=cnt.astype(float); out[f"arr_copy_{nm}"]=np.where(cnt>0,s[:,0]/np.maximum(cnt,1),np.nan); out[f"arr_np_{nm}"]=np.where(cnt>0,s[:,1]/np.maximum(cnt,1),np.nan)
# per-group (op / stand) windows and links
op=d["op"].to_numpy(); st=d["STAND_mvt"].to_numpy(); fnum=d["fnum"].to_numpy()
aop=a["op"].to_numpy(); ast=a["stand"].to_numpy(); afn=a["fnum"].to_numpy()
def grp_idx(keys):
    ix={}
    for i,k in enumerate(keys):
        if k is None: continue
        ix.setdefault(k,[]).append(i)
    return {k:np.array(v) for k,v in ix.items()}
gop=grp_idx(aop); gst=grp_idx(ast)
feat_names=["op6_n","op6_copy","op6_np","st6_n","st6_copy","L1_exists","L1_copy","L1_np","L1_age","L1_sameop","L2_exists","L2_copy","L2_np","L2_age","L2_fdiff","Nx_exists","Nx_gap","Nx_copy"]
F={k:np.full(n,np.nan) for k in feat_names}
for k,ix in gop.items():
    q=np.where(op==k)[0]
    if len(q)==0: continue
    tt=tin[ix]; cnt,s=win(tt,V[ix],T[q],21600); F["op6_n"][q]=cnt; F["op6_copy"][q]=np.where(cnt>0,s[:,0]/np.maximum(cnt,1),np.nan); F["op6_np"][q]=np.where(cnt>0,s[:,1]/np.maximum(cnt,1),np.nan)
for k,ix in gst.items():
    q=np.where(st==k)[0]
    if len(q)==0: continue
    tt=tin[ix]; cnt,s=win(tt,V[ix],T[q],21600); F["st6_n"][q]=cnt; F["st6_copy"][q]=np.where(cnt>0,s[:,0]/np.maximum(cnt,1),np.nan)
    # L1: latest arrival at same stand with tin <= T-600 and >= T-86400 (same year)
    pos=np.searchsorted(tt,T[q]-600,"right")-1
    ok=(pos>=0)
    pp=np.where(ok,pos,0); a_i=ix[pp]; age=T[q]-tin[a_i]
    good=ok&(age<=86400)&(ayr[a_i]==yr[q])
    F["L1_exists"][q]=good.astype(float)
    F["L1_copy"][q]=np.where(good,acopy[a_i],np.nan); F["L1_np"][q]=np.where(good,anp[a_i],np.nan); F["L1_age"][q]=np.where(good,age,np.nan)
    F["L1_sameop"][q]=np.where(good,(aop[a_i]==op[q]).astype(float),np.nan)
    # L2: among arrivals at the same stand in [T-8h, T-600] choose latest with same op
    for jq,qi in enumerate(q):
        if op[qi] is None: F["L2_exists"][qi]=0; continue
        lo=np.searchsorted(tt,T[qi]-28800,"left"); hi=np.searchsorted(tt,T[qi]-600,"right")
        best=-1
        for j in range(hi-1,lo-1,-1):
            if aop[ix[j]]==op[qi] and ayr[ix[j]]==yr[qi]: best=ix[j]; break
        if best<0: F["L2_exists"][qi]=0
        else:
            F["L2_exists"][qi]=1; F["L2_copy"][qi]=acopy[best]; F["L2_np"][qi]=anp[best]; F["L2_age"][qi]=T[qi]-tin[best]
            F["L2_fdiff"][qi]=(fnum[qi]-afn[best]) if not (np.isnan(fnum[qi]) or np.isnan(afn[best])) else np.nan
    # next arrival at the same stand after T (stand reuse)
    pos2=np.searchsorted(tt,T[q],"right"); ok2=pos2<len(tt); p2=np.where(ok2,pos2,0); a2=ix[p2]
    g2=ok2&(ayr[a2]==yr[q])&((tin[a2]-T[q])<=86400)
    F["Nx_exists"][q]=g2.astype(float); F["Nx_gap"][q]=np.where(g2,tin[a2]-T[q],np.nan); F["Nx_copy"][q]=np.where(g2,acopy[a2],np.nan)
for k,v in F.items(): out[k]=v
# dep-side np density (serve-safe input): np share among deps within +-1h by T (excl self) and same op same day
npv=d["np"].to_numpy().astype(np.float64); order=np.argsort(T,kind="stable"); Ts=T[order]; nps=npv[order]
cs=np.r_[0,np.cumsum(nps)]
lo=np.searchsorted(Ts,T-3600,"left"); hi=np.searchsorted(Ts,T+3600,"right")
tot=hi-lo; s=cs[hi]-cs[lo]
rank=np.empty(n,int); rank[order]=np.arange(n)
out["dep_np_1h"]=(s-nps[rank])/np.maximum(tot-1,1)
lo=np.searchsorted(Ts,T-10800,"left"); hi=np.searchsorted(Ts,T+10800,"right"); tot=hi-lo; s=cs[hi]-cs[lo]
out["dep_np_3h"]=(s-nps[rank])/np.maximum(tot-1,1)
# label oracle neighbours (2025 labelled rows only): informative copy share among OTHER informative deps within +-1h, and same op +-12h
inf=((d["gs"].to_numpy()>=1800)&(yr==2025)); cr=d["copy_raw"].to_numpy().astype(np.float64)
ii=np.where(inf)[0]; Ti=T[ii]; oi=np.argsort(Ti,kind="stable"); Tsi=Ti[oi]; ci=cr[ii][oi]; css=np.r_[0,np.cumsum(ci)]
lo=np.searchsorted(Tsi,T-3600,"left"); hi=np.searchsorted(Tsi,T+3600,"right"); cnt=hi-lo; sm=css[hi]-css[lo]
selfc=np.where(inf,cr,0.0); selfn=inf.astype(float)
out["or_copy_1h"]=np.where(yr==2025,(sm-selfc)/np.maximum(cnt-selfn,1),np.nan); out["or_n_1h"]=np.where(yr==2025,cnt-selfn,np.nan)
# same op +-12h
orf=np.full(n,np.nan)
for k in np.unique([x for x in op if x is not None]):
    q=np.where((op==k))[0]
    qi=q[inf[q]]
    if len(qi)==0: continue
    tt=T[qi]; o2=np.argsort(tt,kind="stable"); tts=tt[o2]; cc=cr[qi][o2]; cs2=np.r_[0,np.cumsum(cc)]
    lo=np.searchsorted(tts,T[q]-43200,"left"); hi=np.searchsorted(tts,T[q]+43200,"right"); cnt=hi-lo; sm=cs2[hi]-cs2[lo]
    sc=np.where(inf[q],cr[q],0.0); sn=inf[q].astype(float)
    orf[q]=(sm-sc)/np.maximum(cnt-sn,1); orf[q[(cnt-sn)<1]]=np.nan
orf[yr!=2025]=np.nan
out["or_op12_copy"]=orf
res=d.select("ri","MVT_ID_mvt","yr","mo","np","gs","y","ctrl","prod","op","STAND_mvt","T","S","copy_raw","gl","ga","ge","gi","FLIGHT_mvt","fnum","AIRCRAFT_TYPE_mvt").with_columns([pl.Series(k,v) for k,v in out.items()])
res.write_parquet(R+"data/interim/agent_rome2_sensor.parquet")
print(res.shape)
print(res.group_by("yr").agg(C("L1_exists").mean(),C("L2_exists").mean(),C("arr_copy_w3h").mean()))
print(res.filter(C("np")).group_by("yr").agg(C("L1_exists").mean(),C("L2_exists").mean(),C("L2_age").median(),C("arr_copy_w3h").mean()))
