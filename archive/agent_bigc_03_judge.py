"""experiment bigc 03: judge saved LOMO predictions. python agent_bigc_03_judge.py SEED[,SEED...]"""
import sys, numpy as np
I="data/interim/"; z=np.load(I+"agent_bigc_fold.npz",allow_pickle=True); y=z["y"]; pv=z["pv"]; a=z["apt"]; mon=z["mon"]; N=len(y)
AIR=sorted(set(a.tolist())); H1=np.isin(mon,[1,2,3]); H2=np.isin(mon,[7,8,9]); MONTHS=sorted(set(mon.tolist()))
rm=lambda q,k: np.sqrt(((y-q)[k]**2).mean()); ALL=np.ones(N,bool)
def judge(q):
    d=rm(q,ALL)-rm(pv,ALL); per={ap:rm(q,a==ap)-rm(pv,a==ap) for ap in AIR}
    g=(y-pv)**2-(y-q)**2; kk=np.ones(N,bool); kk[np.argsort(-g)[:50]]=False
    m50=rm(q,kk)-rm(pv,kk); h1=rm(q,H1)-rm(pv,H1); h2=rm(q,H2)-rm(pv,H2); pm={m:rm(q,mon==m)-rm(pv,mon==m) for m in MONTHS}
    nneg=sum(v<0 for v in pm.values())
    clauses=[d<0,max(per.values())<=0.10,m50<0,h1<0 and h2<0,nneg>=6]
    return d,per,m50,h1,h2,pm,nneg,clauses
if __name__=="__main__":
    print("control RMSE %.4f n=%d"%(rm(pv,ALL),N))
    for s in sys.argv[1].split(","):
        L=np.load(I+"agent_bigc_lomo_s%s.npz"%s)
        for rg in ("R1","R2","R3"):
            if rg not in L.files: continue
            for k in (0.25,0.5,0.75,1.0):
                d,per,m50,h1,h2,pm,nneg,cl=judge(pv+k*L[rg])
                print("s%s %s k=%.2f all %+.4f worst %s %+.3f m50 %+.4f JanMar %+.4f JulSep %+.4f months- %d/8 clauses %s %s"%(s,rg,k,d,max(per,key=per.get),max(per.values()),m50,h1,h2,nneg,"".join("Y" if c else "n" for c in cl),"PASS" if all(cl) and d<=-0.10 else ("pass<-0.10?" if all(cl) else "")),flush=True)
