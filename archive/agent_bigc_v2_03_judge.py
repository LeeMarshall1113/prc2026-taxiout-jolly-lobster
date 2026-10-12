"""judge v2 increments: python agent_bigc_v2_03_judge.py MODE SEEDS [REGS]; base b = pv + .75*R1_s"""
import sys, numpy as np
sys.path.insert(0,"archive"); import agent_bigc_03_judge as J
I="data/interim/"; mode=sys.argv[1]; seeds=sys.argv[2].split(","); regs=sys.argv[3].split(",") if len(sys.argv)>3 else None
for s in seeds:
    R1b=J.pv+0.75*np.load(I+"agent_bigc_lomo_s%s.npz"%s)["R1"]; b=J.pv if mode.startswith(("r1x","x3")) else R1b; L=np.load(I+"agent_bigc_v2_%s_s%s%s.npz"%(mode,s,(sys.argv[4] if len(sys.argv)>4 else ""))); C2b=J.pv+0.75*np.load(I+"agent_bigc_v2_r1x_s%s_C3X.npz"%s)["C3X"]  # C3Xb: control for Y runs; print("s%s base b RMSE %.4f"%(s,J.rm(b,J.ALL)))
    for rg in [k for k in L.files if not k.endswith("_bi") and not k.endswith("_imp")]:
        if regs and rg not in regs: continue
        for k in (0.25,0.5,0.75,1.0):
            q=b+k*L[rg]; d=J.rm(q,J.ALL)-J.rm(b,J.ALL); per={ap:J.rm(q,J.a==ap)-J.rm(b,J.a==ap) for ap in J.AIR}
            g=(J.y-b)**2-(J.y-q)**2; kk=np.ones(J.N,bool); kk[np.argsort(-g)[:50]]=False; m50=J.rm(q,kk)-J.rm(b,kk)
            h1=J.rm(q,J.H1)-J.rm(b,J.H1); h2=J.rm(q,J.H2)-J.rm(b,J.H2); pm=[J.rm(q,J.mon==m)-J.rm(b,J.mon==m) for m in J.MONTHS]; nn=sum(v<0 for v in pm)
            cl=[d<0,max(per.values())<=0.10,m50<0,h1<0 and h2<0,nn>=6]
            print("s%s %s k=%.2f all(vs b) %+.4f worst %s %+.3f m50 %+.4f JanMar %+.4f JulSep %+.4f mo- %d/8 %s %s | vs pv %+.4f | vs R1 %+.4f vs C2 %+.4f"%(s,rg,k,d,max(per,key=per.get),max(per.values()),m50,h1,h2,nn,"".join("Y" if c else "n" for c in cl),"PASS" if all(cl) else "",J.rm(q,J.ALL)-J.rm(J.pv,J.ALL),J.rm(q,J.ALL)-J.rm(R1b,J.ALL),J.rm(q,J.ALL)-J.rm(C2b,J.ALL)),flush=True)
