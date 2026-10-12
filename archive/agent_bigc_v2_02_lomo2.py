"""experiment bigc v2 02: ROUND-2 LOMO corrector on r2 = y - (pv + 0.75*R1_LOMO_seed). docs/LAB-NOTES.md
python archive/agent_bigc_v2_02_lomo2.py SEED REGS [MODE]   MODE = global | class ; out data/interim/agent_bigc_v2_{MODE}_s{SEED}.npz"""
import sys, numpy as np, lightgbm as lgb, time, json
I="data/interim/"; seed=int(sys.argv[1]); regs=sys.argv[2].split(","); mode=sys.argv[3] if len(sys.argv)>3 else "global"
z=np.load(I+"agent_bigc_fold.npz",allow_pickle=True); ex=np.load(I+"agent_bigc_extra.npz",allow_pickle=True)
X=np.hstack([z["X"],ex["Xf"]]); cols=list(z["cols"])+list(ex["cols"]); del ex
X3=mode.startswith("x3"); USE_D=mode in ("x3","x3a"); USE_TE=mode in ("x3","x3b")
if X3:
    sys.path.insert(0,"archive"); import agent_bigc_te as TE
    e2=np.load(I+"agent_bigc_extra2.npz",allow_pickle=True); n0=len(X); parts=[X]; nc=list(cols)
    if USE_D: parts.append(e2["Xf"]); nc+=["d_NEW","d_ET"]
    if USE_TE: parts.append(np.full((n0,4),np.nan,np.float32)); nc+=["te_"+c for c in TE.KEYCOLS]
    X=np.hstack(parts); cols=nc; KEYS=TE.keys_from(X,cols) if USE_TE else None
catidx=[int(i) for i in z["cat"]]; y=z["y"]; pv=z["pv"]; mon=z["mon"]; N=len(y)
b=pv+0.75*np.load(I+"agent_bigc_lomo_s%d.npz"%seed)["R1"] if not (mode.startswith("r1x") or mode.startswith("x3")) else pv; r=np.clip(y-b,-900,900).astype(np.float32); months=sorted(set(mon.tolist()))
R={"R1":dict(num_leaves=15,min_data_in_leaf=500,lambda_l2=10.0,R=600),"R2":dict(num_leaves=31,min_data_in_leaf=1000,lambda_l2=50.0,R=600),
   "C3":dict(num_leaves=255,min_data_in_leaf=300,lambda_l2=50.0,R=1500),"C3b":dict(num_leaves=255,min_data_in_leaf=150,lambda_l2=50.0,R=1500),"C4":dict(num_leaves=511,min_data_in_leaf=300,lambda_l2=100.0,R=1500),"C4b":dict(num_leaves=511,min_data_in_leaf=150,lambda_l2=100.0,R=1500),
   "C1":dict(num_leaves=63,min_data_in_leaf=300,lambda_l2=20.0,R=1200),"C2":dict(num_leaves=127,min_data_in_leaf=300,lambda_l2=50.0,R=1200)}
cov=X[:,cols.index("cov_cat")]; cm=json.load(open(I+"agent_bigc_maps.json"))["category"]
cls=np.full(N,3); cls[cov==cm["matched"]]=0; cls[cov==cm["points_no_liftoff"]]=1; cls[cov==cm["no_adsb_points"]]=2
def run(rows,rg):
    """LOMO within `rows` (bool mask); returns prediction array (0 outside), best iters"""
    prm=dict(R[rg]); nr=prm.pop("R"); P=dict(objective="l2",learning_rate=0.03,bagging_fraction=0.5,bagging_freq=1,feature_fraction=0.5,max_bin=63,num_threads=4,verbose=-1,seed=seed,**prm)
    d=np.zeros(N); bi=[]; imp=np.zeros(len(cols)); idx=np.where(rows)[0]
    for m in months:
        nxt=[k for k in months if k!=m]; val=min([k for k in nxt if k>m] or nxt)
        if USE_TE: X[:,-4:]=TE.te_fold(KEYS,r.astype(float),mon,m,months)
        tr=idx[(mon[idx]!=m)&(mon[idx]!=val)]; te=idx[mon[idx]==m]; va=idx[mon[idx]==val]
        dt=lgb.Dataset(X[tr],r[tr],categorical_feature=catidx); dv=lgb.Dataset(X[va],r[va],categorical_feature=catidx,reference=dt)
        bst=lgb.train(P,dt,nr,valid_sets=[dv],callbacks=[lgb.early_stopping(50,verbose=False)]); d[te]=bst.predict(X[te],num_iteration=bst.best_iteration); bi.append(bst.best_iteration); imp+=bst.feature_importance("gain")
        print(mode,rg,"m",m,"bi",bst.best_iteration,flush=True)
    return d,bi,imp
out={}; t0=time.time()
for rg in regs:
    if mode in ("global","r1x","x3","x3a","x3b"): d,bi,imp=run(np.ones(N,bool),rg); out[rg]=d; out[rg+"_bi"]=np.array(bi); out[rg+"_imp"]=imp
    else:
        d=np.zeros(N)
        for c in range(4): dc,bi,_=run(cls==c,rg); d+=dc; out["%s_c%d_bi"%(rg,c)]=np.array(bi)
        out[rg]=d
    np.savez(I+"agent_bigc_v2_%s_s%d_%s.npz"%(mode,seed,"-".join(regs)),**out); print("done",rg,"%.0fs"%(time.time()-t0),flush=True)
