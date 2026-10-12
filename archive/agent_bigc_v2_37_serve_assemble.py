"""Assemble a Modal-fitted serving term (modal/agent_bigc_app.py::serve_one) into a board file.
python archive/agent_bigc_v2_37_serve_assemble.py CFG KAPT.json FOLD_NPZ FOLD_TREAT_NPZ OUT.parquet [BASE=agent_combo_v129_final.parquet] [OLD_DELTA=agent_bigc_v7_serve_delta.npz] [CTRL_FOLD_NPZ=agent_bigc_v7_fold_treat.npz]   (fold npz may carry kold = the old term's per-airport k)
 CFG e.g. C4ff7V3 (seeds 0,1,2 from data/interim/bigc_serve/<cfg>_s<seed>.npz, downloaded by archive/agent_bigc_v2_12_dl.py). Replaces the OLD term (the corrector already inside BASE) by
 kvec x mean-seed corr on the nine airports, leaves every other row (LTFM included) as in BASE; then runs the check, the lens (--sub-ctrl BASE --board-ctrl 222.91) and prints the footprint ratio + group gain shares."""
import sys, json, glob, subprocess, numpy as np, polars as pl
cfg,kapt,foldnpz,foldtreat,out=sys.argv[1:6]; base_f=sys.argv[6] if len(sys.argv)>6 else "agent_combo_v129_final.parquet"; old_f=sys.argv[7] if len(sys.argv)>7 else "agent_bigc_v7_serve_delta.npz"; ctrl_f=sys.argv[8] if len(sys.argv)>8 else "agent_bigc_v7_fold_treat.npz"
I="data/interim/"; ID="MVT_ID_mvt"; T="TAXITIME_SEC_mvt"; AIR=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
fs=sorted(glob.glob(I+"bigc_serve/%s_s*.npz"%cfg)); assert len(fs)>=1, "no serve fits found"; print("seed files:",[f.split("/")[-1] for f in fs])
D=[np.load(f,allow_pickle=True) for f in fs]; corr=np.mean([d["pred"] for d in D],0); imp=np.mean([d["imp"] for d in D],0); names=[str(n) for n in D[0]["names"]]
si=np.load(I+"agent_bigc_serve_in.npz",allow_pickle=True); ids=si["ids"]; apt=si["apt"].astype(str); is9=np.isin(apt,AIR)
KAP=json.load(open(I+kapt)); kvec=np.array([KAP.get(a,0.0) for a in apt]); dp=kvec*corr; dp[~is9]=0
old=np.load(I+old_f,allow_pickle=True); assert (old["ids"]==ids).all(); dpn=dp-old["dp"]
bf=pl.read_parquet(I+base_f).sort(ID); assert (bf[ID].to_numpy()==ids).all(); cur=bf[T].cast(pl.Float64).to_numpy(); o=np.rint(np.maximum(cur+dpn,0)); mv=o!=cur
bf.with_columns(pl.Series(T,o).cast(bf.schema[T])).write_parquet(I+out)
print("rows moved",int(mv.sum()),"mean shift (all rows) %.3f"%float((o-cur).mean()),"rms dp(moved) %.2f"%float(np.sqrt(((o-cur)[mv]**2).mean())),"max %.0f"%np.abs(o-cur).max(),"sum dp^2/row %.2f"%float(((o-cur)**2).mean()))
if len(D)>1: print("seed-to-seed corr of the serve term:",np.round(np.corrcoef([d["pred"][is9] for d in D])[0,1:],4))
fz=np.load(I+foldnpz); af=np.load(I+"agent_bigc_fold.npz",allow_pickle=True)["apt"].astype(str); kf=np.array([dict(zip(fz["air"],fz["kap"])).get(a,0.0) for a in af]); k7f=np.array([dict(zip(fz["air"],fz["kold"])).get(a,0.0) for a in af]) if "kold" in fz.files else np.array([json.load(open(I+"agent_bigc_v7_kapt.json")).get(a,0.0) for a in af])
dpf=kf*fz["f"]-k7f*fz["f7"]; print("fold sum dp^2/row %.2f -> footprint ratio 2026(9apt)/fold %.2f"%(float((dpf**2).mean()),float((dpn[is9]**2).mean()/(dpf**2).mean())))
print("LTFM rows changed vs base:",int((o[apt=="LTFM"]!=cur[apt=="LTFM"]).sum()))
vc=[i for i,n in enumerate(names) if n.startswith("vf_")]; tot=imp.sum()
def grp(pred): return 100*imp[[i for i,n in enumerate(names) if pred(n)]].sum()/tot
print("gain share: vfac total %.1f%% | quantiles %.1f%% | spreads/skew %.1f%% | seed stats %.1f%% | families %.1f%% | key stats (te_/*_mean.. ) %.1f%% | drift %.1f%%"%(grp(lambda n:n.startswith("vf_")),grp(lambda n:n.startswith("vf_d_q")),grp(lambda n:n.startswith("vf_sp_") or n=="vf_skew"),grp(lambda n:"seed_" in n and n.startswith("vf_")),
  grp(lambda n:n in ("vf_d_knn","vf_d_mlp","vf_d_rf","vf_d_ridge")),grp(lambda n:n.startswith("te_") or n.split("_")[-1] in ("mean","med","sd","cnt","p90","tail","ymean")),grp(lambda n:n.startswith("drift_"))))
env=dict(__import__("os").environ,PYTHONIOENCODING="utf-8",POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage")
r=subprocess.run([sys.executable,"archive/agent_smallbundle_06_check.py",I+out,I+base_f],capture_output=True,text=True,env=env); print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in ("FAIL","moved","ALL","shift","corr"))))
ABS=sys.executable.replace("\\","/")
r=subprocess.run([sys.executable,"archive/capped_run.py","--cap-gb","3","--",ABS,"archive/agent_lens_09_predict.py","--fold-ctrl",I+ctrl_f,"--fold-treat",I+foldtreat,"--sub-ctrl",I+base_f,"--sub-treat",I+out,"--board-ctrl","222.91"],capture_output=True,text=True,env=env)
print("\n".join(l for l in r.stdout.splitlines() if "PREDICTED" in l or l.startswith("fold:")))
