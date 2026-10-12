"""bigc ladder step 3 inputs: d_NEW, d_ET chain differences (fold rows + 2026 serve rows), same definitions as agent_meta_01/03. docs/LAB-NOTES.md"""
import numpy as np, polars as pl
I="data/interim/"; ID="MVT_ID_mvt"; T="TAXITIME_SEC_mvt"; AIR=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
L=lambda f: np.load(I+f,allow_pickle=True)
sa=L("agent_bulkr_fold_SA_s0.npz"); keep=np.isin(sa["apts"].astype(str),AIR); z=np.load(I+"agent_bigc_fold.npz",allow_pickle=True); assert (sa["ids"][keep]==z["ids"]).all()
p=sa["both_guard"][keep].astype(float); new=L("agent_bulkr_fold_NEW_s0.npz")["both_guard"][keep]; ctrl=L("agent_etrs_fold_CTRL_s0.npz")["both_guard"][keep]; et=L("agent_etrs_fold_ET_s0.npz")["both_guard"][keep]
Xf=np.column_stack([new-p,et-ctrl]).astype(np.float32)
base=pl.read_parquet(I+"agent_bndl_board_final.parquet").sort(ID); c26=pl.read_parquet(I+"agent_coverage_census_2026.parquet").sort(ID); assert np.array_equal(c26[ID].to_numpy(),base[ID].to_numpy())
a26=c26["apt"].to_numpy().astype(str); PARTS={("EGLL",):"a",("LEMD",):"b",("EDDF","EDDM","EHAM","LEBL","LSZH"):"d",("LFPG","LIRF"):"c"}
rd=lambda f: pl.read_parquet(I+f).sort(ID); dN=np.zeros(base.height); dE=np.zeros(base.height)
for aa,pc in PARTS.items():
    m=np.isin(a26,aa); bk,ref,lg,av=(rd(f) for f in (f"agent_bulkr_t_{pc}_final.parquet",f"agent_sb2_t_{pc}_final.parquet",f"agent_etrs_t_{pc}_final.parquet",f"agent_etrs_t_{pc}_avg_final.parquet"))
    dN[m]=(bk[T].to_numpy().astype(float)-ref[T].to_numpy().astype(float))[m]; dE[m]=(av[T].to_numpy().astype(float)-lg[T].to_numpy().astype(float))[m]
Xs=np.column_stack([dN,dE]).astype(np.float32)
for j,n in enumerate(["d_NEW","d_ET"]): print(n,"fold mean %.2f sd %.2f | serve(9apt) mean %.2f sd %.2f"%(Xf[:,j].mean(),Xf[:,j].std(),Xs[np.isin(a26,AIR),j].mean(),Xs[np.isin(a26,AIR),j].std()))
np.savez(I+"agent_bigc_extra2.npz",Xf=Xf,Xs=Xs,cols=np.array(["d_NEW","d_ET"]))
