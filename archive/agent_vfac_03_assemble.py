"""vfac assemble (run after `agent_vfac_app.py fetch`): fold + serve add arrays -> d_v = add_v - add_ctrl, paired check by airport, value screen,
data/interim/agent_vfac_extra.npz = [stage views from agent_vfac_stage.npz] + [factory variants that pass the pairing rule].
Pairing rule (set before any number): per airport, serve sd / fold sd in [0.6, 1.6] (existing d_NEW 1.24, d_ET 1.26 sit in it) else the airport is NaN in
BOTH fold and serve for that variant. Gate: ctrl fold vs agent_etrs_<piece>_s0 both_guard must agree (rms printed)."""
import glob, os, sys
import numpy as np, polars as pl
from scipy.stats import spearmanr
I = "data/interim/"; M = I + "modal_vfac/"; ID = "MVT_ID_mvt"; AIR = ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]
FOLDP = {"bxTX": ["EGLL","LEMD"], "bxE9f": ["EDDF","EDDM","EHAM","LEBL","LSZH"], "f4TX": ["LFPG","LIRF"]}
SERVP = {"bxTX": "a", "bxE9f": "d", "f4TX": "c"}
z = np.load(I+"agent_bigc_fold.npz", allow_pickle=True); fid = z["ids"]; fapt = z["apt"].astype(str); y = z["y"]; pv = z["pv"]; mon = z["mon"]
base = pl.read_parquet(I+"agent_bndl_board_final.parquet").sort(ID); sid = base[ID].to_numpy().astype(float)
a26 = pl.read_parquet(I+"agent_coverage_census_2026.parquet").sort(ID)["apt"].to_numpy().astype(str)
variants = sorted({os.path.basename(f).split("_")[2] for f in glob.glob(M+"fold_*_*_s0.npz")} - {"ctrl"})
print("variants with fold files:", variants)
def lookup(ids_src, vals, ids_dst):
    o = np.argsort(ids_src); s = ids_src[o]; p = np.searchsorted(s, ids_dst); p = np.clip(p, 0, len(s)-1); ok = s[p] == ids_dst
    out = np.full(len(ids_dst), np.nan); out[ok] = vals[o][p][ok]; return out
cols, XF, XS, report = [], [], [], []
for p, apts in FOLDP.items():
    fc = M+f"fold_{p}_ctrl_s0.npz"
    if os.path.exists(fc):
        c = np.load(fc, allow_pickle=True); e = np.load(I+{"bxTX":"agent_etrs_bxTX_s0","bxE9f":"agent_etrs_bxE9f_s0","f4TX":"agent_etrs_f4TX_s0"}[p]+".npz", allow_pickle=True)
        mine = np.maximum(c["fin"]+c["add_ctrl"], 0); ref = lookup(e["ids"].astype(float), e["both_guard"].astype(float), c["ids"].astype(float))
        m = np.isfinite(ref); print(f"GATE {p}: ctrl(Modal) vs etrs CTRL both_guard rms diff {np.sqrt(np.mean((mine[m]-ref[m])**2)):.4f} max {np.abs(mine[m]-ref[m]).max():.2f} (n {m.sum():,})")
for v in variants:
    df = np.full(len(fid), np.nan); ds = np.full(len(sid), np.nan); keep = []
    for p, apts in FOLDP.items():
        ff, fc = M+f"fold_{p}_{v}_s0.npz", M+f"fold_{p}_ctrl_s0.npz"; fs = M+f"serve_{SERVP[p]}_s0.npz"
        if not (os.path.exists(ff) and os.path.exists(fc) and os.path.exists(fs)): continue
        a, c, s = np.load(ff, allow_pickle=True), np.load(fc, allow_pickle=True), np.load(fs, allow_pickle=True)
        if f"add_{v}_0" not in s.files: continue
        d = a[f"add_{v}"] - c["add_ctrl"]; dfold = lookup(a["ids"].astype(float), d, fid)
        for ap in apts:
            sv = []
            for si in ((1,) if ap == "LIRF" else (0,)):
                ids_s = s[f"ids_{si}"].astype(float); sv = lookup(ids_s, s[f"add_{v}_{si}"]-s[f"add_ctrl_{si}"], sid)
            mf, ms = (fapt == ap), (a26 == ap); f_, s_ = dfold[mf], sv[ms]
            f_, s_ = f_[np.isfinite(f_)], s_[np.isfinite(s_)]
            if len(s_) < 100 or len(f_) < 100: continue
            ratio = s_.std()/max(f_.std(), 1e-9); ok = 0.6 <= ratio <= 1.6
            report.append((v, ap, f_.mean(), f_.std(), s_.mean(), s_.std(), ratio, ok))
            if ok: df[mf] = dfold[mf]; ds[ms] = sv[ms]; keep.append(ap)
    if keep: cols.append("d_"+v); XF.append(df.astype(np.float32)); XS.append(ds.astype(np.float32))
print("variant airport | fold mean sd | serve mean sd | sd ratio | paired")
for r_ in report: print("%-8s %s | %7.2f %6.2f | %7.2f %6.2f | %.2f | %s" % (r_[0], r_[1], r_[2], r_[3], r_[4], r_[5], r_[6], "OK" if r_[7] else "NaN'd"))
st = np.load(I+"agent_vfac_stage.npz", allow_pickle=True)
Xf = np.column_stack([st["Xf"]]+XF) if XF else st["Xf"]; Xs = np.column_stack([st["Xs"]]+XS) if XS else st["Xs"]; names = list(st["cols"])+cols
np.savez(I+"agent_vfac_extra.npz", Xf=Xf.astype(np.float32), Xs=Xs.astype(np.float32), cols=np.array(names)); print("wrote agent_vfac_extra.npz", Xf.shape, Xs.shape, names)
C = np.mean([np.load(I+f"agent_bigc_v2_r1x_s{s}_C4bY.npz")["C4bY"] for s in (0,1,2)], axis=0); r = y-pv-0.75*C; R0 = np.sqrt((r**2).mean())
print("value screen vs residual y - pv - .75*C4bY (3-seed), LOMO per-airport slope (linear: UNDERSTATES trees; baseline d_NEW -0.050, d_ET +0.030)")
for j, n in enumerate(names):
    d = Xf[:, j]; m = np.isfinite(d); gain = 0.0
    for ap in AIR:
        ma = m & (fapt == ap)
        if ma.sum() < 1000: continue
        pr = np.zeros(len(r))
        for h in np.unique(mon):
            tr, te = ma & (mon != h), ma & (mon == h); b = (d[tr]*r[tr]).sum()/((d[tr]**2).sum()+1e-9); pr[te] = b*d[te]
        gain += ((r[ma]-pr[ma])**2).sum()-(r[ma]**2).sum()
    print("%-12s rows %8d spearman %+.4f LOMO-slope dRMSE %+.3f" % (n, m.sum(), spearmanr(d[m], r[m])[0] if m.sum() > 10 else np.nan, np.sqrt(((r**2).sum()+gain)/len(r))-R0))
