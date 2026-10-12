"""refit2 step 4: serve-side pre-corrector base B26 and the new serve input file data/interim/agent_bigc13_serve_in.npz (agent_bigc12_serve_in.npz with pv-dependent entries replaced).
B26 (nine airports) = lv4r - D - kF_old, all in seconds before the tail and before the later Rome rules:
   lv4r = data/interim/agent_lv4r_ltfm_final.parquet (the chain value before the 0.5 x tail step; v132 = rint(max(lv4r + 0.5 ehat, 0)) is asserted in archive/agent_ablate_07_build.py)
   D    = agent_blend_pre_ltfm - v112_backx_final (neighbour blend, being removed; integer valued)
   kF_old = per-airport dose x mean of the 6 shipped C4ff7V5 serve fits (the corrector term inside the current file)
so B26 = stage + rules + M1-M3 + silent + meta, no blend, no corrector: the serve twin of the fold base nb (agent_refit2_02_base.py).  Rewrites: pv column of Xs_pre, pv26, Dsv (drift recomputed on the new fold pv).
Checks printed: identities that make B26 exact (v132 == rint(max(lv4r + 0.5 ehat, 0)); on rows no later rule touched, current file == max(v132 - D, 0)); size of B26 - old pv26 by airport (compare the fold gap).
Run under archive/capped_run.py --cap-gb 3."""
import glob
import json
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
import agent_bigc_drift as DR   # noqa: E402

I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
ld = lambda f: pl.read_parquet(I + f).sort(ID)  # noqa: E731
g = lambda f: ld(f)[T].cast(pl.Float64).to_numpy()  # noqa: E731
si = dict(np.load(I + "agent_bigc12_serve_in.npz", allow_pickle=True)); ids = si["ids"]; apt = si["apt"].astype(str); is9 = np.isin(apt, AIR)
for f in ("agent_bundle_v138_final.parquet", "agent_basefix_final.parquet", "agent_lv4r_ltfm_final.parquet", "agent_bundle_v132_final.parquet"):
    assert (ld(f)[ID].to_numpy() == ids).all(), f
v132, v138, cur, lv4r = g("agent_bundle_v132_final.parquet"), g("agent_bundle_v138_final.parquet"), g("agent_basefix_final.parquet"), g("agent_lv4r_ltfm_final.parquet")
D = g("agent_blend_pre_ltfm.parquet") - g("v112_backx_final.parquet")
ts = np.load(I + "agent_tailc_serve_ehat.npz", allow_pickle=True); assert (ts["ids"] == ids).all(); ehat = ts["ehat"]
fs = sorted(glob.glob(I + "bigc12_serve/C4ff7V5_s*.npz")); assert len(fs) == 6
corr = np.mean([np.load(f, allow_pickle=True)["pred"] for f in fs], 0)
kap = json.load(open(I + "agent_m12_kapt_C4ff7V5.json")); kF = np.array([kap.get(a, 0.0) for a in apt]) * corr; kF[~is9] = 0
assert np.array_equal(np.rint(np.maximum(lv4r + 0.5 * ehat, 0)), v132); print("(a) v132 == rint(max(lv4r + 0.5 ehat, 0)) on all rows: OK")
RR = v138 != v132
nr = is9 & ~RR
exact = (cur[nr] == np.maximum(v132[nr] - D[nr], 0))
print(f"(b) later-rule rows (v138 != v132): {int(RR.sum())} ({int((RR & is9).sum())} on the nine airports); on nine-airport rows no rule touched, current file == max(v132 - D, 0): {int(exact.sum())} of {int(nr.sum())}"
      f" (rows with D != 0 equal: {int(exact[D[nr] != 0].sum())} of {int((D[nr] != 0).sum())})")
B26 = lv4r - D - kF
old = si["pv26"].astype(float)
print("B26 - old pv26 (rules + blend removal + silent/meta; fold analogue in agent_refit2_02_base.py), rms by airport:")
for a in AIR:
    s = apt == a; print(f"  {a}: {np.sqrt(((B26 - old)[s] ** 2).mean()):7.2f}  mean {(B26 - old)[s].mean():+.2f}  rows {int(s.sum())}")
pv_new = np.where(is9, B26, old)
cp = [str(c) for c in si["cols_pre"]]; j = cp.index("pv")
Xs = si["Xs_pre"].copy(); Xs[:, j] = pv_new.astype(Xs.dtype); si["Xs_pre"] = Xs; si["pv26"] = pv_new
k2 = np.load(I + "agent_bigc13_keys2.npz", allow_pickle=True)
f = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True); fc = [str(c) for c in f["cols"]]
span_f = f["X"][:, fc.index("m_e_start")].astype(float); span_s = Xs[:, cp.index("m_e_start")].astype(float)
Dsv = DR.drift_serve(k2["Kf_d"], f["pv"].astype(float), span_f, k2["Ks_d"], pv_new, span_s, si["mon_s"])
old_dsv = si["Dsv"]
print("Dsv new vs shipped: rms change %.3f (sd shipped %.2f, new %.2f)" % (np.sqrt(((Dsv - old_dsv) ** 2).mean()), old_dsv.std(), Dsv.std()))
si["Dsv"] = Dsv
np.savez(I + "agent_bigc13_serve_in.npz", **si)
np.savez(I + "agent_bigc13_serve_base.npz", ids=ids, B26=B26, lv4r=lv4r, D=D, kF_old=kF, ehat=ehat, RR=RR, is9=is9)
print("wrote agent_bigc13_serve_in.npz, agent_bigc13_serve_base.npz")
