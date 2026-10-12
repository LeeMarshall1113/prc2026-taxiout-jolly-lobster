"""lv 04: serving twin of the LTFM corrector WITH variant views (docs/LAB-NOTES.md). Run ONLY after the Stage 1 rule passed.
    python archive/agent_lv_04_serve.py --arm AQC --cfg S2 --k 0.75
Same as archive/agent_lks_04_serve.py (fit on all 7 LTFM fold months, key-stat train features leave-own-month-out, serve stats pool = 7 months of 2025,
ref feature for 2026 rows = v137, snow rows 0, 3 seeds) + view columns from agent_lv_views.make_views: fold = results/ltfmm/ltfmm_{ctrl,treat}_f{1..4}_*,
serve = results/ltfmm/ltfmm_{ctrl,treat}_srv_99_98_* (same function, same ref definition).
Replaces the current LTFM term in v129:  LTFM rows out = max(v128 - M4 + lv_delta, 0)  (v129 LTFM == v128 - M4 + lks_delta exactly, verified); every other row = v129.
Writes data/interim/agent_lv_ltfm_final.parquet (+ agent_lv_serve_delta.npz). BOARD TEST file, not uploaded."""
import argparse
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, "archive")
import agent_lks_te as TE  # noqa: E402
import agent_lv_views as VW  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="AQC"); ap.add_argument("--trainmonths", default=""); ap.add_argument("--catmap", default="sorted", help="sorted (original serve recipe) | phys (= the validated screen: polars physical first-appearance codes of the fold rows, null -> NaN, unseen -> -1)"); ap.add_argument("--lgbcat", action="store_true"); ap.add_argument("--keys", default="agent_lks_keys.npz"); ap.add_argument("--olddelta", default="agent_lv_serve_delta.npz"); ap.add_argument("--base", default="data/interim/agent_m12v4_cand_final.parquet"); ap.add_argument("--out", default="agent_lv2_ltfm_final"); ap.add_argument("--cfg", default="S2"); ap.add_argument("--k", type=float, default=0.75)
a = ap.parse_args()
CFG = {"S1": dict(num_leaves=8, n_estimators=200), "S2": dict(num_leaves=16, n_estimators=300), "S3": dict(num_leaves=31, n_estimators=300)}
C = pl.col; I = "data/interim/"; ID = "MVT_ID_mvt"; T = "TAXITIME_SEC_mvt"; MONTHS = [1, 2, 3, 7, 8, 9, 10]
K = np.load(I + a.keys, allow_pickle=True); fid = K["fid"]; yf = K["yf"]; ref = K["ref"]; monf = K["monf"]; Kall = K["Kall"]; sid = K["sid"]; Nf = len(fid)
if a.keys != "agent_lks_keys.npz": MONTHS = sorted(set(int(m) for m in monf)); print("training months", MONTHS)
cols = ['hour', 'dow', 'minute_of_day', 'gap_sched', 'gap_lobt', 'gap_eobt', 'gap_aobt', 'dep_delay', 'sched_vs_eobt', 'lobt_minus_aobt', 'turn_sched_slack', 'turn_in_delay', 'turn_age', 'turn_taxi_in', 'dep_30min', 'arr_30min', 'dep_60min', 'arr_60min', 'dep_rwy_headway', 'rwy_dep_active', 'rwy_dep_share', 'rwy_arr_active', 'rwy_mixed_mode', 'wake_heavy_60min', 'wake_prev_heavy', 'prev_rwy_gap', 'wx_temp_c', 'wx_wind_kt', 'wx_visibility_km', 'wx_precip', 'wx_lowvis', 'wx_snow', 'wx_freezing', 'has_flight_plan', 'sched_demand_30min', 'late_apt_past60_mean', 'late_apt_past15_mean']
Tt = pl.read_parquet(I + "train_features.parquet", columns=[ID, "RUNWAY_mvt", "WK_TBL_CAT_flt"] + cols)
Xt = pl.DataFrame({ID: fid}).join(Tt, on=ID, how="left", maintain_order="left"); del Tt
if a.catmap == "phys":
    rw_cat = Xt["RUNWAY_mvt"].drop_nulls().unique(maintain_order=True).to_list(); wk_cat = Xt["WK_TBL_CAT_flt"].drop_nulls().unique(maintain_order=True).to_list()
else:
    rw_cat = sorted(set(Xt["RUNWAY_mvt"].drop_nulls().to_list())); wk_cat = sorted(set(Xt["WK_TBL_CAT_flt"].drop_nulls().to_list()))
print("category mapping", a.catmap, {"rwy": rw_cat, "wk": wk_cat})
np.savez(I + a.out + "_catmap.npz", rw=np.array(rw_cat), wk=np.array(wk_cat), mode=a.catmap)


def mat(df, refv):
    nul = np.nan if a.catmap == "phys" else -1
    rw = np.array([rw_cat.index(v) if v in rw_cat else (nul if v is None else -1) for v in df["RUNWAY_mvt"].to_list()], float); wk = np.array([wk_cat.index(v) if v in wk_cat else (nul if v is None else -1) for v in df["WK_TBL_CAT_flt"].to_list()], float)
    Fm = np.column_stack([df.select(cols).to_numpy().astype(float), rw, wk]); return np.column_stack([Fm, refv, Fm[:, cols.index('gap_lobt')] - refv, Fm[:, cols.index('gap_sched')] - refv])


# fold REF = SERVED analogue (v94 bagged + v113 delta), as the Stage 1 screen (--ref served)
if a.keys == "agent_lks_keys.npz":
    ref = np.full(Nf, np.nan)
    for k_, t_ in ((1, "1_7"), (2, "2_8"), (3, "3_9"), (4, "4_10")):
        b = pl.read_parquet(f"results/preds_bulkr_f{k_}_{t_}.parquet").select(ID, "bagged")
        c0 = pl.read_parquet(f"results/preds_ltfmseq_ctrl_f{k_}_{t_}.parquet").select(ID, pl.col("bagged").alias("c0"))
        t1 = pl.read_parquet(f"results/preds_ltfmseq_treat_f{k_}_{t_}.parquet").select(ID, pl.col("bagged").alias("t1"))
        j = b.join(c0, on=ID).join(t1, on=ID); dd = dict(zip(j[ID].to_numpy().astype("float64"), (j["bagged"] + j["t1"] - j["c0"]).to_numpy()))
        sel = np.isin(monf, [int(x) for x in t_.split("_")]); ref[sel] = [dd.get(x, np.nan) for x in fid[sel]]
assert np.isfinite(ref).all(), int((~np.isfinite(ref)).sum())
Ftr = mat(Xt, ref); res = yf - ref
base = pl.read_parquet(I + "v137_ltfmsnow_final.parquet").sort(ID)
rl = pl.read_parquet(I + "agent_bndl_rank_ltfm.parquet").join(base.select(ID, C(T).cast(pl.Float64).alias("v137")), on=ID, how="left", maintain_order="left")
pos = {v: i for i, v in enumerate(sid)}; ii = np.array([pos[v] for v in rl[ID].to_numpy()])
v129n = pl.read_parquet(I + "agent_combo_v129_noM4_final.parquet").sort(ID)
rl = rl.join(v129n.select(ID, C(T).cast(pl.Float64).alias("noM4")), on=ID, how="left", maintain_order="left")
v137 = rl["noM4"].to_numpy(); Fs = mat(rl, v137); snow_s = (Fs[:, cols.index("wx_snow")] == 1) | (Fs[:, cols.index("wx_freezing")] == 1)   # serve REF = served LTFM before any corrector (v128 - M4)
mi = np.searchsorted(np.array(MONTHS), monf); eng = TE.KS(Kall); rf = np.arange(Nf); sr = Nf + ii
tr_y, sv_y = eng.build(None, rf, "y", yf, mi, len(MONTHS), None, serve_rows=sr); tr_r, sv_r = eng.build(None, rf, "r", res, mi, len(MONTHS), None, serve_rows=sr)
parts_tr, parts_sv = [Ftr, tr_y, tr_r], [Fs, sv_y, sv_r]
if a.arm != "ks":
    g = a.arm
    mem = VW.load_fold(fid, monf, "ctrl", VW.MEM); qq = VW.load_fold(fid, monf, "ctrl", VW.QS); tt = VW.load_fold(fid, monf, "treat", VW.MEM)
    ms = VW.load_serve(rl[ID].to_numpy(), "ctrl", VW.MEM); qs = VW.load_serve(rl[ID].to_numpy(), "ctrl", VW.QS); ts = VW.load_serve(rl[ID].to_numpy(), "treat", VW.MEM)
    for nm_, arr in (("fold members", mem), ("fold q", qq), ("fold treat", tt), ("serve members", ms), ("serve q", qs), ("serve treat", ts)):
        need = (nm_.split()[-1] == "members") or (any(c in g for c in "QE") and nm_.endswith(" q")) or ("C" in g and nm_.endswith("treat"))
        if need:
            assert np.isfinite(arr).all(), f"{nm_}: {int((~np.isfinite(arr)).any(1).sum())} rows without checkpoint predictions"
    exf, exs = {}, {}
    for L_, kk in VW.EXTRA.items():
        if L_ in g:
            for k_, c_ in zip(kk, VW.load_fold(fid, monf, "ctrl", kk).T): assert np.isfinite(c_).all(), k_; exf[k_] = c_
            for k_, c_ in zip(kk, VW.load_serve(rl[ID].to_numpy(), "ctrl", kk).T): assert np.isfinite(c_).all(), k_; exs[k_] = c_
    if "R" in g:   # stand-handover columns (agent_lv_07_handover.py): fold rows aligned by id, serve rows in agent_bndl_rank_ltfm order
        HO = np.load(I + "agent_lv_handover.npz", allow_pickle=True); pos_ = {v: i for i, v in enumerate(HO["fold_ids"].tolist())}
        exf["R"] = HO["fold"][np.array([pos_[v] for v in fid.tolist()])]; assert (HO["serve_ids"] == rl[ID].to_numpy()).all(); exs["R"] = HO["serve"]
    qn = any(c in g for c in "QE")
    Vtr, vn = VW.make_views(ref, mem, qq if qn else None, tt if "C" in g else None, g, exf)
    Vsv, _ = VW.make_views(v137, ms, qs if qn else None, ts if "C" in g else None, g, exs)
    print("view columns", vn)
    for j, n in enumerate(vn):   # pairing guard: serve view distribution vs fold
        print(f"  {n:14s} fold mean {Vtr[:, j].mean():9.2f} sd {Vtr[:, j].std():8.2f} | serve mean {Vsv[:, j].mean():9.2f} sd {Vsv[:, j].std():8.2f}")
    parts_tr.append(Vtr); parts_sv.append(Vsv)
Ftr2 = np.hstack(parts_tr).astype(np.float32); Fs2 = np.hstack(parts_sv).astype(np.float32)
print("train", Ftr2.shape, "serve", Fs2.shape, flush=True)
TRM = np.isin(monf, [int(x) for x in a.trainmonths.split(',')]) if a.trainmonths else np.ones(Nf, bool); print('training rows', int(TRM.sum()), 'of', Nf, 'key-stat pool months', MONTHS)
d = np.zeros(len(Fs2)); dfold = np.zeros(len(Ftr2))
for seed in (0, 1, 2):
    gm = lgb.LGBMRegressor(objective='huber', alpha=300, min_child_samples=500, learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=seed, n_jobs=4, verbose=-1, **CFG[a.cfg])
    gm.fit(Ftr2[TRM], res[TRM], **({'categorical_feature': [len(cols), len(cols) + 1]} if a.lgbcat else {})); d += np.clip(gm.predict(Fs2), -600, 600) / 3
d = a.k * d; d[snow_s] = 0
v129 = pl.read_parquet(a.base).sort(ID); v128 = pl.read_parquet(I + "agent_bigc_v4_board_final.parquet").sort(ID)
sd = np.load(I + "agent_bndl_serve_deltas.npz"); assert (sd["ids"] == v128[ID].to_numpy()).all() and (v129[ID].to_numpy() == v128[ID].to_numpy()).all()
p128 = v128[T].cast(pl.Float64).to_numpy(); p129 = v129[T].cast(pl.Float64).to_numpy(); idpos = {x: i for i, x in enumerate(v128[ID].to_numpy())}
new = np.zeros(len(p128)); new[[idpos[x] for x in rl[ID].to_numpy()]] = d; m4 = sd["M4"]
lt = np.zeros(len(p128), bool); lt[[idpos[x] for x in rl[ID].to_numpy()]] = True
oldz = np.load(I + a.olddelta); assert (oldz["ids"] == v128[ID].to_numpy()).all(); old = oldz["new"]
chk = np.rint(np.maximum(v129n[T].cast(pl.Float64).to_numpy() + old, 0))
print("base LTFM == noM4 + AQC delta (rint)? max abs", float(np.abs(p129[lt] - chk[lt]).max()))
out = p129.copy(); out[lt] = np.maximum(p129[lt] - old[lt] + new[lt], 0)   # replace the AQC delta by the new one
net = out - p129
print("rows moved vs v129", int((net != 0).sum()), "mean shift (all rows)", float(net.mean()), "mean on LTFM", float(net[lt].mean()), "rms moved", float(np.sqrt((net[net != 0] ** 2).mean())))
print("new term mean", float(d.mean()), "sd", float(d.std()), "| lks term mean", float(old[lt].mean()), "sd", float(old[lt].std()), "| sum dp^2/row (LTFM) new", float((d ** 2).mean()), "lks", float((old[lt] ** 2).mean()))
res_ = v129.with_columns(pl.Series(T, np.rint(out)).cast(v129.schema[T])); res_.write_parquet(I + a.out + ".parquet")
np.savez(I + a.out + "_delta.npz", ids=v128[ID].to_numpy(), new=new, m4=m4, lt=lt, snow_serve=snow_s)
print("wrote", a.out, "; check vs", a.base)
