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
ap.add_argument("--cols", default="none", help="none | pcd | shuf | rand: extra nine columns"); ap.add_argument("--arm", default="AQC"); ap.add_argument("--anom", default="A", help="A shipped | B window rows dropped | C window labels de-trended"); ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--shrink", type=float, default=30.0); ap.add_argument("--exclude", type=int, default=0, help="servex: leave this fold month out of training AND of the serve key-statistic pool (as the leave-one-month-out fit does for a held month)"); ap.add_argument("--trainmonths", default=""); ap.add_argument("--catmap", default="sorted", help="sorted (original serve recipe) | phys (= the validated screen: polars physical first-appearance codes of the fold rows, null -> NaN, unseen -> -1)"); ap.add_argument("--lgbcat", action="store_true"); ap.add_argument("--keys", default="agent_lks_keys.npz"); ap.add_argument("--olddelta", default="agent_lv_serve_delta.npz"); ap.add_argument("--base", default="data/interim/agent_m12v4_cand_final.parquet"); ap.add_argument("--out", default="agent_lv2_ltfm_final"); ap.add_argument("--cfg", default="S2"); ap.add_argument("--k", type=float, default=0.75)
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
HELD = None if not a.exclude else [MONTHS.index(a.exclude)]
import glob
_raw = pl.concat([pl.read_parquet(f, columns=[ID, "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "PHASE_mvt", "ADEP_mvt"]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
_raw = _raw.with_columns(C("MVT_TIME_UTC_mvt").dt.day().alias("dom"), C("MVT_TIME_UTC_mvt").dt.hour().alias("hr"), C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("ts"))
_J = pl.DataFrame({ID: fid}).join(_raw, on=ID, how="left", maintain_order="left"); assert _J["dom"].null_count() == 0
dom = _J["dom"].to_numpy(); rwy_s = _J["RUNWAY_mvt"].fill_null("?").to_numpy(); hr = _J["hr"].to_numpy(); ts = _J["ts"].to_numpy()
_allt = np.sort(_raw["ts"].to_numpy()); db = (np.searchsorted(_allt, ts + 1800) - np.searchsorted(_allt, ts - 1800)) // 10 * 10
snow_f = Ftr[:, cols.index("wx_snow")] == 1; snow_f |= Ftr[:, cols.index("wx_freezing")] == 1
win = (monf == 7) & (dom >= 5) & (dom <= 26)
_ok = (~snow_f) & (yf < 3600) & ~np.isin(monf, [2, 7])
_key = np.array([f"{r}|{d}|{h}" for r, d, h in zip(rwy_s, db, hr)]); _ku, _ki = np.unique(_key, return_inverse=True)
_cs = np.bincount(_ki[_ok], weights=yf[_ok], minlength=len(_ku)); _cn = np.bincount(_ki[_ok], minlength=len(_ku)).astype(float)
_ex = np.where(_cn[_ki] >= 15, _cs[_ki] / np.maximum(_cn[_ki], 1), np.nan)
_rg = np.where(rwy_s == "35L", "35L", np.where(rwy_s == "36", "36", "oth")); shift = np.zeros(Nf); _okw = win & (~snow_f) & np.isfinite(_ex) & (yf < 3600)
for _d in range(5, 27):
    for _g in ("35L", "36", "oth"):
        _m = win & (dom == _d) & (_rg == _g); _mo = _okw & (dom == _d) & (_rg == _g)
        if _mo.sum(): shift[_m] = max((yf[_mo] - _ex[_mo]).mean() * _mo.sum() / (_mo.sum() + a.shrink), 0.0)
print("anom", a.anom, "window rows", int(win.sum()), "mean shift", float(shift[win].mean()), flush=True)
excl = win.copy() if a.anom == "B" else np.zeros(Nf, bool)
yfc = yf - shift if a.anom == "C" else yf; res = yfc - ref
pool = np.where(~excl)[0]
_ty, sv_y = eng.build(None, pool, "y", yfc[pool], mi[pool], len(MONTHS), HELD, serve_rows=sr); _tr, sv_r = eng.build(None, pool, "r", res[pool], mi[pool], len(MONTHS), HELD, serve_rows=sr)
tr_y = np.full((Nf, _ty.shape[1]), np.nan, np.float32); tr_y[pool] = _ty; tr_r = np.full((Nf, _tr.shape[1]), np.nan, np.float32); tr_r[pool] = _tr
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
ZC = np.load(I + "agent_istpcd_cols.npz", allow_pickle=True)
assert (ZC["fid"] == fid).all() and (ZC["sid"] == rl[ID].to_numpy()).all()
if a.cols == "pcd":
    parts_tr.append(ZC["D"]); parts_sv.append(ZC["Ds"])
elif a.cols == "shuf":
    parts_tr.append(ZC["Dshuf"][0]); parts_sv.append(ZC["Ds"])   # control: shuffled prediction in the fold columns, served columns as built
elif a.cols == "rand":
    parts_tr.append(ZC["Dnoise"][0]); parts_sv.append(np.random.default_rng(3000).normal(0, 1000, ZC["Ds"].shape).astype(np.float32))
print("extra columns", a.cols, flush=True)
Ftr2 = np.hstack(parts_tr).astype(np.float32); Fs2 = np.hstack(parts_sv).astype(np.float32)
print("train", Ftr2.shape, "serve", Fs2.shape, flush=True)
TRM = np.isin(monf, [int(x) for x in a.trainmonths.split(',')]) if a.trainmonths else np.ones(Nf, bool); TRM = TRM & (monf != a.exclude) if a.exclude else TRM; TRM = TRM & ~excl; print('training rows', int(TRM.sum()), 'of', Nf, 'key-stat pool months', MONTHS)
d = np.zeros(len(Fs2)); dfold = np.zeros(len(Ftr2))
SEEDS = [int(x) for x in a.seeds.split(',')]
for seed in SEEDS:
    gm = lgb.LGBMRegressor(objective='huber', alpha=300, min_child_samples=500, learning_rate=0.03, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=seed, n_jobs=4, verbose=-1, **CFG[a.cfg])
    gm.fit(Ftr2[TRM], res[TRM], **({'categorical_feature': [len(cols), len(cols) + 1]} if a.lgbcat else {})); d += np.clip(gm.predict(Fs2), -600, 600) / len(SEEDS)
d = a.k * d; d[snow_s] = 0
old = np.load(I + "agent_lv3a_ltfm_final_delta.npz"); v128 = pl.read_parquet(I + "agent_bigc_v4_board_final.parquet").sort(ID)
assert (old["ids"] == v128[ID].to_numpy()).all(); idpos = {x: i for i, x in enumerate(v128[ID].to_numpy())}
new = np.zeros(len(v128)); new[[idpos[x] for x in rl[ID].to_numpy()]] = d
lt = old["lt"] if "lt" in old.files else None
print("served term: mean %.3f sd %.3f; shipped lv3a term on the same rows: mean %.3f sd %.3f" % (float(d.mean()), float(d.std()), float(old["new"][new != 0].mean()), float(old["new"][new != 0].std())))
dd = new - old["new"]
print("this fit's term minus the shipped lv3a term (all LTFM rows): mean %.4f rms %.4f max abs %.4f (0 = reproduces)" % (float(dd[np.isin(np.arange(len(dd)), [idpos[x] for x in rl[ID].to_numpy()])].mean()), float(np.sqrt((dd ** 2).mean() * len(dd) / len(rl))), float(np.abs(dd).max())))
np.savez(I + a.out + "_delta.npz", ids=v128[ID].to_numpy(), new=new, snow_serve=snow_s, sid=rl[ID].to_numpy(), d=d)
print("wrote", a.out)
