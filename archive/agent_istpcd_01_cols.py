"""istpcd 01: plan-clock distance columns for the ISTANBUL (LTFM) views corrector (experiment "istpcd", 2026-10-05).
Our own construction; idea credited to the public repository https://github.com/EnioAguiar/prc-taxiout-2026 (GPL-3.0), no code copied.
Fold rows = agent_lv_keys8.npz (8 months of 2025, 178,100 LTFM departures); current prediction = its 'ref' (the served-style reference the corrector already receives as a column).
Serve rows = agent_bndl_rank_ltfm.parquet; current prediction = the served LTFM base before any corrector (agent_combo_v129_noM4_final, as agent_istjul_03_serve.py).
Columns (9, same as the nine-airport experiment): d_SCHED d_LOBT d_IOBT d_EOBT d_AOBT3 = pred - (take-off - clock) ; clamp = gap_sched - pred in [5000,50000) else NaN ;
wmin = 3606 - |d_LOBT| ; d_AOBT3_ind / d_AOBT3_der = d_AOBT3 split by the label-free AOBT_3 flag (1 independent, 2 derived).
Controls: shuf = same 9 columns with pred permuted within month (3 draws), rand = 9 N(0,1000) columns (3 draws). No 2026 departure label read."""
import time
import numpy as np
import pandas as pd
import polars as pl

I = "data/interim/"; ID = "MVT_ID_mvt"; C = pl.col
t0 = time.time()
NAMES = ["d_SCHED", "d_LOBT", "d_IOBT", "d_EOBT", "d_AOBT3", "clamp", "wmin", "d_AOBT3_ind", "d_AOBT3_der"]


def P(*a):
    print("[%5.0fs]" % (time.time() - t0), *a, flush=True)


def build(pv, g, flag):
    d = {k: pv - g[k] for k in ("sched", "lobt", "iobt", "eobt", "aobt")}
    gs = g["sched"]
    clamp = np.where((gs >= 5000) & (gs < 50000), gs - pv, np.nan)
    wmin = 3606.0 - np.abs(d["lobt"])
    ind = np.where(flag == 1, d["aobt"], np.nan)
    der = np.where(flag == 2, d["aobt"], np.nan)
    return np.column_stack([d["sched"], d["lobt"], d["iobt"], d["eobt"], d["aobt"], clamp, wmin, ind, der]).astype(np.float32)


def gaps(df):
    f = lambda n: df[n].cast(pl.Float64).to_numpy()  # noqa: E731
    return dict(sched=f("gap_sched"), lobt=f("gap_lobt"), iobt=f("gap_lobt") - f("clk_iobt_minus_lobt"), eobt=f("gap_eobt"), aobt=f("gap_aobt"))


K = np.load(I + "agent_lv_keys8.npz", allow_pickle=True)
fid, ref, monf = K["fid"], K["ref"], K["monf"]
GC = ["gap_sched", "gap_lobt", "gap_eobt", "gap_aobt", "clk_iobt_minus_lobt", "lobt_minus_aobt"]
T = pl.read_parquet(I + "train_features.parquet", columns=[ID] + GC)
X = pl.DataFrame({ID: fid}).join(T, on=ID, how="left", maintain_order="left"); del T
assert X.height == len(fid)
gf = gaps(X)
fl = pl.read_parquet(I + "agent_aobtflag_flags.parquet").filter((C("yr") == 2025) & (C("apt") == "LTFM")).select("id", "flag")
fm = dict(zip(fl["id"].to_list(), fl["flag"].to_list()))
flag = np.array([fm.get(v, np.nan) for v in fid.tolist()], float)
P("fold flag coverage", float(np.isfinite(flag).mean()), "derived share of present", float(np.nanmean(flag[np.isin(flag, (1, 2))] == 2)))
D = build(ref, gf, flag)
Dshuf = np.empty((3,) + D.shape, np.float32)
for s in range(3):
    rng = np.random.default_rng(1000 + s); pvs = ref.copy()
    for m in np.unique(monf):
        ii = np.where(monf == m)[0]; pvs[ii] = ref[rng.permutation(ii)]
    Dshuf[s] = build(pvs, gf, flag)
Dnoise = np.stack([np.random.default_rng(2000 + s).normal(0, 1000, D.shape).astype(np.float32) for s in range(3)])
# serve rows
rl = pl.read_parquet(I + "agent_bndl_rank_ltfm.parquet")
v129n = pl.read_parquet(I + "agent_combo_v129_noM4_final.parquet").sort(ID)
rl = rl.join(v129n.select(ID, C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("noM4")), on=ID, how="left", maintain_order="left")
pvs_ = rl["noM4"].to_numpy(); assert np.isfinite(pvs_).all()
gs_ = gaps(rl)
fs = pl.read_parquet(I + "agent_aobtflag_flags.parquet").filter((C("yr") == 2026) & (C("apt") == "LTFM")).select("id", "flag")
fsm = dict(zip(fs["id"].to_list(), fs["flag"].to_list()))
flag_s = np.array([fsm.get(v, np.nan) for v in rl[ID].to_numpy().astype(float).tolist()], float)
P("serve flag coverage", float(np.isfinite(flag_s).mean()), "derived share of present", float(np.nanmean(flag_s[np.isin(flag_s, (1, 2))] == 2)))
Ds = build(pvs_, gs_, flag_s)
mon_s = rl["month"].to_numpy()
np.savez(I + "agent_istpcd_cols.npz", D=D, Dshuf=Dshuf, Dnoise=Dnoise, Ds=Ds, names=np.array(NAMES), fid=fid, sid=rl[ID].to_numpy(), mon_s=mon_s)
P("saved; fold NaN share", np.round(np.isnan(D).mean(0), 3).tolist(), "serve NaN share", np.round(np.isnan(Ds).mean(0), 3).tolist())
P("serve months", np.unique(mon_s, return_counts=True))


def auc(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return np.nan
    r = pd.Series(np.concatenate([a, b])).rank().values
    v = (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    return max(v, 1 - v)


allc = {n: (D[:, i], Ds[:, i]) for i, n in enumerate(NAMES)}
allc["pred"] = (ref, pvs_)
for n in ("gap_sched", "gap_lobt", "gap_eobt", "gap_aobt"):
    allc[n] = (X[n].to_numpy().astype(float), rl[n].to_numpy().astype(float))
rows = []
for n, (f, s) in allc.items():
    o = {"col": n}
    for m in (1, 7):
        o["auc_m%d" % m] = auc(f[monf == m], s[mon_s == m])
    o["nan_f_JJ"] = float(np.isnan(f[np.isin(monf, (1, 7))]).mean()); o["nan_s_JJ"] = float(np.isnan(s[np.isin(mon_s, (1, 7))]).mean())
    rows.append(o); P(o)
# the two flag classes by themselves: share of derived, fold vs serve, Jan / Jul
for m in (1, 7):
    P("derived share of flagged, month", m, "2025", float(np.mean(flag[(monf == m) & np.isin(flag, (1, 2))] == 2)), "2026", float(np.mean(flag_s[(mon_s == m) & np.isin(flag_s, (1, 2))] == 2)))
pd.DataFrame(rows).to_csv(I + "agent_istpcd_auc.csv", index=False)
# how new is each column? correlation with the inputs the corrector already has (pairs of columns that are the same for a tree: |corr| ~ 1)
ex = np.column_stack([ref, X["gap_lobt"].to_numpy() - ref, X["gap_sched"].to_numpy() - ref]).astype(float)
for i, n in enumerate(NAMES):
    col = D[:, i].astype(float); ok = np.isfinite(col) & np.isfinite(ex).all(1)
    if ok.sum() > 100:
        P("column", n, "|corr| with existing (ref, gap_lobt-ref, gap_sched-ref):", [round(abs(float(np.corrcoef(col[ok], ex[ok, j])[0, 1])), 3) for j in range(3)])
