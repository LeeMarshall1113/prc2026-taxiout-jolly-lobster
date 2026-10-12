"""experiment backx 02: back-extrapolated off-block estimators + the step-1 measurements (docs/LAB-NOTES.md).

tt = e_start - BLOCK = time from the recorded off-block to the first ground fix of the departure segment (label-derived).
Two estimators of tt, both learned OUT OF FOLD with the re-stack's own folds (10 round-robin over the sorted days of the
dump rows in the CAB npz = archive/agent_restack3_03_fold.py's folds), on every training-fold dump row with a first fix:
  E1 bx_c  per (airport, first-fix-to-stand distance bucket) median tt, >= 30 rows, else the airport median
  E2 bx_g  LightGBM (L1 = median) on tt clipped +-3600 from first-fix geometry: airport, first fix lat/lon/heading/gs,
           stand centroid lat/lon, distance and bearing stand->fix, first motion bearing/speed, its angle to the
           stand->fix bearing, fixes in the first 120 s, silence before the segment, hour, plan, taxinm flag, W flag
  off-block estimate bx = e_start - tt_hat.  2026: both fitted on ALL 2025 dump rows, applied label-free.
Writes data/interim/agent_backx_bx_{2025,2026}.parquet (MVT_ID_mvt, bx_c, bx_g, tt_g) and prints the measurements.
"""
import sys
import time

sys.path.insert(0, ".")
import lightgbm as lgb
import numpy as np
import polars as pl

C = pl.col
ID = "MVT_ID_mvt"
I = "data/interim/"
T0 = time.time()
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
DB = [0, 30, 60, 100, 150, 250, 400, 600, 900, 1300, 1800, 2500, 3500, 1e12]
TTCLIP = 3600.0
GP = dict(objective="l1", n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
          subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=0)
GF = ["apt_i", "tx", "hour", "plan_f", "first_gs", "first_dstand", "ff_lat", "ff_lon", "ff_trk", "st_lat", "st_lon",
      "brg_st", "mv_brg", "mv_spd", "rel_mv", "n120", "seg_start_gap", "w_f"]
FFC = ["e_start", "first_gs", "first_dstand", "ff_lat", "ff_lon", "ff_trk", "st_lat", "st_lon", "brg_st", "mv_brg",
       "mv_dt", "mv_spd", "rel_mv", "n120"]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def load(rows_p, ff_p):
    r = pl.read_parquet(rows_p, columns=[ID, "day", "apt", "src", "plan", "hour", "mvt_t", "block_t", "prod_kind",
                                         "prod_t", "e_start", "first_gs", "first_dstand", "seg_start_gap"])
    r = r.with_columns(C(ID).cast(pl.Float64))
    assert r[ID].n_unique() == r.height
    f = pl.read_parquet(ff_p).with_columns(C(ID).cast(pl.Float64)).unique(subset=ID, keep="first")
    f = f.select([ID] + FFC).rename({"e_start": "ff_e_start", "first_gs": "ff_first_gs", "first_dstand": "ff_first_dstand"})
    j = r.join(f, on=ID, how="left")
    assert j.height == r.height
    # identity: the copied first-fix reader must reproduce the offblock dump's own e_start / first_gs / first_dstand
    for a, b in (("e_start", "ff_e_start"), ("first_gs", "ff_first_gs"), ("first_dstand", "ff_first_dstand")):
        x = j[a].cast(pl.Float64).fill_null(np.nan).to_numpy()
        y = j[b].cast(pl.Float64).fill_null(np.nan).to_numpy()
        same = (x == y) | (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) < 1e-6)
        log(f"  identity {a}: {int((~same).sum()):,} of {j.height:,} rows differ ({ff_p})")
    miss = int(j["ff_lat"].is_null().sum())
    log(f"  rows without an ff row: {miss:,}")
    return j


def mat(j):
    g = lambda c: j[c].cast(pl.Float64).fill_null(np.nan).to_numpy()  # noqa: E731
    X = {c: g(c) for c in ["hour", "first_gs", "first_dstand", "ff_lat", "ff_lon", "ff_trk", "st_lat", "st_lon", "brg_st",
                           "mv_brg", "mv_spd", "rel_mv", "n120", "seg_start_gap"]}
    X["apt_i"] = np.array([APTS.index(a) if a in APTS else -1 for a in j["apt"].to_list()], float)
    X["tx"] = (j["src"] == "taxinm").cast(pl.Float64).to_numpy()
    X["plan_f"] = j["plan"].cast(pl.Float64).to_numpy()
    X["w_f"] = j["prod_t"].cast(pl.Float64).fill_null(np.nan).is_nan().cast(pl.Float64).to_numpy()  # W: no off-block measurement (label-free)
    return np.column_stack([X[c] for c in GF])


def dbin(d):
    b = np.digitize(np.nan_to_num(d, nan=-1.0), DB) - 1
    return np.where(np.isnan(d), -1, b)


def cell_fit(apt, db, tt):
    df = pl.DataFrame({"apt": apt, "db": db, "tt": tt})
    c = df.group_by(["apt", "db"]).agg(C("tt").median().alias("m"), pl.len().alias("n")).filter(C("n") >= 30)
    a = df.group_by("apt").agg(C("tt").median().alias("ma"))
    return c, a


def cell_apply(model, apt, db):
    c, a = model
    q = pl.DataFrame({"apt": apt, "db": db}).with_row_index("_i")
    q = q.join(c, on=["apt", "db"], how="left").join(a, on="apt", how="left").sort("_i")
    return q.select(pl.coalesce(C("m"), C("ma"))).to_series().cast(pl.Float64).fill_null(np.nan).to_numpy()


def stats(err):
    e = err[np.isfinite(err)]
    if len(e) == 0:
        return "n 0"
    med = np.median(e)
    mad = np.median(np.abs(e - med))
    return (f"n {len(e):>8,}  bias {med:+7.1f}  MAD {mad:6.1f}  cRMSE {np.sqrt(((e - e.mean()) ** 2).mean()):7.1f}  "
            f"RMSE {np.sqrt((e ** 2).mean()):7.1f}  |e|>300 {np.mean(np.abs(e - med) > 300) * 100:5.1f}%")


# ---------------------------------------------------------------- 2025
j = load(I + "agent_offblock_rows_c238.parquet", I + "agent_backx_ff_2025.parquet")
z = np.load(I + "agent_restack3_CAB.npz", allow_pickle=True)
zid = z["ids"].astype(float)
pos_of = {k: i for i, k in enumerate(zid)}
pos = np.array([pos_of.get(m, -1) for m in j[ID].to_list()])
assert (pos >= 0).all(), "dump rows missing from CAB"
cab, fin, Y = z["both_guard"].astype(float), z["both_guard_prers"].astype(float), z["y"].astype(float)
N_ALL = len(Y)
SSE_ALL = float(((cab - Y) ** 2).sum())
R_ALL = np.sqrt(SSE_ALL / N_ALL)
d_npz = z["days"][pos].astype(str)
assert (d_npz == np.asarray(j["day"].to_list())).all()
ud = sorted(set(d_npz))
fold = {d: i % 10 for i, d in enumerate(ud)}
fo = np.array([fold[d] for d in d_npz])
mvt = j["mvt_t"].to_numpy()
blk = j["block_t"].to_numpy()
y = Y[pos]
assert np.nanmax(np.abs((mvt - blk) - y)) < 1.5, "y != mvt - block"
es = j["e_start"].cast(pl.Float64).fill_null(np.nan).to_numpy()
prod = j["prod_t"].cast(pl.Float64).fill_null(np.nan).to_numpy()
W = np.isnan(prod)
has = np.isfinite(es)
apt = np.asarray(j["apt"].to_list())
tt = es - blk
X = mat(j)
db = dbin(j["first_dstand"].cast(pl.Float64).fill_null(np.nan).to_numpy())
log(f"2025 dump rows {j.height:,}; W (no off-block measurement) {int(W.sum()):,}; W with a first fix {int((W & has).sum()):,}; "
    f"M with a first fix {int((~W & has).sum()):,}; days {len(ud)}; all-rows CAB RMSE {R_ALL:.3f}")

# ---- OOF estimators
tt_c = np.full(len(y), np.nan)
tt_g = np.full(len(y), np.nan)
for k in range(10):
    tr = (fo != k) & has
    te = (fo == k) & has
    cm = cell_fit(apt[tr], db[tr], tt[tr])
    tt_c[te] = cell_apply(cm, apt[te], db[te])
    m = lgb.LGBMRegressor(**GP).fit(X[tr], np.clip(tt[tr], -TTCLIP, TTCLIP), categorical_feature=[0])
    tt_g[te] = m.predict(X[te])
    log(f"  fold {k}: train {int(tr.sum()):,} test {int(te.sum()):,}")
bx_c, bx_g = es - tt_c, es - tt_g
pl.DataFrame({ID: j[ID], "bx_c": bx_c, "bx_g": bx_g, "tt_g": tt_g}).write_parquet(I + "agent_backx_bx_2025.parquet")

# ---- measurements
print("\n== S1a populations (2025 dump, 238 days)")
for a in APTS:
    k = apt == a
    print(f"  {a}: rows {int(k.sum()):>7,}  W {int((k & W).sum()):>7,}  W+first fix {int((k & W & has).sum()):>7,}  "
          f"W+fix+stand {int((k & W & has & np.isfinite(X[:, 5])).sum()):>7,}  "
          f"W share of all-rows CAB SSE {((cab[pos][k & W] - y[k & W]) ** 2).sum() / SSE_ALL * 100:5.1f}%")
print(f"  total W SSE share {((cab[pos][W] - y[W]) ** 2).sum() / SSE_ALL * 100:.1f}%; W with a first fix "
      f"{((cab[pos][W & has] - y[W & has]) ** 2).sum() / SSE_ALL * 100:.1f}%")

print("\n== S1b W rows with a first fix: first-fix distance to stand and tt = t_first - BLOCK by distance bucket")
dist = X[:, 5]
ww = W & has
print(f"  first_dstand quantiles (m) p10/25/50/75/90: "
      + " ".join(f"{v:.0f}" for v in np.nanpercentile(dist[ww], [10, 25, 50, 75, 90]))
      + f"; no stand centroid {np.mean(np.isnan(dist[ww])) * 100:.1f}%")
print(f"  first_gs quantiles (kt) p10/25/50/75/90: " + " ".join(f"{v:.1f}" for v in np.nanpercentile(X[ww, 4], [10, 25, 50, 75, 90])))
print("  bucket(m)        n    tt p10    p25    med    p75    p90    MAD")
for b in range(-1, len(DB) - 1):
    k = ww & (db == b)
    if k.sum() < 50:
        continue
    q = np.percentile(tt[k], [10, 25, 50, 75, 90])
    lab = "no stand" if b < 0 else f"{DB[b]:.0f}-{DB[b + 1]:.0f}" if DB[b + 1] < 1e11 else f">{DB[b]:.0f}"
    print(f"  {lab:>10} {int(k.sum()):>8,} " + " ".join(f"{v:6.0f}" for v in q)
          + f" {np.median(np.abs(tt[k] - q[2])):6.0f}")
print("  M rows with a first fix (reference): tt p10/50/90 "
      + " ".join(f"{v:.0f}" for v in np.percentile(tt[~W & has], [10, 50, 90])))

print("\n== S1c estimator accuracy on W rows with a first fix (err = implied taxi - y; OOF by day-fold)")
Wc = ww & np.isfinite(tt_g)
for nm, p in (("e_start raw (taxi = mvt - t_first)", mvt - es), ("E1 cell median bx_c", mvt - bx_c),
              ("E2 GBM bx_g", mvt - bx_g), ("CAB (current stage + re-stack)", cab[pos]), ("TX_S6 fin (stage)", fin[pos])):
    print(f"  {nm:<36} {stats(p[Wc] - y[Wc])}")
print("  per airport, E2 vs CAB (RMSE / MAD):")
for a in APTS:
    k = Wc & (apt == a)
    if k.sum() < 100:
        continue
    e2, ec = (mvt - bx_g)[k] - y[k], cab[pos][k] - y[k]
    print(f"    {a}: n {int(k.sum()):>7,}  E2 RMSE {np.sqrt((e2 ** 2).mean()):7.1f} MAD {np.median(np.abs(e2 - np.median(e2))):6.1f} "
          f"| CAB RMSE {np.sqrt((ec ** 2).mean()):7.1f} MAD {np.median(np.abs(ec - np.median(ec))):6.1f}")
print("  E2 by distance bucket (W, first fix): RMSE E2 / CAB")
for b in range(-1, len(DB) - 1):
    k = Wc & (db == b)
    if k.sum() < 200:
        continue
    lab = "no stand" if b < 0 else f"{DB[b]:.0f}-{DB[b + 1]:.0f}" if DB[b + 1] < 1e11 else f">{DB[b]:.0f}"
    e2, ec = (mvt - bx_g)[k] - y[k], cab[pos][k] - y[k]
    print(f"    {lab:>10} n {int(k.sum()):>7,}  E2 {np.sqrt((e2 ** 2).mean()):7.1f}  MAD {np.median(np.abs(e2 - np.median(e2))):6.1f}"
          f"  CAB {np.sqrt((ec ** 2).mean()):7.1f}  MAD {np.median(np.abs(ec - np.median(ec))):6.1f}")

print("\n== S1d does bx carry information CAB lacks? (W rows with a first fix; descriptive screen, not a gate)")
r = y - cab[pos]
for nm, bx in (("E1", bx_c), ("E2", bx_g), ("raw e_start", es)):
    x = np.clip((mvt - bx) - cab[pos], -1800, 1800)
    k = Wc & np.isfinite(x)
    cc = np.corrcoef(r[k], x[k])[0, 1]
    gain = np.zeros(len(y))
    for f in range(10):
        tr, te = k & (fo != f), k & (fo == f)
        b = float((r[tr] * x[tr]).sum() / (x[tr] ** 2).sum())
        gain[te] = r[te] ** 2 - (r[te] - b * x[te]) ** 2
    d = np.sqrt((SSE_ALL - gain.sum()) / N_ALL) - R_ALL
    keep = np.argsort(-gain)[50:]
    d50 = np.sqrt((SSE_ALL - gain[keep].sum()) / N_ALL) - R_ALL
    per = {a: np.sqrt((SSE_ALL - gain[apt == a].sum()) / N_ALL) - R_ALL for a in APTS}
    print(f"  {nm}: corr(y - CAB, bx-taxi - CAB) {cc:+.3f}; OOF linear shrink on these rows: all-rows {d:+.3f} s, "
          f"minus-top-50 {d50:+.3f} s; by airport " + " ".join(f"{a} {v:+.3f}" for a, v in per.items()))
fi = lgb.LGBMRegressor(**GP).fit(X[has], np.clip(tt[has], -TTCLIP, TTCLIP), categorical_feature=[0])
print("  E2 full-2025 feature importance (splits): " + " ".join(f"{c}={v}" for c, v in zip(GF, fi.feature_importances_)))
cm_all = cell_fit(apt[has], db[has], tt[has])

# ---------------------------------------------------------------- 2026 (label-free)
j6 = load(I + "agent_offblock_rows_2026.parquet", I + "agent_backx_ff_2026.parquet")
es6 = j6["e_start"].cast(pl.Float64).fill_null(np.nan).to_numpy()
has6 = np.isfinite(es6)
W6 = np.isnan(j6["prod_t"].cast(pl.Float64).fill_null(np.nan).to_numpy())
X6 = mat(j6)
apt6 = np.asarray(j6["apt"].to_list())
tc6 = np.full(j6.height, np.nan)
tg6 = np.full(j6.height, np.nan)
tc6[has6] = cell_apply(cm_all, apt6[has6], dbin(X6[has6, 5]))
tg6[has6] = fi.predict(X6[has6])
pl.DataFrame({ID: j6[ID], "bx_c": es6 - tc6, "bx_g": es6 - tg6, "tt_g": tg6}).write_parquet(I + "agent_backx_bx_2026.parquet")
print("\n== S1e 2026 coverage and parity (label-free)")
print(f"  2026 dump rows {j6.height:,}; W {int(W6.sum()):,}; W with a first fix {int((W6 & has6).sum()):,}")
for a in APTS:
    k6, k5 = (apt6 == a) & W6, (apt == a) & W
    if k6.sum() == 0:
        continue
    print(f"    {a}: 2026 W {int(k6.sum()):>6,} with fix {int((k6 & has6).sum()):>6,} ({np.mean(has6[k6]) * 100:4.0f}% vs 2025 "
          f"{np.mean(has[k5]) * 100:4.0f}%)  first_dstand med {np.nanmedian(X6[k6 & has6, 5]):6.0f} vs {np.nanmedian(X[k5 & has, 5]):6.0f}"
          f"  tt_g med {np.nanmedian(tg6[k6 & has6]):6.0f} vs {np.nanmedian(tt_g[k5 & has]):6.0f}")
log("done")
