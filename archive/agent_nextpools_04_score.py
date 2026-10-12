"""next-pools step 2: score RULE A (LFPG day level, variant a) and RULE B (hour-block slopes, increment over A) exactly as frozen in
docs/LAB-NOTES.md. Frames: month-held-out, leave-one-airline-out, temporal both ways, day bootstrap, Jan/Jul, top-5 days, top-20 rows, control (pub shifted 7 d).
No parameter is changed after seeing a score."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(100)
APTS = ["LFPG", "EGLL", "LEBL", "LEMD", "EDDF", "EDDM", "LIRF"]
ROWRULES = ["R1", "R2", "R3a", "R3b"]
LAMS = [round(0.1 * i, 1) for i in range(11)]
f = frame_2025(APTS)
sv = pl.read_parquet(I + "agent_waithedge_tbl.parquet", columns=["id", "served"]).with_columns(C("id").cast(pl.Float64))
f = f.with_columns(C("id").cast(pl.Float64)).join(sv, on="id", how="inner").with_columns(C("served").alias("pred"))
f = f.filter(C("y").is_not_null() & C("gs").is_not_null())
f = f.with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"), C("rule").fill_null("").is_in(ROWRULES).alias("rr"),
                   C("day_pred").dt.month().alias("dmon"), (C("T").dt.hour() // 4).alias("blk"), C("fs").str.slice(0, 3).alias("air"))
f = f.filter(C("dmon").is_in(FRAME_MONTHS))
print("rows", f.height, f.group_by("apt").len().sort("apt").to_dicts())
pub = load_pub(2025, APTS)
dm = (C("gs") - C("pred")).clip(lower_bound=0) / 60.0
f = f.with_columns(dm.alias("dm"), (~C("amb") & ~C("rr")).alias("sh"))
days = f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum()).rename({"day_pred": "day"}).join(pub, on=["apt", "day"], how="inner")
days = days.filter(((C("n_pub") - C("n")).abs() <= DN_MAX))
parts = {k: g for k, g in f.partition_by("apt", "day_pred", as_dict=True).items()}
rows = []
for r in days.iter_rows(named=True):
    g = parts.get((r["apt"], r["day"]))
    if g is None:
        continue
    m = g["sh"].to_numpy(); gs = g["gs"].to_numpy()[m]; yh = g["pred"].to_numpy()[m]
    rec = {"apt": r["apt"], "day": r["day"]}
    for nm, P in [("d", r["pub"]), ("d7", r["pub7"])]:
        rec[nm] = solve_delta(gs, yh, P - r["fixed"]) if (P is not None and np.isfinite(P)) else np.nan
    rows.append(rec)
dd = pl.DataFrame(rows)
f = f.join(dd, left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
print("days with delta", dd.height, dd.group_by("apt").agg(med=C("d").median(), sd=C("d").std(), med7=C("d7").median()).sort("apt"))
print("LFPG per-month median delta:", f.filter(C("apt") == "LFPG").group_by("dmon").agg(m=C("d").median()).sort("dmon").to_dicts())
f.select("apt", "day_pred", "dmon").unique().height


def fitA(yh, y, d, pos, tr):
    t = tr & pos & np.isfinite(d)
    bias = float(np.median(d[tr & np.isfinite(d)])) if (tr & np.isfinite(d)).any() else 0.0
    best = (None, 0.0)
    for lam in LAMS:
        sse = ((np.maximum(yh[t] + lam * (d[t] - bias), 0) - y[t]) ** 2).sum()
        if best[0] is None or sse < best[0]:
            best = (sse, lam)
    return bias, best[1]


def applyA(yh, d, pos, bias, lam):
    s = pos & np.isfinite(d)
    return np.where(s, np.maximum(yh + lam * (d - bias), 0), yh)


def fitB(yh, y, d, pos, blk, tr, bias):
    t = tr & pos & np.isfinite(d)
    x = np.where(t, d - bias, 0.0); r = y - yh
    sxx = np.array([(x[blk == b] ** 2).sum() for b in range(6)]); sxy = np.array([(x[blk == b] * r[blk == b]).sum() for b in range(6)])
    tau = 0.25 * sxx.mean()
    return (sxy + tau) / (sxx + tau)


def applyB(yh, d, pos, blk, bias, betas):
    s = pos & np.isfinite(d)
    return np.where(s, np.maximum(yh + betas[blk] * (d - bias), 0), yh)


def run_frame(a, dcol, frame, groups=None):
    """returns per-row gains (A, B-increment-over-A) for airport a under the frame."""
    fa = f.filter(C("apt") == a)
    yh = fa["pred"].to_numpy(); y = fa["y"].to_numpy(); d = fa[dcol].to_numpy(); gs = fa["gs"].to_numpy()
    pos = (gs > yh) & fa["sh"].to_numpy(); blk = fa["blk"].to_numpy(); mon = fa["dmon"].to_numpy(); air = fa["air"].to_numpy()
    base = (yh - y) ** 2
    nA = yh.copy(); nB = yh.copy()
    if frame == "mh":
        splits = [(mon != m, mon == m) for m in FRAME_MONTHS]
    elif frame == "temporal":
        splits = [(mon <= 3, mon >= 7), (mon >= 7, mon <= 3)]
    elif frame == "loao":
        vc = fa.group_by("air").len(); keep = set(vc.filter(C("len") >= 0.01 * fa.height)["air"].to_list())
        ag = np.where(np.isin(air, list(keep)), air, "OTHER")
        splits = [(ag != k, ag == k) for k in np.unique(ag)]
    for tr, te in splits:
        bias, lam = fitA(yh, y, d, pos, tr)
        betas = fitB(yh, y, d, pos, blk, tr, bias)
        nA[te] = applyA(yh, d, pos, bias, lam)[te]
        nB[te] = applyB(yh, d, pos, blk, bias, betas)[te]
    gA = base - (nA - y) ** 2; gB = base - (nB - y) ** 2
    slim = fa.select('apt','day_pred','dmon'); del fa
    return slim, gA, gB - gA


def boot(g, key, n=2000, seed=0):
    u, inv = np.unique(key, return_inverse=True); s = np.bincount(inv, weights=g); rng = np.random.default_rng(seed)
    bs = s[rng.integers(0, len(s), size=(n, len(s)))].sum(axis=1) / SE_SCALE
    return float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95)), float((bs > 0).mean()), s


res = {}
for frame in ["mh", "loao", "temporal"]:
    print(f"\n===== frame {frame}: gains in board-s (A = uniform level, B = increment of hour-block slopes over A)")
    tab = []
    for a in APTS:
        fa, gA, gB = run_frame(a, "d", frame)
        res[(frame, a)] = (fa, gA, gB)
        tab.append(dict(apt=a, A=gA.sum() / SE_SCALE, B_inc=gB.sum() / SE_SCALE))
    print(pl.DataFrame(tab).with_columns(pl.col(pl.Float64).round(3)))
    print("B pooled (7 airports) increment", sum(r["B_inc"] for r in tab), " B ex-LFPG", sum(r["B_inc"] for r in tab if r["apt"] != "LFPG"))
# control
print("\n===== control (published shifted 7 days), month-held-out")
tab = []
for a in APTS:
    fa, gA, gB = run_frame(a, "d7", "mh")
    res[("ctl", a)] = (fa, gA, gB)
    tab.append(dict(apt=a, A=gA.sum() / SE_SCALE, B_inc=gB.sum() / SE_SCALE))
print(pl.DataFrame(tab).with_columns(pl.col(pl.Float64).round(3)))
# detailed gate lines
def gate(label, getg, aps):
    gs_ = [getg(a) for a in aps]
    g = np.concatenate(gs_)
    key = np.concatenate([(res[("mh", a)][0]["apt"] + res[("mh", a)][0]["day_pred"].cast(pl.Utf8)).to_numpy() for a in aps])
    mon = np.concatenate([res[("mh", a)][0]["dmon"].to_numpy() for a in aps])
    lo, hi, P, s = boot(g, key)
    ds = np.sort(s)[::-1]
    print(f"{label}: gain {g.sum()/SE_SCALE:+.3f} [{lo:+.3f}, {hi:+.3f}] P {P:.2f}; Jan {g[mon==1].sum()/SE_SCALE:+.3f} Jul {g[mon==7].sum()/SE_SCALE:+.3f}; "
          f"no top-5 days {(g.sum()-ds[:5].sum())/SE_SCALE:+.3f}; no top-20 rows {(g.sum()-np.sort(g)[::-1][:20].sum())/SE_SCALE:+.3f}; days>0 {np.mean(s>0):.2f}")
    return g.sum() / SE_SCALE
print("\n===== gate lines (month-held-out)")
gate("A LFPG", lambda a: res[("mh", a)][1], ["LFPG"])
gate("A six piece-A airports (in production, reference)", lambda a: res[("mh", a)][1], [x for x in APTS if x != "LFPG"])
gate("B increment, all 7", lambda a: res[("mh", a)][2], APTS)
gate("B increment, LFPG", lambda a: res[("mh", a)][2], ["LFPG"])
gate("B increment, five non-Rome (EGLL LEBL LEMD EDDF EDDM)", lambda a: res[("mh", a)][2], ["EGLL", "LEBL", "LEMD", "EDDF", "EDDM"])
gate("B increment, LIRF", lambda a: res[("mh", a)][2], ["LIRF"])
print("control A LFPG:", res[("ctl", "LFPG")][1].sum() / SE_SCALE, " control B all7:", sum(res[("ctl", a)][2].sum() for a in APTS) / SE_SCALE)
# temporal split detail for LFPG A and B-all
for frame in ["temporal", "loao"]:
    print(frame, "A LFPG", res[(frame, "LFPG")][1].sum() / SE_SCALE, "B inc all7", sum(res[(frame, a)][2].sum() for a in APTS) / SE_SCALE)
print("\ntemporal by direction (test 7-11 after fit 1-3 | test 1-3 after fit 7-11):")
for lab, sel in [("A LFPG", [("LFPG", 1)]), ("B inc all7", [(a, 2) for a in APTS])]:
    hi_ = sum(res[("temporal", a)][k][res[("temporal", a)][0]["dmon"].to_numpy() >= 7].sum() for a, k in sel) / SE_SCALE
    lo_ = sum(res[("temporal", a)][k][res[("temporal", a)][0]["dmon"].to_numpy() <= 3].sum() for a, k in sel) / SE_SCALE
    print(lab, f"{hi_:+.3f} | {lo_:+.3f}")
