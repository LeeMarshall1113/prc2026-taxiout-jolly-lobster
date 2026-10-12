"""ruleorder 08: the certain-bounds re-clamp (box edges + hidden-plan lower bound + hidden-plan upper bound) at the END of the chain.
Fold side: top-k robustness, premise on labels, Rome-exempt split, Jan/Jul/pooled with day-bootstrap. Serve side: build
data/interim/agent_ruleorder_final.parquet on the base data/interim/agent_basefix_final.parquet with a row list
data/interim/agent_ruleorder_rows.parquet (MVT_ID_mvt, old, new, rule). No label is used on the serve side. Nothing is uploaded.
"""
import subprocess
import sys

sys.path.insert(0, "archive")
import numpy as np
import polars as pl

I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
zb = np.load("data/interim/tmp_fold_bounds.npz"); zs = np.load("data/interim/tmp_fold_ship.npz")
fr = pl.read_parquet(I + "agent_ablate_frame.parquet", columns=["id", "y", "day", "apt", "mon", "M1", "rule", "eb_mid"]).filter(pl.col("M1").is_not_nan() & pl.col("M1").is_not_null()).filter(pl.col("apt") != "LTFM")
assert np.array_equal(fr["id"].to_numpy().astype(float), zb["ids"])
y = fr["y"].to_numpy().astype(float); mon = fr["mon"].to_numpy().astype(int); day = fr["day"].to_numpy().astype(str)
rome = (fr["rule"].to_numpy().astype(str) != "") | (np.nan_to_num(fr["eb_mid"].to_numpy().astype(float)) > 0.5)
N = len(y); JAN, JUL = mon == 1, mon == 7; JJ = JAN | JUL
bd = {k: zb[k] for k in zb.files}


def cert(x, which=("box", "r2", "ub")):
    x = x.copy()
    if "box" in which: x = np.where(np.isfinite(bd["lo_e"]), np.clip(x, bd["lo_e"], bd["hi_e"]), x)
    if "r2" in which: x = np.where(np.isfinite(bd["r2"]) & (x < bd["r2"]), bd["r2"], x)
    if "ub" in which: x = np.where(np.isfinite(bd["ub"]) & (x > bd["ub"]), bd["ub"], x)
    return np.rint(np.maximum(x, 0))


dayid = {d: i for i, d in enumerate(sorted(set(day[JJ])))}
did = np.array([dayid.get(d, -1) for d in day]); nd = len(dayid); dmon = np.zeros(nd, int)
for d, i in dayid.items(): dmon[i] = int(d[5:7])
rng = np.random.default_rng(11); B = 2000
CNT = np.zeros((B, nd))
for m_ in (1, 7):
    idx = np.where(dmon == m_)[0]; dr = rng.integers(0, len(idx), size=(B, len(idx)))
    for b in range(B): np.add.at(CNT[b], idx[dr[b]], 1)


def perday(e2):
    return np.bincount(did[JJ], weights=e2[JJ], minlength=nd), np.bincount(did[JJ], minlength=nd).astype(float)


def rep(label, pred, ref, mask=None):
    e2, e0 = (pred - y) ** 2, (ref - y) ** 2
    R = lambda e, m: float(np.sqrt(e[m].mean()))  # noqa: E731
    d = {k: R(e2, m) - R(e0, m) for k, m in (("Jan", JAN), ("Jul", JUL), ("JJ", JJ), ("all8", np.ones(N, bool)))}
    s, n = perday(e2); S0, N0 = perday(e0); P = {}
    for nm, msk in (("Jan", dmon == 1), ("Jul", dmon == 7), ("JJ", np.ones(nd, bool))):
        c = CNT * msk[None, :]; P[nm] = float(((np.sqrt((c @ s) / (c @ n)) - np.sqrt((c @ S0) / (c @ N0))) < 0).mean())
    g = (e0 - e2)[JJ]; keep_all = np.ones(JJ.sum(), bool)
    mt = {}
    for k in (1, 3, 5, 10, 50):
        kp = keep_all.copy(); kp[np.argsort(-g)[:k]] = False
        mt[k] = float(np.sqrt(e2[JJ][kp].mean()) - np.sqrt(e0[JJ][kp].mean()))
    ch = np.abs(pred - ref) > 0.5
    print(f"{label:44s} rows {int(ch.sum()):4d} (JJ {int((ch & JJ).sum()):3d}, Rome-rule rows {int((ch & rome).sum()):3d}) d Jan {d['Jan']:+.4f} Jul {d['Jul']:+.4f} JJ {d['JJ']:+.4f} all8 {d['all8']:+.4f} P {P['Jan']:.2f}/{P['Jul']:.2f}/{P['JJ']:.2f} "
          f"months<0 {sum(1 for m_ in (1, 2, 3, 7, 8, 9, 10, 11) if R(e2, mon == m_) - R(e0, mon == m_) < 0)}/8 minus-top1/3/5/10/50 JJ {mt[1]:+.4f}/{mt[3]:+.4f}/{mt[5]:+.4f}/{mt[10]:+.4f}/{mt[50]:+.4f}")
    return d, P


for refname, ship in (("base (blend removed)", zs["ship0"]), ("v138 chain", zs["ship1"])):
    print("\n== reference:", refname)
    for lab, w in (("certain bounds: box + r2 + ub", ("box", "r2", "ub")), ("  box only", ("box",)), ("  r2 only", ("r2",)), ("  ub only", ("ub",)), ("  box + ub", ("box", "ub"))):
        rep(lab, cert(ship, w), ship)
    x = cert(ship); x = np.where(rome, ship, x); rep("certain bounds, Rome-rule rows left alone", x, ship)
    x = cert(ship); x = np.where(~rome, ship, x); rep("certain bounds, ONLY on Rome-rule rows", x, ship)
# premise on labels
for k, f_ in (("box lo edge", y < np.nan_to_num(bd["lo_e"], nan=-1) - 14), ("box hi edge", y > np.nan_to_num(bd["hi_e"], nan=1e9) + 14), ("r2 lower bound", y < np.nan_to_num(bd["r2"], nan=-1) - 14), ("ub upper bound", y > np.nan_to_num(bd["ub"], nan=1e9) + 14)):
    print("labels breaking", k, "(+-14 s slack):", int(f_.sum()), "of", int(np.isfinite({"box lo edge": bd["lo_e"], "box hi edge": bd["hi_e"], "r2 lower bound": bd["r2"], "ub upper bound": bd["ub"]}[k]).sum()), "rows with the bound")
ship = zs["ship0"]; x = cert(ship); ch = x != ship
print("fold: rows changed", int(ch.sum()), "by kind: box", int(((cert(ship, ("box",)) != ship)).sum()), "r2", int((cert(ship, ("r2",)) != ship).sum()), "ub", int((cert(ship, ("ub",)) != ship).sum()))
big = np.argsort(-np.abs((x - y) ** 2 - (ship - y) ** 2) * ch)[:8]
for i in big:
    print("   fold row", int(zb["ids"][i]), fr["apt"][int(i)], "month", mon[i], "ship", ship[i], "-> clamped", x[i], "label", y[i], "rome-rule", bool(rome[i]))

# ------------------- serve build -------------------
from agent_ruleorder_lib import bounds  # noqa: E402
ld = lambda p: pl.read_parquet(p).sort(ID)  # noqa: E731
base = ld(I + "agent_basefix_final.parquet"); ids = base[ID].to_numpy(); cur = base[T].cast(pl.Float64).to_numpy()
bs = bounds(ids, "ranking"); bs.pop("fn")
new = cur.copy(); rule_of = np.array([""] * len(ids), dtype=object)
lo_e, hi_e, r2, ub = bs["lo_e"], bs["hi_e"], bs["r2"], bs["ub"]
m = np.isfinite(lo_e) & (new < lo_e); new[m] = lo_e[m]; rule_of[m] = "box_lo"
m = np.isfinite(hi_e) & (new > hi_e); new[m] = hi_e[m]; rule_of[m] = "box_hi"
m = np.isfinite(r2) & (new < r2); new[m] = np.where(rule_of[m] == "", r2[m], new[m]); rule_of[m] = np.where(rule_of[m] == "", "hidden_lower_r2", rule_of[m] + "+r2")
m = np.isfinite(ub) & (new > ub); new[m] = ub[m]; rule_of[m] = "hidden_upper"
new = np.rint(np.maximum(new, 0))
ch = new != cur
print("\nserve: rows changed on the base:", int(ch.sum()), " by rule", dict(zip(*np.unique(rule_of[ch], return_counts=True))))
v138 = ld("submissions/jolly-lobster_v138.parquet")[T].cast(pl.Float64).to_numpy()
rows = pl.DataFrame({ID: ids[ch], "old": cur[ch], "new": new[ch], "rule": rule_of[ch], "apt": bs["apt"][ch], "v138": v138[ch], "gl": bs["gl"][ch], "lo_e": lo_e[ch], "hi_e": hi_e[ch], "r2": r2[ch], "ub": ub[ch]})
print(rows)
rows.write_parquet(I + "agent_ruleorder_rows.parquet")
out = base.with_columns(pl.Series(T, new).cast(base.schema[T]))
out.write_parquet(I + "agent_ruleorder_final.parquet")
d = new - cur
print("rms move on moved rows %.1f, sum dp^2/row %.4f, max |dp| %.0f, mean move %.4f" % (np.sqrt((d[ch] ** 2).mean()), (d ** 2).sum() / len(d), np.abs(d).max(), d.mean()))
sq = (d ** 2).sum() / len(d)
print("lower bound of board effect if truth sits at the bound for every moved row: dRMSE = -sum(dp^2)/N/(2*219.07) = %.5f s" % (-sq / (2 * 219.0687)))
r = subprocess.run([sys.executable, "archive/agent_smallbundle_06_check.py", I + "agent_ruleorder_final.parquet", I + "agent_basefix_final.parquet"], capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout[-1800:], r.stderr[-300:])
