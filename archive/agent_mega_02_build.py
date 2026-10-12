"""mega step 2: assemble base + (1) certain bounds last + (2) gate w0.1 + (3) Istanbul dose 1.0 [+ (4) corrector refit mix].
    python archive/agent_mega_02_build.py no4|with4
Float deltas (agent_mega_deltas.npz; agent_mega_refit_delta.npz for ingredient 4) added to the base, rounded once, clipped at zero; the certain bounds
(box edges, hidden-plan lower and upper bound, same code and order as agent_ruleorder_08_cand.py) are re-applied LAST on every row.
Writes data/interim/agent_mega_no4_final.parquet or agent_mega_final.parquet and a per-ingredient table; every base-minus-candidate difference is explained."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_ruleorder_lib import bounds
mode = sys.argv[1]; assert mode in ("no4", "with4")
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
base = pl.read_parquet(I + "agent_basefix_final.parquet").sort(ID); ids = base[ID].to_numpy(); cur = base[T].cast(pl.Float64).to_numpy()
z = np.load(I + "agent_mega_deltas.npz"); assert (z["ids"] == ids).all()
dG, dI0, RR = z["dG"], z["dI"], z["rule_rows"]
dI = np.where(RR, 0.0, dI0)
print("ingredient 3: later-rule rows excluded from the Istanbul delta:", int(((dI0 != 0) & RR).sum()), "(delta there %s s)" % np.round(dI0[(dI0 != 0) & RR], 2))
dR = np.zeros(len(ids))
if mode == "with4":
    r = np.load(I + "agent_mega_refit_delta.npz"); assert (r["ids"] == ids).all(); dR = r["dR"]
bs = bounds(ids, "ranking"); apt = bs["apt"]; lo_e, hi_e, r2, ub = bs["lo_e"], bs["hi_e"], bs["r2"], bs["ub"]
def certain(x):
    x = x.copy()
    m = np.isfinite(lo_e) & (x < lo_e); x[m] = lo_e[m]
    m = np.isfinite(hi_e) & (x > hi_e); x[m] = hi_e[m]
    m = np.isfinite(r2) & (x < r2); x[m] = r2[m]
    m = np.isfinite(ub) & (x > ub); x[m] = ub[m]
    return x
def fin(x): return np.rint(np.maximum(x, 0))
# ---- ingredient 1 on the base alone, must equal the experiment file ----
ro = pl.read_parquet(I + "agent_ruleorder_final.parquet").sort(ID)[T].cast(pl.Float64).to_numpy()
assert (fin(certain(cur)) == ro).all(); print("(1) certain-bounds code on the base reproduces agent_ruleorder_final.parquet exactly:", int((ro != cur).sum()), "rows moved")
bound_rows = ro != cur
tot = dG + dI + dR
pre = fin(cur + tot)                       # before the bounds step
out = fin(certain(cur + tot))              # bounds re-applied last, on the float value, rounded once
# any bound violation after the other deltas, before re-clamp
viol_pre = fin(certain(pre)) != pre
print("rows where the other deltas alone push a row outside a certain bound (fixed by the re-clamp):", int((viol_pre & ~bound_rows).sum()))
print("12 bound rows: all still inside bounds in output:", bool(((~np.isfinite(lo_e) | (out >= lo_e)) & (~np.isfinite(hi_e) | (out <= hi_e)) & (~np.isfinite(r2) | (out >= r2)) & (~np.isfinite(ub) | (out <= ub)))[bound_rows].all()))
print("any row of the output outside a certain bound:", int((fin(certain(out)) != out).sum()))
# overlaps
sets = {"1 bounds": bound_rows, "2 gate w0.1": dG != 0, "3 Istanbul dose": dI != 0}
if mode == "with4": sets["4 refit mix"] = dR != 0
ks = list(sets)
print("overlaps (rows with nonzero float delta / bound rows):")
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        print(f"   {ks[i]} & {ks[j]}: {int((sets[ks[i]] & sets[ks[j]]).sum())}")
# ---- explanation of every difference ----
d = out - cur
expl = np.zeros(len(ids), bool)
for k in ks: expl |= sets[k]
print("rows moved vs base:", int((d != 0).sum()), "; moved rows with no ingredient touching them (unexplained):", int(((d != 0) & ~expl).sum()))
free = ~bound_rows & ~viol_pre & (pre > 0)
print("non-bound rows: max | (cand - base) - sum of float deltas | = %.3f (<= 0.5 expected from rounding); clipped-at-zero rows: %d" % (np.abs(d - tot)[free].max(), int((cur + tot < 0).sum())))
assert np.abs(d - tot)[free].max() <= 0.5 + 1e-9
assert ((d != 0) & ~expl).sum() == 0
# per ingredient (alone on the base, rounded)
print("\n| ingredient | rows moved alone (rounded) | rows float delta nonzero | mean move | rms move | max |move| | airports touched | sum dp^2/N/(2*219.07) = loss if wrong |")
print("|---|---|---|---|---|---|---|---|")
tabs = {"1 bounds": ro - cur, "2 gate w0.1": fin(cur + dG) - cur, "3 Istanbul dose": fin(cur + dI) - cur}
if mode == "with4": tabs["4 refit mix"] = fin(cur + dR) - cur
for k, v in tabs.items():
    m = v != 0
    ap = dict(zip(*np.unique(apt[m], return_counts=True)))
    print(f"| {k} | {int(m.sum())} | {int(sets[k].sum())} | {v[m].mean():+.2f} | {np.sqrt((v[m]**2).mean()):.1f} | {np.abs(v).max():.0f} | {ap} | {float((v**2).sum())/len(v)/(2*219.0687):.4f} |")
dm = out - cur; m = dm != 0
print(f"| TOTAL (output) | {int(m.sum())} | | {dm[m].mean():+.2f} | {np.sqrt((dm[m]**2).mean()):.1f} | {np.abs(dm).max():.0f} | {dict(zip(*np.unique(apt[m], return_counts=True)))} | {float((dm**2).sum())/len(dm)/(2*219.0687):.4f} |")
name = "agent_mega_no4_final.parquet" if mode == "no4" else "agent_mega_final.parquet"
base.with_columns(pl.Series(T, out).cast(base.schema[T])).write_parquet(I + name); print("wrote", I + name)
