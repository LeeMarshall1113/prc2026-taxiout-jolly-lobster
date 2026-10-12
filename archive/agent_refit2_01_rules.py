"""refit2 step 1: the post-chain rules (gap clamp, LOBT box, min floor, v35, upper bound, hidden-plan reading) applied to the stage output p of the fold rows that the audit frame does NOT cover
(months 4, 5, 6, 12 of 2025, nine airports), plus a reproduction check on months 1 and 7 against the audit frame column r_full.
Rules are the audit's own functions (head of archive/agent_abl_01_run.py exec'd with the row set swapped; no neighbour blend). Constants come from all of 2025, as in the audit frame.
Writes data/interim/agent_refit2_rules.npz (ids, r = rules(p) - p, mon).  Memory: aim < 3 GB, run under archive/capped_run.py."""
import numpy as np
import polars as pl

I = "data/interim/"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
z = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
mon = z["mon"]; apt = z["apt"].astype(str)
sel = np.isin(apt, AIR) & np.isin(mon, [1, 7, 4, 5, 6, 12])
ids = z["ids"][sel].astype(float); o = np.argsort(ids, kind="stable")
ids = ids[o]; y = z["y"][sel][o].astype(float); p = z["p"][sel][o].astype(float); apts = apt[sel][o]; mons = mon[sel][o]
days = np.array(["2025-%02d-15" % m for m in mons])
CUSTOM = {"ids": ids, "y": y, "days": days, "apts": apts, "both_guard": p}
src = open("archive/agent_abl_01_run.py", encoding="utf-8").read()
cut = src.index('print("computing FULL chain')
a = src.index('c0 = load("agent_stagecap_RS_E9S_s0.npz")')
b = src.index("dnbr = guard(")
b_end = src.index("\n", b)
src = src[:a] + "c0 = CUSTOM" + src[a + len('c0 = load("agent_stagecap_RS_E9S_s0.npz")'):b] + "dnbr = np.zeros(len(ids))" + src[b_end:cut]
g = {"CUSTOM": CUSTOM}
exec(compile(src, "abl_head", "exec"), g)
x = g["rule_gap_clamp"](p)
x = g["rule_lobt_box"](x)
x = g["rule_min_floor"](x)
x = g["rule_v35"](x)[0]
x = g["rule_upper_bound"](x)
x = g["rule_hidden_reading"](x)
r = x - p
fr = pl.read_parquet(I + "agent_ablate_frame.parquet", columns=["id", "r_full"]).filter(pl.col("id").is_in(ids.tolist()))
d = dict(zip(fr["id"].to_numpy().tolist(), fr["r_full"].to_numpy()))
chk = np.array([d.get(v, np.nan) for v in ids]); okc = np.isfinite(chk)
print("reproduction vs audit frame r_full (months 1,7):", int(okc.sum()), "rows; max |diff|", float(np.abs(chk[okc] - r[okc]).max()), "rows differing", int((np.abs(chk[okc] - r[okc]) > 1e-6).sum()))
for m in (1, 7, 4, 5, 6, 12):
    s = mons == m
    print("month", m, "rows", int(s.sum()), "rules rows moved", int((r[s] != 0).sum()), "rms move %.2f" % np.sqrt((r[s] ** 2).mean()))
np.savez(I + "agent_refit2_rules.npz", ids=ids, r=r, mon=mons)
print("wrote agent_refit2_rules.npz")
