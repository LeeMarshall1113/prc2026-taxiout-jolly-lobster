"""experiment seqsplice 02: PRE-REGISTERED judge (docs/LAB-NOTES.md). Clause code copied from
archive/agent_expensive_06_judge.py / archive/agent_smallbundle_05_judge.py (same d, same four clauses).

Inputs: data/interim/agent_restack3_CAB.npz (C0), data/interim/agent_smallbundle_B.npz (seq arm seed 0),
data/interim/agent_seqsplice_seed{1,2}_B.npz (seq arm seeds 1,2; must exist -- built by agent_smallbundle_03_fold.py
before this script runs), data/interim/agent_neighbour_B.npz (neighbour arm, seed 0).
Writes data/interim/agent_seqsplice_verdict.json. Prints BUILD=<base> / BUILD=NONE for the driver shell script to branch on.
"""
import sys

sys.path.insert(0, ".")
import json

import numpy as np

I = "data/interim/"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys}


c0 = load(I + "agent_restack3_CAB.npz")
y, apts, days = c0["y"].astype(float), c0["apts"].astype(str), c0["days"].astype(str)
mon = np.array([d[5:7] for d in days])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])
pq = c0["both_guard"].astype(float)
N = len(y)
rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean())) if m.any() else float("nan")  # noqa: E731


def clauses(pt, base, base_name):
    d = lambda m: rm(pt, m) - rm(base, m)  # noqa: E731
    gain = (base - y) ** 2 - (pt - y) ** 2
    keep = np.ones(N, bool)
    order = np.argsort(-gain)[:50]
    keep[order] = False
    ch = pt != base
    h = [d(m) if (ch & m).any() else 0.0 for m in (H1, H2)]
    per = {a: d(apts == a) for a in AIRPORTS}
    c = {"vs": base_name, "all": d(np.ones(N, bool)), "worst_apt": max(per.values()), "m50": d(keep),
         "h1": h[0], "h2": h[1], "per": per, "rows_changed": int(ch.sum())}
    c["pass"] = c["all"] <= -0.10 and c["worst_apt"] <= 0.10 and c["m50"] < 0 and c["h1"] < 0 and c["h2"] < 0
    return c


def show(name, c):
    print(f"{name} (vs {c['vs']}): all {c['all']:+.4f}  worst airport {c['worst_apt']:+.3f}  minus-top-50 {c['m50']:+.3f}  "
          f"Jan-Mar {c['h1']:+.3f}  Jul-Sep {c['h2']:+.3f}  rows changed {c['rows_changed']:,}  "
          f"4-clause {'PASS' if c['pass'] else 'FAIL'}")
    print("   by airport: " + "  ".join(f"{a} {v:+.3f}" for a, v in c["per"].items()))


res = {}

# ---- 0.2 per-airport splice-set clause (seed 0 B vs C0) ----
b0 = load(I + "agent_smallbundle_B.npz")
assert (b0["ids"] == c0["ids"]).all()
pb0 = b0["both_guard"].astype(float)
gain0 = (pq - y) ** 2 - (pb0 - y) ** 2

per_airport = {}
S0 = []
print("== 0.2 per-airport splice-set clause (seed 0, B vs C0)")
for a in AIRPORTS:
    m = apts == a
    d_all = rm(pb0, m) - rm(pq, m)
    d_h1 = rm(pb0, m & H1) - rm(pq, m & H1) if (m & H1).any() else float("nan")
    d_h2 = rm(pb0, m & H2) - rm(pq, m & H2) if (m & H2).any() else float("nan")
    idx_a = np.flatnonzero(m)
    top10 = idx_a[np.argsort(-gain0[idx_a])[:10]]
    keep = m.copy()
    keep[top10] = False
    d_m10 = rm(pb0, keep) - rm(pq, keep)
    ok1, ok2, ok3 = d_all <= -0.10, (d_h1 < 0 and d_h2 < 0), d_m10 < 0
    included = ok1 and ok2 and ok3
    per_airport[a] = {"n": int(m.sum()), "d_all": d_all, "d_h1": d_h1, "d_h2": d_h2, "d_minus_top10": d_m10,
                       "clause1_le_-0.10": bool(ok1), "clause2_both_halves_neg": bool(ok2),
                       "clause3_minus_top10_neg": bool(ok3), "included": bool(included)}
    print(f"  {a}: n={m.sum():,} d_all={d_all:+.3f} (<=-0.10 {ok1})  Jan-Mar={d_h1:+.3f} Jul-Sep={d_h2:+.3f} (both<0 {ok2})  "
          f"minus-top10={d_m10:+.3f} (<0 {ok3})  -> {'IN' if included else 'OUT'}")
    if included:
        S0.append(a)
res["per_airport_seed0"] = per_airport
res["S0"] = S0
print(f"SPLICE SET S0 = {S0 or '(empty)'}")


def splice(base, treat, apts_arr, aset):
    return np.where(np.isin(apts_arr, aset), treat, base) if aset else base.copy()


if not S0:
    res["verdict_b_seed0"] = None
    res["overall"] = "FAIL_EMPTY_SPLICE_SET"
    print("VERDICT: empty splice set (0.2) -> FAIL, nothing built. No fallback set.")
    json.dump(res, open(I + "agent_seqsplice_verdict.json", "w"), indent=1)
    print("BUILD=NONE")
    sys.exit(0)

# ---- 0.3 combined splice verdict, seed 0 ----
p_s0 = splice(pq, pb0, apts, S0)
c_b0 = clauses(p_s0, pq, "C0")
print("== 0.3 combined splice verdict (seed 0)")
show("P(S0) seed0", c_b0)
res["verdict_b_seed0"] = c_b0

if not c_b0["pass"]:
    res["overall"] = "FAIL_0.3"
    print("VERDICT: 0.3 combined splice (seed 0) FAILS -> nothing built. No threshold change, no set change.")
    json.dump(res, open(I + "agent_seqsplice_verdict.json", "w"), indent=1)
    print("BUILD=NONE")
    sys.exit(0)

# ---- 0.4 seed-stability (seeds 1, 2, SAME S0) ----
print("== 0.4 seed-stability (same S0, seeds 1 and 2)")
seed_pass = {"0": True}
for s in (1, 2):
    fp = I + f"agent_seqsplice_seed{s}_B.npz"
    bs = load(fp)
    assert (bs["ids"] == c0["ids"]).all(), f"seed {s} npz id mismatch"
    pbs = bs["both_guard"].astype(float)
    p_s = splice(pq, pbs, apts, S0)
    cs = clauses(p_s, pq, "C0")
    show(f"P(S0) seed{s}", cs)
    res[f"verdict_b_seed{s}"] = cs
    seed_pass[str(s)] = bool(cs["pass"])

all_seeds_pass = all(seed_pass.values())
res["seed_pass"] = seed_pass
if not all_seeds_pass:
    res["overall"] = "FAIL_0.4_SEED_STABILITY"
    print(f"VERDICT: 0.4 seed-stability FAILS ({seed_pass}) -> nothing built. No seed dropped, no S0 change.")
    json.dump(res, open(I + "agent_seqsplice_verdict.json", "w"), indent=1)
    print("BUILD=NONE")
    sys.exit(0)

print("0.2, 0.3, 0.4 ALL PASS. Proceeding to 0.5 neighbour interaction (decides build base only).")

# ---- 0.5 neighbour interaction ----
nb = load(I + "agent_neighbour_B.npz")
assert (nb["ids"] == c0["ids"]).all()
pnb = nb["both_guard"].astype(float)
p_nbseq = splice(pnb, pb0, apts, S0)
c_int_vs_nb = clauses(p_nbseq, pnb, "NB(neighbour arm B)")
c_int_vs_c0 = clauses(p_nbseq, pq, "C0")
print("== 0.5 neighbour interaction (NB+seq vs NB -- the gating comparison)")
show("NB+seq", c_int_vs_nb)
print("== 0.5 neighbour interaction, context only (NB+seq vs C0, not gating)")
show("NB+seq", c_int_vs_c0)
res["neighbour_interaction_vs_NB"] = c_int_vs_nb
res["neighbour_interaction_vs_C0_context"] = c_int_vs_c0

if c_int_vs_nb["pass"]:
    base = "data/interim/v110_neighbour_final.parquet"
    base_fold = I + "agent_neighbour_B.npz"
    print("NB+seq vs NB PASSES -> build base = v110_neighbour_final.parquet")
else:
    base = "data/interim/combo_T1_CE_final.parquet"
    base_fold = I + "agent_restack3_CAB.npz"
    print("NB+seq vs NB FAILS -> the splice does not stack on the neighbour arm; build base = combo_T1_CE_final.parquet")

res["build_base"] = base
res["build_base_fold"] = base_fold
res["S0_csv"] = ",".join(S0)
res["overall"] = "PASS"
json.dump(res, open(I + "agent_seqsplice_verdict.json", "w"), indent=1)
print(f"BUILD={base}|{base_fold}|{','.join(S0)}")
