"""arith step 12: per-airport choice (2025 arrays only). For each of the nine airports compare the candidate term against the term SERVED NOW in v146 at the served dose:
pcd at the eight airports EDDF EDDM EGLL EHAM LEBL LEMD LIRF LSZH (EHAM since v144), the OLD term at LFPG. Rule fixed before looking: candidate is taken where all-eight-month RMSE difference < 0,
January and July both < 0, and better in at least 6 of 8 months; otherwise the airport keeps the shipped term. Output data/interim/agent_arith_airports.json.
usage: python agent_arith_12_select.py SEEDS CAND   (e.g. 0,1,2 ABCDE-pw-s0-r0)"""
import sys, json
import numpy as np
sys.path.insert(0, "archive")
import agent_basefix_lib as L   # noqa: E402
I = "data/interim/"; ALLM = (1, 2, 3, 7, 8, 9, 10, 11)
z = np.load(I + "agent_bigc13_fold.npz", allow_pickle=True); fold_ids = z["ids"].astype(float); monf = z["mon"]
pos = {v: i for i, v in enumerate(fold_ids.tolist())}; ix = np.array([pos[v] for v in L.ids.tolist()])
NB = np.load(I + "agent_bigc13_base.npz")["nb"][ix]; y = L.y
seeds = [int(s) for s in sys.argv[1].split(",")]; cand = sys.argv[2]


def arith(s):
    F = np.full(len(fold_ids), np.nan)
    for m in ALLM:
        d = np.load(I + "agent_arith_out/%s_s%d_m%d.npz" % (cand, s, m)); F[d["rows"]] = d["pred"]
    return F[ix]


def pcdarr(s):
    arr = np.zeros(len(fold_ids))
    for m in range(1, 13):
        d = np.load(I + f"agent_pcd_modal/C4ff7V5_s{s}_m{m}.npz"); arr[d["rows"]] = d["pred"]
    return arr[ix]


C = np.mean([arith(s) for s in seeds], 0); P = np.mean([pcdarr(s) for s in (0, 1, 2)], 0)
O = np.mean([np.load(I + f"agent_bigc12_r1x_s{s}_C4ff7V5.npz")["C4ff7V5"][ix] for s in (0, 1, 2)], 0)
assert np.isfinite(C).all()


def served(corr):
    x = np.maximum(NB + corr * L.K5, 0); x = np.maximum(x + 0.5 * L.TAIL, 0); out = x.copy()
    out[L.mid] = x[L.mid] + L.W3[L.mid] * (L.MIX[L.mid] - x[L.mid])
    for rr in ("R1", "R2", "R3b"):
        s_ = L.rule == rr; out[s_] = L.ebpred[s_]
    return np.rint(np.maximum(out, 0))


rm = lambda p, m: float(np.sqrt(((p - y)[m] ** 2).mean()))   # noqa: E731
Pc = served(C)
choice = {}
for a in L.AIR:
    cur = O if a == "LFPG" else P
    Pk = served(cur); mk = L.apt == a
    mm = {m: rm(Pc, mk & (L.mon == m)) - rm(Pk, mk & (L.mon == m)) for m in ALLM}
    all8 = rm(Pc, mk) - rm(Pk, mk)
    nb = sum(v < 0 for v in mm.values())
    ok = all8 < 0 and mm[1] < 0 and mm[7] < 0 and nb >= 6
    choice[a] = bool(ok)
    print("%s vs %s: all8 %+.3f Jan %+.3f Jul %+.3f months better %d/8 -> %s" % (a, "old" if a == "LFPG" else "pcd", all8, mm[1], mm[7], nb, "CANDIDATE" if ok else "keep shipped"))
json.dump(choice, open(I + "agent_arith_airports.json", "w")); print(choice)
