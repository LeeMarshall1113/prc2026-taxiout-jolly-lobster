"""refit2 step 2: the NEW FIT BASE `nb` for every row of the 12-month corrector input (agent_bigc12_fold.npz, 1,723,537 rows), and its verification against the audit's held-out chain.
nb = held-out serve-like prediction BEFORE the corrector term:
    x = max(p + rules(p), 0)                      post-chain rules in served order (gap clamp, LOBT box, min floor, v35, upper bound, hidden-plan reading); no neighbour blend
    x = max(x + M1 + M2 + M3, 0)                  (= pv - p of the fold file, exactly)
    x = x + silent + meta                         (fold treat-minus-control; 0 on months 4,5,6,12 which have no nudge fold)
Not in nb (applied after the corrector, as served): corrector x dose, 0.5 x tail, Rome no-plan mixture / hidden-plan readings / lever B / long-gap step.
Old months (1,2,3,7,8,9,10,11): rules, silent, meta from agent_ablate_frame.parquet; new months (4,5,6,12): rules from agent_refit2_01_rules.py output (same functions, reproduced the frame to 0.0 on months 1, 7).
Checks printed: (1) nb - pv_old rms by airport; (2) the audit's held-out chain with the blend (Jan 253.91 / Jul 249.03 / pooled 251.01) and without it; (3) the chain rebuilt from nb equals the audit chain
with the blend off to 0.0 on every row.   Writes data/interim/agent_bigc13_base.npz.   Run under archive/capped_run.py --cap-gb 3."""
import sys
import numpy as np
sys.path.insert(0, "archive")
import agent_basefix_lib as L   # noqa: E402  (frame + audit chain + day-bootstrap judge)

I = "data/interim/"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
z = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
ids = z["ids"].astype(float); pv = z["pv"].astype(float); p = z["p"].astype(float); mon = z["mon"]; apt = z["apt"].astype(str); N = len(ids)
M = pv - p
pos = {v: i for i, v in enumerate(ids.tolist())}
r = np.zeros(N); sil = np.zeros(N); meta = np.zeros(N)
ix = np.array([pos[v] for v in L.ids.tolist()])
r[ix] = L.R; sil[ix] = L.SIL; meta[ix] = L.META
print("max |(pv-p) - (M1+M2+M3)| on old rows:", np.abs(M[ix] - (L.M1 + L.M2 + L.M3)).max())
rz = np.load(I + "agent_refit2_rules.npz")
new = np.isin(mon, [4, 5, 6, 12]); inz = np.array([pos[v] for v in rz["ids"].tolist()]); sel = new[inz]
r[inz[sel]] = rz["r"][sel]
assert (np.isin(mon[ix], [1, 2, 3, 7, 8, 9, 10, 11])).all()
x = np.maximum(p + r, 0); x = np.maximum(x + M, 0); nb = x + sil + meta
print("rows", N, "old-month rows", len(ix), "new-month rows", int(new.sum()), "| rules move rms old %.2f new %.2f" % (np.sqrt((r[ix] ** 2).mean()), np.sqrt((r[new] ** 2).mean())))
print("nb - pv_old rms by airport (the fit-base gap the refit removes):")
for a in AIR:
    s = apt == a
    print(f"  {a}: all months rms {np.sqrt(((nb - pv)[s] ** 2).mean()):7.2f} | old months {np.sqrt(((nb - pv)[s & ~new] ** 2).mean()):7.2f} | new months {np.sqrt(((nb - pv)[s & new] ** 2).mean()):7.2f}")
# ---- (2) audit numbers
def rm(pred, m): return float(np.sqrt(((pred - L.y) ** 2)[m].mean()))
print("audit chain WITH blend: Jan %.2f Jul %.2f pooled %.2f (audit: 253.91 / 249.03 / 251.01) all8 %.2f (audit 201.38)" % (rm(L.SHIP, L.JAN), rm(L.SHIP, L.JUL), rm(L.SHIP, L.JJ), rm(L.SHIP, L.ALL)))
nob = L.chain(nb=0.0)
print("audit chain NO blend  : Jan %.2f Jul %.2f pooled %.2f   delta vs shipped Jan %+.3f Jul %+.3f pooled %+.3f (audit -0.188 / -0.207 / -0.199)" % (
    rm(nob, L.JAN), rm(nob, L.JUL), rm(nob, L.JJ), rm(nob, L.JAN) - rm(L.SHIP, L.JAN), rm(nob, L.JUL) - rm(L.SHIP, L.JUL), rm(nob, L.JJ) - rm(L.SHIP, L.JJ)))
# ---- (3) chain rebuilt from nb
nbf = nb[ix]
y_ = np.maximum(nbf + L.KF, 0); y_ = np.maximum(y_ + 0.5 * L.TAIL, 0); out = y_.copy()
out[L.mid] = y_[L.mid] + L.W3[L.mid] * (L.MIX[L.mid] - y_[L.mid])
for rr in ("R1", "R2", "R3b"):
    s = L.rule == rr; out[s] = L.ebpred[s]
out = np.rint(np.maximum(out, 0))
print("chain rebuilt from nb vs audit chain (blend off): max |diff| %.1f, rows differing %d of %d" % (np.abs(out - nob).max(), int((out != nob).sum()), len(out)))
np.savez(I + "agent_bigc13_base.npz", ids=ids, nb=nb, r=r, M=M, silent=sil, meta=meta, pv_old=pv, mon=mon, apt=apt)
print("wrote agent_bigc13_base.npz")
