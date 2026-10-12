"""pcdx step 2: frame numbers for the candidate changes (EHAM only; EHAM+LFPG) against v143 as served, plus scored-set scaling inputs. Imports the arms by re-running the setup of step 1 (exec of its head)."""
import sys
src = open("archive/agent_pcdx_01_judge.py").read().split('CUR = served(')[0]
exec(compile(src, "pcdx01head", "exec"))
CUR = served(np.where(is7, P, O), K41)
eh = apt == "EHAM"; lf = apt == "LFPG"
for tag, mk in (("EHAM only", eh), ("LFPG only", lf), ("EHAM+LFPG", eh | lf)):
    C2 = served(np.where(is7 | mk, P, O), K41)
    print(line(C2, CUR, tag + " (frame, all nine airports pooled)"))
    for s in S3:
        print("   " + line(served(np.where(is7 | mk, P_[s], O_[s]), K41), served(np.where(is7, P_[s], O_[s]), K41), f"seed {s}"))
    print("   whole-frame RMSE v143 J+J %.3f all8 %.3f; changed rows %d" % (np.sqrt(((CUR - y)[jj] ** 2).mean()), np.sqrt(((CUR - y) ** 2).mean()), int((C2 != CUR).sum())))
C2 = served(np.where(is7 | eh, P, O), K41)
split(C2, CUR, jj, "EHAM-only J+J"); split(C2, CUR, np.ones(N, bool), "EHAM-only all8")
print("EHAM per month (pcd - old, EHAM rows):", {mm: round(cmp(C2, CUR, [mm], eh)[0], 3) for mm in MONS})
print("whole-frame per month:", {mm: round(cmp(C2, CUR, [mm])[0], 3) for mm in MONS})
# worst month bootstrap: probability-weighted lower tail
g, p, lo, hi = cmp(C2, CUR, [1, 7]); print("EHAM-only J+J 90%% [%+.3f, %+.3f]" % (lo, hi))
# scaling to the January/July file: EHAM share of its rows, frame share
print("EHAM frame share %.3f; scored share %.3f" % (eh.mean(), 38182 / 344841))
