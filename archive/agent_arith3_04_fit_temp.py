"""arith3 temporal frame (experiment corrnext takeover, 2026-10-06). Library use of agent_arith_02_fit (F) and agent_arith3_02_fit (F3); neither is edited.
usage: python agent_arith3_04_fit_temp.py JOBS   JOBS = comma list of ARM:temp:SIDE:SEED   (SIDE a = fit months 1-6 predict 7-12; b = fit 7-12 predict 1-6)
ARM as in agent_arith3_02_fit.py (e.g. 0-pw-s0-r0 control, KYQGSR-pw-s0-r0 candidate). Output <OUT>/<ARM>_s<seed>_t<side>.npz, judged by agent_arith3_03_judge.py temp (frame 'temp' reads ta / tb)."""
import os
import sys
import numpy as np
os.environ.setdefault("ARITH_OUT", "data/interim/agent_arith3_out/")
sys.path.insert(0, "archive")
import agent_arith3_02_fit as F3   # noqa: E402
F = F3.F


def temporal3(arm, side, seed):
    path = F.OUT + "%s_s%d_t%s.npz" % (arm, seed, side)
    if os.path.exists(path):
        F.P("skip", path)
        return
    mon = F.mon
    fit_m = [1, 2, 3, 4, 5, 6] if side == "a" else [7, 8, 9, 10, 11, 12]
    val = 6 if side == "a" else 7
    Ts = np.where(np.isin(mon, fit_m))[0]
    Hs = np.where(~np.isin(mon, fit_m))[0]
    tr_i = Ts[mon[Ts] != val]
    va_i = Ts[mon[Ts] == val]
    perm = np.concatenate([tr_i, va_i, Hs])
    T2 = F.TE.Te2(F.Kf[Ts])
    s4 = np.empty((F.N, 4), np.float32)
    s112 = np.empty((F.N, 112), np.float32)
    s4[Ts] = F.TE.te_loo(F.kf[Ts], F.r[Ts], mon[Ts], fit_m)
    s112[Ts] = T2.fold(F.r[Ts], F.y[Ts], F.ymu_f[Ts], mon[Ts], None, fit_m)
    s4[Hs] = F.TE.te_serve(F.kf[Ts], F.r[Ts], F.kf[Hs])
    s112[Hs] = T2.serve(F.r[Ts], F.y[Ts], mon[Ts], fit_m, F.Kf[Hs], F.ymu_f[Hs])
    use = np.isin(mon, fit_m)
    pool_mask = np.array([m in fit_m for m in F.months])
    F3.run3(arm, perm, len(tr_i), len(va_i), seed, (s4, s112), use, pool_mask, Hs, path)


if __name__ == "__main__":
    for j in sys.argv[1].split(","):
        p = j.split(":")
        assert p[1] == "temp"
        temporal3(p[0], p[2], int(p[3]))
