"""arith3 step 3 (experiment arith3, 2026-10-05): corrector replica of the round-2 experiment (agent_arith_02_fit.py used as a library, not edited) with the BASE = the nine pcd columns + the round-2 groups A-E winsorised
(0.5 / 99.5 percentiles of the non-held rows) ALWAYS present, plus the frozen round-3 groups (agent_arith3_02_cols.py).
usage: python agent_arith3_02_fit.py JOBS   JOBS = comma list of ARM:lomo:HELD:SEED
ARM = "<groups>-pw-s<shuf>-r<reduced>" with groups = letters from KYQGSR (round-3 groups) or "0" (base only, = round-2 arm ABCDE-pw refit here, used for seed-1 bases);
shuf = 1: the whole new round-3 block permuted jointly within airport-month (base columns stay true), draw = seed. reduced = 1: learning rate 0.10 (screen), 0: 0.03 (full).
Output data/interim/agent_arith3_out/<ARM>_s<seed>_m<held>.npz in the round-2 judge layout (pred, rows, bi, gate, cmask, bnames)."""
import os
import sys
import gc
import numpy as np
import lightgbm as lgb

os.environ.setdefault("ARITH_OUT", "data/interim/agent_arith3_out/")
sys.path.insert(0, "archive")
import agent_arith_02_fit as F   # noqa: E402  (library; loads the corrector inputs at import)
import agent_arith3_02_cols as C3   # noqa: E402

P = F.P
D3 = C3.load_inputs()
DAY = C3.day_block(D3)
KI = F.KI
assert D3["KN"] == F.KN and np.array_equal(D3["ids"], F.ids)
SORT_BASE = list("ABCDE")


def parse3(arm):
    t = arm.split("-")
    assert t[1] == "pw", arm
    return dict(groups=[] if t[0] == "0" else list(t[0]), shuf=int(t[2][1:]), red=int(t[3][1:]))


def winsor(blk, use):
    sub = np.where(use)[0][::7]
    lo, hi = np.nanpercentile(blk[sub], [0.5, 99.5], axis=0)
    return np.clip(blk, lo, hi)


def run3(arm, perm, ntr, nva, seed, stat, use, pool_mask, h_rows, path):
    a = parse3(arm)
    s4, s112 = stat
    Xv = {}
    for n in ["vf_d_" + v for v in F.VPOINT + F.VQ] + list(F.ADSB):
        src = F.vf["Xf"] if n.startswith("vf_") else (F.z["X"] if n in F.cols else F.ex["Xf"])
        if n.startswith("vf_"):
            j = F.NI[n] - 318
        elif n in F.cols:
            j = F.cols.index(n)
        else:
            j = [str(c) for c in F.ex["cols"]].index(n)
        Xv[n] = src[:, j].astype(np.float32)
    base, bn0 = F.col_block(SORT_BASE, Xv, s112, use, pool_mask)
    base = winsor(base, use)
    del Xv
    blk, bn = None, []
    if a["groups"]:
        blk, bn = C3.build3(D3, a["groups"], use, pool_mask, s112, KI, day=DAY)
        if a["shuf"]:
            rng = np.random.default_rng(5000 + seed)
            for ii in F.SHUF_GROUPS:
                blk[ii] = blk[rng.permutation(ii)]
        blk = winsor(blk, use)
    nx = 0 if blk is None else blk.shape[1]
    N = F.N
    X = np.empty((N, 370 + 9 + 52 + nx), np.float32)
    X[:, :140] = F.z["X"][perm]
    X[:, 140:192] = F.ex["Xf"][perm]
    X[:, 192:194] = F.e2["Xf"][perm]
    X[:, 194:198] = s4[perm]
    X[:, 198:310] = s112[perm]
    X[:, 310:318] = F.kk["Df"][perm]
    X[:, 318:370] = F.vf["Xf"][perm]
    X[:, 370:379] = F.Dpcd[perm]
    X[:, 379:431] = base[perm]
    if nx:
        X[:, 431:] = blk[perm]
    del base, blk
    rp = F.r[perm].astype(np.float32)
    dt = lgb.Dataset(X[:ntr], rp[:ntr], categorical_feature=F.catidx, params=dict(max_bin=63, verbose=-1))
    dv = lgb.Dataset(X[ntr:ntr + nva], rp[ntr:ntr + nva], categorical_feature=F.catidx, reference=dt)
    dt.construct()
    dv.construct()
    Xe = X[ntr + nva:].copy()
    del X
    gc.collect()
    bst = lgb.train(F.params(seed, a["red"]), dt, F.NR, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = bst.predict(Xe, num_iteration=bst.best_iteration)
    gate = np.abs(F.pv[h_rows] - s112[h_rows, KI["st_rwy"] * 7 + 6]) < 900
    cm = (F.gap_sched[h_rows] > 3600) | (F.gap_lobt[h_rows] > 3600) | (F.hasfp[h_rows] == 0)
    np.savez(path + ".tmp.npz", pred=pred, rows=h_rows, bi=bst.best_iteration, gate=gate, cmask=cm, bnames=np.array(bn))
    os.replace(path + ".tmp.npz", path)
    P("fit", arm, "seed", seed, "rounds", bst.best_iteration, "ncols", Xe.shape[1])


def lomo3(arm, h, seed):
    path = F.OUT + "%s_s%d_m%d.npz" % (arm, seed, h)
    if os.path.exists(path):
        P("skip", path)
        return
    months = F.months
    nxt = [k for k in months if k != h]
    val = min([k for k in nxt if k > h] or nxt)
    tr_i = np.where((F.mon != h) & (F.mon != val))[0]
    va_i = np.where(F.mon == val)[0]
    te_i = np.where(F.mon == h)[0]
    perm = np.concatenate([tr_i, va_i, te_i])
    T2 = F.TE.Te2(F.Kf)
    s4 = F.TE.te_fold(F.kf, F.r, F.mon, h, months)
    s112 = T2.fold(F.r, F.y, F.ymu_f, F.mon, h, months)
    use = F.mon != h
    pool_mask = np.array([m != h for m in months])
    run3(arm, perm, len(tr_i), len(va_i), seed, (s4, s112), use, pool_mask, te_i, path)


if __name__ == "__main__":
    for j in sys.argv[1].split(","):
        if os.environ.get("JOB_PREEMPT_FILE") and os.path.exists(os.environ["JOB_PREEMPT_FILE"]):
            P("preempt requested")
            sys.exit(10)
        p = j.split(":")
        assert p[1] == "lomo"
        lomo3(p[0], int(p[2]), int(p[3]))
