"""arith4 step X (experiment arith4, 2026-10-06): round-4 corrector fit = the round-3 recipe (agent_arith3_02_fit.py used as a library: pcd nine + round-2 ABCDE winsorised + round-3 KYQGSR
winsorised, all ALWAYS present = the base) plus the FROZEN round-4 groups of agent_arith4_X_cols.py (T, L, W, D; 88 columns).
usage: python archive/agent_arith4_X_fit.py JOBS      JOBS = comma list of ARM:lomo:HELD:SEED  or  ARM:temp:SIDE:SEED  (SIDE a = fit months 1-6 predict 7-12, b = the reverse)
ARM = "<groups>-pw-s<shuf>-r<reduced>" with groups = letters from TLWD (round-4 groups) or "0" (base only = the round-3 arm KYQGSR-pw refit with this code; used for the control
      of the temporal frame and for a reproduction check); shuf = 1: the round-4 block permuted jointly within airport-month (base columns true), draw = seed; reduced = 1: lr 0.10.
Optional trailing tokens: "-mT" / "-mH" = coverage-mask variant (step d; see mask_rows: track-derived inputs blanked so each airport-month's tracked share equals the 2026 rate from
data/interim/agent_arith4_avail.json; mT masks training and held rows, mH the held rows only = the control); "-w<k>" = wiring job (every k-th row, 5 rounds).
Output $ARITH_OUT/<ARM>_s<seed>_m<held>.npz (or _t<side>.npz) in the judge layout (pred, rows, bi, gate, cmask, bnames). Default ARITH_OUT data/interim/agent_arith4_out/.
Wiring checks: ARITH_NR=<rounds> caps the rounds; ARITH4_SUB=<k> fits on every k-th row of each split (small matrix, no slot needed); ARITH4_SCRAMBLE=1 scrambles the held
month's labels before building the columns and asserts that no held-row column changes (label-leak test of the builder)."""
import os
import sys
import gc
import numpy as np
import lightgbm as lgb

os.environ.setdefault("ARITH_OUT", "data/interim/agent_arith4_out/")
sys.path.insert(0, "archive")
if os.environ.get("ARITH4_MMAP") == "1":
    import agent_arith4_mmap_shim   # noqa: E402,F401  (local wiring checks only, see that file)
import agent_arith3_02_fit as R3   # noqa: E402  (library; loads the corrector inputs at import; its ARITH_OUT default is overridden above)
import agent_arith4_X_cols as C4   # noqa: E402

F = R3.F
C3 = R3.C3
P = F.P
KI = F.KI
OUT = F.OUT
os.makedirs(OUT, exist_ok=True)
SUB = int(os.environ.get("ARITH4_SUB", 1))
SCRAMBLE = os.environ.get("ARITH4_SCRAMBLE", "0") == "1"
R3_GROUPS = list("KYQGSR")
# ---- coverage-mask variant (step d): track-derived inputs blanked on a random share of tracked rows so that each airport-month's tracked share equals the 2026 rate
MASK = {"mode": None}   # None | "T" (train + held rows) | "H" (held rows only = the control: a 2025-trained model served into 2026 availability)
_c0, _c1 = F.cols.index("P_span"), F.cols.index("tt_g")
MASK_FOLD = list(range(_c0, _c1 + 1))                                             # NaN on every untracked row (P_span .. tt_g)
_exc = [str(c) for c in F.ex["cols"]]
MASK_EX = [_exc.index(c) for c in _exc if (_exc.index("first_m") <= _exc.index(c) <= _exc.index("start_d")) or c.startswith("tf_") or c.startswith("osm_")]
COV, OBS, OBK = F.cols.index("cov_cat"), F.cols.index("ob_src"), F.cols.index("ob_prod_kind")


def mask_rows(h):
    """fold rows whose track columns are blanked: per airport x month, tracked rows (m_e_start present) drawn at random (fixed draw 777) so the tracked share drops to the 2026 rate
    (January rate for months 10-3, July rate for 4-9; data/interim/agent_arith4_avail.json, inputs only). Where 2026 has MORE tracks than the 2025 month nothing is masked."""
    import json
    av = json.load(open(F.I + "agent_arith4_avail.json"))
    X = F.z["X"]
    tracked = np.isfinite(X[:, F.cols.index("m_e_start")])
    del X
    rng = np.random.default_rng(777)
    m = np.zeros(F.N, bool)
    for a in sorted(set(F.apt.tolist())):
        for mo in F.months:
            ii = np.where((F.apt == a) & (F.mon == mo) & tracked)[0]
            n_all = ((F.apt == a) & (F.mon == mo)).sum()
            if len(ii) == 0:
                continue
            s25 = len(ii) / n_all
            tgt = av[a]["m_e_start_2026"]["1" if mo in (10, 11, 12, 1, 2, 3) else "7"]
            if s25 > tgt:
                k = int(round(len(ii) * (1 - tgt / s25)))
                m[rng.choice(ii, k, replace=False)] = True
    if MASK["mode"] == "H":
        m &= F.mon == h
    P("coverage mask (%s): %d rows masked (%.3f of fold rows; held month %d: %.3f)" % (MASK["mode"], m.sum(), m.mean(), h, m[F.mon == h].mean()))
    return m


def untracked_cov_code():
    """per airport, the most common cov_cat among untracked rows (what a masked row is relabelled to)"""
    X = F.z["X"]
    tracked = np.isfinite(X[:, F.cols.index("m_e_start")])
    cc = X[:, COV]
    del X
    out = {}
    for a in sorted(set(F.apt.tolist())):
        v = cc[(F.apt == a) & ~tracked]
        u, c = np.unique(v[np.isfinite(v)], return_counts=True)
        out[a] = float(u[np.argmax(c)]) if len(u) else np.nan
    return out


def parse4(arm):
    t = arm.split("-")
    assert t[1] == "pw", arm
    return dict(groups=[] if t[0] == "0" else list(t[0]), shuf=int(t[2][1:]), red=int(t[3][1:]))


def wiring_token(arm):
    """an arm ending in "-w<k>" is a WIRING job: every k-th row, 5 rounds (so a cloud wiring job needs no environment change); returns the arm without the token"""
    global SUB
    t = arm.split("-")
    if len(t) >= 5 and t[-1].startswith("w") and t[-1][1:].isdigit():
        SUB = int(t[-1][1:])
        F.NR = 5
        t = t[:-1]
    MASK["mode"] = None
    if len(t) >= 5 and t[-1] in ("mT", "mH"):   # coverage-mask variant: mT = masked in training and in the held month, mH = held month only (control)
        MASK["mode"] = t[-1][1]
        t = t[:-1]
    return "-".join(t), len(t) != len(arm.split("-"))


def base_block(s112, use, pool_mask, mrow=None):
    """pcd + ABCDE-pw + KYQGSR-pw as agent_arith3_02_fit.run3 builds them (same code path); mrow = coverage-mask rows (their ADS-B inputs blanked before the C group is built)"""
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
        if mrow is not None and not n.startswith("vf_"):
            Xv[n][mrow] = np.nan
    b2, _ = F.col_block(R3.SORT_BASE, Xv, s112, use, pool_mask)
    b2 = R3.winsor(b2, use)
    del Xv
    b3, _ = C3.build3(R3.D3, R3_GROUPS, use, pool_mask, s112, KI, day=R3.DAY)
    b3 = R3.winsor(b3, use)
    return b2, b3


def run4(arm, perm, ntr, nva, seed, stat, use, pool_mask, h_rows, path, frame="lomo"):
    a = parse4(arm)
    s4, s112 = stat
    mrow = None
    if MASK["mode"]:
        assert frame == "lomo"
        mrow = mask_rows(int(F.mon[h_rows[0]]))
        covcode = untracked_cov_code()
    b2, b3 = base_block(s112, use, pool_mask, mrow)
    blk, bn = None, []
    if a["groups"]:
        blk, bn = C4.build4(a["groups"], F.pv, F.Kf, KI, s112, F.mon, F.months, use, pool_mask)
        if SCRAMBLE and frame == "lomo":   # label-leak test (lomo frame only: it rebuilds the fold key statistics): rebuild with the held month's labels scrambled (the key statistics for the held rows must not change)
            rng = np.random.default_rng(1)
            y2 = F.y.copy()
            y2[h_rows] = rng.permutation(y2[h_rows])
            r2 = np.clip(y2 - F.pv, -900, 900)
            held = F.months[int(np.searchsorted(F.months, F.mon[h_rows[0]]))]
            T2 = F.TE.Te2(F.Kf)
            s112b = T2.fold(r2, y2, F.ymu_f, F.mon, held, F.months)
            blkb, _ = C4.build4(a["groups"], F.pv, F.Kf, KI, s112b, F.mon, F.months, use, pool_mask)
            dmax = np.nanmax(np.abs(blkb[h_rows] - blk[h_rows]))
            P("SCRAMBLE test: max abs change of a held-row column after scrambling the held month's labels = %.6f (must be 0)" % dmax)
            assert dmax == 0.0
            del blkb, s112b
        if a["shuf"]:
            rng = np.random.default_rng(5000 + seed)
            for ii in F.SHUF_GROUPS:
                blk[ii] = blk[rng.permutation(ii)]
        blk = R3.winsor(blk, use)
        nanshare = np.isnan(blk).mean(0)
        P("round-4 block: %d columns; NaN share by column min %.3f max %.3f; columns over 0.3: %s" % (blk.shape[1], nanshare.min(), nanshare.max(), [(bn[i], round(float(nanshare[i]), 2)) for i in np.where(nanshare > 0.3)[0]]))
    nx = 0 if blk is None else blk.shape[1]
    N = F.N
    if SUB > 1:   # wiring mode: every SUB-th row of each split
        tr_i, va_i, te_i = perm[:ntr][::SUB], perm[ntr:ntr + nva][::SUB], perm[ntr + nva:][::SUB]
        perm = np.concatenate([tr_i, va_i, te_i])
        ntr, nva = len(tr_i), len(va_i)
        h_rows = te_i
    n_ = len(perm)
    X = np.empty((n_, 370 + 9 + 52 + 45 + nx), np.float32)
    X[:, :140] = F.z["X"][perm]
    X[:, 140:192] = F.ex["Xf"][perm]
    X[:, 192:194] = F.e2["Xf"][perm]
    X[:, 194:198] = s4[perm]
    X[:, 198:310] = s112[perm]
    X[:, 310:318] = F.kk["Df"][perm]
    X[:, 318:370] = F.vf["Xf"][perm]
    X[:, 370:379] = F.Dpcd[perm]
    X[:, 379:431] = b2[perm]
    X[:, 431:476] = b3[perm]
    if nx:
        X[:, 476:] = blk[perm]
    del b2, b3, blk
    if mrow is not None:
        mp = np.where(mrow[perm])[0]
        X[np.ix_(mp, MASK_FOLD)] = np.nan
        X[np.ix_(mp, [140 + j for j in MASK_EX])] = np.nan
        X[mp, COV] = np.array([covcode[a_] for a_ in F.apt[perm[mp]]], np.float32)
        X[mp, OBS] = 1.0
        X[mp, OBK] = -1.0
        P("coverage mask applied to %d matrix rows (%d fold columns, %d extra columns, cov_cat/ob_src/ob_prod_kind relabelled)" % (len(mp), len(MASK_FOLD), len(MASK_EX)))
    gc.collect()
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
    P("fit", arm, "seed", seed, "rounds", bst.best_iteration, "ncols", Xe.shape[1], "rows", n_)


def lomo4(arm, h, seed):
    arm0, wiring = wiring_token(arm)
    path = OUT + "%s_s%d_m%d.npz" % (arm, seed, h)
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
    run4(arm0, perm, len(tr_i), len(va_i), seed, (s4, s112), use, pool_mask, te_i, path)


def temporal4(arm, side, seed):
    """temporal frame as agent_arith_02_fit.temporal: fit months 1-6 predict 7-12 (side a) or the reverse (b); the pool for every statistic is the fit half only"""
    path = OUT + "%s_s%d_t%s.npz" % (arm, seed, side)
    if os.path.exists(path):
        P("skip", path)
        return
    N = F.N
    fit_m = [1, 2, 3, 4, 5, 6] if side == "a" else [7, 8, 9, 10, 11, 12]
    val = 6 if side == "a" else 7
    Ts = np.where(np.isin(F.mon, fit_m))[0]
    Hs = np.where(~np.isin(F.mon, fit_m))[0]
    tr_i = Ts[F.mon[Ts] != val]
    va_i = Ts[F.mon[Ts] == val]
    perm = np.concatenate([tr_i, va_i, Hs])
    T2 = F.TE.Te2(F.Kf[Ts])
    s4 = np.empty((N, 4), np.float32)
    s112 = np.empty((N, 112), np.float32)
    s4[Ts] = F.TE.te_loo(F.kf[Ts], F.r[Ts], F.mon[Ts], fit_m)
    s112[Ts] = T2.fold(F.r[Ts], F.y[Ts], F.ymu_f[Ts], F.mon[Ts], None, fit_m)
    s4[Hs] = F.TE.te_serve(F.kf[Ts], F.r[Ts], F.kf[Hs])
    s112[Hs] = T2.serve(F.r[Ts], F.y[Ts], F.mon[Ts], fit_m, F.Kf[Hs], F.ymu_f[Hs])
    use = np.isin(F.mon, fit_m)
    pool_mask = np.array([m in fit_m for m in F.months])
    run4(arm, perm, len(tr_i), len(va_i), seed, (s4, s112), use, pool_mask, Hs, path, frame="temp")


if __name__ == "__main__":
    for j in sys.argv[1].split(","):
        if os.environ.get("JOB_PREEMPT_FILE") and os.path.exists(os.environ["JOB_PREEMPT_FILE"]):
            P("preempt requested")
            sys.exit(10)
        p = j.split(":")
        if p[1] == "lomo":
            lomo4(p[0], int(p[2]), int(p[3]))
        else:
            temporal4(p[0], p[2], int(p[3]))
