"""arith3 step 6 (experiment arith3-serve, 2026-10-06): SERVING fit of a round-three arm (all 12 months of 2025, predicts the scored 2026 rows).
Everything that is not new comes from the round-two serve code path, unchanged: agent_arith_14_fit_serve.serve("ABCDE-pw-s0-r<red>", seed, 0) returns the fold and serve matrices
(base 370 columns, key statistics with leave-own-month-out for fold rows and all-2025 for serve rows, nine plan-clock columns, groups A-E with group D served from the 2026 flags, winsorised on the fit rows).
The round-three block is appended from agent_arith3_05_servecols.build_pair (unchanged build3 on fold + serve rows; bounds from the fit rows applied to both). Column order = the held-out fit matrix (agent_arith3_02_fit.run3).
usage: python agent_arith3_06_fit_serve.py ARM:serve:ROUNDS:SEED[,ARM:serve:ROUNDS:SEED...]
  ARM e.g. KYQGSR-pw-s0-r0 (shuffled arms are not served). ROUNDS = 0: the mean best iteration of the arm's stored held-out fits ('bi' of <ARM>_s*_m*.npz, searched in $ARITH3_HELD,
  <ARITH_I>/agent_arith3_out/, <ARITH_OUT>/../out_arith3/, <ARITH_OUT>); a positive number is used as given.
  Output <ARITH_OUT>/<ARM>_serve_s<seed>.npz: pred (serve_in row order), ids, names (all feature names), imp (gain), rounds. Existing files are skipped.
  ARITH_I (inputs, default data/interim/), ARITH_OUT (default <ARITH_I>/agent_arith3_serve/), BIGC_THREADS as the other fit scripts.
  ARITH3_NR=<n>: WIRING CHECK. Builds the matrices, runs the checks (shapes, order, NaN pattern, the served round-two columns, the saved round-three serve columns), fits n rounds if n > 0
  and writes <ARM>_serve_s<seed>_wire<n>.npz (never a real name).
Needs: the fit inputs + agent_bigc13_serve_in.npz, agent_pcd_cols_app.npz, agent_arith3_time_serve.npz (and the yr 2026 rows of agent_aobtflag_flags.parquet, same file as the fits use)."""
import os
import sys
import gc
import glob
import numpy as np
import lightgbm as lgb

os.environ.setdefault("ARITH_PCD", "agent_pcd_cols_app.npz")       # same plan-clock columns as agent_pcd_cols.npz (asserted below) plus the serve ids
_I = os.environ.get("ARITH_I", "data/interim/")
os.environ.setdefault("ARITH_OUT", _I + "agent_arith3_serve/")
sys.path.insert(0, "archive")
import agent_arith_14_fit_serve as A   # noqa: E402  (round-two fit + serve; library, loads the corrector inputs at import)
import agent_arith3_02_cols as C3   # noqa: E402
import agent_arith3_05_servecols as S5   # noqa: E402

P = A.P
I, OUT = A.I, A.OUT
WIRE = os.environ.get("ARITH3_NR")
NBASE = 370 + 9 + 52
_M = {}


def parse3(arm):
    t = arm.split("-")
    assert t[1] == "pw" and t[2] == "s0", "only winsorised, unshuffled arms are served: " + arm
    g = [] if t[0] == "0" else list(t[0])
    assert all(q in C3.GROUPS3 for q in g), arm
    return dict(groups=g, red=int(t[3][1:]))


def rounds_of(arm):
    for d in [os.environ.get("ARITH3_HELD"), I + "agent_arith3_out/", os.path.join(os.path.dirname(OUT.rstrip("/\\")), "out_arith3") + "/", OUT]:
        if not d:
            continue
        fs = sorted(glob.glob(os.path.join(d, arm + "_s*_m*.npz")))
        if fs:
            bis = [int(np.load(f)["bi"]) for f in fs]
            assert len(bis) == 24, "expected the 24 held-out fits (3 seeds x 8 months) of %s in %s, found %d" % (arm, d, len(bis))
            nr = int(round(float(np.mean(bis))))
            P("rounds: mean best iteration of", len(bis), "held-out fits in", d, "=", nr, "(min %d, max %d)" % (min(bis), max(bis)))
            return nr
    raise SystemExit("no held-out fits of %s found: give the rounds in the job string (ARM:serve:ROUNDS:SEED)" % arm)


def all_names(names_b, names3):
    exn = [str(c) for c in A.ex["cols"]]
    n = list(A.cols) + exn + ["d_NEW", "d_ET"] + list(A.names[194:370]) + [str(c) for c in A.C["names"]] + list(names_b) + list(names3)
    assert list(A.names[:140]) == list(A.cols) and list(A.names[140:192]) == exn and list(A.names[192:194]) == ["d_NEW", "d_ET"], "feature name list does not match the matrix layout"
    return n


def matrices(arm):
    """(XF, XS, names) for the arm; built once per process (the matrices do not depend on the seed)"""
    if arm in _M:
        return _M[arm]
    a = parse3(arm)
    _, names_b, XF, XS = A.serve("ABCDE-pw-s0-r%d" % a["red"], 0, 0)      # the round-two serve path, wiring return (no fit)
    N, ns = A.N, len(XS)
    assert XF.shape == (N, NBASE) and XS.shape == (ns, NBASE) and ns == 344841
    assert names_b == sum([A.GNAMES[g] for g in "ABCDE"], [])
    names3 = []
    if a["groups"]:
        col = 198 + A.KI["st_rwy"] * 7 + 2
        sdf, sds = XF[:, col].copy(), XS[:, col].copy()
        D3 = C3.load_inputs()
        assert D3["KN"] == A.KN and np.array_equal(D3["ids"], A.ids)
        assert np.array_equal(D3["Dpcd"], A.Dpcd, equal_nan=True), "plan-clock columns of the two files differ"
        assert np.array_equal(D3["pv"], A.pv) and np.array_equal(XF[:, 370:379], A.Dpcd, equal_nan=True)
        Ds = S5.load_serve()
        assert np.array_equal(XS[:, 370:379], Ds["Dpcd"], equal_nan=True)
        hS = XS[:, A.cols.index("hour")]
        assert np.array_equal(hS, Ds["hour"], equal_nan=True)
        dayf, days = C3.day_block(D3), C3.day_block(Ds)
        bF, bS, names3, (lo, hi) = S5.build_pair(D3, Ds, sdf, sds, dayf, days, a["groups"])
        assert names3 == sum([C3.GROUPS3[g] for g in a["groups"]], [])
        if WIRE is not None:
            wiring_checks(XS, bF, bS, names3, lo, hi, D3["mon"])
        del D3, Ds, dayf, days
        nx = bF.shape[1]
        X = np.empty((N, NBASE + nx), np.float32)
        X[:, :NBASE] = XF
        del XF
        gc.collect()
        X[:, NBASE:] = np.clip(bF, lo, hi)
        del bF
        Xs2 = np.empty((ns, NBASE + nx), np.float32)
        Xs2[:, :NBASE] = XS
        Xs2[:, NBASE:] = np.clip(bS, lo, hi)
        del XS, bS
        XF, XS = X, Xs2
        gc.collect()
    names = all_names(names_b, names3)
    assert len(names) == XF.shape[1] == XS.shape[1]
    P("matrices", arm, XF.shape, XS.shape)
    _M.clear()
    _M[arm] = (XF, XS, names)
    return _M[arm]


def wiring_checks(XS, bF, bS, names3, lo, hi, mon):
    print("WIRING CHECKS", flush=True)
    p = I + "agent_arith_serve_cols.npz"          # the 9 + 52 serve columns saved when the round-two term in v150 was served
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        S = z["S"]
        same = np.array_equal(S, XS[:, 370:NBASE], equal_nan=True)
        print("  round-two serve columns (9 plan-clock + 52) equal to the saved v150 serve columns:", same, flush=True)
        assert same
    p = I + S5.SCOLS
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        if [str(n) for n in z["names"]] == names3:
            d = np.nanmax(np.abs(z["S"].astype(np.float64) - bS), 0)
            nn = int((np.isnan(z["S"]) != np.isnan(bS)).sum())
            print("  round-three serve columns vs the saved table-run columns: max abs diff %.3g, NaN mismatches %d" % (float(d.max()), nn), flush=True)
            assert d.max() < 1e-3 and nn == 0
    fn, sn = np.isnan(bF).mean(0), np.isnan(bS).mean(0)
    print("  column | NaN share fit rows | NaN share serve rows | share of serve rows clipped by the fit-row bounds")
    for j, n in enumerate(names3):
        v = bS[:, j]
        print("  %-18s %.4f %.4f %.4f" % (n, fn[j], sn[j], float(((v < lo[j]) | (v > hi[j])).mean())))
    assert np.isfinite(lo).all() and np.isfinite(hi).all() and (hi >= lo).all()


def serve3(arm, nr, seed):
    a = parse3(arm)
    wire = WIRE is not None
    path = OUT + "%s_serve_s%d%s.npz" % (arm, seed, "_wire%s" % WIRE if wire else "")
    if os.path.exists(path) and not wire:
        P("skip", path)
        return
    if wire:
        nr = int(WIRE)
    elif nr <= 0:
        nr = rounds_of(arm)
    XF, XS, names = matrices(arm)
    si_ids = np.load(I + os.environ.get("ARITH_SERVE", "agent_bigc13_serve_in.npz"), allow_pickle=True)["ids"]
    if nr == 0:
        P("wiring only, no fit")
        return
    b = lgb.train(A.params(seed, a["red"]), lgb.Dataset(XF, A.r.astype(np.float32), categorical_feature=A.catidx, params=dict(max_bin=63, verbose=-1)), nr)
    pred = b.predict(XS)
    assert np.isfinite(pred).all() and len(pred) == 344841 == len(si_ids)
    np.savez(path + ".tmp.npz", pred=pred, ids=si_ids, names=np.array(names), imp=b.feature_importance("gain"), rounds=nr, arm=arm, seed=seed)
    os.replace(path + ".tmp.npz", path)
    P("serve fit", arm, "seed", seed, "rounds", nr, "ncols", XF.shape[1], "pred mean %.3f sd %.3f" % (pred.mean(), pred.std()))


if __name__ == "__main__":
    for j in sys.argv[1].split(","):
        if os.environ.get("JOB_PREEMPT_FILE") and os.path.exists(os.environ["JOB_PREEMPT_FILE"]):
            P("preempt requested")
            sys.exit(10)
        p = j.split(":")
        assert p[1] == "serve", j
        serve3(p[0], int(p[2]), int(p[3]))
