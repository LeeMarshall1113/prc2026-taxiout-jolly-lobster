"""Stage 41: serve fits of the residual correctors for the departures prepared by final/40_serve_inputs.py.

The corrector is fitted on the 12 months of 2025 and predicts the served rows; it does not depend on which months are served,
so the fit code of January/July 2026 is reused as shipped and only the serve-side inputs change. Two shipped fits:

  plan-clock corrector (the v143 refit, 12 seeds)   modal/agent_pcd_app.py serve_one("C4ff7V5", seed)
  arithmetic rounds two and three (v150, v152)      archive/agent_arith3_06_fit_serve.py, arm KYQSR-pw-s0-r0 (round three
                                                    without the view-geometry group, as shipped), 12 seeds

Both functions below are parameterised copies of that code (row count free, input directory free); the fit parameters, seeds,
round counts and column layout are untouched. Round counts are those of the stored January/July serve fits (pcd 775, arith 845).

Sub-commands
  stage     build <outdir>/<tag>_stage/ : the 2025-side files hard-linked from data/interim, the serve-side files of stage 40
            written under the names the fit code reads (keys, views, plan-clock columns, flags, time, serve_in).
  check     compare the staged serve-side arrays with the stored January/July files for the same ids (only for a stage 40
            run on ranking.parquet rows, profile stored).
  arith     the arithmetic serve fits ->  <stage>/out/arith/KYQSR-pw-s0-r0_serve_s<seed>.npz   (pred, ids, names, imp, rounds)
  pcd       the plan-clock serve fits  ->  <stage>/out/pcd/C4ff7V5_s<seed>.npz
  v5        the earlier 12-month corrector (stage built with --prefix agent_bigc12, serve file made with the old pv)
            -> <stage>/out/v5/C4ff7V5_s<seed>.npz  (3 seeds feed the tail component of final/42_small_stages.py)
  summary   mean and sd of the corrections per airport-month of finished fits

    python final/41_corrector_serve.py stage --serve-dir data/final/J9_serve --tag J10
    python final/41_corrector_serve.py arith --stage data/final/J10_stage --seeds 0-11 [--wire 3]
    python final/41_corrector_serve.py pcd   --stage data/final/J10_stage --seeds 0-11

Machine: each fit builds a 1.72M x 479 float32 matrix (3.3 GB) next to the shipped import of the 2025 inputs; the shipped
round-three run needed about 10 GB and 7 min to build + 8-10 min per seed (6 threads), the plan-clock fit about the same.
Local, one seed after another in one process: about 10 GB slot, 3.5 to 4 h for the 24 fits. Cloud: modal/final_corrector_app.py.
--wire N builds the matrices, runs the shape checks and fits N rounds (no real fit) to prove the wiring.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
INTERIM = REPO / "data" / "interim"
sys.path.insert(0, str(REPO / "archive"))
T0 = time.time()
ARM, ARM_ROUNDS = "KYQSR-pw-s0-r0", 845
PCD_CFG, PCD_ROUNDS, V5_ROUNDS = "C4ff7V5", 775, 676
# 2025-side files the fit code reads from its input directory (hard-linked, never rewritten)
FOLD_FILES = ["agent_bigc13_fold.npz", "agent_bigc12_extra.npz", "agent_bigc12_extra2.npz", "agent_yearfp_names.json", "agent_arith3_time.npz",
              "agent_pcd_cols.npz", "agent_bigc12_maps.json"]


def P(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


def seeds_of(text: str) -> list[int]:
    out = []
    for t in text.split(","):
        a, _, b = t.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def link(src: Path, dst: Path):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


# ------------------------------------------------------------------------------------------------ stage
def stage(serve_dir: Path, outdir: Path, tag: str, prefix: str = "agent_bigc13"):
    """prefix agent_bigc13 = the plan-clock / arithmetic correctors (serve file made with the B26 pv); agent_bigc12 = the earlier
    12-month corrector C4ff7V5 whose serve preds feed the tail component (serve file made with --fold-prefix agent_bigc12)."""
    st = outdir / f"{tag}_stage"
    st.mkdir(parents=True, exist_ok=True)
    for f in FOLD_FILES:
        link(INTERIM / f.replace("agent_bigc13_fold", f"{prefix}_fold"), st / f.replace("agent_bigc13_fold", f"{prefix}_fold"))
    sin = np.load(serve_dir / f"{prefix}_serve_in.npz", allow_pickle=True)
    ids = sin["ids"].astype(float)
    n = len(ids)
    link(serve_dir / f"{prefix}_serve_in.npz", st / f"{prefix}_serve_in.npz")
    link(serve_dir / "agent_arith3_time_serve.npz", st / "agent_arith3_time_serve.npz")
    if prefix == "agent_bigc13":   # pcd app file: fold half unchanged, serve half replaced
        pc = np.load(INTERIM / "agent_pcd_cols_app.npz", allow_pickle=True)
        ps = np.load(serve_dir / "agent_pcd_cols_app.npz", allow_pickle=True)
        (st / "agent_pcd_cols_app.npz").unlink(missing_ok=True)
        np.savez(st / "agent_pcd_cols_app.npz", D=pc["D"], Ds=ps["Ds"], ids=pc["ids"], ids_s=ps["ids_s"], names=pc["names"])
        del pc
    sk = np.load(serve_dir / "serve_keys.npz", allow_pickle=True)
    k2 = dict(np.load(INTERIM / f"{prefix}_keys2.npz", allow_pickle=True))
    assert [str(x) for x in sk["names2"]] == [str(x) for x in k2["names"]]
    k2["Ks"], k2["Ks_d"] = sk["Ks2"], sk["Ks_d"]
    (st / f"{prefix}_keys2.npz").unlink(missing_ok=True)
    np.savez(st / f"{prefix}_keys2.npz", **k2)
    del k2
    k1 = dict(np.load(INTERIM / "agent_bigc12_keys.npz", allow_pickle=True))
    assert [str(x) for x in sk["names1"]] == [str(x) for x in k1["names"]]
    k1["Ks"], k1["apt_s"] = sk["Ks"], sk["apt_s"]
    (st / "agent_bigc12_keys.npz").unlink(missing_ok=True)
    np.savez(st / "agent_bigc12_keys.npz", **k1)
    del k1
    vz = np.load(serve_dir / "serve_views.npz", allow_pickle=True)
    assert np.array_equal(vz["ids"].astype(float), ids)
    v = dict(np.load(INTERIM / "agent_bigc12_vfac_clean5.npz", allow_pickle=True))
    v["Xs"] = vz["Xs"]
    (st / "agent_bigc12_vfac_clean5.npz").unlink(missing_ok=True)
    np.savez(st / "agent_bigc12_vfac_clean5.npz", **v)
    del v
    old = pl.read_parquet(INTERIM / "agent_aobtflag_flags.parquet").filter(pl.col("yr") == 2025)
    new = pl.read_parquet(serve_dir / "serve_flags.parquet")
    new = new.select([pl.col(c).cast(old.schema[c]) if c in new.columns else pl.lit(None, old.schema[c]).alias(c) for c in old.columns])
    pl.concat([old, new]).write_parquet(st / "agent_aobtflag_flags.parquet")
    P(f"staged {n:,} serve rows in {st}")
    print("served rows per airport (nine corrector airports first):")
    apt = sin["apt"].astype(str)
    mo = sin["mon_s"]
    for a in sorted(set(apt.tolist())):
        print("  %s %s" % (a, {int(m): int(((apt == a) & (mo == m)).sum()) for m in sorted(set(mo.tolist()))}))
    return st


def check(stage_dir: Path):
    """The staged serve-side arrays equal the stored January/July files on the staged ids."""
    sys.path.insert(0, str(HERE))
    import fserve as F
    z = np.load(stage_dir / "agent_bigc13_serve_in.npz", allow_pickle=True)
    ids = z["ids"].astype(float)
    i = F.id_index(np.load(INTERIM / "agent_bigc13_serve_in.npz", allow_pickle=True)["ids"].astype(float), ids)
    assert (i >= 0).all(), "staged ids are not January/July rows"
    rows = []
    pairs = [("agent_bigc13_keys2.npz", ["Ks", "Ks_d"]), ("agent_bigc12_keys.npz", ["Ks"]), ("agent_bigc12_vfac_clean5.npz", ["Xs"]), ("agent_pcd_cols_app.npz", ["Ds"])]
    for f, ks in pairs:
        a, b = np.load(stage_dir / f, allow_pickle=True), np.load(INTERIM / f, allow_pickle=True)
        for k in ks:
            x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)[i]
            bad = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            rows.append((f"{f}:{k}", float(np.nanmax(np.abs(x - y))) if x.size else 0.0, int(bad.any(1).sum()) if bad.ndim > 1 else int(bad.sum())))
    t = np.load(stage_dir / "agent_arith3_time_serve.npz")
    t0 = np.load(INTERIM / "agent_arith3_time_serve.npz")
    rows.append(("time", float(np.abs(t["t"] - t0["t"][i]).max()), int((t["t"] != t0["t"][i]).sum())))
    print("STAGE CHECK vs stored January/July (%d rows): array | max abs difference | differing rows" % len(ids))
    for r in rows:
        print("  %-48s %-12.6g %d" % r)
    return all(r[2] == 0 for r in rows)


# ------------------------------------------------------------------------------------------------ arithmetic serve fits
def load_serve(S5, C3, I, n_expect=None):
    """Copy of agent_arith3_05_servecols.load_serve without the full-file row-count assert."""
    names = json.load(open(I + "agent_yearfp_names.json"))["names"]
    NI = {n: i for i, n in enumerate(names)}
    si = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True)
    ids = si["ids"].astype(float)
    colsS = [str(c) for c in si["cols_pre"]]
    Xs = si["Xs_pre"]
    kk = np.load(I + "agent_bigc13_keys2.npz", allow_pickle=True)
    Cc = np.load(I + "agent_pcd_cols_app.npz", allow_pickle=True)
    assert np.array_equal(Cc["ids_s"].astype(float), ids), "pcd serve columns are not in serve_in row order"
    vs = np.load(I + "agent_bigc12_vfac_clean5.npz", allow_pickle=True)["Xs"]
    tz = np.load(I + "agent_arith3_time_serve.npz")
    assert np.array_equal(tz["ids"], ids), "serve time file is not in serve_in row order"
    D = S5.serve_dict(si["pv26"], si["apt"], Xs[:, colsS.index("hour")], Xs[:, colsS.index("RUNWAY_mvt")], Xs[:, colsS.index("ADEP_mvt")],
                      Xs[:, colsS.index("is_weekend")], kk["Ks"], Cc["Ds"], {n: vs[:, NI["vf_d_" + n] - 318] for n in C3.VPOINT + C3.VQ}, tz["t"])
    D["ids"] = ids
    D["mon_s"] = si["mon_s"]
    D["is9"] = si["is9"].astype(bool)
    return D


def arith(stage_dir: Path, seeds: list[int], rounds: int, wire: int):
    import gc
    I = str(stage_dir).replace("\\", "/") + "/"
    out = stage_dir / "out" / "arith"
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(ARITH_I=I, ARITH_OUT=str(out).replace("\\", "/") + "/", ARITH_PCD="agent_pcd_cols_app.npz",
                      ARITH3_HELD=str(INTERIM / "agent_arith3_out").replace("\\", "/") + "/")
    os.chdir(REPO)
    import lightgbm as lgb
    import agent_arith_14_fit_serve as A   # loads the 2025 side from I
    import agent_arith3_02_cols as C3
    import agent_arith3_05_servecols as S5
    S5.C3.I = I
    P = A.P
    NBASE = 370 + 9 + 52
    a = dict(groups=list(ARM.split("-")[0]), red=int(ARM.split("-")[3][1:]))
    assert ARM.split("-")[1:3] == ["pw", "s0"]
    _, names_b, XF, XS = A.serve("ABCDE-pw-s0-r%d" % a["red"], 0, 0)
    ns = len(XS)
    assert XF.shape == (A.N, NBASE) and XS.shape == (ns, NBASE)
    col = 198 + A.KI["st_rwy"] * 7 + 2
    sdf, sds = XF[:, col].copy(), XS[:, col].copy()
    D3 = C3.load_inputs()
    assert D3["KN"] == A.KN and np.array_equal(D3["ids"], A.ids)
    assert np.array_equal(D3["Dpcd"], A.Dpcd, equal_nan=True) and np.array_equal(D3["pv"], A.pv) and np.array_equal(XF[:, 370:379], A.Dpcd, equal_nan=True)
    Ds = load_serve(S5, C3, I)
    assert np.array_equal(XS[:, 370:379], Ds["Dpcd"], equal_nan=True)
    assert np.array_equal(XS[:, A.cols.index("hour")], Ds["hour"], equal_nan=True)
    dayf, days = C3.day_block(D3), C3.day_block(Ds)
    bF, bS, names3, (lo, hi) = S5.build_pair(D3, Ds, sdf, sds, dayf, days, a["groups"])
    assert names3 == sum([C3.GROUPS3[g] for g in a["groups"]], [])
    del D3, Ds, dayf, days
    nx = bF.shape[1]
    X = np.empty((A.N, NBASE + nx), np.float32)
    X[:, :NBASE] = XF
    del XF
    gc.collect()
    X[:, NBASE:] = np.clip(bF, lo, hi)
    del bF
    X2 = np.empty((ns, NBASE + nx), np.float32)
    X2[:, :NBASE] = XS
    X2[:, NBASE:] = np.clip(bS, lo, hi)
    del XS, bS
    gc.collect()
    exn = [str(c) for c in A.ex["cols"]]
    names = list(A.cols) + exn + ["d_NEW", "d_ET"] + list(A.names[194:370]) + [str(c) for c in A.C["names"]] + list(names_b) + list(names3)
    assert len(names) == X.shape[1] == X2.shape[1]
    P("matrices", ARM, X.shape, X2.shape)
    sid = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True)["ids"]
    # share of served values outside the fit-row winsorising bounds, per added column (a support check)
    for s in seeds:
        path = out / f"{ARM}_serve_s{s}{'_wire%d' % wire if wire else ''}.npz"
        if path.exists() and not wire:
            P("skip", path.name)
            continue
        nr = wire or rounds
        b = lgb.train(A.params(s, a["red"]), lgb.Dataset(X, A.r.astype(np.float32), categorical_feature=A.catidx, params=dict(max_bin=63, verbose=-1)), nr)
        pred = b.predict(X2)
        assert np.isfinite(pred).all() and len(pred) == len(sid) == ns
        np.savez(str(path) + ".tmp.npz", pred=pred, ids=sid, names=np.array(names), imp=b.feature_importance("gain"), rounds=nr, arm=ARM, seed=s)
        os.replace(str(path) + ".tmp.npz", path)
        P("serve fit", ARM, "seed", s, "rounds", nr, "pred mean %.3f sd %.3f" % (pred.mean(), pred.std()))


# ------------------------------------------------------------------------------------------------ plan-clock corrector serve fits
def pcd(stage_dir: Path, seeds: list[int], rounds: int, wire: int, prefix: str = "agent_bigc13"):
    """Copy of the V branch of modal/agent_pcd_app.py serve_one for cfg C4ff7V5 (fold from the stage directory). With prefix
    agent_bigc12 it is the earlier 12-month corrector of modal/agent_bigc12_app.py (no plan-clock columns), whose serve preds
    are an input of the tail component."""
    with_pcd = prefix == "agent_bigc13"
    import lightgbm as lgb
    import agent_bigc_te as TE
    I = stage_dir
    out = stage_dir / "out" / ("pcd" if with_pcd else "v5")
    out.mkdir(parents=True, exist_ok=True)
    ld = lambda n: np.load(I / n, allow_pickle=True)  # noqa: E731
    z, ex, e2, si = ld(f"{prefix}_fold.npz"), ld("agent_bigc12_extra.npz"), ld("agent_bigc12_extra2.npz"), ld(f"{prefix}_serve_in.npz")
    X = np.hstack([z["X"], ex["Xf"], e2["Xf"]])
    cols = list(z["cols"]) + list(ex["cols"]) + ["d_NEW", "d_ET"]
    Xs = si["Xs_pre"]
    assert list(map(str, si["cols_pre"])) == [str(c) for c in cols], "serve/fold column mismatch"
    y, pv, mon = z["y"], z["pv"], z["mon"]
    catidx = [int(i) for i in z["cat"]]
    months = sorted(set(mon.tolist()))
    r = np.clip(y - pv, -900, 900)
    kf, ks = TE.keys_from(X, cols), TE.keys_from(Xs, cols)
    X = np.hstack([X, TE.te_loo(kf, r, mon, months)])
    Xs = np.hstack([Xs, TE.te_serve(kf, r, ks)])
    kk, k0 = ld(f"{prefix}_keys2.npz"), ld("agent_bigc12_keys.npz")
    yy = y.astype(float)
    ymu_apt = {a: yy[k0["apt_f"] == a].mean() for a in set(k0["apt_f"].tolist())}
    T2 = TE.Te2(kk["Kf"])
    F2f = T2.fold(r, yy, np.array([ymu_apt[a] for a in k0["apt_f"]]), mon, None, months)
    F2s = T2.serve(r, yy, mon, months, kk["Ks"], np.array([ymu_apt.get(a, yy.mean()) for a in k0["apt_s"]]))
    X = np.hstack([X, F2f, kk["Df"]])
    Xs = np.hstack([Xs, F2s, si["Dsv"]])
    del F2f, F2s
    vf = ld("agent_bigc12_vfac_clean5.npz")
    X = np.hstack([X, vf["Xf"]])
    Xs = np.hstack([Xs, vf["Xs"]])
    if with_pcd:
        pc = ld("agent_pcd_cols_app.npz")
        assert np.array_equal(pc["ids"].astype(float), z["ids"].astype(float)) and np.array_equal(pc["ids_s"].astype(float), si["ids"].astype(float))
        X = np.hstack([X, pc["D"]])
        Xs = np.hstack([Xs, pc["Ds"]])
    assert X.shape[1] == Xs.shape[1], (X.shape, Xs.shape)
    # the 'C4ff7' parameters of modal/agent_pcd_app.py CFG
    prm = dict(objective="l2", learning_rate=0.03, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.7, max_bin=63,
               num_threads=int(os.environ.get("BIGC_THREADS", 8)), verbose=-1, num_leaves=511, min_data_in_leaf=150, lambda_l2=100.0)
    for s in seeds:
        path = out / f"{PCD_CFG}_s{s}{'_wire%d' % wire if wire else ''}.npz"
        if path.exists() and not wire:
            P("skip", path.name)
            continue
        nr = wire or rounds
        b = lgb.train(dict(prm, seed=s), lgb.Dataset(X, r.astype(np.float32), categorical_feature=catidx, params=dict(max_bin=63, verbose=-1)), nr)
        pred = b.predict(Xs)
        np.savez(str(path) + ".tmp.npz", pred=pred, ids=si["ids"], imp=b.feature_importance("gain"), rounds=nr)
        os.replace(str(path) + ".tmp.npz", path)
        P("pcd serve fit seed", s, "rounds", nr, "features", X.shape[1], "pred mean %.3f sd %.3f" % (pred.mean(), pred.std()))


def summary(stage_dir: Path):
    sin = np.load(stage_dir / "agent_bigc13_serve_in.npz", allow_pickle=True)
    apt, mo, is9 = sin["apt"].astype(str), sin["mon_s"], sin["is9"]
    for kind, pat in (("arith", f"out/arith/{ARM}_serve_s*.npz"), ("pcd", f"out/pcd/{PCD_CFG}_s*.npz")):
        fs = [f for f in sorted(glob.glob(str(stage_dir / pat))) if "_wire" not in f]
        if not fs:
            continue
        pr = np.mean([np.load(f, allow_pickle=True)["pred"] for f in fs], 0)
        sd_between = np.std([np.load(f, allow_pickle=True)["pred"] for f in fs], 0).mean() if len(fs) > 1 else float("nan")
        print(f"{kind}: {len(fs)} seeds, mean over seeds; seed sd (row mean) {sd_between:.2f}")
        print("  airport month rows mean sd p1 p99")
        for a in sorted(set(apt[is9].tolist())):
            for m in sorted(set(mo.tolist())):
                s = (apt == a) & (mo == m)
                if s.any():
                    q = pr[s]
                    print("  %s %d %d %.2f %.2f %.1f %.1f" % (a, m, s.sum(), q.mean(), q.std(), np.percentile(q, 1), np.percentile(q, 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stage", "check", "arith", "pcd", "v5", "summary"])
    ap.add_argument("--prefix", default="agent_bigc13", choices=["agent_bigc13", "agent_bigc12"])
    ap.add_argument("--serve-dir", default=str(REPO / "data/final/J9_serve"))
    ap.add_argument("--outdir", default=str(REPO / "data/final"))
    ap.add_argument("--tag", default="J10")
    ap.add_argument("--stage", default=None)
    ap.add_argument("--seeds", default="0-11")
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--wire", type=int, default=0)
    a = ap.parse_args()
    st = Path(a.stage) if a.stage else Path(a.outdir) / f"{a.tag}_stage"
    if a.cmd == "stage":
        stage(Path(a.serve_dir), Path(a.outdir), a.tag, a.prefix)
    elif a.cmd == "check":
        sys.exit(0 if check(st) else 1)
    elif a.cmd == "arith":
        arith(st, seeds_of(a.seeds), a.rounds or ARM_ROUNDS, a.wire)
    elif a.cmd == "pcd":
        pcd(st, seeds_of(a.seeds), a.rounds or PCD_ROUNDS, a.wire)
    elif a.cmd == "v5":
        pcd(st, seeds_of(a.seeds), a.rounds or V5_ROUNDS, a.wire, "agent_bigc12")
    else:
        summary(st)


if __name__ == "__main__":
    main()
