"""experiment ltreg judge (docs/LAB-NOTES.md/s1b rules). Reads only fold prediction files; no fitting.

    python archive/agent_ltreg_judge.py drift                       # lr_ctrl_f1_s0 vs preds_ltfmseq_treat_f1_1_7
    python archive/agent_ltreg_judge.py judge <arm> [--partial]     # arm = snow | regime | combo

Control r=0 at folds 1,2,3,4,6 = results/preds_ltfmseq_treat_f{k}_* iff the drift check passed (logs/agent_ltreg_drift.json),
else results/preds_lr_ctrl_f{k}_s0_*. Treatment = results/preds_lr_<arm>_f{k}_s{r}_*.
W = treatment on LTFM rows, control elsewhere; d = RMSE(W) - RMSE(ctrl) over ALL rows of the arm's population months.
"""
import glob
import json
import sys

import numpy as np
import polars as pl

POP = {"snow": list(range(1, 13)), "regime": [7, 8, 9, 10, 11], "combo": None}
HALVES = {"snow": ([1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]), "regime": ([7, 8, 9], [10, 11])}
FOLDS = {"snow": [1, 2, 3, 4, 5, 6], "regime": [1, 2, 3, 4, 5]}


def one(pattern):
    h = sorted(glob.glob(pattern))
    return h[0] if len(h) == 1 else None


def months():
    return pl.scan_parquet("data/raw/training_2025-*.parquet").filter(pl.col("PHASE_mvt") == "DEP").select(
        "MVT_ID_mvt", pl.col("MVT_TIME_UTC_mvt").dt.month().alias("mo"), pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day")).collect()


def drift():
    a = pl.read_parquet(one("results/preds_lr_ctrl_f1_s0_1_7.parquet"))
    b = pl.read_parquet("results/preds_ltfmseq_treat_f1_1_7.parquet")
    j = a.select("MVT_ID_mvt", pl.col("bagged").alias("a")).join(b.select("MVT_ID_mvt", pl.col("bagged").alias("b")), on="MVT_ID_mvt")
    assert j.height == a.height == b.height, (j.height, a.height, b.height)
    d = (j["a"] - j["b"]).abs()
    lt = j.join(a.select("MVT_ID_mvt", "ADEP_mvt"), on="MVT_ID_mvt").filter(pl.col("ADEP_mvt") == "LTFM")
    res = {"rows": j.height, "max_abs": float(d.max()), "mean_abs": float(d.mean()),
           "ltfm_max_abs": float((lt["a"] - lt["b"]).abs().max()), "pass": bool(d.max() < 1e-6)}
    print("DRIFT", res)
    json.dump(res, open("logs/agent_ltreg_drift.json", "w"), indent=1)


def ctrl_path(k, r):
    ok = json.load(open("logs/agent_ltreg_drift.json"))["pass"] if glob.glob("logs/agent_ltreg_drift.json") else False
    if r == 0 and ok and k != 5:
        return one(f"results/preds_ltfmseq_treat_f{k}_{k}_*.parquet")
    return one(f"results/preds_lr_ctrl_f{k}_s{r}_{k}_*.parquet")


def rmse(se):
    return float(np.sqrt(se.mean()))


def judge(arm, partial):
    base = arm if arm in POP and POP[arm] is not None else "snow"
    pop, (h1, h2), folds = POP[base], HALVES[base], FOLDS[base]
    mo = months()
    verdicts, deltas = [], []
    for r in (0, 1, 2):
        parts, missing = [], []
        for k in folds:
            c, t = ctrl_path(k, r), one(f"results/preds_lr_{arm}_f{k}_s{r}_{k}_*.parquet")
            if c is None or t is None:
                missing.append((k, c is None, t is None)); continue
            C = pl.read_parquet(c).select("MVT_ID_mvt", "ADEP_mvt", "y", pl.col("bagged").alias("c"))
            T = pl.read_parquet(t).select("MVT_ID_mvt", pl.col("bagged").alias("t"), pl.col("y").alias("y2"))
            J = C.join(T, on="MVT_ID_mvt", how="inner")
            assert J.height == C.height == T.height and (J["y"] == J["y2"]).all(), (k, r)
            parts.append(J.drop("y2"))
        if missing:
            print(f"seed {r}: missing (fold, ctrl?, treat?) {missing}")
            if not partial or not parts:
                continue
        D = pl.concat(parts).join(mo, on="MVT_ID_mvt", how="left")
        assert D["mo"].null_count() == 0
        D = D.filter(pl.col("mo").is_in(pop)).with_columns(
            w=pl.when(pl.col("ADEP_mvt") == "LTFM").then(pl.col("t")).otherwise(pl.col("c")))
        y, c, w = D["y"].to_numpy(), D["c"].to_numpy(), D["w"].to_numpy()
        sc, sw = (c - y) ** 2, (w - y) ** 2
        d_all = rmse(sw) - rmse(sc)
        m = D["mo"].to_numpy()
        d_h = [rmse(sw[np.isin(m, h)]) - rmse(sc[np.isin(m, h)]) for h in (h1, h2)]
        gain = sc - sw
        keep = np.ones(len(y), bool); keep[np.argsort(-gain)[:50]] = False
        d_m50 = rmse(sw[keep]) - rmse(sc[keep])
        ok = d_all <= -0.10 and d_h[0] < 0 and d_h[1] < 0 and d_m50 < 0
        lt = D["ADEP_mvt"].to_numpy() == "LTFM"
        print(f"\n== {arm} seed {r}: {len(y):,} rows ({lt.sum():,} LTFM) folds {sorted(set(folds) - {x[0] for x in missing})}")
        print(f"   all-rows d {d_all:+.4f} | halves {h1[0]}-{h1[-1]} {d_h[0]:+.4f}  {h2[0]}-{h2[-1]} {d_h[1]:+.4f} | minus-top-50 {d_m50:+.4f}"
              f"  -> {'PASS' if ok else 'FAIL'}{' (PARTIAL)' if missing else ''}")
        print(f"   LTFM RMSE {rmse(sc[lt]):.3f} -> {rmse(sw[lt]):.3f} ({rmse(sw[lt]) - rmse(sc[lt]):+.3f}); fold dp^2/row (all rows) "
              f"{float(((w - c) ** 2).mean()):.2f}")
        for mm in sorted(set(m[lt].tolist())):
            s = lt & (m == mm)
            print(f"   m{mm:>2} LTFM n {s.sum():>6,} rmse {rmse(sc[s]):7.2f} -> {rmse(sw[s]):7.2f} ({rmse(sw[s]) - rmse(sc[s]):+6.2f})  "
                  f"bias c {np.mean(c[s] - y[s]):+7.2f} w {np.mean(w[s] - y[s]):+7.2f}   all-rows d(month) "
                  f"{rmse(sw[m == mm]) - rmse(sc[m == mm]):+.4f}")
        verdicts.append({"seed": r, "d_all": d_all, "d_h1": d_h[0], "d_h2": d_h[1], "d_m50": d_m50, "pass": ok,
                         "partial": bool(missing), "fold_dp2_row": float(((w - c) ** 2).mean()),
                         "dsse_per_ltfm_row": float((sw[lt] - sc[lt]).mean())})
        deltas.append(D.select("MVT_ID_mvt", "ADEP_mvt", "mo", "y", "c", "w").with_columns(seed=pl.lit(r)))
    if deltas:
        pl.concat(deltas).write_parquet(f"data/interim/agent_ltreg_{arm}_fold.parquet")
    allpass = len(verdicts) == 3 and all(v["pass"] and not v["partial"] for v in verdicts)
    if verdicts:
        dl = np.mean([v["dsse_per_ltfm_row"] for v in verdicts])
        print(f"\nfirst-order board (mean over seeds): dSSE/LTFM row {dl:+.1f} -> {dl * 47581 / 344841 / (2 * 224.88):+.3f} s at 1x")
    print(f"\nLTREG_{arm.upper()}_VERDICT: {'PASS' if allpass else 'FAIL' if len(verdicts) == 3 and not any(v['partial'] for v in verdicts) else 'INCOMPLETE'}")
    json.dump({"arm": arm, "verdicts": verdicts, "pass": allpass}, open(f"logs/agent_ltreg_{arm}_verdict.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "drift":
        drift()
    else:
        judge(sys.argv[2], "--partial" in sys.argv)
