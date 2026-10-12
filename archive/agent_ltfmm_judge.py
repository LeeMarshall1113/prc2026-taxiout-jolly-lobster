"""Judge for lever L1 (docs/LAB-NOTES.md): LTFM members fitted WITH the take-off-sequence columns.
Local, numpy only (< 1 GB, seconds).  Rule PRE-REGISTERED in the scope note s3 BEFORE any arm ran.

Inputs
  results/ltfmm/ltfmm_{ctrl,treat}_f{1,2,3}_{a}_{b}_{c,p3,r,e}.npz   (ids, pred)  LTFM-only member fits, seq off / on
  results/preds_bulkr_f{1,2,3}_{a}_{b}.parquet                         v94 fold predictions (`bagged`) + members m_c m_p3 m_r m_e
  results/preds_ltfmseq_{ctrl,treat}_f{1,2,3}_{a}_{b}.parquet          the v113 single-CatBoost pair (global, seq off/on) = the SHIPPED design
Arms (all: LTFM rows only differ; every other row = v94 `bagged`)
  BASE = v94 bagged.
  SERVED = v94 bagged + (treat_cat - ctrl_cat)            the v113 REPLACE arithmetic as shipped (delta of ONE global CatBoost, weight 1)
  NEW    = v94 bagged + mean_m(treat_m - ctrl_m)          member_delta arithmetic; every member fitted with the seq columns
  NEW_c  = v94 bagged + (treat_c + treat_p3 - ctrl_c - ctrl_p3) / 4   (CatBoost members only: which member carries the gain)
  NEW_re = v94 bagged + (treat_r + treat_e - ctrl_r - ctrl_e) / 4     (LightGBM members only)
GATING = NEW against SERVED (the increment over what is already on the board), the four clauses of the 238-day rule, on the rows of
folds 1-3 pooled: 1. all rows d <= -0.10 s | 2. no airport worse than +0.10 s | 3. d < 0 after deleting the 50 rows NEW helps most |
4. d < 0 on Jan-Mar rows AND on Jul-Sep rows.  Identical to archive/agent_ltfm_judge.py's clause code.
ALSO GATING: NEW against BASE must pass clause 1 and 4 too (an increment over a bad SERVED is not a win).  REPORTED: NEW vs BASE,
SERVED vs BASE (the fold view of v113), NEW_c, NEW_re, LTFM-only RMSE per fold and month, sd of the delta.
IDENTITY (must hold or the verdict is void): ctrl_c vs preds_bulkr m_c on the LTFM rows, max |diff| <= 1.0 s (same routed fit,
same rows); reported per member.
"""
import glob
import sys

import numpy as np
import polars as pl

TAG = {1: "1_7", 2: "2_8", 3: "3_9"}
MEM = ["c", "p3", "r", "e"]
ID = "MVT_ID_mvt"


def ck(arm, k, m):
    z = np.load(f"results/ltfmm/ltfmm_{arm}_f{k}_{TAG[k]}_{m}.npz")
    return z["ids"].astype(float), z["pred"].astype(float)


frames, ident = [], {}
for k in (1, 2, 3):
    b = pl.read_parquet(f"results/preds_bulkr_f{k}_{TAG[k]}.parquet").select(ID, "ADEP_mvt", "y", "bagged", "m_c", "m_p3", "m_r", "m_e")
    sc = pl.read_parquet(f"results/preds_ltfmseq_ctrl_f{k}_{TAG[k]}.parquet").select(ID, pl.col("bagged").alias("s_ctrl"))
    st = pl.read_parquet(f"results/preds_ltfmseq_treat_f{k}_{TAG[k]}.parquet").select(ID, pl.col("bagged").alias("s_treat"))
    d = b.join(sc, on=ID).join(st, on=ID).with_columns(pl.lit(k).alias("fold"))
    lt = d.filter(pl.col("ADEP_mvt") == "LTFM")
    add = {}
    for m in MEM:
        ids_c, p_c = ck("ctrl", k, m)
        ids_t, p_t = ck("treat", k, m)
        assert (ids_c == ids_t).all(), f"fold {k} {m}: ctrl/treat rows differ"
        dm = pl.DataFrame({ID: ids_c, f"d_{m}": p_t - p_c, f"ctrl_{m}": p_c})
        lt = lt.join(dm, on=ID, how="left")
        ident.setdefault(m, []).append(float(np.abs(lt[f"m_{m}"].to_numpy() - lt[f"ctrl_{m}"].to_numpy()).max()) if m != "e" else float("nan"))
    frames.append(d.join(lt.select(ID, *[f"d_{m}" for m in MEM]), on=ID, how="left").with_columns([pl.col(f"d_{m}").fill_null(0.0) for m in MEM]))
D = pl.concat(frames).sort(ID)
raw = pl.concat([pl.scan_parquet(f).select(ID, "MVT_TIME_UTC_mvt").collect() for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
mon = D.join(raw, on=ID, how="left", maintain_order="left")["MVT_TIME_UTC_mvt"].dt.month().to_numpy()
y = D["y"].to_numpy().astype(float)
apt = D["ADEP_mvt"].to_numpy().astype(str)
base = np.clip(D["bagged"].to_numpy().astype(float), 0, None)
lt = apt == "LTFM"
dser = np.where(lt, D["s_treat"].to_numpy() - D["s_ctrl"].to_numpy(), 0.0)
dmem = {m: D[f"d_{m}"].to_numpy() for m in MEM}
arms = {"BASE": base,
        "SERVED": np.clip(base + dser, 0, None),
        "NEW": np.clip(base + sum(dmem[m] for m in MEM) / 4, 0, None),
        "NEW_c": np.clip(base + (dmem["c"] + dmem["p3"]) / 4, 0, None),
        "NEW_re": np.clip(base + (dmem["r"] + dmem["e"]) / 4, 0, None)}
print("identity ctrl_m vs v94 m_m on LTFM rows, max|diff| per fold (e is LTFM-only here, not comparable):", {m: np.round(v, 3).tolist() for m, v in ident.items()})
ident_ok = all(max(v) <= 1.0 for m, v in ident.items() if m != "e")
print("IDENTITY", "PASS" if ident_ok else "FAIL -> verdict void")

rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean()))  # noqa: E731
allm = np.ones(len(y), bool)
jm, js = np.isin(mon, [1, 2, 3]), np.isin(mon, [7, 8, 9])


def judge(pt, pc, label, gate=True):
    d_of = lambda m: rm(pt, m) - rm(pc, m)  # noqa: E731
    dsq = (pt - y) ** 2 - (pc - y) ** 2
    d50 = float(np.sqrt(((pc - y) ** 2).mean() + (dsq.sum() - np.sort(dsq)[:50].sum()) / len(y)) - rm(pc, allm))
    byapt = {a: d_of(apt == a) for a in sorted(set(apt))}
    checks = {"1. all rows <= -0.10 s": (d_of(allm), d_of(allm) <= -0.10),
              "2. no airport worse than +0.10 s": (max(byapt.values()), max(byapt.values()) <= 0.10),
              "3. negative minus top-50": (d50, d50 < 0),
              "4a. Jan-Mar rows negative": (d_of(jm), d_of(jm) < 0),
              "4b. Jul-Sep rows negative": (d_of(js), d_of(js) < 0)}
    print(f"\n== {label}: {len(y):,} rows; RMSE {rm(pc, allm):.3f} -> {rm(pt, allm):.3f}")
    for k, (v, ok) in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k:<34} {v:+.3f}")
    print("  LTFM only: %.2f -> %.2f (%+.2f s); by fold %s; by month %s" % (
        rm(pc, lt), rm(pt, lt), rm(pt, lt) - rm(pc, lt),
        {k: round(rm(pt, lt & (D['fold'].to_numpy() == k)) - rm(pc, lt & (D['fold'].to_numpy() == k)), 2) for k in (1, 2, 3)},
        {int(m): round(rm(pt, lt & (mon == m)) - rm(pc, lt & (mon == m)), 2) for m in sorted(set(mon[lt]))}))
    ok = all(v for _, v in checks.values())
    print(f"  VERDICT ({label}): {'PASS' if ok else 'FAIL'}")
    return ok, checks


print("\nLTFM delta scale: sd(NEW delta) %.1f s, sd(SERVED delta) %.1f s, corr %.3f" % (
    np.std((arms['NEW'] - base)[lt]), np.std((arms['SERVED'] - base)[lt]), np.corrcoef((arms['NEW'] - base)[lt], (arms['SERVED'] - base)[lt])[0, 1]))
judge(arms["SERVED"], arms["BASE"], "REPORTED: SERVED (v113 as shipped) vs BASE", gate=False)
judge(arms["NEW_c"], arms["SERVED"], "REPORTED: NEW_c (CatBoost members only) vs SERVED", gate=False)
judge(arms["NEW_re"], arms["SERVED"], "REPORTED: NEW_re (LightGBM members only) vs SERVED", gate=False)
ok_b, ck_b = judge(arms["NEW"], arms["BASE"], "NEW vs BASE (clauses 1 and 4 also gate)")
ok_s, ck_s = judge(arms["NEW"], arms["SERVED"], "PRIMARY: NEW vs SERVED")
final = ident_ok and ok_s and ck_b["1. all rows <= -0.10 s"][1] and ck_b["4a. Jan-Mar rows negative"][1] and ck_b["4b. Jul-Sep rows negative"][1]
print(f"\nFINAL VERDICT L1: {'PASS' if final else 'FAIL'} (identity {ident_ok}, primary {ok_s})")
sys.exit(0 if final else 2)
