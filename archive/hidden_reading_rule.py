"""Where a schedule copy is impossible, predict the hidden plan's own off-block reading.

    .venv/Scripts/python.exe archive/hidden_reading_rule.py

v36's scope (Rule B): a no-plan departure whose plan is recovered from the destination's arrival row,
with gl = MVT - LOBT >= 3,606s and |gs - gl| < 3,606s. There BLOCK = SCHED would have joined the
plan, so the row is NOT a schedule copy -- yet at Rome the model still hedges toward the copy: on the
36 gated-fold rows in scope the truth runs 358-1,671s while the model predicts up to 7,107s. v36 only
caps them at gl - 3,606s, which is loose. The same recovered plan carries AOBT_3, and on 2025 Rule-B
rows the reading a = MVT - AOBT_3 sits within 600s of y on 94 per cent of LIRF rows (RMSE 285s).

RULE: in scope, prediction := clip(a, 0, gl - 3,606s)   (R, replace)
                 or := min(0.5 * (chain + a), gl - 3,606s)   (H, half blend)
The chain it is applied to is production's: gap clamp, the Rome airline rule, the hidden-plan lower
bound and the v36 upper bound, all as shipped. It goes LAST, and it can only act inside Rule B scope.

SCOPES: S1 LIRF only; S2 all ten airports. Both reported; nothing else is tried.

PASS RULE, FIXED BEFORE THIS RAN, per (scope, variant):
  1. no fold positive (Rome judged only on the gate-inclusive folds (1,7) and (2,8));
  2. pooled all-rows change <= -0.10s;
  3. pooled still negative after deleting the 5 rows it helps most;
  4. (S2 only) no airport's pooled change worse than +0.02s.
Adopt the passing (scope, variant) with the most negative pooled change. Then 2026: the rows it
changes on top of v36.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, TARGET, W = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 3606.0
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
BOUND_APTS = ["LIRF", "LTFM"]
FL = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase()
OP = FL.str.extract(r"^([A-Z]{2,3})", 1)
TESTS = [("1_7", [1, 7], "results/preds_q_gate_1_7.parquet", None),
         ("2_8", [2, 8], "results/preds_q_gate_2_8.parquet", None),
         ("3_9", [3, 9], "results/preds_n3_bulk_3_9.parquet", "LIRF"),
         ("4_10", [4, 10], "results/preds_s4_ctrl_4_10.parquet", "LIRF"),
         ("5_11", [5, 11], "results/preds_c_ctrl_5_11.parquet", "LIRF")]


def recover(src):
    dep = (pl.scan_parquet(src).filter((C("PHASE_mvt") == "DEP") & C("AOBT_3_flt").is_null()
                                       & C("ADES_mvt").is_in(APTS) & C("FLIGHT_mvt").is_not_null())
           .select(C(ID), C("ADEP_mvt").alias("adep"), C("ADES_mvt").alias("ades"), FL.alias("flight"),
                   C("MVT_TIME_UTC_mvt").alias("mvt"), C("MVT_TIME_UTC_mvt").dt.month().alias("mo"),
                   (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"))
           .collect().sort("mvt"))
    arr = (pl.scan_parquet(src).filter((C("PHASE_mvt") == "ARR") & C("LOBT_flt").is_not_null())
           .select(C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("arr_t"),
                   C("LOBT_flt").alias("lobt"), C("ADEP_flt").alias("plan_adep"), C("AOBT_3_flt").alias("aobt"))
           .collect().sort("arr_t"))
    return (dep.join_asof(arr, left_on="mvt", right_on="arr_t", by=["ades", "flight"], strategy="forward", tolerance="8h")
               .filter(C("lobt").is_not_null() & (C("plan_adep") == C("adep")))
               .with_columns((C("mvt") - C("lobt")).dt.total_seconds().alias("gl"),
                             (C("mvt") - C("aobt")).dt.total_seconds().alias("rd")))


TRAIN = "data/raw/training_*.parquet"
r25 = recover(TRAIN)
tr = (pl.scan_parquet(TRAIN).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_null())
      .select(OP.alias("op"), ((C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60).alias("copy"))
      .collect())
habitual = set(tr.group_by("op").agg(pl.len().alias("n"), C("copy").mean().alias("r"))
                 .filter((C("n") >= 8) & (C("r") >= 0.97))["op"].to_list())
info = (pl.scan_parquet(TRAIN).filter(C("PHASE_mvt") == "DEP")
        .select(C(ID), C("ADEP_mvt").alias("adep"), OP.alias("op"), C("AOBT_3_flt").is_null().alias("noplan"),
                (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs")).collect())


def chain(P):
    """Production's post-processing on fold predictions, in shipping order."""
    b = np.clip(P["bagged"].to_numpy().astype(float), 0, None)
    g = P["gs"].fill_null(np.nan).to_numpy().astype(float)
    b = np.where((g >= 5000) & (g < 50000) & (b > g), g, b)                             # gap clamp
    air = ((P["adep"] == "LIRF").fill_null(False).to_numpy() & P["noplan"].fill_null(False).to_numpy()
           & P["op"].is_in(list(habitual)).fill_null(False).to_numpy() & (g < 50000))
    b = np.where(air, np.maximum(g, 0), b)                                               # v35 rule 1
    ids = P[ID].to_list()
    rec = r25.filter(C("adep").is_in(BOUND_APTS))
    lo = dict(zip(*[rec.filter(C("gl") < W)[c].to_list() for c in (ID, "gl")]))
    hi = dict(zip(*[rec.filter((C("gl") >= W) & ((C("gs") - C("gl")).abs() < W))[c].to_list() for c in (ID, "gl")]))
    b = np.array([max(v, lo[m] + W) if m in lo else v for m, v in zip(ids, b)])          # v35 rule 2
    b = np.array([min(v, hi[m] - W) if m in hi else v for m, v in zip(ids, b)])          # v36
    return np.clip(b, 0, None)


scope25 = r25.filter((C("gl") >= W) & ((C("gs") - C("gl")).abs() < W) & C("rd").is_not_null())
folds = []
for label, mo, path, excl in TESTS:
    P = pl.read_parquet(path).select(ID, "y", "bagged").join(info, on=ID, how="left").sort(ID)
    base = chain(P)
    sc = scope25.filter(C("mo").is_in(mo))
    if excl:
        sc = sc.filter(C("adep") != excl)
    folds.append((label, P, base, sc))
    print(f"fold {label}: chain RMSE {np.sqrt(((P['y'].to_numpy() - base) ** 2).mean()):.3f}; Rule-B rows in scope {sc.height}")

verdicts = {}
for scope_name, apts in (("S1 LIRF", ["LIRF"]), ("S2 all", APTS)):
    for var in ("R", "H"):
        per, ys, bs, ns, aps = [], [], [], [], []
        for label, P, base, sc in folds:
            s = sc.filter(C("adep").is_in(apts))
            rd = dict(zip(s[ID].to_list(), s["rd"].to_list()))
            cp = dict(zip(s[ID].to_list(), (s["gl"] - W).to_list()))
            new = base.copy()
            for i, m in enumerate(P[ID].to_list()):
                if m in rd:
                    v = rd[m] if var == "R" else 0.5 * (base[i] + rd[m])
                    new[i] = float(np.clip(v, 0, cp[m]))
            y = P["y"].to_numpy().astype(float)
            d = float(np.sqrt(((y - new) ** 2).mean()) - np.sqrt(((y - base) ** 2).mean()))
            ch = new != base
            per.append((label, d, int(ch.sum())))
            ys.append(y), bs.append(base), ns.append(new), aps.append(P["adep"].to_numpy())
        y, b, n, ap = (np.concatenate(a) for a in (ys, bs, ns, aps))
        pooled = float(np.sqrt(((y - n) ** 2).mean()) - np.sqrt(((y - b) ** 2).mean()))
        gain = (y - b) ** 2 - (y - n) ** 2
        keep = np.ones(len(y), bool)
        keep[np.argsort(gain)[::-1][:5]] = False
        minus5 = float(np.sqrt(((y[keep] - n[keep]) ** 2).mean()) - np.sqrt(((y[keep] - b[keep]) ** 2).mean()))
        N = len(y)
        by_apt = {a: float(np.sqrt(((y - b) ** 2).mean() + (gain[ap == a] * -1).sum() / N) - np.sqrt(((y - b) ** 2).mean()))
                  for a in apts}
        checks = [("1 no fold positive", all(d <= 0 for _, d, _ in per), " ".join(f"{l} {d:+.3f}({k})" for l, d, k in per)),
                  ("2 pooled <= -0.10s", pooled <= -0.10, f"{pooled:+.3f}"),
                  ("3 negative minus top-5", minus5 < 0, f"{minus5:+.3f}")]
        if scope_name.startswith("S2"):
            worst = max(by_apt.items(), key=lambda kv: kv[1])
            checks.append(("4 no airport > +0.02s", worst[1] <= 0.02, f"worst {worst[0]} {worst[1]:+.3f}"))
        ok = all(c[1] for c in checks)
        verdicts[(scope_name, var)] = (ok, pooled)
        print(f"\n{scope_name} / {var}: rows changed {int((n != b).sum())}")
        for nm, good, val in checks:
            print(f"  {'PASS' if good else 'FAIL'}  {nm:<26} {val}")
        print("  by airport (pooled share): " + "  ".join(f"{a} {v:+.3f}" for a, v in by_apt.items()))
        print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")

passing = {k: v[1] for k, v in verdicts.items() if v[0]}
print()
if not passing:
    print("no (scope, variant) passes")
    sys.exit(0)
best = min(passing, key=passing.get)
print(f"ADOPT {best[0]} / {best[1]} ({passing[best]:+.3f}s pooled)")

r26 = recover("data/raw/ranking.parquet")
s26 = r26.filter((C("gl") >= W) & ((C("gs") - C("gl")).abs() < W) & C("rd").is_not_null())
s26 = s26.filter(C("adep").is_in(["LIRF"] if best[0].startswith("S1") else APTS))
v36 = pl.read_parquet("submissions/jolly-lobster_v36.parquet").select(ID, C(TARGET).alias("v36"))
k = s26.join(v36, on=ID, how="inner")
v = k["v36"].to_numpy().astype(float)
rd, cp = k["rd"].to_numpy().astype(float), (k["gl"] - W).to_numpy()
nv = np.clip(rd if best[1] == "R" else 0.5 * (v + rd), 0, cp)
mv = nv - v
print(f"2026 on top of v36: {k.height} rows in scope; moves median {np.median(mv):+,.0f}s, mean {mv.mean():+,.0f}s, "
      f"{int((np.abs(mv) > 0.5).sum())} rows change; by airport {dict(k.group_by('adep').len().sort('adep').iter_rows())}")
print(f"  if the truth sat on the reading, squared error removed per scored row {((v - rd) ** 2 - (nv - rd) ** 2).sum() / 344841:.0f} s^2")
