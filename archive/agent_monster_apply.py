"""Monster-row rules, applied AFTER the whole shipped chain on a built submission (experiment 'monster', 2026-09-23).

    ./.venv/Scripts/python.exe archive/agent_monster_apply.py --in submissions/jolly-lobster_v99.parquet --out X.parquet
        [--rules r1,r2,r3] [--r1-mixture P]

Every rule was judged on 2025 LABELS only (docs/LAB-NOTES.md); none was chosen by uploading.

R1  DAY-GAP BOUND. For a departure with gs = MVT - SCHED in [88,000, 129,600)s, y <= gs: a copy gives y = gs, a
    one-day-early stamp gives 86,400 + taxi < gs, an ordinary taxi is far below. 2025: 125 departures with gs >= 88,000s
    at every airport, ZERO with y > gs + 60. Prediction := min(prediction, gs). A bound: it can only reduce error.
    (--r1-mixture P additionally moves those rows to P x (86,400 + day offset) + (1 - P) x gs. NOT validated: 2025 has
    4 day-early / 1 copy at gs >= 70,000s at Rome and the only row above 100,000s was a copy. Off by default.)
R2  ROME 'LOW' HIDDEN PLAN -> COPY. LIRF no-plan departure whose plan is recovered from the destination's ARR row
    (same flight number, landing 0-8h later, plan ADEP = LIRF; apply_v35_rules.py's recipe) with h_gl = MVT - LOBT <
    3,606s (v35 rule 2's scope: y > h_gl + 3,606s) AND gs > h_gl + 3,606s (so a schedule copy is possible).
    Prediction := gs. 2025: 23 rows, 21 copies (91.3%), 1 day-early (y ~87k, above gs, so moving to gs still helps),
    1 long taxi (BAW557H, y 4,868 vs bound 3,610 vs gs 6,604: costs 1.4e6 s^2). Folds (1,7) + (4,10): -0.378s pooled on
    11 rows, -0.111s without the best row. Break-even for a hard override at LIRF gs 5-10k: 85.6%.
R3  FIRST-PUSH READING on the LOBT box's HARD-BINDING rows (plan rows with gl - 3,606 - ga > 1,800s: the recorded
    off-block is not the push NM saw -- return-to-stand / first push). Where the prediction still sits on the box's
    hard-binding target (+/-2s: nothing upstream placed it), and the aircraft's ADS-B shows a departure session (first
    ground fix after the arrival's parking: a silence or stop >= 1,200s after touchdown) whose start lies inside the
    box, prediction := MVT - that first fix. Airports: production's ADS-B skip list (LIRF, LFPG) excluded.
    2025: 24 rows, within 600s of y on 70.8%, SSE -2.6e7 against the box target, every airport better.
The script prints every row it changes with its rule and refuses to write if anything else moved.
"""
import argparse
import glob
import sys
from datetime import timedelta

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
W_EXACT, W_BOX, HARD_S = 3606.0, 3620.0, 1800.0
DAY = 86400.0
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
R3_SKIP = ["LIRF", "LFPG"]
FL = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase()
S = lambda a, b: (C(a) - C(b)).dt.total_seconds().cast(pl.Float64)  # noqa: E731

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rules", default="r1,r2,r3")
ap.add_argument("--r1-mixture", type=float, default=None)
args = ap.parse_args()
rules = set(args.rules.split(","))
R = "data/raw/ranking.parquet"

sub = pl.read_parquet(args.inp)
if set(sub.columns) != {ID, TARGET}:
    sys.exit(f"unexpected submission columns: {sub.columns}")
dep = (pl.scan_parquet(R).filter(C("PHASE_mvt") == "DEP")
       .select(C(ID), C("ADEP_mvt").alias("apt"), C("ADES_mvt").alias("ades"), FL.alias("flight"),
               C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("cs"),
               C("MVT_TIME_UTC_mvt").alias("mvt"), C("AOBT_3_flt").is_not_null().alias("plan"),
               S("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").alias("gs"), S("MVT_TIME_UTC_mvt", "LOBT_flt").alias("gl"),
               S("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("ga"))
       .collect())
assert dep[ID].n_unique() == dep.height
j = sub.join(dep, on=ID, how="left", maintain_order="left")
assert j.height == sub.height and j["apt"].null_count() == 0, "a submission row is not a scored departure"
p = j[TARGET].to_numpy().astype(float)
new = p.copy()
tag = np.array([""] * len(p), dtype=object)
gs = j["gs"].to_numpy()

# ---------------- R1: day-gap bound
if "r1" in rules:
    m1 = (gs >= 88000) & (gs < 129600)
    if args.r1_mixture is not None:
        tr = (pl.scan_parquet("data/raw/training_*.parquet")
              .filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_null())
              .select(S("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt").alias("y")).collect())
        off = float(tr.filter(C("y").is_between(DAY + 60, DAY + 5000))["y"].median() - DAY)
        tgt = args.r1_mixture * (DAY + off) + (1 - args.r1_mixture) * gs
        r1 = m1 & (j["apt"] == "LIRF").to_numpy() & ~j["plan"].to_numpy()
        new = np.where(r1, tgt, new)
        tag[r1 & (np.abs(tgt - p) > 0.5)] = "R1mix"
    r1b = m1 & (new > gs)
    new = np.where(r1b, np.floor(gs), new)
    tag[r1b] = "R1"

# ---------------- R2: Rome 'low' hidden plan -> copy
if "r2" in rules:
    npd = (j.filter((C("apt") == "LIRF") & ~C("plan") & C("ades").is_in(APTS) & C("flight").is_not_null())
           .select(ID, "ades", "flight", "mvt", "gs").sort("mvt"))
    arr = (pl.scan_parquet(R).filter((C("PHASE_mvt") == "ARR") & C("LOBT_flt").is_not_null())
           .select(C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("arr_t"),
                   C("LOBT_flt").alias("lobt"), C("ADEP_flt").alias("plan_adep")).collect().sort("arr_t"))
    rec = (npd.join_asof(arr, left_on="mvt", right_on="arr_t", by=["ades", "flight"], strategy="forward", tolerance="8h")
           .filter(C("lobt").is_not_null() & (C("plan_adep") == "LIRF"))
           .with_columns(S("mvt", "lobt").alias("h_gl"))
           .filter((C("h_gl") < W_EXACT) & (C("gs") > C("h_gl") + W_EXACT)))
    assert rec[ID].n_unique() == rec.height
    r2ids = set(rec[ID].to_list())
    r2 = np.array([m in r2ids for m in j[ID].to_list()])
    ch = r2 & (np.abs(gs - new) > 0.5)
    new = np.where(r2, gs, new)
    tag[ch] = np.where(tag[ch] == "", "R2", tag[ch] + "+R2")
    print(f"R2 scope: {int(r2.sum())} rows (LIRF no-plan, hidden plan h_gl < 3,606s, gs > h_gl + 3,606s)")

# ---------------- R3: first-push reading on hard-binding rows
if "r3" in rules:
    tr = (pl.scan_parquet("data/raw/training_*.parquet")
          .filter((C("PHASE_mvt") == "DEP") & C("AOBT_3_flt").is_not_null() & C("LOBT_flt").is_not_null())
          .select(S("MVT_TIME_UTC_mvt", "LOBT_flt").alias("gl"), S("MVT_TIME_UTC_mvt", "AOBT_3_flt").alias("ga"),
                  S("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt").alias("y"))
          .filter(C("gl") - W_EXACT - C("ga") > HARD_S).collect())
    M = float((tr["y"] - (tr["gl"] - W_EXACT)).median())
    gl, ga = j["gl"].to_numpy(), j["ga"].to_numpy()
    plan = j["plan"].to_numpy() & ~np.isnan(gl)
    hi = np.where(plan, np.floor(gl + W_BOX), np.inf)
    target = np.minimum(np.rint(gl - W_EXACT + M * 0.5), hi)
    hard = plan & ~np.isnan(ga) & (gl - W_EXACT - ga > HARD_S) & ~j["apt"].is_in(R3_SKIP).to_numpy()
    cand = hard & (np.abs(p - target) <= 2)
    cj = j.filter(pl.Series(cand)).with_columns(C("mvt").dt.date().alias("d1"),
                                                 (C("mvt") - timedelta(hours=8)).dt.date().alias("d0"))
    fixes = {}
    for d in sorted(set(cj["d1"].to_list()) | set(cj["d0"].to_list())):
        f = f"data/adsb/points/{d}.parquet"
        todo = cj.filter((C("d1") == d) | (C("d0") == d))
        if not glob.glob(f) or todo.height == 0:
            continue
        pts = (pl.scan_parquet(f).filter(C("apt").is_in(todo["apt"].unique().to_list()))
               .select("apt", "icao", C("callsign").str.strip_chars().str.to_uppercase().alias("callsign"), "t", "alt", "gs")
               .collect())
        for r in todo.iter_rows(named=True):
            mv = r["mvt"].timestamp()
            ic = pts.filter((C("apt") == r["apt"]) & (C("callsign") == r["cs"]) & ((C("t") - mv).abs() <= 900))["icao"].unique()
            if ic.len():
                fixes.setdefault(r[ID], []).append(
                    pts.filter((C("apt") == r["apt"]) & C("icao").is_in(ic.to_list()) & (C("t") >= mv - 8 * 3600)
                               & (C("t") <= mv + 600)).select("t", "alt", "gs"))

    def first_departure_fix(t, alt, gg, t_lift):
        ground = alt == -1
        idx = np.arange(len(t))
        td = [i for i in idx[1:] if ground[i] and not ground[i - 1]]
        gi = [i for i in idx[(td[-1] if td else 0):] if ground[i] and t[i] <= t_lift]
        if len(gi) < 2:
            return None
        k = 0
        while k + 1 < len(gi):
            a, b = gi[k], gi[k + 1]
            if t[b] - t[a] >= 1200:
                return float(t[b])
            if gg[a] < 1:
                jj = k
                while jj + 1 < len(gi) and gg[gi[jj + 1]] < 1 and t[gi[jj + 1]] - t[gi[jj]] < 1200:
                    jj += 1
                if t[gi[jj]] - t[a] >= 1200 and jj + 1 < len(gi):
                    return float(t[gi[jj + 1]])
                k = jj + 1
                continue
            k += 1
        return float(t[gi[0]]) if not td else None

    reading = {}
    for r in cj.iter_rows(named=True):
        fr = fixes.get(r[ID])
        if not fr:
            continue
        fx = pl.concat(fr).unique().sort("t")
        t, alt, gg = fx["t"].to_numpy(), fx["alt"].to_numpy(), fx["gs"].fill_null(0.0).to_numpy()
        mv = r["mvt"].timestamp()
        lift = t[(alt == -1) & (t <= mv + 600)]
        if len(lift) == 0:
            continue
        t0 = first_departure_fix(t, alt, gg, float(lift[-1]))
        if t0 is None:
            continue
        yr = mv - t0
        if r["gl"] - W_EXACT <= yr <= r["gl"] + W_EXACT:
            reading[r[ID]] = yr
    r3 = np.array([m in reading for m in j[ID].to_list()])
    val = np.array([reading.get(m, np.nan) for m in j[ID].to_list()])
    ch = r3 & (np.abs(val - new) > 0.5)
    new = np.where(r3, val, new)
    tag[ch] = np.where(tag[ch] == "", "R3", tag[ch] + "+R3")
    print(f"R3: 2025 median lift M = {M:,.0f}s; {int(hard.sum())} hard-binding rows outside {R3_SKIP}, "
          f"{int(cand.sum())} still on the box target, {len(reading)} with an in-box first-push reading")

new = np.clip(np.rint(new), 0, None)
changed = new != p
if (changed & (tag == "")).any():
    sys.exit("refusing: a row changed without a rule tag")
K = 2 * len(p) * 238.14
print(f"\n{int(changed.sum())} rows changed ({', '.join(f'{t} {int((tag[changed] == t).sum())}' for t in sorted(set(tag[changed])))}):")
cj = j.with_columns(pl.Series("new", new), pl.Series("rule", tag.astype(str))).filter(pl.Series(changed))
for r in cj.sort(["rule", "apt"]).iter_rows(named=True):
    print(f"  {r['rule']:<6} {int(r[ID])} {r['apt']} {str(r['flight'] or r['cs']):<9} gs {r['gs']:>9,.0f}  "
          f"{int(r[TARGET]):>8,} -> {int(r['new']):>8,}  ({r['new'] - r[TARGET]:+,.0f})")
out = sub.with_columns(pl.Series(TARGET, new).cast(sub.schema[TARGET]))
assert out.height == sub.height and out[ID].equals(sub[ID])
assert int((out[TARGET] != sub[TARGET]).sum()) == int(changed.sum())
out.write_parquet(args.out)
print(f"wrote {args.out}: {out.height:,} rows, {int(changed.sum())} changed, all other rows identical")
