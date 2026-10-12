"""experiment abl 01: cheap post-re-stack chain ablation on the v114 fold analogue (docs/LAB-NOTES.md).

Reimplements the six post-re-stack rules (apply_gap_clamp / apply_lobt_box / apply_min_floor /
apply_v35_rules / apply_upper_bound / apply_hidden_reading) against data/raw/training_2025 in place
of data/raw/ranking.parquet, applied to CTRL_0 = agent_stagecap_RS_E9S_s0.npz + Dnbr (the v114 fold
analogue, 1,352,077 rows, 9 airports; exactly archive/agent_blend2_02_fit.py's CTRL_0). No prc/ or
existing-script edits; new file only. Read-only against training_*.parquet and the three existing npz.

Writes data/interim/agent_abl_verdict.json.
"""
import json

import numpy as np
import polars as pl

C = pl.col
ID = "MVT_ID_mvt"
I = "data/interim/"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
BOUND_APTS = ["LIRF", "LTFM"]
W = 3606.0
FL = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase()
OP = FL.str.extract(r"^([A-Z]{2,3})", 1)


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(I + p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys}


print("loading fold analogue (CTRL_0)...", flush=True)
c0 = load("agent_stagecap_RS_E9S_s0.npz")
ids, y, days, apts = c0["ids"], c0["y"].astype(float), c0["days"].astype(str), c0["apts"].astype(str)
N = len(y)
mon = np.array([d[5:7] for d in days])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])


def guard(path):
    z = load(path, ("ids", "both_guard"))
    assert np.array_equal(z["ids"], ids), f"{path}: id mismatch"
    return z["both_guard"].astype(float)


dnbr = guard("agent_blend_oof_seedA.npz") - guard("agent_blend_v109analogue.npz")
ctrl = c0["both_guard"].astype(float) + dnbr
print(f"CTRL_0: {N:,} rows, 9 airports {sorted(set(apts))}", flush=True)

# --- pull the row-level context every rule needs, from data/raw/training_2025, joined to our ids ---
print("joining training context...", flush=True)
dep_ctx = (pl.scan_parquet("data/raw/training_*.parquet")
           .filter(C("PHASE_mvt") == "DEP")
           .select(C(ID).cast(pl.Float64), C("ADEP_mvt").alias("adep"), C("ADES_mvt").alias("ades"),
                   FL.alias("flight"), OP.alias("op"), C("MVT_TIME_UTC_mvt").alias("mvt"),
                   C("AOBT_3_flt").alias("aobt3"), C("LOBT_flt").alias("lobt"),
                   (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                   (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
                   (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"))
           .collect())
idf = pl.DataFrame({ID: ids}).with_columns(pl.int_range(pl.len()).alias("_row"))
j = idf.join(dep_ctx, on=ID, how="left", maintain_order="left").sort("_row")
assert j.height == N and (j[ID].to_numpy() == ids).all()
apt = j["adep"].fill_null("").to_numpy()
ades = j["ades"].fill_null("").to_numpy()
flight = j["flight"].to_numpy()
op = j["op"].fill_null("").to_numpy()
mvt = j["mvt"].to_numpy()
plan = j["aobt3"].is_not_null().to_numpy()
gs = j["gs"].fill_null(-1.0).to_numpy().astype(float)
gl = j["gl"].to_numpy().astype(float)
ga = j["ga"].to_numpy().astype(float)
assert (apt == apts).all(), "ADEP from training join disagrees with npz apts -- id alignment bug"

# arrival rows, for the destination-arrival-plan recovery used by v35 r2 / upper_bound / hidden_reading
arr_ctx = (pl.scan_parquet("data/raw/training_*.parquet")
           .filter((C("PHASE_mvt") == "ARR") & C("LOBT_flt").is_not_null())
           .select(C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("arr_t"),
                   C("LOBT_flt").alias("lobt"), C("ADEP_flt").alias("plan_adep"), C("AOBT_3_flt").alias("aobt"))
           .collect().sort("arr_t"))


_RECOVER_CACHE = {}
_IDX = {m: i for i, m in enumerate(ids)}  # built once, reused everywhere (avoids repeated 1.35M-entry dict builds)


def recover_plan(scope_apts, need_aobt=False):
    key = (tuple(scope_apts), need_aobt)
    if key in _RECOVER_CACHE:
        return _RECOVER_CACHE[key]
    dep = (pl.DataFrame({ID: ids, "adep": apt, "ades": ades, "flight": flight, "mvt": mvt, "gs": gs, "plan": plan})
           .with_columns(C("mvt").dt.replace_time_zone("UTC"))
           .filter(C("adep").is_in(scope_apts) & C("flight").is_not_null() & (C("flight") != "") & ~C("plan"))
           .sort("mvt"))
    rec = (dep.join_asof(arr_ctx, left_on="mvt", right_on="arr_t", by=["ades", "flight"], strategy="forward", tolerance="8h")
              .filter(C("lobt").is_not_null() & (C("plan_adep") == C("adep"))))
    if need_aobt:
        rec = rec.filter(C("aobt").is_not_null())
    rec = rec.with_columns((C("mvt") - C("lobt")).dt.total_seconds().alias("gl_r"))
    if need_aobt:
        rec = rec.with_columns((C("mvt") - C("aobt")).dt.total_seconds().alias("rd"))
    _RECOVER_CACHE[key] = rec
    return rec


rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean())) if m.any() else float("nan")  # noqa: E731


def clauses(pt, base_p, kind):
    d = lambda m: rm(pt, m) - rm(base_p, m)  # noqa: E731
    gain = (base_p - y) ** 2 - (pt - y) ** 2
    keep = np.ones(N, bool)
    keep[np.argsort(-gain)[:50]] = False
    per = {a: d(apts == a) for a in AIRPORTS}
    c = {"all": d(np.ones(N, bool)), "worst_apt": max(per.values()), "m50": d(keep),
         "jan_mar": d(H1), "jul_sep": d(H2), "per_apt": per, "rows_changed": int((pt != base_p).sum())}
    if kind == "removal":
        c["pass"] = (c["all"] <= -0.05 and c["worst_apt"] <= 0.10 and c["m50"] < 0
                     and c["jan_mar"] < 0 and c["jul_sep"] < 0)
    else:
        c["pass"] = (c["all"] <= -0.10 and c["worst_apt"] <= 0.10 and c["m50"] < 0
                     and c["jan_mar"] < 0 and c["jul_sep"] < 0)
    return c


# --- the six rules, each: (pred_in) -> pred_out ---

def rule_gap_clamp(p, hi=50000.0, lo=5000.0):
    use = (gs >= lo) & (gs < hi) & (p > gs) & (gs >= 0)
    return np.where(use, np.floor(gs), p)


def rule_lobt_box(p, blend=0.5, w=3620.0, w_exact=3606.0, hard_s=1800.0):
    box = plan & np.isfinite(gl)
    lo_e = np.where(box, np.ceil(gl - w), -np.inf)
    hi_e = np.where(box, np.floor(gl + w), np.inf)
    new = np.clip(np.clip(p, lo_e, hi_e), 0, None)
    tr = (pl.scan_parquet("data/raw/training_*.parquet")
          .filter((C("PHASE_mvt") == "DEP") & C("AOBT_3_flt").is_not_null() & C("LOBT_flt").is_not_null())
          .select((C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
                  (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
                  (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("yy"))
          .filter(C("gl") - w_exact - C("ga") > hard_s).collect())
    med = float((tr["yy"] - (tr["gl"] - w_exact)).median())
    hard = box & np.isfinite(ga) & (gl - w_exact - ga > hard_s)
    target = np.minimum(np.rint(gl - w_exact + med * blend), hi_e)
    lift = hard & (new < target)
    return np.where(lift, target, new)


FLOORS_CACHE = {}


def rule_min_floor(p, q=0.01):
    if q not in FLOORS_CACHE:
        gen = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
               .select(C("ADEP_mvt").alias("apt"),
                       (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gsx"),
                       (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("yy"))
               .collect().filter((C("yy") - C("gsx")).abs() > 60))
        FLOORS_CACHE[q] = dict(gen.group_by("apt").agg(C("yy").quantile(q).alias("f")).sort("apt").iter_rows())
    floors = FLOORS_CACHE[q]
    f = np.array([floors.get(a, 0.0) for a in apt], float)
    box = plan & np.isfinite(gl)
    f = np.where(box, np.minimum(f, gl + W), f)
    return np.maximum(p, np.clip(f, 0, None))


def rule_v35(p, min_n=8, rate=0.97):
    tr = (pl.scan_parquet("data/raw/training_*.parquet")
          .filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_null())
          .select(OP.alias("op"), ((C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60).alias("copy"))
          .collect())
    habitual = set(tr.group_by("op").agg(pl.len().alias("n"), C("copy").mean().alias("r"))
                     .filter((C("n") >= min_n) & (C("r") >= rate))["op"].to_list())
    new = p.copy()
    r1 = (apt == "LIRF") & (~plan) & np.isin(op, list(habitual)) & (gs < 50000) & (gs >= 0)
    new = np.where(r1, np.maximum(gs, 0.0), new)
    rec = recover_plan(BOUND_APTS).filter(C("gl_r") < W)
    bound = dict(zip(rec[ID].to_list(), (rec["gl_r"] + W).to_list()))
    idx = _IDX
    r2n = 0
    for m, b in bound.items():
        i = idx.get(m)
        if i is None:
            continue
        if new[i] < b:
            new[i] = b
            r2n += 1
    return new, len(habitual), r2n


def rule_upper_bound(p):
    rec = recover_plan(BOUND_APTS).filter((C("gl_r") >= W) & ((C("gs") - C("gl_r")).abs() < W))
    cap = dict(zip(rec[ID].to_list(), (rec["gl_r"] - W).to_list()))
    idx = _IDX
    new = p.copy()
    for m, v in cap.items():
        i = idx.get(m)
        if i is not None and new[i] > v:
            new[i] = v
    return np.clip(new, 0, None)


def rule_hidden_reading(p):
    rec = recover_plan(["LIRF"], need_aobt=True).filter((C("gl_r") >= W) & ((C("gs") - C("gl_r")).abs() < W))
    target = dict(zip(rec[ID].to_list(), np.clip(rec["rd"].to_numpy(), 0, (rec["gl_r"] - W).to_numpy()).tolist()))
    idx = _IDX
    new = p.copy()
    for m, v in target.items():
        i = idx.get(m)
        if i is not None:
            new[i] = v
    return new


print("computing FULL chain (production order, current strengths)...", flush=True)
p1 = rule_gap_clamp(ctrl)
p2 = rule_lobt_box(p1)
p3 = rule_min_floor(p2)
p4, n_hab, n_r2 = rule_v35(p3)
p5 = rule_upper_bound(p4)
p6 = rule_hidden_reading(p5)
FULL = p6
print(f"  habitual ops {n_hab}, v35 r2 rows {n_r2}", flush=True)
print(f"  FULL vs CTRL_0: all rows RMSE delta {rm(FULL, np.ones(N,bool)) - rm(ctrl, np.ones(N,bool)):+.4f}s "
      f"(negative = the current 6-rule chain still helps vs no chain at all)", flush=True)

out = {"full_vs_ctrl_all": rm(FULL, np.ones(N, bool)) - rm(ctrl, np.ones(N, bool)), "toggles": {}}

# --- T1-T6: removal (leave-one-out vs FULL) ---
print("T1-T6 removals...", flush=True)
loo = {}
loo["gap_clamp"] = rule_hidden_reading(rule_upper_bound(rule_v35(rule_min_floor(rule_lobt_box(ctrl)))[0]))
loo["lobt_box"] = rule_hidden_reading(rule_upper_bound(rule_v35(rule_min_floor(rule_gap_clamp(ctrl)))[0]))
loo["min_floor"] = rule_hidden_reading(rule_upper_bound(rule_v35(rule_lobt_box(rule_gap_clamp(ctrl)))[0]))
loo["v35_rules"] = rule_hidden_reading(rule_upper_bound(rule_min_floor(rule_lobt_box(rule_gap_clamp(ctrl)))))
loo["upper_bound"] = rule_hidden_reading(rule_v35(rule_min_floor(rule_lobt_box(rule_gap_clamp(ctrl))))[0])
loo["hidden_reading"] = rule_upper_bound(rule_v35(rule_min_floor(rule_lobt_box(rule_gap_clamp(ctrl))))[0])
for name, pt in loo.items():
    c = clauses(pt, FULL, "removal")
    out["toggles"][f"REMOVE_{name}"] = c
    print(f"  REMOVE {name}: all {c['all']:+.4f} worst {c['worst_apt']:+.3f} m50 {c['m50']:+.3f} "
          f"JanMar {c['jan_mar']:+.3f} JulSep {c['jul_sep']:+.3f} changed {c['rows_changed']:,} "
          f"-> {'WORTH REMOVING' if c['pass'] else 'keep'}", flush=True)

# --- T7-T10: strength variants vs FULL ---
print("T7-T10 strength variants...", flush=True)


def full_with(step_name, replacement_out):
    # rebuild FULL with one step swapped for `replacement_out`, rest unchanged & in order
    seq = {"gap_clamp": p1, "lobt_box": p2, "min_floor": p3, "v35": p4}
    if step_name == "gap_clamp":
        a = replacement_out
    else:
        a = p1
    b = rule_lobt_box(a) if step_name != "lobt_box" else replacement_out
    c = rule_min_floor(b) if step_name != "min_floor" else replacement_out
    d, _, _ = rule_v35(c) if step_name != "v35" else (replacement_out, 0, 0)
    e = rule_upper_bound(d)
    f = rule_hidden_reading(e)
    return f


variants = {
    "gap_clamp_0.5x_hi25000": full_with("gap_clamp", rule_gap_clamp(ctrl, hi=25000.0)),
    "gap_clamp_1.5x_hi75000": full_with("gap_clamp", rule_gap_clamp(ctrl, hi=75000.0)),
    "gap_clamp_2x_hi100000": full_with("gap_clamp", rule_gap_clamp(ctrl, hi=100000.0)),
    "lobt_box_0.5x_blend0.25": full_with("lobt_box", rule_lobt_box(p1, blend=0.25)),
    "lobt_box_1.5x_blend0.75": full_with("lobt_box", rule_lobt_box(p1, blend=0.75)),
    "lobt_box_2x_blend1.0": full_with("lobt_box", rule_lobt_box(p1, blend=1.0)),
    "min_floor_0.5x_q0.005": full_with("min_floor", rule_min_floor(p2, q=0.005)),
    "min_floor_1.5x_q0.015": full_with("min_floor", rule_min_floor(p2, q=0.015)),
    "min_floor_2x_q0.02": full_with("min_floor", rule_min_floor(p2, q=0.02)),
    "v35_r1_0.5x_minn4": full_with("v35", rule_v35(p3, min_n=4)[0]),
    "v35_r1_2x_minn16": full_with("v35", rule_v35(p3, min_n=16)[0]),
}
for name, pt in variants.items():
    c = clauses(pt, FULL, "strength")
    out["toggles"][name] = c
    print(f"  {name}: all {c['all']:+.4f} worst {c['worst_apt']:+.3f} m50 {c['m50']:+.3f} "
          f"JanMar {c['jan_mar']:+.3f} JulSep {c['jul_sep']:+.3f} changed {c['rows_changed']:,} "
          f"-> {'PASS (ship)' if c['pass'] else 'fail'}", flush=True)

with open(I + "agent_abl_verdict.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print("wrote data/interim/agent_abl_verdict.json", flush=True)
