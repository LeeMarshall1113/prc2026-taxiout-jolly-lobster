"""ruleorder lib: build the hard-constraint bound arrays (lower / upper) for a set of departure rows, either the 2026 scoring rows
(source='ranking') or 2025 training rows (source='training'), by exec'ing the rule functions of agent_abl_01_run.py (same code the
chain audit used) against this row set. Read only.
bounds(ids, source) -> dict of arrays aligned to `ids` (float ids):
  gs, gl, ga, plan, apt, flo (floor), lo_e, hi_e (box edges, nan where no box), lift (hard-lift target or nan), r2 (hidden-plan lower bound or nan),
  ub (hidden-plan upper bound or nan), gcap (gap clamp value floor(gs) where it applies or nan), rd (hidden-reading target or nan), hab (habitual-copy set rows), op
plus the rule functions themselves in dict 'fn'.
"""
import numpy as np
import polars as pl

C = pl.col
ID = "MVT_ID_mvt"
BIG = 1e9


def bounds(ids, source):
    src = open("archive/agent_abl_01_run.py", encoding="utf-8").read()
    a = src.index("_RECOVER_CACHE = {}")
    b = src.index('print("computing FULL chain')
    glob = "data/raw/ranking.parquet" if source == "ranking" else "data/raw/training_*.parquet"
    FL = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase()
    OP = FL.str.extract(r"^([A-Z]{2,3})", 1)
    dep_ctx = (pl.scan_parquet(glob).filter(C("PHASE_mvt") == "DEP")
               .select(C(ID).cast(pl.Float64), C("ADEP_mvt").alias("adep"), C("ADES_mvt").alias("ades"), FL.alias("flight"), OP.alias("op"),
                       C("MVT_TIME_UTC_mvt").alias("mvt"), C("AOBT_3_flt").alias("aobt3"), C("LOBT_flt").alias("lobt"),
                       (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                       (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
                       (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga")).collect())
    idf = pl.DataFrame({ID: ids}).with_columns(pl.int_range(pl.len()).alias("_row"))
    j = idf.join(dep_ctx, on=ID, how="left", maintain_order="left").sort("_row")
    assert j.height == len(ids) and (j[ID].to_numpy() == ids).all()
    arr_ctx = (pl.scan_parquet(glob).filter((C("PHASE_mvt") == "ARR") & C("LOBT_flt").is_not_null())
               .select(C("ADES_mvt").alias("ades"), FL.alias("flight"), C("MVT_TIME_UTC_mvt").alias("arr_t"), C("LOBT_flt").alias("lobt"),
                       C("ADEP_flt").alias("plan_adep"), C("AOBT_3_flt").alias("aobt")).collect().sort("arr_t"))
    g = dict(np=np, pl=pl, C=C, ID=ID, FL=FL, OP=OP, W=3606.0, BOUND_APTS=["LIRF", "LTFM"], ids=ids, N=len(ids), arr_ctx=arr_ctx,
             apt=j["adep"].fill_null("").to_numpy(), ades=j["ades"].fill_null("").to_numpy(), flight=j["flight"].to_numpy(),
             op=j["op"].fill_null("").to_numpy(), mvt=j["mvt"].to_numpy(), plan=j["aobt3"].is_not_null().to_numpy(),
             gs=j["gs"].fill_null(-1.0).to_numpy().astype(float), gl=j["gl"].to_numpy().astype(float), ga=j["ga"].to_numpy().astype(float))
    exec(compile(src[a:b], "abl_rules", "exec"), g)
    W = 3606.0
    gs, gl, plan, apt = g["gs"], g["gl"], g["plan"], g["apt"]
    out = dict(gs=gs, gl=gl, ga=g["ga"], plan=plan, apt=apt, op=g["op"], fn={k: g[k] for k in ("rule_gap_clamp", "rule_lobt_box", "rule_min_floor", "rule_v35", "rule_upper_bound", "rule_hidden_reading", "recover_plan")})
    big = np.full(len(ids), BIG)
    gc_ = g["rule_gap_clamp"](big)
    out["gcap"] = np.where(gc_ < BIG, gc_, np.nan)
    box = plan & np.isfinite(gl)
    out["lo_e"] = np.where(box, np.maximum(np.ceil(gl - 3620.0), 0), np.nan)
    out["hi_e"] = np.where(box, np.floor(gl + 3620.0), np.nan)
    lb = g["rule_lobt_box"](np.full(len(ids), -BIG))
    hard = lb != np.where(box, np.maximum(np.ceil(gl - 3620.0), 0), 0.0)
    out["lift"] = np.where(hard, lb, np.nan)
    flo = g["rule_min_floor"](np.full(len(ids), -BIG)); out["flo"] = flo
    rec = g["recover_plan"](["LIRF", "LTFM"]).filter(C("gl_r") < W)
    r2 = np.full(len(ids), np.nan); idx = g["_IDX"]
    for m, bnd in zip(rec[ID].to_list(), (rec["gl_r"] + W).to_list()):
        i = idx.get(m)
        if i is not None: r2[i] = bnd
    out["r2"] = r2
    rec = g["recover_plan"](["LIRF", "LTFM"]).filter((C("gl_r") >= W) & ((C("gs") - C("gl_r")).abs() < W))
    ub = np.full(len(ids), np.nan)
    for m, cap in zip(rec[ID].to_list(), (rec["gl_r"] - W).to_list()):
        i = idx.get(m)
        if i is not None: ub[i] = cap
    out["ub"] = ub
    rd = g["rule_hidden_reading"](big)
    out["rd"] = np.where(rd < BIG, rd, np.nan)
    return out
