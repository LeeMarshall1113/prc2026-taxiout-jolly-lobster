"""Rule layers that act on a prediction through input columns: general rules, fitted on 2025, valid for any month.

Each rule is one function  rule(inputs, pred) -> pred  with a one-sentence docstring, the 2025 evidence in a line, and a
counter of rows moved per airport and month. `inputs` is the table built by `load_inputs` (one row per departure, only input
columns, never a label). `pred` is a float array aligned to that table. No flight string, row id or scored-set count is used.

Derived from the shipped scripts (named above each function): scripts/apply_*.py of the post-chain, the row-level bundles
agent_bundle_v133..v148 and the window re-application of v158. The constants are the shipped ones; where a shipped choice was
made on January and July 2026 only, it is implemented as shipped and the month-agnostic alternative sits behind a flag
(listed at the end of this docstring).

Stages (the order is the shipped order):
  pre    rome_day_early_bands, rome_plan_gate        applied to the model output, before the ADS-B stage
  post   gap_clamp, plan_window (with hard-binding lift), floor, habitual_copy_operators, hidden_plan_lower,
         hidden_plan_upper, hidden_plan_reading    applied after the re-stack
  rows   hidden_plan_callsign_reach, rome_hidden_plan_late_clock, paris_duplicate_pairs, rome_calendar_condition,
         istanbul_noplan_bound, madrid_ceiling, rome_planband, rome_hard_binding_window
  last   bounds_last   (gap clamp, plan window at the exact edge, hidden-plan lower bound, re-applied)

Flags (month-agnostic alternatives; defaults are the shipped behaviour):
  --madrid-only        the one-hour ceiling for Madrid (LEMD) only. The shipped code also covers Barcelona (LEBL); the
                       documentation says Madrid only.
  --jitter-allow S     seconds added to the plan-clock window for the take-off jitter (default 12, as the final file needs;
                       use 0 to reproduce the shipped chain on data/raw/ranking.parquet).
  --no-hard-binding    the plan window moves a prediction to the nearer edge only, without the 2025-median lift.

Not in this file (other experiments): the Rome no-plan and with-plan mixtures and their stored footprints, the correctors,
the neighbour blend, the day-level layers (final/32_published_daily.py), the rows that name 2026 flights (final/31_scored_row_fixes.py).

Usage
  python final/30_general_rules.py --pred P.parquet --ranking data/raw/final_ranking.parquet --out Q.parquet \
        --stage post,rows,last [--sample 0.01] [--months 2,6]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL, OUT, RAW, TARGET_COL  # noqa: E402

C = pl.col
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
W_EXACT = 3606.0          # largest |off-block - LOBT| seen on 2,062,577 plan departures of 2025, plus 6 s of timestamp jitter
BOX_SLACK = 14.0          # shipped slack of the post-chain box (3,620 = 3,606 + 14)
HARD_S = 1800.0           # plan edge above the network manager's own taxi by more than this = hard binding
TRAIN = str(RAW / "training_*.parquet")
FL = C("FLIGHT_mvt").str.strip_chars().str.to_uppercase()
CS = C("CALLSIGN_flt").str.strip_chars().str.to_uppercase()
HIDDEN_KEEP = ["LEBL", "LEMD", "LFPG", "LTFM", "EDDM", "EGLL", "LIRF"]   # airports of the callsign-extended upper bound
TZ_ROME = "Europe/Rome"


# ---------------------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------------------
class Counters:
    """Rows moved per rule, airport and calendar month (month of the take-off time)."""

    def __init__(self) -> None:
        self.n: dict[str, dict[tuple[str, int], int]] = defaultdict(lambda: defaultdict(int))

    def add(self, rule: str, inp: pl.DataFrame, moved: np.ndarray) -> int:
        if moved.any():
            g = (inp.filter(pl.Series(moved)).group_by("apt", "month").len())
            for apt, mon, k in g.iter_rows():
                self.n[rule][(apt, int(mon))] += int(k)
        return int(moved.sum())

    def table(self, months: list[int] | None = None) -> pl.DataFrame:
        rows = [dict(rule=r, apt=a, month=m, rows=k) for r, d in self.n.items() for (a, m), k in d.items()
                if months is None or m in months]
        return pl.DataFrame(rows, schema=dict(rule=pl.Utf8, apt=pl.Utf8, month=pl.Int64, rows=pl.Int64)).sort("rule", "apt", "month")

    def print(self, months: list[int] | None = None) -> None:
        t = self.table(months)
        print("rows moved per rule, airport and month:")
        if t.is_empty():
            print("  (none)")
            return
        for rule in t["rule"].unique(maintain_order=True):
            s = t.filter(C("rule") == rule)
            print(f"  {rule:32s} total {int(s['rows'].sum()):6,}   " + "  ".join(f"{a} m{m:02d}:{k}" for a, m, k in s.select("apt", "month", "rows").iter_rows()))


def load_inputs(ranking: Path, months: list[int] | None = None, sample: float | None = None) -> pl.DataFrame:
    """One row per departure of the ranking file, sorted by id, input columns only (no departure label is selected).

    Columns: id, apt, ades, month, mvt/sched/lobt/aobt (epoch s, null if absent), gs = take-off minus schedule,
    gl = take-off minus LOBT, ga = take-off minus AOBT_3, plan = AOBT_3 present, has_lobt, flight, fid_null, day (UTC take-off date).
    """
    sel = [C(ID_COL).alias("id"), C("ADEP_mvt").alias("apt"), C("ADES_mvt").alias("ades"), C("FLIGHT_mvt").alias("flight_raw"),
           FL.alias("flight"), C("FLIGHT_ID_mvt").is_null().alias("fid_null"),
           C("MVT_TIME_UTC_mvt").alias("t_mvt"), C("SCHED_TIME_UTC_mvt").alias("t_sched"), C("LOBT_flt").alias("t_lobt"), C("AOBT_3_flt").alias("t_aobt")]
    d = (pl.scan_parquet(ranking).filter(C("PHASE_mvt") == "DEP").select(sel).collect().sort("id"))
    d = d.with_columns(C("t_mvt").dt.month().alias("month"), C("t_mvt").dt.date().cast(pl.Utf8).alias("day"),
                       (C("t_mvt") - C("t_sched")).dt.total_seconds().cast(pl.Float64).alias("gs"),
                       (C("t_mvt") - C("t_lobt")).dt.total_seconds().cast(pl.Float64).alias("gl"),
                       (C("t_mvt") - C("t_aobt")).dt.total_seconds().cast(pl.Float64).alias("ga"),
                       C("t_mvt").dt.epoch("s").alias("mvt"), C("t_aobt").is_not_null().alias("plan"), C("t_lobt").is_not_null().alias("has_lobt"))
    if months:
        d = d.filter(C("month").is_in(months))
    if sample:
        d = d.sample(fraction=sample, seed=0).sort("id")
    return d


def recover_hidden_plans(ranking: Path, dep: pl.DataFrame) -> pl.DataFrame:
    """Plan of a departure without its own plan, read from the destination's arrival record (flight string or callsign key).

    Same flight string (or the plan's callsign), landing 0 to 8 h after take-off, plan origin equal to the departure airport.
    Returns id, route ('str' preferred, else 'cs'), lobt, aobt as epoch seconds. Shipped in apply_v35_rules.py and
    agent_detrules_19_callsign.py. Needs the arrival rows of the whole file, so it is always built on the full ranking file.
    """
    arr = (pl.scan_parquet(ranking).filter((C("PHASE_mvt") == "ARR") & C("LOBT_flt").is_not_null())
           .select(C("ADES_mvt").alias("ades"), FL.alias("fkey"), CS.alias("ckey"), C("MVT_TIME_UTC_mvt").alias("arr_t"),
                   C("LOBT_flt").alias("lobt"), C("ADEP_flt").alias("plan_adep"), C("AOBT_3_flt").alias("aobt"))
           .collect())
    left = (dep.filter(C("ades").is_in(APTS)).select("id", "apt", "ades", "flight", C("t_mvt").alias("t")).drop_nulls("flight").sort("t"))
    outs = []
    for route, key in (("str", "fkey"), ("cs", "ckey")):
        a = arr.filter(C(key).is_not_null()).select("ades", C(key).alias("flight"), "arr_t", "lobt", "plan_adep", "aobt").sort("arr_t")
        r = (left.join_asof(a, left_on="t", right_on="arr_t", by=["ades", "flight"], strategy="forward", tolerance="8h")
             .filter(C("lobt").is_not_null() & (C("plan_adep") == C("apt")) & ((C("arr_t") > C("t")) | (route == "str")))
             .select("id", pl.lit(route).alias("route"), C("lobt").dt.epoch("s").alias("h_lobt"), C("aobt").dt.epoch("s").alias("h_aobt")))
        outs.append(r)
    both = pl.concat(outs).sort("id", pl.col("route") == "cs")       # 'str' rows first within an id
    return both.unique(subset="id", keep="first", maintain_order=True)


class Ctx:
    """Everything a rule needs besides the prediction; built once."""

    def __init__(self, ranking: Path, months: list[int] | None = None, sample: float | None = None,
                 jitter_allow: float = 12.0, madrid_only: bool = False, hard_binding: bool = True,
                 with_hidden: bool = True, gate_seeds: int = 20) -> None:
        self.ranking = ranking
        self.inp = load_inputs(ranking, months, sample)
        self.jitter = jitter_allow
        self.madrid_only = madrid_only
        self.hard_binding = hard_binding
        self.gate_seeds = gate_seeds
        self.counters = Counters()
        self.hidden = recover_hidden_plans(ranking, self.inp) if with_hidden else pl.DataFrame(schema=dict(id=pl.Float64, route=pl.Utf8, h_lobt=pl.Int64, h_aobt=pl.Int64))
        self._train: pl.DataFrame | None = None

    @property
    def w_exact(self) -> float:
        return W_EXACT + self.jitter

    @property
    def w_box(self) -> float:
        return W_EXACT + BOX_SLACK + self.jitter

    def train(self) -> pl.DataFrame:
        """2025 departures: apt, label y, schedule gap gs, plan gaps (labels are 2025 training data and are used for constants only)."""
        if self._train is None:
            self._train = (pl.scan_parquet(TRAIN).filter(C("PHASE_mvt") == "DEP")
                           .select(C("ADEP_mvt").alias("apt"), FL.alias("flight"), C("AOBT_3_flt").is_not_null().alias("plan"),
                                   (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                                   (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"),
                                   (C("MVT_TIME_UTC_mvt") - C("AOBT_3_flt")).dt.total_seconds().alias("ga"),
                                   (C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y"),
                                   (C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("b_s"))
                           .collect())
        return self._train


def _round(x: np.ndarray) -> np.ndarray:
    return np.rint(x)


def _fire(ctx: Ctx, name: str, old: np.ndarray, new: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Count the rows a rule moved (|new - old| > tol), return the new values."""
    n = ctx.counters.add(name, ctx.inp, np.abs(new - old) > tol)
    print(f"  {name}: {n:,} rows moved")
    return new


# ---------------------------------------------------------------------------------------------------------------------
# pre-stage Rome rules (applied to the model output, before the ADS-B stage)
# ---------------------------------------------------------------------------------------------------------------------
def rome_day_early_bands(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """At Rome, a schedule gap of 50,000 to 100,000 s is served at the expected truth of its band: P(day-early) x (86,400 + offset) + P(copy) x gap.

    From archive/apply_rome_dayearly.py. 2025: bands single-digit counts (no-plan 50-60k 6 copy / 3 day-early, 60-70k 2 / 4,
    70-100k 0 / 4; plan 70-100k pooled), add-half smoothing, rebuilt from 2025 at run time.
    """
    DAY = 86400.0
    tr = ctx.train().filter(C("apt") == "LIRF").with_columns(C("y").is_between(DAY + 60, DAY + 5000).alias("r"),
                                                              ((C("b_s")).abs() <= 60).alias("s"))
    r_all = tr.filter(C("r"))
    if r_all.height < 5:
        raise SystemExit(f"only {r_all.height} day-early rows in 2025")
    offset = float(np.median(r_all["y"].to_numpy() - DAY))
    inp = ctx.inp
    gs = inp["gs"].to_numpy()
    is_lirf = (inp["apt"] == "LIRF").to_numpy()
    plan = inp["plan"].to_numpy()
    out = p.copy()
    for lo, hi, with_plan in [(50000.0, 60000.0, False), (60000.0, 70000.0, False), (70000.0, 100000.0, True)]:
        b = tr.filter(C("gs").is_between(lo, hi, closed="left"))
        if not with_plan:
            b = b.filter(~C("plan"))
        n_r = int(b["r"].sum())
        n_s = int((b["s"] & ~b["r"]).sum())
        p_r = (n_r + 0.5) / (n_r + n_s + 1.0)
        m = is_lirf & (gs >= lo) & (gs < hi)
        if not with_plan:
            m &= ~plan
        out[m] = p_r * (DAY + offset) + (1 - p_r) * gs[m]
    return _fire(ctx, "rome_day_early_bands", p, _round(out))


def rome_plan_gate(ctx: Ctx, p: np.ndarray, *, gap_cap: float = 10800.0, blend: float = 0.60, seeds: tuple[int, ...] | None = None,
                   features: pl.DataFrame | None = None) -> np.ndarray:
    """Rome plan rows with a schedule gap up to the cap take (1 - w) x prediction + w x a calibrated copy mixture (classifier x gap + regressor).

    From archive/apply_rome_plan_gate.py as run by the bagged Rome path of archive/merged_chain.sh (cap 10,800 s, weight 0.60,
    seeds 1..20, mixtures averaged). 2025 evidence: per fold -0.53, -1.19, -1.15 s at cap 7,200 s; uncapped it failed.
    `features` is the prc.features matrix of the departures to serve (final/10_base.py builds it one calendar month at a time); when it is
    None the matrix is built here from the ranking file, month by month. The extra weights added to the chain later
    (0.1 on a bundle of certain bounds, 0.2 on July with-plan rows not seen on the ground) are not part of this function.
    """
    import lightgbm as lgb
    seeds = seeds if seeds is not None else tuple(range(1, ctx.gate_seeds + 1))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from prc.features import CATEGORICAL, FEATURES, TARGET, build
    from prc.config import INTERIM_DIR

    feat = pl.scan_parquet(INTERIM_DIR / "train_features.parquet")
    use = [f for f in FEATURES if f in set(feat.collect_schema().names()) and f != TARGET]
    cats = [c for c in use if c in CATEGORICAL]
    raw = (pl.scan_parquet(TRAIN).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_not_null())
           .select(C(ID_COL), (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap_s"),
                   ((C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60).alias("copy")).collect())
    tr = feat.filter(C(ID_COL).is_in(raw[ID_COL].implode())).select([ID_COL, TARGET, *use]).collect().join(raw, on=ID_COL)

    def pandas(frame: pl.DataFrame):
        return frame.with_columns([C(c).cast(pl.Utf8).cast(pl.Categorical) for c in cats]).select(use).to_pandas()

    if features is None:
        full = pl.read_parquet(ctx.ranking)
        parts = []
        for m in sorted(set(ctx.inp["month"].to_list())):
            part = full.filter(C("MVT_TIME_UTC_mvt").dt.month() == m)
            parts.append(build(part, with_target=False))
        features = pl.concat(parts)
        del full
    inp = ctx.inp
    rows = inp.filter((C("apt") == "LIRF") & C("plan") & (C("gs") <= gap_cap)).select("id", C("gs").alias("gap_s"))
    te = features.rename({ID_COL: "id"}) if ID_COL in features.columns else features
    te = te.join(rows, on="id", how="inner")
    if te.is_empty():
        print("  rome_plan_gate: no Rome plan rows in scope")
        return p
    Xte = pandas(te.rename({"id": ID_COL}))
    Xall = pandas(tr)
    copy = tr["copy"].to_numpy()
    y = tr[TARGET].to_numpy().astype(float)
    gap = tr["gap_s"].to_numpy()
    mixes = []
    for sd in seeds:
        params = dict(learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.8, verbose=-1, random_state=sd)
        rng = np.random.default_rng(sd)
        idx = rng.permutation(tr.height)
        hold, fit = idx[: tr.height // 5], idx[tr.height // 5:]
        clf_h = lgb.LGBMClassifier(n_estimators=400, **params).fit(Xall.iloc[fit], copy[fit])
        nfit = np.intersect1d(fit, np.flatnonzero(~copy))
        reg_h = lgb.LGBMRegressor(n_estimators=600, **params).fit(Xall.iloc[nfit], np.clip(y[nfit], 0, 7200))
        ph = clf_h.predict_proba(Xall.iloc[hold])[:, 1]
        mix_h = ph * gap[hold] + (1 - ph) * reg_h.predict(Xall.iloc[hold])
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(hold)), mix_h]), np.clip(y[hold], -3600, 20000), rcond=None)[0]
        clf = lgb.LGBMClassifier(n_estimators=400, **params).fit(Xall, copy)
        reg = lgb.LGBMRegressor(n_estimators=600, **params).fit(Xall[~copy], np.clip(y[~copy], 0, 7200))
        pc = clf.predict_proba(Xte)[:, 1]
        mixes.append(coef[0] + coef[1] * (pc * te["gap_s"].to_numpy() + (1 - pc) * reg.predict(Xte)))
    mix = np.mean(mixes, axis=0)
    pos = dict(zip(inp["id"].to_list(), range(inp.height)))
    idx = np.array([pos[i] for i in te["id"].to_list()])
    out = p.copy()
    out[idx] = np.clip((1.0 - blend) * p[idx] + blend * mix, 0, None)
    return _fire(ctx, "rome_plan_gate", p, _round(out))


# ---------------------------------------------------------------------------------------------------------------------
# post-chain rules
# ---------------------------------------------------------------------------------------------------------------------
def gap_clamp(ctx: Ctx, p: np.ndarray, name: str = "gap_clamp") -> np.ndarray:
    """A departure taking off 5,000 to 50,000 s after its schedule is predicted at most its schedule gap (a copy equals the gap, a real taxi is far below).

    From archive/apply_gap_clamp.py. 2025: 93,993 rows in scope, the label exceeded the gap by more than 60 s on 58 (0.062%).
    """
    gs = ctx.inp["gs"].fill_null(-1.0).to_numpy()
    use = (gs >= 5000.0) & (gs < 50000.0) & (p > gs)
    return _fire(ctx, name, p, _round(np.where(use, np.floor(gs), p)))


def _hard_binding_median(ctx: Ctx) -> float:
    """Median of (label - exact lower plan edge) over the 2025 plan rows whose edge is hard-binding (about 359 rows)."""
    t = ctx.train().filter(C("plan") & C("gl").is_not_null() & ((C("gl") - W_EXACT - C("ga")) > HARD_S))
    if t.height < 200:
        raise SystemExit(f"only {t.height} hard-binding rows in 2025; expected about 359")
    return float((t["y"] - (t["gl"] - W_EXACT)).median())


def plan_window(ctx: Ctx, p: np.ndarray, *, blend: float = 0.5, exact: bool = False, name: str = "plan_window") -> np.ndarray:
    """A prediction of a row with a flight plan moves into [take-off - LOBT - W, take-off - LOBT + W]; hard-binding rows are lifted to the lower edge plus the 2025 median.

    From archive/apply_lobt_box.py (run with --hard-binding-blend 0.5 in the chain) and, with exact=True, the exact-edge
    re-application of agent_bundle_v158.py / agent_logicB_20_build.py. 2025: 0 exceptions on 2,062,577 plan rows.
    W = 3,606 + 14 s slack + the jitter allowance (exact edge: 3,606 + the jitter allowance).
    """
    inp = ctx.inp
    gl = inp["gl"].to_numpy()
    ga = inp["ga"].to_numpy()
    plan = inp["plan"].to_numpy() & ~np.isnan(gl)
    W = ctx.w_exact if exact else ctx.w_box
    lo = np.where(plan, np.ceil(gl - W), -np.inf)
    hi = np.where(plan, np.floor(gl + W), np.inf)
    new = np.clip(np.clip(p, lo, hi), 0, None)
    if ctx.hard_binding and not exact:
        M = _hard_binding_median(ctx)
        hard = plan & ~np.isnan(ga) & (gl - W_EXACT - ga > HARD_S)
        target = np.minimum(np.rint(gl - W_EXACT + M * blend), hi)
        lift = hard & (new < target)
        new = np.where(lift, target, new)
    if exact:
        # the exact edge never goes below zero and never reads an unplanned row
        fin = np.isfinite(gl)
        new = np.where(fin, np.clip(p, np.maximum(gl - W, 0.0), gl + W), p)
    return _fire(ctx, name, p, _round(new))


def floor(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """A prediction is raised to the first percentile of non-copy 2025 taxi times at its airport (capped at the plan edge for plan rows).

    From archive/apply_min_floor.py. 2025: out-of-fold -0.055 s pooled, every fold negative; ships as a rider (worth about zero).
    """
    t = ctx.train().filter((C("y") - C("gs")).abs() > 60)
    fl = dict(t.group_by("apt").agg(C("y").quantile(0.01).alias("f")).iter_rows())
    inp = ctx.inp
    f = np.array([fl.get(a, 0.0) for a in inp["apt"].to_list()], float)
    gl = inp["gl"].fill_null(np.nan).to_numpy()
    box = inp["plan"].to_numpy() & np.isfinite(gl)
    f = np.where(box, np.minimum(f, gl + W_EXACT), f)
    return _fire(ctx, "floor", p, _round(np.maximum(p, np.clip(f, 0, None))))


def habitual_copy_operators(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """A Rome no-plan departure with a schedule gap under 50,000 s whose operator prefix copied the schedule in at least 97% of at least 8 rows of 2025 is predicted at its gap.

    From archive/apply_v35_rules.py rule 1. 2025: 97.9% right out of fold; fired 22 and 23 rows in the two folds tested.
    """
    op = lambda e: e.str.extract(r"^([A-Z]{2,3})", 1)  # noqa: E731
    t = (ctx.train().filter((C("apt") == "LIRF") & ~C("plan"))
         .select(op(C("flight")).alias("op"), (C("b_s").abs() <= 60).alias("copy")))
    hab = set(t.group_by("op").agg(pl.len().alias("n"), C("copy").mean().alias("r")).filter((C("n") >= 8) & (C("r") >= 0.97))["op"].drop_nulls().to_list())
    inp = ctx.inp.with_columns(op(C("flight")).alias("op"))
    m = ((inp["apt"] == "LIRF") & ~inp["plan"] & (inp["gs"] < 50000) & inp["op"].is_in(sorted(hab))).fill_null(False).to_numpy()
    new = p.copy()
    tgt = np.maximum(inp["gs"].fill_null(0.0).to_numpy(), 0.0)
    new[m] = tgt[m]
    return _fire(ctx, "habitual_copy_operators", p, _round(new), tol=0.5)


def _hidden(ctx: Ctx, routes: tuple[str, ...] = ("str",)) -> pl.DataFrame:
    """Departures without their own plan joined to the recovered plan (lobt, aobt) by the chosen route(s)."""
    h = ctx.hidden.filter(C("route").is_in(list(routes)))
    return ctx.inp.join(h, on="id", how="left")


def hidden_plan_lower(ctx: Ctx, p: np.ndarray, name: str = "hidden_plan_lower") -> np.ndarray:
    """A no-plan Rome or Istanbul departure whose recovered plan has take-off minus LOBT under W is predicted at least that plus W (the stamp cannot lie before the plan window).

    From archive/apply_v35_rules.py rule 2 (flight-string route). 2025: 0 violations on 57 rows (LIRF 32, LTFM 25).
    """
    j = _hidden(ctx)
    gl = (j["mvt"] - j["h_lobt"]).cast(pl.Float64).to_numpy()
    m = (~j["plan"] & j["apt"].is_in(["LIRF", "LTFM"]) & j["ades"].is_in(APTS) & j["h_lobt"].is_not_null()).fill_null(False).to_numpy() & (gl < W_EXACT)
    bound = np.where(m, gl + W_EXACT, -np.inf)
    return _fire(ctx, name, p, _round(np.maximum(p, bound)))


def hidden_plan_upper(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """A no-plan departure whose recovered plan lies after its schedule window (take-off minus LOBT at least W, schedule within W of LOBT) is predicted at most that minus W.

    From archive/apply_upper_bound.py (Rome and Istanbul, flight-string key) extended by agent_detrules_30_build.py to seven
    airports and the callsign key. 2025: 0 of 132 violations at Rome and Istanbul, 1 of 1,278 over seven airports.
    """
    j = _hidden(ctx, ("str", "cs"))
    W = W_EXACT
    gl = (j["mvt"] - j["h_lobt"]).cast(pl.Float64).to_numpy()
    ls = (j["h_lobt"] - j["t_sched"].dt.epoch("s")).cast(pl.Float64).to_numpy()
    base = (~j["has_lobt"] & j["h_lobt"].is_not_null()).fill_null(False).to_numpy()
    m = base & j["apt"].is_in(HIDDEN_KEEP).fill_null(False).to_numpy() & (gl >= W) & (np.abs(ls) < W)
    # the flight-string route of the first two airports uses the plan-origin filter only; the callsign route needs the later landing (already in recover)
    cap = np.where(m, gl - W, np.inf)
    return _fire(ctx, "hidden_plan_upper", p, _round(np.clip(np.minimum(p, cap), 0, None)))


def hidden_plan_reading(ctx: Ctx, p: np.ndarray, callsign_reach: bool = True) -> np.ndarray:
    """A Rome no-plan departure whose recovered plan makes a schedule copy impossible is predicted at the plan's own off-block reading, clipped to [0, take-off - LOBT - W].

    From archive/apply_hidden_reading.py and its callsign reach (archive/agent_fix2_01_rome_callsign.py). 2025: 128 rows,
    label within 600 s of the reading on 94%; pooled -0.139 s out of fold.
    """
    j = _hidden(ctx, ("str", "cs") if callsign_reach else ("str",))
    W = W_EXACT
    gl = (j["mvt"] - j["h_lobt"]).cast(pl.Float64).to_numpy()
    rd = (j["mvt"] - j["h_aobt"]).cast(pl.Float64).to_numpy()
    ls = (j["h_lobt"] - j["t_sched"].dt.epoch("s")).cast(pl.Float64).to_numpy()
    m = (~j["plan"] & (j["apt"] == "LIRF") & j["h_aobt"].is_not_null() & j["ades"].is_in(APTS)).fill_null(False).to_numpy() & (gl >= W) & (np.abs(ls) < W)
    new = np.where(m, np.clip(rd, 0, gl - W), p)
    return _fire(ctx, "hidden_plan_reading", p, _round(new), tol=0.5)


# ---------------------------------------------------------------------------------------------------------------------
# row-level rules defined on input columns
# ---------------------------------------------------------------------------------------------------------------------
def rome_hidden_plan_late_clock(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """A Rome no-plan departure whose recovered plan has its own off-block at least W after LOBT (so no copy is needed to explain the missing plan) is predicted at take-off minus that off-block.

    From archive/agent_rome2_54_build.py (lever B) and agent_rome2_33_degarble.py (plan found on an orphan arrival by the
    de-garbled flight string or the callsign, flight time 1,500-20,000 s). 2025: 7 of 7 real taxis (8 of 8 with the callsign key).
    """
    ranking = ctx.ranking
    arr = (pl.scan_parquet(ranking).filter((C("PHASE_mvt") == "ARR") & (C("ADEP_flt") == "LIRF") & C("LOBT_flt").is_not_null() & C("AOBT_3_flt").is_not_null())
           .select(C("FLIGHT_ID_mvt").alias("fid"), C("MVT_TIME_UTC_mvt").alias("arr_t"), C("LOBT_flt").alias("lobt"), C("AOBT_3_flt").alias("aobt"),
                   CS.alias("k1"), FL.alias("k2")).collect())
    own = (pl.scan_parquet(ranking).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("FLIGHT_ID_mvt").is_not_null())
           .select("FLIGHT_ID_mvt").collect()["FLIGHT_ID_mvt"].unique().to_list())
    arr = arr.filter(~C("fid").is_in(own))

    def fix(s: str | None) -> str | None:
        if s is None:
            return None
        s = s.strip().upper()
        if len(s) >= 6 and s[:3].isalpha():
            p3 = s[:3]
            for k in (2, 1):
                if len(s) > 3 + k and s[3:3 + k] == p3[3 - k:3] and s[3 + k].isdigit():
                    return p3 + s[3 + k:]
        return s

    dep = ctx.inp.filter((C("apt") == "LIRF") & ~C("plan") & C("flight_raw").is_not_null())
    if dep.is_empty() or arr.is_empty():
        return p
    dep = dep.with_columns(pl.Series("k", [fix(x) for x in dep["flight_raw"].to_list()]))
    parts = []
    for key in ("k1", "k2"):
        parts.append(dep.join(arr.select(C(key).alias("k"), "arr_t", "lobt", "aobt"), on="k", how="inner"))
    j = pl.concat(parts)
    j = j.with_columns((C("arr_t") - C("t_mvt")).dt.total_seconds().alias("ft"), (C("t_mvt") - C("aobt")).dt.total_seconds().alias("ta"),
                       (C("t_mvt") - C("lobt")).dt.total_seconds().alias("gl2"), (C("aobt") - C("lobt")).dt.total_seconds().alias("a_l"))
    j = j.filter((C("ft") >= 1500) & (C("ft") <= 20000) & (C("ta") >= -120) & (C("ta") <= 7200))
    j = j.sort("id", "ft").unique(subset="id", keep="first", maintain_order=True)
    W = W_EXACT
    sel = j.filter(~((C("gl2") >= W) & ((C("gs") - C("gl2")).abs() < W)) & ~(C("gl2") < W) & ~(C("a_l").abs() < W))
    tgt = dict(zip(sel["id"].to_list(), np.clip(sel["ta"].to_numpy(), 0, None).tolist()))
    pos = ctx.inp["id"].to_list()
    new = np.array([tgt.get(i, v) for i, v in zip(pos, p)])
    return _fire(ctx, "rome_hidden_plan_late_clock", p, _round(new))


def paris_duplicate_pairs(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """At Paris, the plan record of a duplicate pair (same flight string and take-off date, partner without a plan within 1,800 s) is moved 75% toward the schedule gap less 750 s.

    From archive/agent_multipush_08_pairrule.py (pairs) and agent_multipush_19_build.py (value). 2025: 107 pairs, the plan record has the larger
    label on 101 (94.4%); leave-one-month-out +0.87 s on months 1 and 7, all 12 months better.
    """
    d = (ctx.inp.filter((C("apt") == "LFPG") & C("flight_raw").is_not_null())
         .select("id", C("flight_raw").alias("fl"), "day", "mvt", "plan", "gs").sort("fl", "day", "mvt"))
    d = d.with_columns(C("mvt").diff().over("fl", "day").alias("dm_p"), (-C("mvt").diff(-1).over("fl", "day")).alias("dm_n"),
                       C("plan").shift(1).over("fl", "day").alias("pl_p"), C("plan").shift(-1).over("fl", "day").alias("pl_n"),
                       pl.len().over("fl", "day").alias("n"))
    d = d.with_columns(((C("n") == 2) & (((C("dm_p") < 1800) & (C("pl_p") != C("plan"))) | ((C("dm_n") < 1800) & (C("pl_n") != C("plan"))))).fill_null(False).alias("pair"))
    q = d.filter(C("pair") & C("plan") & ((C("gs") - 780) <= 14400))
    if q.is_empty():
        return p
    pos = dict(zip(ctx.inp["id"].to_list(), range(ctx.inp.height)))
    new = p.copy()
    for i, g in zip(q["id"].to_list(), q["gs"].to_list()):
        k = pos[i]
        new[k] = np.rint(0.75 * np.clip(g - 750.0, 60.0, 14400.0) + 0.25 * p[k])
    return _fire(ctx, "paris_duplicate_pairs", p, new)


def rome_calendar_condition(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """A Rome departure without a plan, a schedule gap of 50,000 to 100,000 s and a take-off on the same Europe/Rome calendar day as its schedule is predicted at its gap.

    From archive/agent_bundle_v144_v145.py. 2025: 0 day-early stamps in about 159,600 same-day Rome departures; all 12 day-early records took off on a later local day.
    """
    inp = ctx.inp
    slip = (inp["t_mvt"].dt.convert_time_zone(TZ_ROME).dt.date() - inp["t_sched"].dt.convert_time_zone(TZ_ROME).dt.date()).dt.total_days().to_numpy()
    m = ((inp["apt"] == "LIRF") & ~inp["has_lobt"]).to_numpy() & (inp["gs"].to_numpy() >= 50000) & (inp["gs"].to_numpy() <= 100000) & (slip == 0)
    new = np.where(m, np.rint(inp["gs"].fill_null(0.0).to_numpy()), p)
    return _fire(ctx, "rome_calendar_condition", p, new)


def istanbul_noplan_bound(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """An Istanbul departure without a plan that took off at or after its schedule is predicted at most its schedule gap plus 1,200 s.

    From archive/agent_logicC_10_ltfm_noplan_bound.py. 2025: 3,585 rows, maximum excess over the gap 903 s, 0 above 1,200 s (all months, snow month included).
    """
    inp = ctx.inp
    m = ((inp["apt"] == "LTFM") & ~inp["plan"]).to_numpy() & (inp["gs"].fill_null(-1.0).to_numpy() >= 0)
    new = np.where(m, np.minimum(p, inp["gs"].fill_null(0.0).to_numpy() + 1200.0), p)
    return _fire(ctx, "istanbul_noplan_bound", p, _round(new))


def madrid_ceiling(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """A Madrid (and, as shipped, Barcelona) prediction is at most 3,606 s: the off-block stamp is not kept more than an hour before take-off.

    From archive/agent_logicC_12_madrid_ceiling.py. 2025: LEMD maximum 3,598 s in 212,242 rows, LEBL 3,604 s in 179,705 rows.
    The shipped code covers LEMD and LEBL; --madrid-only restricts it to LEMD as the documentation states.
    """
    apts = ["LEMD"] if ctx.madrid_only else ["LEMD", "LEBL"]
    m = ctx.inp["apt"].is_in(apts).to_numpy()
    return _fire(ctx, "madrid_ceiling", p, np.where(m, np.minimum(p, W_EXACT), p))


def rome_planband(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """A Rome plan row with a schedule gap of 70,000 to 100,000 s is predicted at take-off minus the plan's own off-block, inside the plan window.

    From archive/agent_logic_06_build.py (b). 2025: plan rows with gap of 50,000 s or more (436, nine airports): label within 600 s of this reading on 90.6%.
    """
    inp = ctx.inp
    gl, ga, gs = inp["gl"].to_numpy(), inp["ga"].to_numpy(), inp["gs"].to_numpy()
    m = ((inp["apt"] == "LIRF") & inp["plan"]).to_numpy() & ~np.isnan(gl) & (gs >= 70000) & (gs < 100000)
    W = ctx.w_exact
    new = np.where(m, np.clip(ga, gl - W, gl + W), p)
    return _fire(ctx, "rome_planband", p, _round(new))


def rome_hard_binding_window(ctx: Ctx, p: np.ndarray, w: float = 1.0) -> np.ndarray:
    """A Rome plan row whose lower plan edge exceeds 2,800 s and lies more than 1,800 s above the network manager's own reading is moved to take-off minus LOBT (weight w).

    From archive/agent_romenear_05_build_item1.py (weight 1.0: least squares 1.07 on 37 rows of 2025, capped at 1). 2025: 38 rows, label within 60 s of take-off minus LOBT on 21, none at the edge.
    """
    inp = ctx.inp
    gl, ga = inp["gl"].to_numpy(), inp["ga"].to_numpy()
    lo = gl - W_EXACT
    m = ((inp["apt"] == "LIRF") & inp["plan"]).to_numpy() & ~np.isnan(gl) & ~np.isnan(ga) & (lo > 2800.0) & (lo - ga > HARD_S)
    new = np.where(m, np.rint(np.clip(p + w * (gl - p), gl - ctx.w_exact, gl + ctx.w_exact)), p)
    return _fire(ctx, "rome_hard_binding_window", p, new)


def bounds_last(ctx: Ctx, p: np.ndarray) -> np.ndarray:
    """The three bounds that hold by construction are applied again at the end: schedule-gap clamp, plan window at the exact edge, hidden-plan lower bound.

    From archive/agent_logic_06_build.py (a), agent_logicB_20_build.py and agent_bundle_v158.py. Later additive terms push some predictions across them.
    """
    q = gap_clamp(ctx, p, name="bounds_last/gap_clamp")
    q = plan_window(ctx, q, exact=True, name="bounds_last/plan_window")
    q = hidden_plan_lower(ctx, q, name="bounds_last/hidden_plan_lower")
    return q


STAGES = {
    "pre": [rome_day_early_bands, rome_plan_gate],
    "post": [gap_clamp, plan_window, floor, habitual_copy_operators, hidden_plan_lower, hidden_plan_upper, hidden_plan_reading],
    "rows": [rome_hidden_plan_late_clock, paris_duplicate_pairs, rome_calendar_condition, istanbul_noplan_bound,
             madrid_ceiling, rome_planband, rome_hard_binding_window],
    "last": [bounds_last],
}


def apply_stages(ctx: Ctx, p: np.ndarray, stages: list[str], skip: set[str] | None = None) -> np.ndarray:
    skip = skip or set()
    for s in stages:
        print(f"stage {s}")
        for fn in STAGES[s]:
            if fn.__name__ in skip:
                continue
            p = fn(ctx, p)
    return p


def align(pred: pl.DataFrame, inp: pl.DataFrame) -> np.ndarray:
    """Prediction values in the row order of the inputs table; every input id must be present."""
    j = inp.select(C("id")).join(pred.select(C(ID_COL).alias("id"), C(TARGET_COL).cast(pl.Float64).alias("p")), on="id", how="left")
    if j["p"].null_count():
        raise SystemExit(f"{j['p'].null_count()} input ids have no prediction")
    return j["p"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, help="parquet with MVT_ID_mvt and TAXITIME_SEC_mvt (the prediction to start from)")
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--stage", default="post,rows,last", help="comma list of pre, post, rows, last")
    ap.add_argument("--skip", default="", help="comma list of rule names to skip")
    ap.add_argument("--months", default="", help="restrict to these UTC take-off months, e.g. 2,6")
    ap.add_argument("--sample", type=float, default=None, help="plumbing mode: a random fraction of the departures (rows without a prediction are dropped)")
    ap.add_argument("--jitter-allow", type=float, default=12.0)
    ap.add_argument("--madrid-only", action="store_true")
    ap.add_argument("--no-hard-binding", action="store_true")
    ap.add_argument("--gate-seeds", type=int, default=20, help="bagging seeds of the Rome plan gate (shipped 20; 1 for a plumbing test)")
    a = ap.parse_args()
    months = [int(t) for t in a.months.split(",") if t.strip()] or None
    ctx = Ctx(a.ranking, months, a.sample, a.jitter_allow, a.madrid_only, not a.no_hard_binding, gate_seeds=a.gate_seeds)
    pred = pl.read_parquet(a.pred) if a.pred else None
    if pred is None:
        sys.exit("--pred is required")
    pred = pred.filter(C(ID_COL).is_in(ctx.inp["id"].implode()))
    ctx.inp = ctx.inp.filter(C("id").is_in(pred[ID_COL].implode()))
    p0 = align(pred, ctx.inp)
    p = apply_stages(ctx, p0.copy(), [s for s in a.stage.split(",") if s], set(t for t in a.skip.split(",") if t))
    ctx.counters.print(months)
    print(f"rows {len(p):,}; moved overall {int((p != p0).sum()):,}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out = ctx.inp.select(C("id").alias(ID_COL)).with_columns(pl.Series(TARGET_COL, np.rint(p)).cast(pred.schema[TARGET_COL]))
        out.write_parquet(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
