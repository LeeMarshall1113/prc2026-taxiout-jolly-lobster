"""Reproduction test for final/30_general_rules.py and final/32_published_daily.py on the January/July 2026 chain.

Each step takes the stored file that precedes a rule in the shipped lineage (data/interim/agent_bundle_v*_final.parquet,
the stage output of the post-chain), applies the new function with the shipped settings (jitter allowance 0, ranking file
data/raw/ranking.parquet) and compares with the stored file that follows it. The comparison is by id; it prints the maximum
absolute difference and the differing rows. A second block compares the rows each rule owns with submissions/jolly-lobster_v162.parquet.

  python final/39_rules_repro_test.py [--steps a,b,c]

Inputs only: no 2026 departure label is read. 2025 labels enter through the constants the functions rebuild.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
I = REPO / "data" / "interim"
RANK = REPO / "data" / "raw" / "ranking.parquet"


def load(name: str):
    spec = importlib.util.spec_from_file_location("mod_" + name[:2], HERE / name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def rd(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path).sort(ID)


def compare(label: str, got: pl.DataFrame, want: pl.DataFrame, apt: pl.DataFrame, mask_ids: set | None = None) -> dict:
    """got/want: id, value. Prints rows that differ; mask_ids restricts to the rows a step owns."""
    j = got.join(want, on=ID, suffix="_w").join(apt, on=ID, how="left")
    if mask_ids is not None:
        j = j.filter(C(ID).is_in(list(mask_ids)))
    d = (j[T].cast(pl.Float64) - j[T + "_w"].cast(pl.Float64))
    nz = j.with_columns(d.alias("d")).filter(C("d") != 0)
    print(f"  {label}: rows compared {j.height:,}, max |diff| {float(d.abs().max() or 0):.1f} s, differing rows {nz.height}")
    if nz.height:
        print(nz.select("ADEP_mvt", ID, T, T + "_w", "d").sort(C("d").abs(), descending=True).head(15))
    return dict(label=label, rows=j.height, maxdiff=float(d.abs().max() or 0), differing=nz.height)


def step_paris_pairs(g, apt):
    base = rd(I / "agent_bundle_v136_final.parquet")
    want = rd(I / "agent_multipush_final.parquet")
    ctx = g.Ctx(RANK, jitter_allow=0.0, with_hidden=False)
    p = base[T].cast(pl.Float64).to_numpy()
    out = g.paris_duplicate_pairs(ctx, p)
    got = base.with_columns(pl.Series(T, out))
    # pairs owned by the rule: rows the stored piece changed
    owned = set(want.join(base, on=ID, suffix="_b").filter(C(T) != C(T + "_b"))[ID].to_list())
    print(f"  stored piece changed {len(owned)} rows; rule fired on {int((out != p).sum())}")
    return compare("paris_duplicate_pairs vs agent_multipush_final", got, want, apt)


def step_detrules(g, apt):
    base = rd(I / "agent_bundle_v136_final.parquet")
    want = rd(I / "agent_detrules_final.parquet")
    ctx = g.Ctx(RANK, jitter_allow=0.0)
    p = base[T].cast(pl.Float64).to_numpy()
    out = g.hidden_plan_upper(ctx, p)
    got = base.with_columns(pl.Series(T, out))
    return compare("hidden_plan_upper vs agent_detrules_final", got, want, apt)


def step_lever_b(g, apt):
    base = rd(I / "agent_bundle_v135_final.parquet")
    want = rd(I / "agent_bundle_v136_final.parquet")
    ctx = g.Ctx(RANK, jitter_allow=0.0, with_hidden=False)
    p = base[T].cast(pl.Float64).to_numpy()
    out = g.rome_hidden_plan_late_clock(ctx, p)
    got = base.with_columns(pl.Series(T, out))
    return compare("rome_hidden_plan_late_clock vs v136", got, want, apt)


def step_calendar(g, apt):
    base = rd(I / "agent_bundle_v144_final.parquet")
    want = rd(I / "agent_bundle_v145_final.parquet")
    ctx = g.Ctx(RANK, jitter_allow=0.0, with_hidden=False)
    p = base[T].cast(pl.Float64).to_numpy()
    out = g.rome_calendar_condition(ctx, p)
    return compare("rome_calendar_condition vs v145", base.with_columns(pl.Series(T, out)), want, apt)


def step_v146(g, apt):
    base = rd(I / "agent_bundle_v145_final.parquet")
    want = rd(I / "agent_bundle_v146_final.parquet")
    ctx = g.Ctx(RANK, jitter_allow=0.0)
    p = base[T].cast(pl.Float64).to_numpy()
    q = g.istanbul_noplan_bound(ctx, p)
    q = g.madrid_ceiling(ctx, q)
    q = g.rome_planband(ctx, q)
    q = g.bounds_last(ctx, q)
    return compare("six bounds (30) vs v146", base.with_columns(pl.Series(T, q)), want, apt)


def step_v148(g, apt):
    base = rd(I / "agent_bundle_v146_final.parquet")
    want = rd(I / "agent_romenear_hardbind_final.parquet")
    ctx = g.Ctx(RANK, jitter_allow=0.0, with_hidden=False)
    p = base[T].cast(pl.Float64).to_numpy()
    q = g.rome_hard_binding_window(ctx, p)
    return compare("rome_hard_binding_window vs agent_romenear_hardbind_final", base.with_columns(pl.Series(T, q)), want, apt)


def step_v158(g, apt):
    base = rd(I / "agent_bundle_v156_final.parquet")
    want = rd(I / "agent_bundle_v158_final.parquet")
    ctx = g.Ctx(RANK, jitter_allow=0.0, with_hidden=False)
    p = base[T].cast(pl.Float64).to_numpy()
    q = g.plan_window(ctx, p, exact=True, name="plan_window_exact")
    return compare("plan_window (exact edge) vs v158", base.with_columns(pl.Series(T, q)), want, apt)


def step_postchain(g, apt):
    """Counts of the post-chain on the stage output named in docs/BUILD-TRACE.md section 1a."""
    f = REPO / "data" / "interim" / "sb2serve" / "ta_v105_s1.parquet"
    if not f.exists():
        print("  (stage output ta_v105_s1.parquet not on disk)")
        return None
    base = rd(f)
    ctx = g.Ctx(RANK, jitter_allow=0.0)
    p = base[T].cast(pl.Float64).to_numpy()
    out = g.apply_stages(ctx, p.copy(), ["post"])
    ctx.counters.print()
    print("  documented counts (BUILD-TRACE 1a): gap clamp 39, window 82 raised (40 lifted), floor 1,307, habitual 6 + lower bound 9, upper bound 8 lowered, reading 18")
    return None


def step_owned_v162(g, apt):
    """Rows each general rule owns, compared with the last uploaded file."""
    v162 = rd(REPO / "submissions" / "jolly-lobster_v162.parquet")
    ctx = g.Ctx(RANK, jitter_allow=0.0)
    p = v162[T].cast(pl.Float64).to_numpy()
    for fn in (g.paris_duplicate_pairs, g.rome_hidden_plan_late_clock, g.istanbul_noplan_bound, g.madrid_ceiling, g.rome_planband,
               g.rome_calendar_condition, g.rome_hard_binding_window, g.hidden_plan_upper, g.hidden_plan_reading, g.bounds_last):
        q = fn(ctx, p.copy())
        mv = np.flatnonzero(q != p)
        if len(mv):
            print(f"     largest moves of {fn.__name__}: " + ", ".join(f"{p[k]:.0f}->{q[k]:.0f}" for k in mv[np.argsort(-np.abs(q[mv] - p[mv]))[:6]]))
    return None


def _daily(pieces, airports, base_path, want_path, label, own_rows=None, window=3606.0, fixes=False, **kw):
    spec = importlib.util.spec_from_file_location("mod_32", HERE / "32_published_daily.py")
    m32 = importlib.util.module_from_spec(spec)
    sys.modules["mod_32"] = m32
    spec.loader.exec_module(m32)
    base = rd(base_path)
    want = rd(want_path)
    out, notes = m32.run(base, RANK, pieces=pieces, months=None, sample=None, lam_cap=None, own_bias=False, guard=False, level_airports=airports, window=window, **kw)
    for n in notes:
        print("  note:", n)
    got = base.join(out.rename({T: "n"}), on=ID, how="left").with_columns(pl.coalesce(C("n"), C(T)).cast(base.schema[T]).alias(T)).select(ID, T)
    if fixes:
        spec31 = importlib.util.spec_from_file_location("mod_31", HERE / "31_scored_row_fixes.py")
        m31 = importlib.util.module_from_spec(spec31)
        spec31.loader.exec_module(m31)
        got = m31.apply_fixes(base.select(ID, T), got, RANK)
        label += " + row fixes (31)"
    return compare(label, got, want, APT_TABLE[0], mask_ids=own_rows)


APT_TABLE = []


def step_level6(g, apt):
    APT_TABLE[:] = [apt]
    return _daily({"level"}, ["EDDF", "EDDM", "EGLL", "LEBL", "LEMD", "LIRF"], I / "agent_bundle_v150_final.parquet", I / "agent_dayeq_level_final.parquet", "daily level, six airports vs agent_dayeq_level_final", window=None)


def step_types(g, apt):
    APT_TABLE[:] = [apt]
    return _daily({"types"}, [], I / "agent_bundle_v150_final.parquet", I / "agent_dayeq_types_final.parquet", "rome record types vs agent_dayeq_types_final", window=None, fixes=True)


def step_ltfm(g, apt):
    APT_TABLE[:] = [apt]
    ids = set(apt.filter(C("ADEP_mvt") == "LTFM")[ID].to_list())
    return _daily({"ltfm"}, ["LTFM"], I / "agent_bundle_v158_final.parquet", I / "agent_bundle_v159_final.parquet", "daily level Istanbul vs v159 (Istanbul rows)", ids, fixes=True)


def step_lszh(g, apt):
    APT_TABLE[:] = [apt]
    ids = set(apt.filter(C("ADEP_mvt") == "LSZH")[ID].to_list())
    return _daily({"lszh"}, ["LSZH"], I / "agent_bundle_v159_final.parquet", I / "agent_bundle_v160_final.parquet", "daily level Zurich IFR vs v160 (Zurich rows)", ids)


def step_lfpg(g, apt):
    APT_TABLE[:] = [apt]
    ids = set(apt.filter(C("ADEP_mvt") == "LFPG")[ID].to_list())
    return _daily({"lfpg"}, ["LFPG"], I / "agent_bundle_v160_final.parquet", REPO / "submissions" / "jolly-lobster_v161.parquet", "daily level Paris vs v161 (Paris rows)", ids, fixes=True)


def step_cells(g, apt):
    APT_TABLE[:] = [apt]
    ids = set(apt.filter(C("ADEP_mvt").is_in(["LFPG", "EGLL", "EDDF", "EDDM", "LIRF"]))[ID].to_list())
    return _daily({"level", "lfpg"}, ["EDDF", "EDDM", "EGLL", "LIRF", "LFPG"], I / "agent_bundle_v150_final.parquet", REPO / "submissions" / "jolly-lobster_v162.parquet",
                  "cell split chain at five airports (v150, level, Paris, cells) vs v162", ids, fixes=True, cell_split=True, census=I / "agent_coverage_census_2026.parquet")


def step_capped_jj(g, apt):
    """January/July rebuilt with the capped form (lam <= 0.5, own-month bias, snow days, rule hedge, null flights updated) from the pre-layer file v152,
    next to the shipped form on the same file: rows moved and mean / rms difference per airport."""
    spec = importlib.util.spec_from_file_location("mod_32", HERE / "32_published_daily.py")
    m32 = importlib.util.module_from_spec(spec)
    sys.modules["mod_32"] = m32
    spec.loader.exec_module(m32)
    base = rd(I / "agent_bundle_v152_final.parquet")
    air = ["EDDF", "EDDM", "EGLL", "LEBL", "LEMD", "LIRF", "LTFM", "LFPG"]
    pieces = {"level", "types", "ltfm", "lfpg"}
    ship, _ = m32.run(base, RANK, pieces=pieces, months=None, sample=None, lam_cap=None, own_bias=False, guard=False, level_airports=air, window=3606.0)
    cap, notes = m32.run(base, RANK, pieces=pieces, months=None, sample=None, lam_cap=0.5, own_bias=True, guard=False, level_airports=air, window=3606.0,
                         rule_hedge=True, update_null_flight=True, stop_days="signature")
    for n in notes:
        print("  note:", n)
    j = (base.rename({T: "b"}).join(ship.rename({T: "ship"}), on=ID).join(cap.rename({T: "cap"}), on=ID).join(apt, on=ID)
         .with_columns((C("ship") - C("b")).alias("m_ship"), (C("cap") - C("b")).alias("m_cap"), (C("cap") - C("ship")).alias("diff")))
    out = (j.group_by("ADEP_mvt").agg(rows=pl.len(), moved_shipped=(C("m_ship") != 0).sum(), moved_capped=(C("m_cap") != 0).sum(),
                                     rms_move_shipped=(C("m_ship") ** 2).mean().sqrt(), rms_move_capped=(C("m_cap") ** 2).mean().sqrt(),
                                     mean_diff=C("diff").mean(), rms_diff=(C("diff") ** 2).mean().sqrt()).sort("ADEP_mvt")
           .with_columns(pl.col(pl.Float64).round(2)))
    print(out)
    return None


STEPS = dict(lever_b=step_lever_b, paris=step_paris_pairs, detrules=step_detrules, calendar=step_calendar, v146=step_v146,
             v148=step_v148, v158=step_v158, postchain=step_postchain, owned=step_owned_v162,
             level6=step_level6, types=step_types, ltfm=step_ltfm, lszh=step_lszh, lfpg=step_lfpg, cells=step_cells, capped_jj=step_capped_jj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=",".join(STEPS))
    a = ap.parse_args()
    g = load("30_general_rules.py")
    apt = pl.read_parquet(RANK, columns=[ID, "ADEP_mvt", "PHASE_mvt"]).filter(C("PHASE_mvt") == "DEP").select(ID, "ADEP_mvt")
    res = []
    for s in a.steps.split(","):
        print(f"== {s}")
        r = STEPS[s](g, apt)
        if r:
            res.append(r)
    print("\nsummary")
    for r in res:
        print(f"  {r['label']:70s} rows {r['rows']:,}  max|diff| {r['maxdiff']:.1f}  differing {r['differing']}")


if __name__ == "__main__":
    main()
