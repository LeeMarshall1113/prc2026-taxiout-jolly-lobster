"""Final assembly of the submitted file from the intermediate prediction files, and a check of the result.

    python archive/reproduce_final.py              # assemble, compare with the stored final file, run the label-free checks
    python archive/reproduce_final.py --out X.parquet   # also write the assembled file
    python archive/reproduce_final.py --check-file submissions/jolly-lobster_v156.parquet   # only the label-free checks on one file

WHAT THIS IS. The submitted file (upload v156, the best scored file) was never produced by one command. The model fits (base, ADS-B stage,
re-stack, residual corrector, Istanbul models) are expensive, partly seeded and partly run on rented machines; they are
described in docs/BUILD-TRACE.md and docs/PIPELINE.md and are NOT re-run here. What this script does is the last part of the
chain, from the file of upload v132 to the file of upload v156: assembly steps, each of the form
"base + (component - reference)" (DELTA), "base + a sum of deltas" (DELTAS) or "base with the rows a component changed replaced
by the component's values" (REPLACE). Inputs are files in data/interim/ that the earlier stages wrote (they are derived from the
competition data and are not part of the repository); nothing is fitted, no label is read, the ranking file is read only for the
airport code and the input columns of the label-free checks.

THE STEPS, IN ORDER (roles as in docs/BUILD-TRACE.md section 2)
  base   agent_bundle_v132_final       the file of upload v132: core model, Istanbul views corrector, tail component
  v133   REPLACE  agent_rome2_A_final    Rome no-plan mid-gap mixture at half weight (259 rows)
                  agent_fix2_c_final     callsign-keyed reach of the hidden-plan reading (11 rows)
  v135   REPLACE  agent_rome3_final      Rome no-plan mixture, round 2: new copy indicators, per-bucket weights (the same 259 rows)
  v136   REPLACE  agent_rome3B_final     hard rule for five Rome hidden-plan rows (take-off minus the plan's own off-block)
  v137   REPLACE  agent_multipush_final  Paris duplicate-record rule (16 rows)
                  agent_detrules_final   hidden-plan upper bound by callsign (6 rows)
  v141   DELTA    agent_mega_final       main residual corrector refitted at seven airports ...
                  minus agent_mega_no4_final   ... minus the same build with the shipped corrector (reference)
  v142   DELTAS   (A) agent_mega_no4_final minus agent_basefix_final       certain bounds re-applied last, Rome plan-gate extra
                                                                          weight 0.1, Istanbul corrector dose 1.0 on non-snow rows
                  (B) agent_regime_gatejul_final minus the v141 file      Rome plan-gate extra weight 0.2 on July with-plan rows
                                                                          not seen on the ground; where B moves a row, A's delta
                                                                          on that row is dropped
                  (C) agent_detrules_final minus agent_bundle_v136_final  hidden-plan upper bound, only on rows v141 left at the
                                                                          v136 value (no row: the bound was already in v141)
                  (D) agent_mega_refit12_delta minus agent_mega_refit_delta   corrector serve term from 12 seeds instead of 6
                  The float deltas are summed on the v141 file, rounded once, clipped at zero.
  v143   DELTAS   (E) agent_pcd_refit12_delta minus agent_mega_refit12_delta  corrector refitted with nine plan-clock distance
                                                                          columns, 12 serve seeds, same seven airports and doses
                  (F) agent_rome1c_final minus the v141 file              Rome with-plan two-sided copy mixture, dose 0.5
                  The float deltas are summed on the v142 file, rounded once, clipped at zero.
  v144   DELTAS   (H) agent_apt_EDDM_final minus the v143 file       Munich January level shift removed (7,062 rows)
                  (I) agent_pcdx_final minus the v143 file           Amsterdam on the plan-clock corrector term (31,254 rows)
                  The two row sets are disjoint (asserted); the deltas are summed on the v143 file, rounded once, clipped at zero.
  v145   RULE     Rome calendar condition: LIRF no-plan departures with schedule gap 50,000-100,000 s whose take-off is on the same
                  Europe/Rome local day as the schedule take the schedule gap (one scored row). Defined on input columns only.
  v146   BOUNDS   six piece files (data/interim/agent_logicC_LTFM_noplan-early-bound, agent_logicC_LEMD_ceiling3606, agent_logic_planband,
                  agent_logic_bounds, agent_logicB_window, agent_logicB_hiddenlower; 51 rows in all), each equal to v145 except where its
                  bound applies; where two move one row the value farthest from v145 is taken (they must agree in direction).
                  Each is a bound on input columns; the rule and its 2025 count are in docs/ROW-RULES.md.
  v148   PIECE   agent_romenear_hardbind_final  Rome plan rows whose plan window binds hard served at take-off minus LOBT (14 scored rows);
                 the stored piece is v146 plus this rule, so the step adds the piece's difference from v146 to v146.
  v150   DELTA   agent_arith_final minus v146   residual corrector refitted with 52 arithmetic columns at six airports (189,076 rows);
                 applied on v148, row sets disjoint (asserted). The piece is a stored output of the serve fits (rented CPU, docs/REPRODUCE.md).
  v152   PIECE   agent_arith3_final_noG         corrector round three (45 further columns without the view-geometry group) at seven airports,
                 stored as v150 with the corrector term of those airports replaced; v152 is that file. The step asserts it equals v150 outside
                 the seven airports. (Not in the chain: v147, v149, v151 and v157, which lost and were not carried forward.)
  v153   DELTA   agent_dayeq_level_final minus v150   day-level calibration of delayed flights from the published daily delay identity
                 (six airports, 139,724 rows), on v152. Uses published figures about the scored months (docs/PIPELINE.md section 5).
  v154   DELTA   agent_dayeq_types_final minus v150   Rome no-plan record-type posteriors from the same identity (168 rows), on v152.
  v155   BOTH    v153's change with the rows of the types piece set to their v154 values; arithmetic of archive/agent_bundle_dayeq.py.
  v156   DELTA   agent_istnext_final minus v150       Istanbul corrector with the nine-airport column lists on Istanbul cells (42,538 non-snow rows), on v155.
The REPLACE components were each built on an earlier file of the same lineage, so a component's rows are "the rows where it
differs from that earlier file": for v135 and v136 that earlier file is agent_bundle_v133_final (as in the build scripts);
for the others it is the running base. Row sets of components applied in one step must be disjoint (asserted). The v142 and v143
steps are the arithmetic of archive/agent_bundle_v142.py and archive/agent_bundle_v143.py, which sort by id; the assembled file
is therefore in id order from v142 on, and every comparison below is made by id.

At the end each intermediate is compared with its stored file (data/interim/agent_bundle_v133_final ... v146_final), and the files of v143 to v146
with data/interim/agent_bundle_v14N_final.parquet and with submissions/jolly-lobster_v14N.parquet (when present): the
maximum absolute difference per airport and the number of rows that differ. The expected result is zero everywhere.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
I = ROOT / "data" / "interim"
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
C = pl.col


def load(stem: str) -> pl.DataFrame:
    p = I / f"{stem}.parquet"
    if not p.exists():
        sys.exit(f"missing intermediate file {p} (these are produced by the earlier stages; see docs/BUILD-TRACE.md)")
    return pl.read_parquet(p)


def load_npz(stem: str):
    p = I / f"{stem}.npz"
    if not p.exists():
        sys.exit(f"missing intermediate file {p} (these are produced by the earlier stages; see docs/BUILD-TRACE.md)")
    return np.load(p)


def same_ids(a: pl.DataFrame, b: pl.DataFrame, name: str):
    assert a.height == b.height and (a[ID] == b[ID]).all(), f"{name}: id order differs from the base"


def replace_step(base: pl.DataFrame, comps: list, ref: pl.DataFrame, label: str) -> pl.DataFrame:
    """REPLACE: rows where a component differs from `ref` take the component's value; the row sets must be disjoint."""
    masks = []
    for name, df in comps:
        same_ids(df, base, name)
        masks.append((name, df[T] != ref[T]))
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            assert int((masks[i][1] & masks[j][1]).sum()) == 0, f"{label}: overlap between {masks[i][0]} and {masks[j][0]}"
    out_vals = base[T].to_numpy().copy()
    for (name, m), (_, df) in zip(masks, comps):
        mm = m.to_numpy()
        out_vals[mm] = df[T].to_numpy()[mm]
    out = base.with_columns(pl.Series(T, out_vals).cast(base[T].dtype))
    parts = ", ".join(f"{name}: {int(m.sum())} rows" for name, m in masks)
    print(f"  {label:5s} REPLACE   {parts}; changed vs running base {int((out[T] != base[T]).sum())}")
    return out


def delta_step(base: pl.DataFrame, comp: pl.DataFrame, ref: pl.DataFrame, label: str) -> pl.DataFrame:
    """DELTA: base + (component - reference), whole seconds, clipped at zero (as archive/agent_v14z_build.py)."""
    if not (base[ID] == ref[ID]).all():
        base = base.join(ref.select(ID).with_row_index("o"), on=ID).sort("o").drop("o")
    assert (base[ID] == ref[ID]).all() and (comp[ID] == ref[ID]).all()
    i = lambda s: s.cast(pl.Int64)  # noqa: E731
    d = i(comp[T]) - i(ref[T])
    out = base.with_columns((i(base[T]) + d).clip(0, None).cast(base[T].dtype).alias(T))
    print(f"  {label:5s} DELTA     component - reference nonzero on {int((d != 0).sum())} rows "
          f"(mean {d.filter(d != 0).mean():+.1f} s, min {d.min()}, max {d.max()}); changed vs base {int((out[T] != base[T]).sum())}")
    return out


def deltas_step(base: pl.DataFrame, parts: list, label: str) -> pl.DataFrame:
    """DELTAS: base + the sum of float deltas, rounded once to whole seconds, clipped at zero (the arithmetic of the v142 and v143 build scripts)."""
    b = base.sort(ID)
    cur = b[T].cast(pl.Float64).to_numpy()
    new = np.rint(np.maximum(cur + sum(d for _, d in parts), 0))
    for n, d in parts:
        m = d != 0
        print(f"  {label:5s} {n}: rows {int(m.sum())}, mean {d[m].mean() if m.any() else 0:+.2f}, "
              f"rms {np.sqrt((d[m] ** 2).mean()) if m.any() else 0:.2f}, max |d| {np.abs(d).max():.1f}")
    print(f"  {label:5s} DELTAS    rows changed vs base {int((new != cur).sum())}, mean shift {(new - cur).mean():+.3f} s, "
          f"rms {np.sqrt(((new - cur) ** 2).mean()):.2f}, max |move| {np.abs(new - cur).max():.0f}")
    return b.with_columns(pl.Series(T, new).cast(b[T].dtype))


def v142_step(b41: pl.DataFrame) -> pl.DataFrame:
    """v142 = v141 + (A) low-risk pieces + (B) Rome plan-gate weight on July rows + (C) hidden-plan bound + (D) 12 serve seeds."""
    b = b41.sort(ID)
    ids, cur = b[ID].to_numpy(), b[T].cast(pl.Float64).to_numpy()

    def col(stem):
        d = load(stem).sort(ID)
        assert (d[ID].to_numpy() == ids).all(), stem
        return d[T].cast(pl.Float64).to_numpy()
    dA = col("agent_mega_no4_final") - col("agent_basefix_final")
    dB = col("agent_regime_gatejul_final") - cur
    v136 = col("agent_bundle_v136_final")
    dC = col("agent_detrules_final") - v136
    dC[cur != v136] = 0.0
    z6, z12 = load_npz("agent_mega_refit_delta"), load_npz("agent_mega_refit12_delta")
    assert (z6["ids"] == ids).all() and (z12["ids"] == ids).all()
    dD = z12["dR"] - z6["dR"]
    dA[dB != 0] = 0.0
    return deltas_step(b, [("A passengers        ", dA), ("B gate July         ", dB),
                           ("C hidden-plan bound ", dC), ("D 12 serve seeds    ", dD)], "v142")


def v143_step(b42: pl.DataFrame, b41: pl.DataFrame) -> pl.DataFrame:
    """v143 = v142 + (E) corrector refitted with plan-clock distance columns + (F) Rome with-plan two-sided mixture."""
    b = b42.sort(ID)
    ids = b[ID].to_numpy()
    zp, zs = load_npz("agent_pcd_refit12_delta"), load_npz("agent_mega_refit12_delta")
    assert (zp["ids"] == ids).all() and (zs["ids"] == ids).all()
    dE = zp["dR"] - zs["dR"]
    rome, v41 = load("agent_rome1c_final").sort(ID), b41.sort(ID)
    assert (rome[ID].to_numpy() == ids).all() and (v41[ID].to_numpy() == ids).all()
    dF = rome[T].cast(pl.Float64).to_numpy() - v41[T].cast(pl.Float64).to_numpy()
    print(f"  v143  rows moved by both E and F: {int(((dE != 0) & (dF != 0)).sum())}")
    return deltas_step(b, [("E plan-clock corrector ", dE), ("F Rome with-plan mixture", dF)], "v143")


def v144_step(b43: pl.DataFrame) -> pl.DataFrame:
    """v144 = v143 + (H) Munich January level shift removed + (I) Amsterdam on the plan-clock corrector term (arithmetic of archive/agent_bundle_v144_v145.py)."""
    b = b43.sort(ID)
    ids, cur = b[ID].to_numpy(), b[T].cast(pl.Float64).to_numpy()

    def col(stem):
        d = load(stem).sort(ID)
        assert (d[ID].to_numpy() == ids).all(), stem
        return d[T].cast(pl.Float64).to_numpy()
    dH, dI = col("agent_apt_EDDM_final") - cur, col("agent_pcdx_final") - cur
    assert int(((dH != 0) & (dI != 0)).sum()) == 0, "v144: Munich and Amsterdam pieces overlap"
    return deltas_step(b, [("H Munich January shift removed", dH), ("I Amsterdam plan-clock term   ", dI)], "v144")


def v145_step(b44: pl.DataFrame) -> pl.DataFrame:
    """v145 = v144 + calendar condition on the Rome day-early band rule: no-plan LIRF departures with gap 50,000-100,000 s whose take-off is on
    the SAME local (Europe/Rome) day as the schedule take the schedule gap. Defined on input columns only."""
    b = b44.sort(ID)
    r = (pl.scan_parquet(ROOT / "data" / "raw" / "ranking.parquet")
         .filter((C("ADEP_mvt") == "LIRF") & (C("PHASE_mvt") == "DEP") & C("LOBT_flt").is_null())
         .select(ID, "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").collect()
         .with_columns(gap=(C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                       slip=(C("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date()
                             - C("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date()).dt.total_days())
         .filter(C("gap").is_between(50000, 100000) & (C("slip") == 0)))
    pos = {int(i): k for k, i in enumerate(b[ID].to_numpy())}
    vals = b[T].cast(pl.Float64).to_numpy().copy()
    for x in r.iter_rows(named=True):
        vals[pos[int(x[ID])]] = round(x["gap"])
    print(f"  v145  calendar rule: rows set to the schedule gap {r.height}, changed vs v144 {int((vals != b[T].cast(pl.Float64).to_numpy()).sum())}")
    return b.with_columns(pl.Series(T, vals).cast(b[T].dtype))


V146_PIECES = ["agent_logicC_LTFM_noplan-early-bound_final", "agent_logicC_LEMD_ceiling3606_final", "agent_logic_planband_final",
               "agent_logic_bounds_final", "agent_logicB_window_final", "agent_logicB_hiddenlower_final"]


def v146_step(b45: pl.DataFrame) -> pl.DataFrame:
    """v146 = v145 + six row-level bounds (51 rows). Each piece file equals v145 except where its bound applies; where two pieces move one row
    the value farthest from v145 is taken (all pieces are bounds; the pieces must point the same way). Arithmetic of archive/agent_bundle_v146.py."""
    b = b45.sort(ID)
    out = b.select(ID, C(T).alias("v145"))
    for p in V146_PIECES:
        d = load(p).sort(ID)
        assert (d[ID] == b[ID]).all(), p
        out = out.with_columns(d[T].alias(p))
    vals, nconf = [], 0
    for r in out.iter_rows(named=True):
        ch = {p: r[p] for p in V146_PIECES if r[p] != r["v145"]}
        if not ch:
            vals.append(r["v145"])
            continue
        v = set(ch.values())
        if len(v) > 1:
            nconf += 1
            up, dn = [x for x in v if x > r["v145"]], [x for x in v if x < r["v145"]]
            assert not (up and dn), "pieces disagree in direction"
            vals.append(max(up) if up else min(dn))
        else:
            vals.append(v.pop())
    res = b.with_columns(pl.Series(T, vals).cast(b[T].dtype))
    print("  v146  " + ", ".join(f"{p.replace('agent_', '').replace('_final', '')}: {int((out[p] != out['v145']).sum())}" for p in V146_PIECES))
    print(f"  v146  rows changed vs v145 {int((res[T] != b[T]).sum())}, rows moved by more than one piece {nconf}")
    return res


def piece_delta(base: pl.DataFrame, piece: str, ref_stem: str, label: str) -> pl.DataFrame:
    """base + (piece - reference), where the piece's rows must not already have been moved in the base (the arithmetic of
    archive/agent_bundle_v150_v151.py and archive/agent_bundle_v156_v157.py: join by id, add the integer difference)."""
    p = load(piece).rename({T: "p"})
    r = load(ref_stem).rename({T: "r"})
    j = base.join(p, on=ID).join(r, on=ID).with_columns((C("p") - C("r")).alias("d"))
    assert j.height == base.height
    ov = j.filter((C("d") != 0) & (C(T) != C("r"))).height
    assert ov == 0, f"{label}: {ov} rows of {piece} were already moved in the base"
    out = base.select(ID).join(j.select(ID, (C(T) + C("d")).cast(base.schema[T]).alias(T)), on=ID, how="left")
    assert (out[ID] == base[ID]).all() and int((out[T] < 0).sum()) == 0
    d = out[T] - base[T]
    print(f"  {label:5s} PIECE     {piece}: rows changed {int((d != 0).sum())}, mean {float(d.mean()):+.3f}, max {int(d.max())}, min {int(d.min())}")
    return out


def v152_step(b150: pl.DataFrame) -> pl.DataFrame:
    """v152 = v150 with the corrector term of seven airports replaced by the round-three term without the view-geometry group
    (stored result of archive/agent_arith3_07_build.py, suffix _noG; logs/auto_v152.log names it). Asserted: equal to v150 outside those airports."""
    f = load("agent_arith3_final_noG")
    b = b150
    assert (f[ID] == b[ID]).all()
    apt = pl.scan_parquet(ROOT / "data" / "raw" / "ranking.parquet").select(C(ID), C("ADEP_mvt").alias("apt")).collect()
    j = b.join(apt, on=ID, how="left", maintain_order="left").with_columns(f[T].alias("f"))
    outside = j.filter(~C("apt").is_in(["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LFPG", "LEMD"]))
    assert int((outside["f"] != outside[T]).sum()) == 0, "round-three file differs from v150 outside its seven airports"
    print(f"  v152  PIECE     agent_arith3_final_noG: rows changed vs v150 {int((f[T] != b[T]).sum())}; equal to v150 outside the seven airports")
    return f


def dayeq_steps(b152: pl.DataFrame) -> dict:
    """v153, v154, v155 from v152 by the arithmetic of archive/agent_bundle_dayeq.py: A = base + level change, B = base + types change,
    AB = A with the rows of the types piece set to their B values (the posterior supersedes the level shift on those rows)."""
    b = b152
    r = load("agent_bundle_v150_final").rename({T: "r"})
    lv = load("agent_dayeq_level_final").rename({T: "lv"})
    ty = load("agent_dayeq_types_final").rename({T: "ty"})
    j = b.join(r, on=ID).join(lv, on=ID).join(ty, on=ID).with_columns((C("lv") - C("r")).alias("dl"), (C("ty") - C("r")).alias("dt"))
    j = j.with_columns((C(T) + C("dl")).alias("A"), (C(T) + C("dt")).alias("B"))
    j = j.with_columns(pl.when(C("dt") != 0).then(C("B")).otherwise(C("A")).alias("AB"))
    assert (j[ID] == b[ID]).all()
    res = {}
    for v, col in ((153, "A"), (154, "B"), (155, "AB")):
        res[v] = j.select(ID, pl.max_horizontal(C(col), pl.lit(0)).round().cast(b.schema[T]).alias(T))
        d = res[v][T] - b[T]
        print(f"  v{v}  DAYEQ     ({col}) rows changed vs v152 {int((d != 0).sum())}, mean {float(d.mean()):+.3f}, max {int(d.max())}, min {int(d.min())}")
    print("  v155  rows in both pieces:", j.filter((C("dl") != 0) & (C("dt") != 0)).height)
    return res


def vs_stored(df: pl.DataFrame, stem: str):
    """Report whether an intermediate result equals the stored file of the same upload (compared by id, when that file is present)."""
    p = I / f"{stem}.parquet"
    if p.exists():
        o = pl.read_parquet(p)
        j = df.select(ID, C(T).cast(pl.Int64).alias("a")).join(o.select(ID, C(T).cast(pl.Int64).alias("b")), on=ID, how="left")
        print(f"        equals stored {stem}: {bool(o.height == df.height and j['b'].null_count() == 0 and (j['a'] == j['b']).all())}")


def assemble() -> dict:
    print("assembly (docs/BUILD-TRACE.md section 2):")
    b32 = load("agent_bundle_v132_final")
    b33 = replace_step(b32, [("rome2_A", load("agent_rome2_A_final")), ("fix2_c", load("agent_fix2_c_final"))], b32, "v133")
    vs_stored(b33, "agent_bundle_v133_final")
    b35 = replace_step(b33, [("rome3", load("agent_rome3_final"))], b33, "v135")
    vs_stored(b35, "agent_bundle_v135_final")
    # v136: the component (rome3B) was built on v133; its rows are the rows where it differs from v133; applied on v135
    b36 = replace_step(b35, [("rome3B", load("agent_rome3B_final"))], b33, "v136")
    vs_stored(b36, "agent_bundle_v136_final")
    b37 = replace_step(b36, [("multipush", load("agent_multipush_final")), ("detrules", load("agent_detrules_final"))], b36, "v137")
    vs_stored(b37, "agent_bundle_v137_final")
    b41 = delta_step(b37, load("agent_mega_final"), load("agent_mega_no4_final"), "v141")
    vs_stored(b41, "agent_bundle_v141_final")
    b42 = v142_step(b41)
    vs_stored(b42, "agent_bundle_v142_final")
    b43 = v143_step(b42, b41)
    vs_stored(b43, "agent_bundle_v143_final")
    b44 = v144_step(b43)
    vs_stored(b44, "agent_bundle_v144_final")
    b45 = v145_step(b44)
    vs_stored(b45, "agent_bundle_v145_final")
    b46 = v146_step(b45)
    vs_stored(b46, "agent_bundle_v146_final")
    b48 = piece_delta(b46, "agent_romenear_hardbind_final", "agent_bundle_v146_final", "v148")
    vs_stored(b48, "agent_bundle_v148_final")
    b50 = piece_delta(b48, "agent_arith_final", "agent_bundle_v146_final", "v150")
    vs_stored(b50, "agent_bundle_v150_final")
    b52 = v152_step(b50)
    vs_stored(b52, "agent_bundle_v152_final")
    dq = dayeq_steps(b52)
    for v in (153, 154, 155):
        vs_stored(dq[v], f"agent_bundle_v{v}_final")
    b56 = piece_delta(dq[155], "agent_istnext_final", "agent_bundle_v150_final", "v156")
    vs_stored(b56, "agent_bundle_v156_final")
    return {143: b43, 144: b44, 145: b45, 146: b46, 148: b48, 150: b50, 152: b52, 153: dq[153], 154: dq[154], 155: dq[155], 156: b56}


def compare(out: pl.DataFrame, other: pl.DataFrame, name: str, apt: pl.DataFrame):
    if not (out[ID] == other[ID]).all():
        other = other.join(out.select(ID).with_row_index("o"), on=ID).sort("o").drop("o")
    assert out.height == other.height and (out[ID] == other[ID]).all()
    d = (out[T].cast(pl.Int64) - other[T].cast(pl.Int64)).abs()
    tab = (out.select(ID).with_columns(d.alias("d")).join(apt, on=ID, how="left", maintain_order="left")
           .group_by("apt").agg(C("d").max().alias("max_abs_diff"), (C("d") > 0).sum().alias("rows_differing"), pl.len().alias("rows"))
           .sort("apt"))
    print(f"\ncomparison with {name}: maximum absolute difference {int(d.max())} s over {out.height:,} rows; rows differing {int((d > 0).sum())}")
    print(tab)
    return int(d.max())


def checks(out: pl.DataFrame):
    """Label-free checks on the assembled file: ids, no negatives, and the two bounds that hold by construction on the database."""
    print("\nlabel-free checks on the assembled file (inputs only: schedule, take-off time, plan clocks):")
    rk = (pl.scan_parquet(ROOT / "data" / "raw" / "ranking.parquet").filter(C("PHASE_mvt") == "DEP")
          .select(C(ID), C("ADEP_mvt").alias("apt"), C("AOBT_3_flt").is_not_null().alias("plan"),
                  (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap"),
                  (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl")).collect())
    j = out.join(rk, on=ID, how="left", maintain_order="left")
    p = j[T].to_numpy().astype(float)
    print(f"  rows {out.height:,}; ids unique {out[ID].n_unique() == out.height}; ids all in the ranking file {int(j['apt'].null_count()) == 0}")
    print(f"  negative predictions: {int((p < 0).sum())}")
    g = j["gap"].fill_null(-1).to_numpy().astype(float)
    v = (g >= 5000) & (g < 50000) & (p > g + 1)
    print(f"  gap clamp (5,000 <= gap < 50,000 s): predictions above the gap: {int(v.sum())}")
    gl = j["gl"].to_numpy().astype(float)
    plan = j["plan"].fill_null(False).to_numpy() & ~np.isnan(gl)
    lo, hi = gl - 3606, gl + 3606
    out_box = plan & ((p < lo - 14) | (p > hi + 14))
    print(f"  plan-clock window (+-3,606 s, 14 s slack): plan rows {int(plan.sum()):,}, predictions outside: {int(out_box.sum())}"
          + ("" if not out_box.any() else " (by airport: " + ", ".join(f"{a} {int(n)}" for a, n in
             j.filter(pl.Series(out_box)).group_by('apt').len().sort('apt').iter_rows()) + ")"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="also write the assembled file here (parquet)")
    ap.add_argument("--check-file", help="run only the label-free checks on this file (needs data/raw/ranking.parquet, no intermediates)")
    ap.add_argument("--no-checks", action="store_true", help="skip the label-free checks (they read data/raw/ranking.parquet)")
    a = ap.parse_args()
    if a.check_file:
        checks(pl.read_parquet(a.check_file))
        return
    steps = assemble()
    out = steps[156]
    apt = pl.scan_parquet(ROOT / "data" / "raw" / "ranking.parquet").select(C(ID), C("ADEP_mvt").alias("apt")).collect()
    worst = 0
    for v in (143, 144, 145, 146, 148, 150, 152, 153, 154, 155, 156):
        worst = max(worst, compare(steps[v], load(f"agent_bundle_v{v}_final"), f"data/interim/agent_bundle_v{v}_final.parquet", apt))
        sub = ROOT / "submissions" / f"jolly-lobster_v{v}.parquet"
        if sub.exists():
            worst = max(worst, compare(steps[v], pl.read_parquet(sub), f"submissions/jolly-lobster_v{v}.parquet", apt))
    if not a.no_checks:
        checks(out)
    if a.out:
        out.write_parquet(a.out)
        print(f"\nwrote {a.out}")
    print("\nRESULT:", "identical to the stored final file" if worst == 0 else f"DIFFERENT (maximum absolute difference {worst} s)")
    sys.exit(0 if worst == 0 else 1)


if __name__ == "__main__":
    main()
