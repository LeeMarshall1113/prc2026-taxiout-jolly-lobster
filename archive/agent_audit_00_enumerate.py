"""AUDIT STEP 1 -- ENUMERATE. What per-row quantity exists at serve time, and how is it aggregated?

Read-only, light (polars lazy aggregates only). Prints:

  1. every raw column, its null share on 2025 DEPARTURES, 2025 ARRIVALS at the ten airports, and
     2026 ranking departures/arrivals -- so a quantity that looks available but is withheld at serve
     time (BLOCK_TIME/TAXITIME on departures) is visible before any feature reads it;
  2. the degenerate-quantity check: for each clock difference, the share exactly zero and the share
     on a whole minute, per airport -- a quantity that is constant (e.g. AOBT_3 == EOBT_1 at an
     airport that writes the plan into the actual) cannot carry a neighbour signal there;
  3. the arrival-side clocks at the ten airports;
  4. the (quantity x aggregation) grid, hand-compiled from prc/features.py, with the cells that
     already exist marked and whether they SHIP (final.DROP minus the v26 --keep list).

    PYTHONPATH=. .venv/Scripts/python.exe archive/agent_audit_00_enumerate.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import polars as pl

C = pl.col
TEN = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
TRAIN = "data/raw/training_*.parquet"
RANK = "data/raw/ranking.parquet"


def secs(a, b):
    return (C(a) - C(b)).dt.total_seconds()


def two_lines(t, key, fmt):
    rows = list(t.iter_rows(named=True))
    for part in (rows[:5], rows[5:]):
        print("      " + "  ".join(fmt(r) for r in part))


def main() -> None:
    cols = pl.scan_parquet(RANK).collect_schema().names()
    print("=" * 100)
    print("1. NULL SHARE BY COLUMN -- 2025 training vs 2026 ranking, departures and arrivals separately")
    print(f"   {'column':<24}{'2025 DEP':>10}{'2025 ARR':>10}{'2026 DEP':>10}{'2026 ARR':>10}")
    res = {}
    for tag, src in (("2025", TRAIN), ("2026", RANK)):
        for ph in ("DEP", "ARR"):
            key = "ADEP_mvt" if ph == "DEP" else "ADES_mvt"
            q = (pl.scan_parquet(src).filter((C("PHASE_mvt") == ph) & C(key).is_in(TEN))
                 .select([C(c).is_null().mean().alias(c) for c in cols]).collect())
            res[(tag, ph)] = q.row(0, named=True)
    for c in cols:
        print(f"   {c:<24}" + "".join(f"{res[k][c]:>10.3f}" for k in
                                     (("2025", "DEP"), ("2025", "ARR"), ("2026", "DEP"), ("2026", "ARR"))))

    print()
    print("=" * 100)
    print("2. CLOCK DIFFERENCES ON 2025 DEPARTURES: z = share exactly 0, m = share on a whole minute,")
    print("   median and IQR in seconds, per airport")
    diffs = {
        "AOBT3-EOBT1 (dep_delay)": secs("AOBT_3_flt", "EOBT_1_flt"),
        "MVT-AOBT3 (gap_aobt)": secs("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
        "MVT-SCHED (gap_sched)": secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        "SCHED-EOBT1 (sched_vs_eobt)": secs("SCHED_TIME_UTC_mvt", "EOBT_1_flt"),
        "LOBT-SCHED": secs("LOBT_flt", "SCHED_TIME_UTC_mvt"),
        "EOBT1-LOBT": secs("EOBT_1_flt", "LOBT_flt"),
        "IOBT-LOBT": secs("IOBT_flt", "LOBT_flt"),
        "AOBT3-LOBT": secs("AOBT_3_flt", "LOBT_flt"),
        "ARVT3-ARVT1": secs("ARVT_3_flt", "ARVT_1_flt"),
        "ARVT1-EOBT1 (planned g2g)": secs("ARVT_1_flt", "EOBT_1_flt"),
        "ARVT3-AOBT3 (NM g2g)": secs("ARVT_3_flt", "AOBT_3_flt"),
    }
    dep = pl.scan_parquet(TRAIN).filter((C("PHASE_mvt") == "DEP") & C("ADEP_mvt").is_in(TEN))
    for name, e in diffs.items():
        t = (dep.select(C("ADEP_mvt"), e.alias("d")).drop_nulls("d")
             .group_by("ADEP_mvt").agg((C("d") == 0).mean().alias("zero"),
                                        ((C("d") % 60) == 0).mean().alias("minute"),
                                        C("d").median().alias("med"),
                                        (C("d").quantile(0.75) - C("d").quantile(0.25)).alias("iqr"))
             .sort("ADEP_mvt").collect())
        print(f"   {name}")
        two_lines(t, "ADEP_mvt", lambda r: f"{r['ADEP_mvt']}: z{r['zero']:.2f} m{r['minute']:.2f} "
                                           f"med{r['med']:>6.0f} iqr{r['iqr']:>6.0f}")
    t = (dep.select(C("ADEP_mvt"), (C("AOBT_3_flt").dt.second() == 0).alias("a0"),
                    (C("MVT_TIME_UTC_mvt").dt.second() == 0).alias("m0"))
         .group_by("ADEP_mvt").agg(C("a0").mean(), C("m0").mean()).sort("ADEP_mvt").collect())
    print("   seconds-of-minute == 0 share, AOBT_3 / MVT:")
    two_lines(t, "ADEP_mvt", lambda r: f"{r['ADEP_mvt']} {r['a0']:.2f}/{r['m0']:.2f}")

    print()
    print("=" * 100)
    print("3. ARRIVALS AT THE TEN (2025): the clocks an arrival row carries, at the DESTINATION airport")
    print("   median [p10, p90] seconds")
    arr = pl.scan_parquet(TRAIN).filter((C("PHASE_mvt") == "ARR") & C("ADES_mvt").is_in(TEN))
    adiffs = {
        "BLOCK-MVT (taxi-in)": secs("BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"),
        "BLOCK-SCHED (on-block delay)": secs("BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        "MVT-SCHED (landing vs sched)": secs("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
        "MVT-ARVT1 (landing vs first est.)": secs("MVT_TIME_UTC_mvt", "ARVT_1_flt"),
        "MVT-ARVT3": secs("MVT_TIME_UTC_mvt", "ARVT_3_flt"),
        "AOBT3-EOBT1 at origin": secs("AOBT_3_flt", "EOBT_1_flt"),
        "(MVT-AOBT3)-(ARVT1-EOBT1) airborne excess": (secs("MVT_TIME_UTC_mvt", "AOBT_3_flt")
                                                      - secs("ARVT_1_flt", "EOBT_1_flt")),
    }
    for name, e in adiffs.items():
        t = (arr.select(C("ADES_mvt"), e.alias("d")).drop_nulls("d")
             .group_by("ADES_mvt").agg(C("d").median().alias("med"),
                                        C("d").quantile(0.1).alias("p10"),
                                        C("d").quantile(0.9).alias("p90"))
             .sort("ADES_mvt").collect())
        print(f"   {name}")
        two_lines(t, "ADES_mvt", lambda r: f"{r['ADES_mvt']}:{r['med']:>6.0f} [{r['p10']:>6.0f},{r['p90']:>6.0f}]")
    t = (arr.select(C("ADES_mvt"), (C("BLOCK_TIME_UTC_mvt").dt.second() == 0).alias("b0"))
         .group_by("ADES_mvt").agg(C("b0").mean()).sort("ADES_mvt").collect())
    print("   arrival BLOCK seconds == 0 share:")
    two_lines(t, "ADES_mvt", lambda r: f"{r['ADES_mvt']} {r['b0']:.2f}")

    print()
    print("=" * 100)
    print("4. THE (QUANTITY x AGGREGATION) GRID, compiled by reading prc/features.py (2026-09-21).")
    print("   S = exists and SHIPS (v26 model, reused through v41);  b = built in features.py but in")
    print("   final.DROP and NOT shipped;  L = the late_* family under fold test tonight;")
    print("   . = never built (a candidate cell);  dead = closed in the ledger;  - = not meaningful")
    rows = [
        ("taxi proxy MVT-AOBT3", "S", "S", "S", ".", ".", "b", ".", "dead"),
        ("lateness AOBT3-EOBT1", "S", "L", "L", ".", ".", ".", ".", "dead"),
        ("takeoff delay MVT-SCHED", "S", ".", ".", ".", ".", ".", ".", "dead"),
        ("MVT-EOBT1 (=taxi+late)", "S", "dup", "dup", ".", ".", ".", ".", "dead"),
        ("MVT-LOBT", "S", ".", ".", ".", ".", ".", ".", "dead"),
        ("SCHED-EOBT1", "S", ".", ".", ".", ".", ".", ".", "dead"),
        ("LOBT-SCHED", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("EOBT1-LOBT", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("IOBT-LOBT", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("AOBT3-LOBT", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("AOBT3-SCHED", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("ARVT3-ARVT1", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("ARVT3-AOBT3 (NM g2g)", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("AOBT3 on :00 (clk_sec)", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("MVT on :00", "b", ".", ".", ".", ".", ".", ".", "dead"),
        ("dep count", "-", "S", "S", ".", ".", ".", ".", "dead"),
        ("sched demand", "-", "S", ".", ".", ".", ".", ".", "dead"),
        ("heavy share", "b", ".", "b", ".", ".", ".", ".", "dead"),
        ("no-plan share", "S", "dead", ".", ".", ".", ".", ".", "dead"),
        ("ARR taxi-in", "-", "S/b", ".", ".", ".", "dead", "S", "dead"),
        ("ARR on-block delay", "-", ".", ".", ".", ".", ".", "b", "dead"),
        ("ARR landing delay", "-", ".", ".", ".", ".", ".", ".", "dead"),
        ("ARR airborne excess", "-", ".", ".", ".", ".", ".", ".", "dead"),
        ("ARR origin lateness", "-", ".", ".", ".", ".", ".", ".", "dead"),
        ("ARR substituted", "-", "b(day)", ".", ".", ".", "b(day)", ".", "b"),
        ("ARR BLOCK on :00", "-", ".", ".", ".", ".", ".", ".", "dead"),
        ("ARR count", "-", "S", "S(flag)", ".", ".", ".", ".", "dead"),
        ("turn: AOBT3 - inbound on-block", ".", "-", "-", "-", "-", "-", ".", "-"),
    ]
    hdr = ["quantity", "per-row", "apt win", "rwy win", "op win", "dest win", "stand", "turn", "day"]
    print("   " + f"{hdr[0]:<32}" + "".join(f"{h:>9}" for h in hdr[1:]))
    for r in rows:
        print("   " + f"{r[0]:<32}" + "".join(f"{x:>9}" for x in r[1:]))


if __name__ == "__main__":
    main()
