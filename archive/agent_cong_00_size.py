"""SIZING, before either congestion feature is built. Read-only, no training, no slot.

TWO CANDIDATE FEATURES, both verified absent from prc/features.py:

  A. AIRPORT-WIDE TAXI SPEED from ADS-B in a trailing WALL-CLOCK window, as a congestion
     signal (Wang et al., TR-C 2021: average speed is one of five features that generalise
     across Zurich, Manchester and Hong Kong). `gs` is in every points file and is read only
     by prc/adsb_match.py, per flight, for anchoring / dwell / the taxi floor. Nothing
     aggregates it across aircraft.
     It escapes the CLOSED queue family (ledger: "any feature that counts events between
     off-block and wheels-up is a re-reading of gap_aobt") only if the window is a FIXED
     wall-clock length that does not depend on this flight's own off-block -> wheels-up
     interval. That is a design constraint, not a hope, and it is honoured in step 01.

  B. NEIGHBOURHOOD MEAN OF OFF-BLOCK LATENESS. `_nm_neighbours` aggregates MVT - AOBT_3 (a
     taxi proxy) over 15/60-minute windows; `dep_delay` = AOBT_3 - EOBT_1 exists per row and
     is never aggregated over neighbours. Needs no ADS-B, so it reaches every airport.

WHAT THIS SCRIPT DOES, and why it runs first. The working constraint on A is that ADS-B is
blind at LIRF, LFPG and LTFM, where most of the remaining error lives. "Blind" has never
been quantified for the AIRPORT-WIDE aggregate -- every coverage number in the lab record is
about matching a PARTICULAR departure to its own track, which is a far stricter requirement
than "are there any other aircraft visible taxiing here right now". So:

  1. share of squared error by airport, pooled over the five folds, through production's
     chain (clip0 -> gap clamp -> LOBT box) on out-of-fold predictions;
  2. ADS-B ground coverage per airport per month in 2025 -- distinct aircraft seen taxiing
     per hour, which is what an airport-wide speed needs;
  3. the same for the 62 extracted 2026 scored days, because a feature that cannot be
     computed at serve time is worth nothing whatever it measures in 2025;
  4. the resulting BOUND on A: share of squared error at airports where the aggregate is
     computable on both sides.

PRE-DECLARED BARS FOR THE WHOLE PROBE (fixed here, before any number is looked at).

  Falsifier, inherited from the runway-queue longshot's placebo: partial |rho| against the
  chained residual must exceed 0.019. That is what ZERO reads as in this instrument.

  B1  best construction reaches |partial rho| >= 0.019 pooled on eligible rows, after the
      controls, with the SAME SIGN on folds (1,7) and (2,8).
  B2  its own placebo -- identical arithmetic, window contents borrowed from a different
      flight at the same airport and month -- comes in at LESS THAN HALF the real figure.
  B3  implied pooled RMSE reduction from a linear use of it is >= 1.0s on eligible rows.
      Deliberately a low bar: "is there anything here at all", not "is it worth shipping".
  B4  concentration: the implied gain still stands after deleting the top 1, 5, 20 and 100
      rows it helps most. Reported for both; a reversal at top-100 is a FAIL of B4 alone.
  Positive control: inject a synthetic signal sized to exactly -1.000s and -0.250s and
  confirm the instrument returns it. Without that, "we measured nothing" and "our instrument
  measures nothing" are the same reading.

  B1 fails  -> dead. Report, no slot, no fold run.
  B1+B2 but not B3 -> real and too small. Report, no slot.
  B1+B2+B3  -> file a slot and run a real fold test; B4 decides how it is reported.

  SIZING ARITHMETIC used for B3, from the project's standing numbers: scored rows,
  1.67e8 s^2 of squared error per board second, bulk-model changes survive to the board at
  0.38x (docs/LAB-NOTES.md).

    PYTHONPATH=. .venv/Scripts/python.exe archive/agent_cong_00_size.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, W = "MVT_ID_mvt", 3606.0
POINTS = Path("data/adsb/points")

FOLDS = [("1_7", "results/preds_q_gate_1_7.parquet", False),
         ("2_8", "results/preds_q_gate_2_8.parquet", False),
         ("3_9", "results/preds_n3_bulk_3_9.parquet", True),
         ("4_10", "results/preds_s4_ctrl_4_10.parquet", True),
         ("5_11", "results/preds_c_ctrl_5_11.parquet", True)]

# a taxiing fix: on the ground, moving, but not on the take-off roll
TAXI_LO, TAXI_HI = 3.0, 40.0


def chain(bagged, gs, gl, plan):
    """production's chain, lifted verbatim from archive/screen_rule.py."""
    p = np.clip(bagged, 0, None)
    p = np.where(np.isfinite(gs) & (gs >= 5000) & (gs < 50000) & (p > gs), gs, p)
    box = plan & np.isfinite(gl)
    p = np.where(box, np.clip(p, np.where(box, gl - W, -np.inf), np.where(box, gl + W, np.inf)), p)
    return np.clip(p, 0, None)


def main() -> None:
    dep = (pl.scan_parquet("data/raw/training_*.parquet")
           .filter(C("PHASE_mvt") == "DEP")
           .select(C(ID),
                   C("ADEP_mvt").alias("adep"),
                   C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
                   C("AOBT_3_flt").is_not_null().alias("plan"),
                   (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                   (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"))
           .collect())
    ids = dep[ID].to_numpy()
    adep = dep["adep"].to_numpy().astype(str)
    gs = dep["gs"].fill_null(np.nan).to_numpy().astype(float)
    gl = dep["gl"].fill_null(np.nan).to_numpy().astype(float)
    plan = dep["plan"].fill_null(False).to_numpy()
    pos = {m: i for i, m in enumerate(ids)}

    print("=" * 96)
    print("1. WHERE THE SQUARED ERROR SITS, pooled over five folds, through production's chain")
    print("   (out-of-fold `bagged` -> clip0 -> gap clamp -> LOBT box; LIRF taken from the two")
    print("    gate-inclusive folds only, exactly as screen_rule.py does)")
    sse: dict[str, float] = {}
    rows: dict[str, int] = {}
    for label, path, drop_lirf in FOLDS:
        P = pl.read_parquet(path).select(ID, "y", "bagged")
        loc = np.array([pos[m] for m in P[ID].to_numpy()])
        y = P["y"].to_numpy().astype(float)
        base = chain(P["bagged"].to_numpy().astype(float), gs[loc], gl[loc], plan[loc])
        a = adep[loc]
        keep = np.ones(len(a), dtype=bool) if not drop_lirf else (a != "LIRF")
        for apt in np.unique(a[keep]):
            m = keep & (a == apt)
            sse[apt] = sse.get(apt, 0.0) + float(((y[m] - base[m]) ** 2).sum())
            rows[apt] = rows.get(apt, 0) + int(m.sum())
    tot = sum(sse.values())
    print(f"   {'apt':<6} {'rows':>9} {'RMSE':>8} {'share of sq err':>16}")
    for apt in sorted(sse, key=lambda k: -sse[k]):
        print(f"   {apt:<6} {rows[apt]:>9,} {np.sqrt(sse[apt] / rows[apt]):>8.1f} "
              f"{100 * sse[apt] / tot:>15.2f}%")

    print()
    print("=" * 96)
    print("2. ADS-B GROUND COVERAGE FOR AN AIRPORT-WIDE AGGREGATE")
    print("   Per airport-month: hours in which at least MIN_AC distinct aircraft were seen")
    print(f"   taxiing ({TAXI_LO:.0f}-{TAXI_HI:.0f} kt, on-ground flag), as a share of the month's hours.")
    print("   This is a much weaker requirement than matching a departure to its own track,")
    print("   which is what every coverage figure in the lab record measures.")
    days = sorted(p.stem for p in POINTS.glob("*.parquet"))
    d2025 = [d for d in days if d.startswith("2025")]
    d2026 = [d for d in days if d.startswith("2026")]
    print(f"   {len(d2025)} extracted 2025 days, {len(d2026)} extracted 2026 days")

    for tag, dayset in (("2025", d2025), ("2026 scored days", d2026)):
        acc: dict[tuple[str, str], list[int]] = {}
        for d in dayset:
            p = (pl.scan_parquet(POINTS / f"{d}.parquet")
                 .filter((C("alt") == -1) & C("gs").is_between(TAXI_LO, TAXI_HI))
                 .select(C("apt"), (C("t") // 3600).cast(pl.Int64).alias("hr"), C("icao"))
                 .unique()
                 .group_by("apt", "hr").agg(pl.len().alias("ac"))
                 .collect())
            ym = d[:7]
            for apt, ac in zip(p["apt"].to_list(), p["ac"].to_list()):
                acc.setdefault((apt, ym), [0, 0, 0])
                acc[(apt, ym)][0] += 1                     # hours with any aircraft
                acc[(apt, ym)][1] += 1 if ac >= 3 else 0   # hours with >= 3
                acc[(apt, ym)][2] += ac                    # total aircraft-hours
        months = sorted({k[1] for k in acc})
        apts = sorted({k[0] for k in acc} | set(sse))
        print(f"\n   --- {tag}: share of the month's 24*ndays hours with >= 3 taxiing aircraft ---")
        hdr = "   " + f"{'apt':<6}" + "".join(f"{m[2:]:>8}" for m in months)
        print(hdr)
        ndays = {m: sum(1 for d in dayset if d[:7] == m) for m in months}
        for apt in apts:
            line = f"   {apt:<6}"
            for m in months:
                v = acc.get((apt, m))
                line += f"{'  -  ':>8}" if v is None else f"{100 * v[1] / (24 * ndays[m]):>7.0f}%"
            print(line)

    print()
    print("=" * 96)
    print("3. THE BOUND ON FEATURE A")
    covered = set()
    for d in d2026:
        p = (pl.scan_parquet(POINTS / f"{d}.parquet")
             .filter((C("alt") == -1) & C("gs").is_between(TAXI_LO, TAXI_HI))
             .select(C("apt"), (C("t") // 3600).cast(pl.Int64).alias("hr"), C("icao"))
             .unique().group_by("apt", "hr").agg(pl.len().alias("ac"))
             .filter(C("ac") >= 3).group_by("apt").agg(pl.len().alias("hrs")).collect())
        for apt, hrs in zip(p["apt"].to_list(), p["hrs"].to_list()):
            if hrs >= 8:
                covered.add(apt)
    inside = sum(v for k, v in sse.items() if k in covered)
    print(f"   airports with >= 8 usable hours on at least one 2026 day: {sorted(covered)}")
    print(f"   share of 2025 fold squared error at those airports: {100 * inside / tot:.1f}%")
    print(f"   share at the rest (the cap):                          {100 * (1 - inside / tot):.1f}%")
    print("   Feature B has no such cap: it is competition data and reaches all ten.")


if __name__ == "__main__":
    main()
