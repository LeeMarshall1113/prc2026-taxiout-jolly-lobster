"""Post-process a submission for Rome's extreme schedule gaps (day-early and schedule-copy records).

    .venv/Scripts/python.exe archive/apply_rome_dayearly.py --in submissions/X.parquet --out submissions/Y.parquet

WHAT. At LIRF, in the schedule-gap bands where 2025 shows day-early records, replace the
prediction with the band's expected truth:

    E[y] = P(R) * (86,400 + offset) + P(S) * gap

where R = day-early BLOCK (y = 86,400 + an ordinary taxi) and S = BLOCK copied from SCHED
(y = gap). Bands and probabilities are rebuilt from 2025 at run time, never from the scored
set or the leaderboard:

    no-plan rows, gap 50,000-60,000 / 60,000-70,000 / 70,000-100,000
    plan rows,    gap 70,000-100,000   (pooled with no-plan: the only band where 2025 has
                                        a plan row, and it was ~87,000)

WHY ONLY THESE BANDS. drop_day_offsets deletes day-early records from TRAINING, correctly --
their label is corrupt -- so no model has ever seen one. Below 50,000s there are no R rows
in 2025 and the gate already predicts optimally (no-plan 25-50k: 100% substituted, gate
predicts the gap; plan rows 14.4-50k: 100% ordinary taxis, bulk model predicts ~1,200s).
Replacing the model there with a band average would throw away the row-level
discrimination it does have. Above 50,000s it has no information about R at all.

2025 CLASS MIX (LIRF): no-plan 50-60k S6/R3, 60-70k S2/R4, 70-100k S0/R4; plan 80-90k one
row, y = 87,002. A public 2025 analysis by another team gives the same table. Add-half
smoothing on the counts because every band is single digits.

EVIDENCE. The original single-band version (60-100k, P(R) 0.80) beat the gate in 5 of 7
months leave-one-month-out and read -2.45s on fold (1,7). In the January/July file it touches 6
rows; the gate is already near-right on all of them (no catastrophic miss), so the expected
gain is ~-1.5s: sum over rows of (current - E[y])^2. Full detail:
docs/LAB-NOTES.md, sections 10 and 11.
"""
import argparse
import sys

import numpy as np
import polars as pl

C = pl.col
DAY = 86400.0
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
# (lo, hi, applies to plan rows?)
BANDS = [(50000.0, 60000.0, False), (60000.0, 70000.0, False), (70000.0, 100000.0, True)]

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()


def lirf(src, labelled):
    sel = [C(ID), C("AOBT_3_flt").is_not_null().alias("plan"),
           (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap")]
    if labelled:
        sel += [(C("MVT_TIME_UTC_mvt") - C("BLOCK_TIME_UTC_mvt")).dt.total_seconds().alias("y"),
                ((C("BLOCK_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60).alias("s")]
    return (pl.scan_parquet(src).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF"))
            .select(sel).collect())


tr = lirf("data/raw/training_*.parquet", True).with_columns(
    C("y").is_between(DAY + 60, DAY + 5000).alias("r"))
r_all = tr.filter(C("r"))
if r_all.height < 5:
    sys.exit(f"only {r_all.height} day-early rows in 2025 -- too little to estimate anything")
offset = float(np.median(r_all["y"].to_numpy() - DAY))

table = []
print(f"2025 LIRF day-early offset: median {offset:,.0f}s over {r_all.height} rows -> R truth ~{DAY + offset:,.0f}s")
print(f"  {'band':<18}{'rows':>8}{'S':>4}{'R':>4}{'other':>7}{'P(R)':>7}{'P(S)':>7}   plan rows too")
for lo, hi, with_plan in BANDS:
    b = tr.filter(C("gap").is_between(lo, hi, closed="left"))
    if not with_plan:
        b = b.filter(~C("plan"))
    n_r = int(b["r"].sum())
    n_s = int((b["s"] & ~b["r"]).sum())
    other = b.height - n_r - n_s
    p_r = (n_r + 0.5) / (n_r + n_s + 1.0)          # add-half smoothing; 'other' is 0 in every band
    table.append((lo, hi, with_plan, p_r))
    print(f"  {f'{lo:,.0f}-{hi:,.0f}':<18}{b.height:>8}{n_s:>4}{n_r:>4}{other:>7}{p_r:>7.2f}{1 - p_r:>7.2f}   "
          f"{'yes' if with_plan else 'no'}")
    if other:
        print(f"    warning: {other} rows in this band are neither S nor R in 2025")

sub = pl.read_parquet(args.inp)
if set(sub.columns) != {ID, TARGET}:
    sys.exit(f"unexpected submission columns: {sub.columns}")
rank = lirf("data/raw/ranking.parquet", False)
j = sub.join(rank, on=ID, how="left")
if j.height != sub.height:
    sys.exit(f"join to ranking.parquet changed the row count: {j.height:,}")

gap = j["gap"].to_numpy().astype(float)
plan = j["plan"].to_numpy()
is_lirf = np.isfinite(gap)                 # non-LIRF rows found no match in the LIRF frame
before = j[TARGET].to_numpy().astype(float)
after = before.copy()
hit = np.zeros(len(before), bool)
for lo, hi, with_plan, p_r in table:
    m = is_lirf & (gap >= lo) & (gap < hi)
    if not with_plan:
        m &= ~plan.astype(bool) if plan.dtype == bool else (plan == False)  # noqa: E712
    after[m] = p_r * (DAY + offset) + (1 - p_r) * gap[m]
    hit |= m

print(f"\n{int(hit.sum())} of {sub.height:,} scored departures changed:")
print(f"  {'gap':>9}{'plan':>6}{'before':>10}{'after':>10}{'moved':>9}")
order = np.argsort(gap[hit])
for g, p_, b, a in zip(gap[hit][order], plan[hit][order], before[hit][order], after[hit][order]):
    print(f"  {g:>9,.0f}{('yes' if p_ else 'no'):>6}{b:>10,.0f}{a:>10,.0f}{a - b:>+9,.0f}")
exp_gain = float(((before[hit] - after[hit]) ** 2).sum())
base_mse = 271.969 ** 2
print(f"  expected squared-error reduction {exp_gain:.3e} s^2 -> about "
      f"{np.sqrt(max(base_mse - exp_gain / sub.height, 0)) - 271.969:+.2f}s at our current level "
      f"(expectation under the 2025 class mix; the realised figure depends on 6 rows)")

# Round, as final.py does (np.rint then int32). A bare cast truncates toward zero.
out = sub.with_columns(pl.Series(TARGET, np.rint(after)).cast(sub.schema[TARGET]))
assert out.height == sub.height and out[ID].equals(sub[ID])
assert (out[TARGET] >= 0).all() and out[TARGET].null_count() == 0
changed = int((out[TARGET] != sub[TARGET]).sum())
assert changed <= int(hit.sum()), f"changed {changed} rows, but only {int(hit.sum())} were targeted"
out.write_parquet(args.out)
print(f"\nwrote {args.out}: {out.height:,} rows, {changed} changed, all other rows identical")
