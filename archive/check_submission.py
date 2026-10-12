"""Compare a newly built submission with the last one, before any upload decision.

    .venv/Scripts/python.exe archive/check_submission.py submissions/<old>.parquet submissions/<new>.parquet

Checks the new file against the template (every ID exactly once, no nulls, no
negatives), then says how many predictions changed and WHERE: Rome no-plan
rows, other no-plan rows, flight-plan rows. A change meant to touch one pocket
that moves rows everywhere is not the change it claims to be. Also prints the
Rome no-plan rows with gap_sched above 60,000s -- the handful the board's tail
rides on -- before and after, and the largest single changes.
"""
import sys

import numpy as np
import polars as pl

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
old_p, new_p = sys.argv[1], sys.argv[2]
old, new = pl.read_parquet(old_p), pl.read_parquet(new_p)
tpl = pl.read_parquet("data/raw/submitting.parquet", columns=[ID])

problems = []
if set(new.columns) != {ID, T}:
    problems.append(f"columns {new.columns}")
if new.height != tpl.height:
    problems.append(f"{new.height:,} rows against a {tpl.height:,}-row template")
if new[ID].n_unique() != new.height:
    problems.append("duplicate IDs")
if set(new[ID].to_list()) != set(tpl[ID].to_list()):
    problems.append("ID set differs from the template")
if new[T].null_count():
    problems.append(f"{new[T].null_count():,} null predictions")
if (new[T] < 0).sum():
    problems.append(f"{int((new[T] < 0).sum()):,} negative predictions")
print(f"{new_p}: {new.height:,} rows, dtype {new[T].dtype} -- " + ("template checks pass" if not problems else "PROBLEMS: " + "; ".join(problems)))

r = (pl.read_parquet("data/raw/ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP")
     .select(ID, "ADEP_mvt", pl.col("AOBT_3_flt").is_null().alias("noplan"),
             (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs")))
j = (old.rename({T: "old"}).join(new.rename({T: "new"}), on=ID, how="inner").join(r, on=ID, how="inner"))
d = (j["new"] - j["old"]).to_numpy().astype(float)
lirf = j["ADEP_mvt"].to_numpy() == "LIRF"
noplan = j["noplan"].to_numpy()
segments = (("Rome no-plan", lirf & noplan), ("other no-plan", ~lirf & noplan), ("flight-plan rows", ~noplan))
print(f"\n  {'segment':<18}{'rows':>9}{'changed':>9}{'mean |change|':>15}{'max |change|':>14}")
for name, k in segments:
    ch = k & (d != 0)
    print(f"  {name:<18}{int(k.sum()):>9,}{int(ch.sum()):>9,}{np.abs(d[k]).mean():>14,.1f}s{np.abs(d[k]).max() if k.any() else 0:>13,.0f}s")

sig = lirf & noplan & (np.nan_to_num(j["gs"].to_numpy().astype(float)) > 60000)
print("\n  Rome no-plan rows with gap_sched > 60,000s")
for gs, o, n in zip(j["gs"].to_numpy()[sig], j["old"].to_numpy()[sig], j["new"].to_numpy()[sig]):
    print(f"    gap_sched {gs:>9,.0f}   old {o:>8,}   new {n:>8,}   change {n - o:>+8,}")

top = np.argsort(-np.abs(d))[:8]
print("\n  largest single changes")
for i in top:
    seg = next(name for name, k in segments if k[i])
    print(f"    {seg:<18} old {j['old'][int(i)]:>8,}  new {j['new'][int(i)]:>8,}  change {d[i]:>+9,.0f}")
