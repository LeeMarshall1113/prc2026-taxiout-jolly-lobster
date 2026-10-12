"""experiment npadsb 02: does wiring `air_cs_all` into the no-plan DERIVED-CALLSIGN match recover more rows?

CODE FINDING (read directly, section 2 of docs/LAB-NOTES.md): `air_cs_all` (every airborne
callsign in the window, not just the first) is wired into PLAN-row matching in both copies of the code
that have it (archive/agent_postchain_adsb_match_aircs.py, archive/agent_postchain_07_stage1_eval_aircs.py's
match_departures_ext). Neither wires it into `noplan_callsign` (no-plan rows matched on the callsign
DERIVED from FLIGHT_mvt, Gates flag, currently OFF; K13 in docs/LAB-NOTES.md measured it alone
at -0.03s / 55 more matches on 30 days, a run that PREDATES air_cs_all entirely). Both copies explicitly
`.drop("cs_all")` before building the no-plan `free`/`keys` frame. Since the mechanism that made air_cs_all
worth +52% EDDF / +141% LFPG matches for plan rows (stale ground callsigns, correct callsign only heard
later airborne) applies identically to a no-plan aircraft's OWN airborne callsign, this is untested.

Three arms on the SAME `free` (runs not claimed by a plan match, air_cs_all computed once per day):
  A  baseline: time (120s) + type only (production default, noplan_callsign=False)
  B  + derived callsign via ground callsign / first-airborne callsign only (current noplan_callsign=True
     implementation, reproducing K13 minus the days/window this uses)
  C  + derived callsign via ground / first-airborne / EVERY airborne callsign in the window (NEW)
B and C's callsign match uses g.match_s (600s), matching prc/adsb_match.py's no-plan-callsign branch, not
the narrower noplan_match_s (120s) used for time+type.

Precision check (no stand map needed): departure_floor()'s hard floor against the label y, the same
raise-only-floor precision test moredata used to validate FB_A/FB_B (violation = y < lb - 60).
Match-rate / estimator-kind sizing (needs the stand map + estimate()) is deferred to a follow-up if this
passes the precision bar -- see the note's running log.

    OMP_NUM_THREADS=4 .venv/Scripts/python.exe archive/agent_npadsb_02_csall_noplan.py
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, ".")
from archive.agent_postchain_adsb_match_aircs import (Gates, add_derived_callsign, departure_floor,
                                                        flight_callsign_maps, runs_and_tracks)

C = pl.col
pl.Config.set_tbl_rows(30)

DAYS = sorted(pl.read_parquet("results/agent_moredata_funnel_2025.parquet")["day"].unique().to_list())
print(f"{len(DAYS)} days: {DAYS[0]}..{DAYS[-1]}")

G = Gates(air_cs_all=True, air_cs_skip=("LIRF",), airborne_callsign=True, noplan_callsign_skip=("LIRF",))

C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)
all_dep = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
           .select(C("MVT_ID_mvt"), C("ADEP_mvt").alias("apt"), C_DAY.alias("day"),
                   C("AOBT_3_flt").is_not_null().alias("plan"),
                   C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                   C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                   C("AIRCRAFT_TYPE_mvt").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("type"),
                   C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
                   C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t"))
           .filter(C("day").is_in(DAYS)).collect())
cs_maps = flight_callsign_maps([pl.scan_parquet(s).filter(C("PHASE_mvt") == "DEP")
                                 .select(C("AOBT_3_flt").is_not_null().alias("plan"),
                                         C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                                         C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight")).collect()
                                 for s in ("data/raw/training_*.parquet", "data/raw/ranking.parquet")])
all_dep = add_derived_callsign(all_dep, cs_maps)
npl = all_dep.filter(~C("plan"))
print(f"no-plan rows on these {len(DAYS)} days: {npl.height:,}; derived callsign (cs2) present for "
      f"{npl['cs2'].is_not_null().mean():.1%}")

labels = (pl.read_parquet("results/agent_moredata_funnel_2025.parquet")
            .select("MVT_ID_mvt", "y", C("cls").alias("cls0")))


def match_plan_and_free(deps_day, runs, g):
    d = deps_day.filter(C("plan") & C("callsign").is_not_null())
    has_all = g.air_cs_all and "cs_all" in runs.columns
    parts = [runs.with_columns(C("callsign").alias("cs_key")).drop([c for c in ("cs_all",) if c in runs.columns])]
    if "callsign_air" in runs.columns:
        parts.append(runs.filter(C("callsign_air").is_not_null() & (C("callsign_air") != C("callsign").fill_null("")))
                         .with_columns(C("callsign_air").alias("cs_key")).drop([c for c in ("cs_all",) if c in runs.columns]))
    if has_all:
        parts.append(runs.filter(C("cs_all").is_not_null()).explode("cs_all")
                         .with_columns(C("cs_all").alias("cs_key")).drop("cs_all"))
    long = pl.concat(parts, how="vertical_relaxed").drop("callsign")
    if has_all:
        long = long.unique(subset=["apt", "icao", "liftoff_t", "cs_key"], maintain_order=True)
    cand = (d.with_columns(C("callsign").alias("cs_key")).join(long, on=["apt", "cs_key"], how="inner").drop("cs_key")
             .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dmatch"))
             .filter(C("dmatch") <= g.match_s).sort("dmatch")
             .unique(subset="MVT_ID_mvt", keep="first").unique(subset=["apt", "icao", "liftoff_t"], keep="first"))
    free = runs.join(cand.select("apt", "icao", "liftoff_t"), on=["apt", "icao", "liftoff_t"], how="anti")
    return cand, free


def noplan_time_type(dn, free, g):
    nc = (dn.drop("callsign").join(free, on="apt", how="inner")
            .filter(((C("liftoff_t") - C("mvt_t")).abs() <= g.noplan_match_s) & (C("adsb_type") == C("type"))))
    once_row = nc.group_by("MVT_ID_mvt").len().filter(C("len") == 1).select("MVT_ID_mvt")
    once_run = nc.group_by(["apt", "icao", "liftoff_t"]).len().filter(C("len") == 1).select("apt", "icao", "liftoff_t")
    return (nc.join(once_row, on="MVT_ID_mvt", how="semi").join(once_run, on=["apt", "icao", "liftoff_t"], how="semi")
              .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dmatch")))


def noplan_callsign_match(dcs, free, g, use_all):
    parts = [free.with_columns(C("callsign").alias("cs_key")).drop([c for c in ("cs_all",) if c in free.columns])]
    if "callsign_air" in free.columns:
        parts.append(free.filter(C("callsign_air").is_not_null() & (C("callsign_air") != C("callsign").fill_null("")))
                         .with_columns(C("callsign_air").alias("cs_key")).drop([c for c in ("cs_all",) if c in free.columns]))
    if use_all and "cs_all" in free.columns:
        parts.append(free.filter(C("cs_all").is_not_null()).explode("cs_all")
                         .with_columns(C("cs_all").alias("cs_key")).drop("cs_all"))
    keys = pl.concat(parts, how="vertical_relaxed")
    if use_all:
        keys = keys.unique(subset=["apt", "icao", "liftoff_t", "cs_key"], maintain_order=True)
    ncs = (dcs.drop("callsign").with_columns(C("cs2").alias("cs_key"))
              .join(keys, on=["apt", "cs_key"], how="inner").drop("cs_key")
              .filter((C("liftoff_t") - C("mvt_t")).abs() <= g.match_s))
    once = ncs.group_by("MVT_ID_mvt").len().filter(C("len") == 1).select("MVT_ID_mvt")
    once_run = ncs.group_by(["apt", "icao", "liftoff_t"]).len().filter(C("len") == 1).select("apt", "icao", "liftoff_t")
    ncs = (ncs.join(once, on="MVT_ID_mvt", how="semi").join(once_run, on=["apt", "icao", "liftoff_t"], how="semi")
              .with_columns((C("liftoff_t") - C("mvt_t")).abs().alias("dmatch")))
    if g.noplan_callsign_skip:
        ncs = ncs.filter(~C("apt").is_in(list(g.noplan_callsign_skip)))
    return ncs


rows_out = []
for day in DAYS:
    runs, touch, tracks = runs_and_tracks(day, G)
    deps_day = all_dep.filter(C("day") == day)
    cand, free = match_plan_and_free(deps_day, runs, G)
    dn_all = deps_day.filter(~C("plan") & C("type").is_not_null())
    dcs = deps_day.filter(~C("plan") & C("cs2").is_not_null())

    # Arm A: time+type only
    ncA = noplan_time_type(dn_all, free, G)

    # Arm B: + callsign (ground / first-airborne only), then remaining by time+type
    ncsB = noplan_callsign_match(dcs, free, G, use_all=False)
    freeB = free.join(ncsB.select("apt", "icao", "liftoff_t"), on=["apt", "icao", "liftoff_t"], how="anti")
    dnB = dn_all.join(ncsB.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
    ncB_tt = noplan_time_type(dnB, freeB, G)

    # Arm C: + callsign (ground / first-airborne / EVERY airborne in window), then remaining by time+type
    ncsC = noplan_callsign_match(dcs, free, G, use_all=True)
    freeC = free.join(ncsC.select("apt", "icao", "liftoff_t"), on=["apt", "icao", "liftoff_t"], how="anti")
    dnC = dn_all.join(ncsC.select("MVT_ID_mvt"), on="MVT_ID_mvt", how="anti")
    ncC_tt = noplan_time_type(dnC, freeC, G)

    ids = {"A": set(ncA["MVT_ID_mvt"].to_list()),
           "B": set(ncsB["MVT_ID_mvt"].to_list()) | set(ncB_tt["MVT_ID_mvt"].to_list()),
           "C": set(ncsC["MVT_ID_mvt"].to_list()) | set(ncC_tt["MVT_ID_mvt"].to_list())}
    id_row = {r["MVT_ID_mvt"]: r for r in
              pl.concat([ncsB.select("MVT_ID_mvt", "apt", "icao", "liftoff_t", "dmatch"),
                         ncsC.select("MVT_ID_mvt", "apt", "icao", "liftoff_t", "dmatch")], how="vertical_relaxed")
                .unique(subset="MVT_ID_mvt").iter_rows(named=True)}
    for tag, s in ids.items():
        extra = s - ids["A"] if tag != "A" else set()
        for mid in (extra if tag != "A" else s):
            apt_icao_lo = id_row.get(mid)
            if apt_icao_lo is None:
                continue  # part of ncA / ncB_tt / ncC_tt only (time+type), floor check skipped here
            lb = departure_floor({"apt": apt_icao_lo["apt"], "icao": apt_icao_lo["icao"],
                                   "start_t": apt_icao_lo["liftoff_t"] - 1, "liftoff_t": apt_icao_lo["liftoff_t"],
                                   "mvt_t": apt_icao_lo["liftoff_t"] + 1}, tracks, G)
            rows_out.append(dict(day=day, MVT_ID_mvt=mid, apt=apt_icao_lo["apt"], arm=tag,
                                  extra=(mid in extra) if tag != "A" else True, dmatch=apt_icao_lo["dmatch"], lb=lb))
    print(f"{day}: no-plan matched A={len(ids['A'])} B={len(ids['B'])} (+{len(ids['B']-ids['A'])}) "
          f"C={len(ids['C'])} (+{len(ids['C']-ids['A'])}, of which +{len(ids['C']-ids['B'])} beyond B)", flush=True)

res = pl.DataFrame(rows_out) if rows_out else pl.DataFrame()
if res.height:
    res = res.join(labels, on="MVT_ID_mvt", how="left")
    print("\n===== summary: no-plan derived-callsign matches beyond arm A (time+type), by arm")
    print(res.filter(C("extra")).group_by("arm").agg(
        pl.len().alias("n"), C("apt").n_unique().alias("apts"),
        C("dmatch").median().alias("dmatch_med"),
        (C("y") < C("lb") - 60).mean().round(4).alias("floor_violation_rate"),
        C("y").is_not_null().sum().alias("n_labelled")))
    print("\nby airport, arm C (new: cs_all wired into no-plan matching)")
    print(res.filter(C("extra") & (C("arm") == "C")).group_by("apt").agg(
        pl.len().alias("n"), (C("y") < C("lb") - 60).mean().round(4).alias("floor_violation_rate")).sort("n", descending=True))
    res.write_parquet("results/agent_npadsb_csall_noplan.parquet")
else:
    print("\nno extra callsign-derived no-plan matches found on these days")
print("\ndone")
