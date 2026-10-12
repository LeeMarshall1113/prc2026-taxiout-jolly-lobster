"""Does OPDI's taxiway geometry give a better off-block than our own adsb.lol pipeline?

    .venv/Scripts/python.exe archive/opdi_probe.py

WHY THIS MIGHT BE THE NEXT ADS-B. OPDI (EUROCONTROL's Open Performance Data Initiative) publishes
flight events derived from the OPENSKY NETWORK -- a DIFFERENT receiver network from adsb.lol, which
is the one our pipeline reads. Two things follow. First, the geometry is already done: an
`entry-taxiway` event carries the OSM taxiway id the aircraft joined, so the first one after a
departure's stand is essentially the end of pushback, which is what we spend a whole module
reconstructing. Second, and more important, a different receiver network has DIFFERENT COVERAGE.
Section 26 measured that adsb.lol has 0.0 per cent surface coverage at Rome across all 30 labelled
2025 days and 7.4 per cent at Paris -- the two airports holding most of our remaining error.

WHAT DECIDES IT. Section 28's lesson: coverage is worthless unless the estimate BEATS THE MODEL.
Rome's adsb.lol estimator has an RMSE of 445s against the model's 367s, which is why v31 gained
nothing. So this measures, per airport, the SPREAD of (event time - true BLOCK) -- a constant bias
calibrates away, scatter does not -- and reports it next to the adsb.lol stand estimator's known
MAD and the model's RMSE on the same rows.

The sample on disk is 2025-01-05..15 only, so this is a feasibility read, not a fold test. If the
spread is tight at LIRF or LFPG, the next question is whether OPDI publishes 2026, because our
scored set is Jan and Jul 2026 -- and that is a question about the archive, not about this code.
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
EV = "data/raw/opdi/flight_events_20250105_20250115.parquet"
FL = "data/raw/opdi/flight_list_202501.parquet"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
# adsb.lol stand-estimator MAD and model RMSE on the same rows, from the 30-day eval (lab record)
KNOWN = {"EDDF": (76, 154), "EDDM": (48, 155), "EGLL": (71, 197), "EHAM": (57, 148), "LEBL": (59, 216),
         "LEMD": (88, 185), "LFPG": (None, 178), "LIRF": (117, 367), "LSZH": (120, 148), "LTFM": (None, None)}

fl = (pl.scan_parquet(FL)
      .filter(C("adep").is_in(APTS) & C("flt_id").is_not_null())
      .select(C("id").alias("flight_id"), C("adep").alias("apt"),
              C("flt_id").str.strip_chars().str.to_uppercase().alias("callsign"),
              C("first_seen"), C("last_seen"))
      .collect())
print(f"OPDI flights departing our ten airports in the January list: {fl.height:,}")

ev = (pl.scan_parquet(EV).filter(C("type").is_in(["entry-taxiway", "exit-taxiway", "take-off", "first_seen"]))
      .select("flight_id", "type", "event_time").collect())
print(f"OPDI events of the four types we care about: {ev.height:,}")

dep = (pl.scan_parquet("data/raw/training_*.parquet")
       .filter((C("PHASE_mvt") == "DEP") & C("ADEP_mvt").is_in(APTS))
       .select(C("ADEP_mvt").alias("apt"),
               C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
               # OPDI stamps are naive UTC; ours carry the zone, so drop it to compare
               C("MVT_TIME_UTC_mvt").dt.replace_time_zone(None).alias("mvt_t"),
               C("BLOCK_TIME_UTC_mvt").dt.replace_time_zone(None).alias("block_t"))
       .filter(C("callsign").is_not_null() & C("block_t").is_not_null()
               & C("mvt_t").is_between(pl.datetime(2025, 1, 5), pl.datetime(2025, 1, 16)))
       .collect())
print(f"our labelled departures in the sample window: {dep.height:,}\n")

# Join our departure to the OPDI flight with the same airport+callsign whose take-off is nearest
# the movement stamp -- the same identity rule the adsb.lol matcher uses.
to = (ev.filter(C("type") == "take-off").group_by("flight_id").agg(C("event_time").min().alias("takeoff_t"))
      .join(fl, on="flight_id", how="inner"))
m = (dep.join(to, on=["apt", "callsign"], how="inner")
        .with_columns((C("takeoff_t") - C("mvt_t")).dt.total_seconds().abs().alias("dm"))
        .filter(C("dm") <= 600).sort("dm").unique(subset=["apt", "callsign", "mvt_t"], keep="first"))
print(f"matched to an OPDI flight by callsign and take-off within 600s: {m.height:,} "
      f"({m.height / max(dep.height, 1):.1%} of our departures in the window)\n")

first_tw = (ev.filter(C("type") == "entry-taxiway").group_by("flight_id")
            .agg(C("event_time").min().alias("tw_t")))
m = m.join(first_tw, on="flight_id", how="left")

print(f"{'apt':6} {'our deps':>9} {'matched':>8} {'match%':>7} | {'entry-taxiway vs BLOCK':^30} | {'adsb.lol':>9} {'model':>7}")
print(f"{'':6} {'':>9} {'':>8} {'':>7} | {'n':>6} {'median':>8} {'IQR':>7} {'MAD':>6} | {'stand MAD':>9} {'RMSE':>7}")
for apt in APTS:
    d = dep.filter(C("apt") == apt)
    s = m.filter(C("apt") == apt)
    t = s.filter(C("tw_t").is_not_null())
    if not d.height:
        continue
    row = f"{apt:6} {d.height:9,} {s.height:8,} {s.height / d.height:6.1%} | "
    if t.height >= 15:
        e = (t["tw_t"] - t["block_t"]).dt.total_seconds().to_numpy().astype(float)
        e = e[np.abs(e) <= 3600]
        q1, q3 = np.percentile(e, [25, 75])
        row += f"{len(e):6,} {np.median(e):+8.0f} {q3 - q1:7.0f} {np.median(np.abs(e - np.median(e))):6.0f} | "
    else:
        row += f"{t.height:6,} {'too few':>24} | "
    mad, mrmse = KNOWN[apt]
    row += f"{mad if mad else 'n/a':>9} {mrmse if mrmse else 'n/a':>7}"
    print(row)
print("\nadsb.lol stand MAD and model RMSE are from the 30-day evaluation, shown for scale only --")
print("they are different rows and a different window, so read the SPREAD column, not a difference.")
print("\nThe bar that matters (section 28): an estimator earns its place only if it beats the MODEL.")
