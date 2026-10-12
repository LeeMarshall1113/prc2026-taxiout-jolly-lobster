"""LFPG NM-callsign taxi-phase matching, NO lift-off requirement (tag lfpgnewstand, LEVER 1).

docs/LAB-NOTES.md. prc.adsb_match.runs_and_tracks only keeps a ground run as a
candidate for match_departures if it is followed by an airborne fix within Gates.liftoff_s (600s
by default). At LFPG the reception dead zone near the runway means many real ground tracks never
produce such a run, so production's matched-departure count at LFPG is ~1.5% of scored rows
(docs/LAB-NOTES.md section 4) despite 29.3% of 2026 LFPG departures having an ADS-B
ground fix at all (docs/LAB-NOTES.md section "fresh quantification"). This module
re-derives ground evidence for a departure's NM callsign directly from the day's points file, keyed
on the scheduled MVT time instead of a detected lift-off -- exactly the method
archive/agent_big_06_lfpg_taxi.py already validated (RMSE 227.56 -> 224.88 fold, Sep16-Oct31 2025,
31,007 LFPG departures): first ground fix, first fix >= 5kt (the validated floor anchor), last
ground fix, fix count, first speed, median fix spacing.

Independent of prc/adsb_match.py and prc/adsb.py -- no production file is touched. Used only for
LFPG (apt == "LFPG"); every other airport is untouched by construction (the caller only invokes
this on LFPG rows).
"""
import numpy as np
import polars as pl

C = pl.col

LFPGNM_COLS = ("lfnm_first_s", "lfnm_move_s", "lfnm_last_s", "lfnm_nfix", "lfnm_gsfirst", "lfnm_dtmed")
ELEV_LFPG = 392.0  # field elevation, ft (prc.adsb_match.ELEVATION_FT["LFPG"])


def lfpg_nm_match(deps_day: pl.DataFrame, points_day: pl.DataFrame) -> pl.DataFrame:
    """deps_day: MVT_ID_mvt, callsign, mvt_t for LFPG departures on ONE day (any match status --
    called on every LFPG row in `universe`, not just production-matched ones).
    points_day: that day's points already filtered to apt == 'LFPG' (icao, callsign, t, alt, gs).

    Returns one row per input departure: MVT_ID_mvt, LFPGNM_COLS (NaN where no NM-callsign match
    or no ground fix in the window), and lfnm_floor = mvt_t - t(first fix >= 5kt) where available
    (NaN otherwise) -- the validated floor y >= MVT - t(first >= 5kt), agent_big_06_lfpg_taxi.py.
    """
    if deps_day.height == 0:
        return pl.DataFrame(schema={"MVT_ID_mvt": pl.Int64, **{c: pl.Float64 for c in LFPGNM_COLS}, "lfnm_floor": pl.Float64})
    p = points_day.with_columns(
        ((C("alt") == -1) | (((C("alt").cast(pl.Float64) - ELEV_LFPG).abs() <= 800) & (C("gs") <= 40)))
        .fill_null(False).alias("ground"),
        C("callsign").str.strip_chars().str.to_uppercase().alias("cs"))
    d = deps_day.with_columns(C("callsign").str.strip_chars().str.to_uppercase().alias("cs"))
    cs = (p.filter(C("cs").is_not_null()).select("cs", "icao", "t")
          .join(d.select("MVT_ID_mvt", "cs", "mvt_t"), on="cs", how="inner")
          .filter((C("t") >= C("mvt_t") - 5400) & (C("t") <= C("mvt_t") + 900))
          .group_by(["MVT_ID_mvt", "icao"]).len().sort("len", descending=True)
          .unique(subset="MVT_ID_mvt", keep="first"))
    g = p.filter(C("ground")).select("icao", "t", "gs").sort(["icao", "t"])
    tr = {k[0]: (v["t"].to_numpy(), v["gs"].fill_null(0).to_numpy()) for k, v in g.partition_by("icao", as_dict=True).items()}
    icao_of = dict(zip(cs["MVT_ID_mvt"].to_list(), cs["icao"].to_list()))
    mvt_of = dict(zip(d["MVT_ID_mvt"].to_list(), d["mvt_t"].to_list()))
    rows = []
    for mid, mt in mvt_of.items():
        ic = icao_of.get(mid)
        first_s = move_s = last_s = nfix = gsfirst = dtmed = np.nan
        if ic is not None and ic in tr:
            t, gs = tr[ic]
            k = np.flatnonzero((t >= mt - 5400) & (t <= mt + 60))
            if len(k):
                tt, gg = t[k], gs[k]
                brk = np.flatnonzero(np.diff(tt) > 1200)
                if len(brk):
                    tt, gg = tt[brk[-1] + 1:], gg[brk[-1] + 1:]
                if len(tt):
                    mv = np.flatnonzero(gg >= 5)
                    first_s = mt - tt[0]
                    move_s = (mt - tt[mv[0]]) if len(mv) else np.nan
                    last_s = mt - tt[-1]
                    nfix = float(len(tt))
                    gsfirst = float(gg[0])
                    dtmed = float(np.median(np.diff(tt))) if len(tt) > 1 else np.nan
        rows.append((mid, first_s, move_s, last_s, nfix, gsfirst, dtmed))
    out = pl.DataFrame(rows, schema=["MVT_ID_mvt", *LFPGNM_COLS], orient="row")
    return out.with_columns(C("lfnm_move_s").alias("lfnm_floor"))
