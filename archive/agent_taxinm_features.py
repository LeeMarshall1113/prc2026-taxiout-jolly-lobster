"""Generalized taxi-phase NM-callsign matching, NO lift-off requirement, ANY airport.

LEVER 1 generalized (tag taxinm, team instruction 2026-09-23, docs/LAB-NOTES.md).
LFPG-specific origin: archive/agent_lfpgnewstand_features.py (itself archive/agent_big_06_lfpg_taxi.py's
method, ported into a floor-stack feature/new-row mechanism and validated: Modal splice PASS, LFPG
-1.09s / board ~-0.11s at 0.77x). Same idea generalized to any of the ADS-B-covered airports: LIRF and
LFPG are matched by a lift-off run only if the ground track is followed by an airborne fix within
liftoff_s (600s) -- at reception-limited airports this gate itself is the binding constraint, not
callsign correctness (which --air-cs-all already fixes) or coverage generally. This module re-derives
ground evidence for a departure's NM callsign directly from the day's points file, keyed on the
scheduled MVT time instead of a detected lift-off, for ANY named airport.

Independent of prc/adsb_match.py and prc/adsb.py -- no production file is touched.
"""
import numpy as np
import polars as pl

from prc.adsb_match import ELEVATION_FT

C = pl.col

TAXINM_COLS = ("txnm_first_s", "txnm_move_s", "txnm_last_s", "txnm_nfix", "txnm_gsfirst", "txnm_dtmed")


def taxinm_match(deps_day: pl.DataFrame, points_day: pl.DataFrame, apt: str) -> pl.DataFrame:
    """deps_day: MVT_ID_mvt, callsign, mvt_t for `apt` departures on ONE day (any match status --
    call on every row at `apt`, not just already-matched ones).
    points_day: that day's points already filtered to apt == `apt` (icao, callsign, t, alt, gs).

    Returns one row per input departure: MVT_ID_mvt, TAXINM_COLS (NaN where no NM-callsign match or
    no ground fix in the window), and txnm_floor = mvt_t - t(first fix >= 5kt) where available (NaN
    otherwise) -- the validated floor y >= MVT - t(first >= 5kt), agent_big_06_lfpg_taxi.py.
    """
    if deps_day.height == 0:
        return pl.DataFrame(schema={"MVT_ID_mvt": pl.Int64, **{c: pl.Float64 for c in TAXINM_COLS}, "txnm_floor": pl.Float64})
    elev = ELEVATION_FT[apt]
    p = points_day.with_columns(
        ((C("alt") == -1) | (((C("alt").cast(pl.Float64) - elev).abs() <= 800) & (C("gs") <= 40)))
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
    out = pl.DataFrame(rows, schema=["MVT_ID_mvt", *TAXINM_COLS], orient="row")
    return out.with_columns(C("txnm_move_s").alias("txnm_floor"))
