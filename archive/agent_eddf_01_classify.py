"""experiment eddf 01: classify WHY EDDF's points_no_liftoff population stays unmatched after taxinm (tag "eddf",
2026-09-24). Task: docs/LAB-NOTES.md. EDDF 2026 matched share fell to 0.72 (0.94 in 2025); ~8,906 2026
board departures have the aircraft seen in ADS-B (points_no_liftoff, archive/agent_coverage_01_census.py) but
taxinm (archive/agent_taxinm_stage1_eval.py --floor-stack-taxinm-apts) recovers only ~2,942 of them. This finds
out why the other ~6,000 stay unmatched, using ONLY what's on disk already (the census parquets, read-only) plus
fresh ADS-B track diagnostics.

Reuses scripts/agent_taxinm_features.taxinm_match UNMODIFIED to get the row's real recovery status (so "not
recovered" here means the SAME thing agent_taxinm_stage1_eval.py means), then re-derives the intermediate state
of that same function (icao resolution, ground fixes in its window, the >1200s break truncation, the >=5kt move
gate) to see exactly which step drops each row. Population: EDDF, category=='points_no_liftoff' from the
already-computed data/interim/agent_coverage_census_{2025,2026}.parquet (read-only, NP1/production gates,
archive/agent_coverage_01_census.py). 2026 = unlabelled board population (the target); 2025 = same class, with
truth (MVT_TIME_UTC - BLOCK_TIME_UTC), for validating any new rule before it ever touches board rows.

Also classifies the last-observed-ground-fix position against OSM stands/taxiways/runways
(data/external/osm/{stands,ways}.parquet) -- nearest of (gate, taxiway point, runway point), and which runway if
that's nearest -- to answer "stand area / taxiway / near which runway".

Read-only on data/raw, data/adsb/points, data/interim/agent_coverage_census_*.parquet, data/external/osm/*.
Writes only to data/interim/agent_eddf_*.parquet. No model fit. No edits to prc/ or any existing archive/*.py.

    ./.venv/Scripts/python.exe archive/capped_run.py --cap-gb 3 -- \
        ./.venv/Scripts/python.exe archive/agent_eddf_01_classify.py [--days-2026 N] [--days-2025 N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import polars as pl

from archive.agent_noplan_adsb_match import ELEVATION_FT, add_derived_callsign, flight_callsign_maps
from archive.agent_taxinm_features import taxinm_match

C = pl.col
APT = "EDDF"
ELEV = ELEVATION_FT[APT]
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--days-2026", type=int, default=0, help="debug: truncate 2026 day list")
ap.add_argument("--days-2025", type=int, default=0, help="debug: truncate 2025 day list")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# OSM geometry: nearest-point classifier (gate / taxiway / runway[+ref])
# ---------------------------------------------------------------------------
def metres(lat, lon, lat0, lon0):
    dy = (lat - lat0) * 110_540.0
    dx = (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0))
    return np.sqrt(dx * dx + dy * dy)


stands = pl.read_parquet("data/external/osm/stands.parquet").filter(C("apt") == APT)
ways = pl.read_parquet("data/external/osm/ways.parquet").filter(C("apt") == APT)
GATE_LL = (stands["lat"].to_numpy(), stands["lon"].to_numpy())
TAXI_LL = (ways.filter(C("aeroway") == "taxiway")["lat"].to_numpy(), ways.filter(C("aeroway") == "taxiway")["lon"].to_numpy())
rw = ways.filter(C("aeroway") == "runway")
RW_LL = (rw["lat"].to_numpy(), rw["lon"].to_numpy())
RW_REF = rw["ref"].fill_null("?").to_numpy()
log(f"OSM EDDF: {len(GATE_LL[0])} gates, {len(TAXI_LL[0])} taxiway pts, {len(RW_LL[0])} runway pts")


def nearest_feature(lat, lon):
    """('gate'|'taxiway'|'runway'[/ref]|'none', metres) for one point, nan-safe."""
    if lat is None or lon is None or (isinstance(lat, float) and np.isnan(lat)):
        return "none", float("nan")
    best_kind, best_d, best_ref = "none", float("inf"), None
    for kind, (la, lo), refs in (("gate", GATE_LL, None), ("taxiway", TAXI_LL, None), ("runway", RW_LL, RW_REF)):
        if len(la) == 0:
            continue
        d = metres(la, lo, lat, lon)
        i = int(np.argmin(d))
        if d[i] < best_d:
            best_d, best_kind = float(d[i]), kind
            best_ref = refs[i] if refs is not None else None
    label = f"runway {best_ref}" if best_kind == "runway" and best_ref else best_kind
    return label, best_d


# ---------------------------------------------------------------------------
# universe (callsign, cs2, plan, mvt_t[, y]) for the census's points_no_liftoff MVT_ID_mvt set
# ---------------------------------------------------------------------------
cs_src = [pl.scan_parquet(s).filter(C("PHASE_mvt") == "DEP")
            .select(C("AOBT_3_flt").is_not_null().alias("plan"),
                    C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
                    C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight")).collect()
          for s in ("data/raw/training_*.parquet", "data/raw/ranking.parquet")]
CS_MAPS = flight_callsign_maps(cs_src)
log("flight_callsign_maps built")

C_DAY = C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8)


def load_universe(src, ids, with_truth):
    cols = [C("MVT_ID_mvt"), C_DAY.alias("day"), C("AOBT_3_flt").is_not_null().alias("plan"),
            C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("callsign"),
            C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("flight"),
            C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("mvt_t")]
    if with_truth:
        cols.append(C("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("block_t"))
    df = (pl.scan_parquet(src).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == APT)).select(*cols).collect())
    if with_truth:
        df = df.with_columns(C("MVT_ID_mvt").cast(pl.Float64))
    df = df.join(ids, on="MVT_ID_mvt", how="semi")
    df = add_derived_callsign(df, CS_MAPS)
    df = df.with_columns(pl.coalesce(C("callsign"), C("cs2")).alias("key"))
    if with_truth:
        df = df.with_columns((C("mvt_t") - C("block_t")).alias("y"))
    return df


def classify_day(deps_day: pl.DataFrame, points_day: pl.DataFrame, apt: str = APT, elev: float = ELEV) -> pl.DataFrame:
    """Re-derives taxinm_match's own intermediate state for every row in deps_day (MVT_ID_mvt, key, mvt_t)."""
    p = points_day.with_columns(
        ((C("alt") == -1) | (((C("alt").cast(pl.Float64) - elev).abs() <= 800) & (C("gs") <= 40)))
        .fill_null(False).alias("ground"),
        C("callsign").str.strip_chars().str.to_uppercase().alias("cs"))
    d = deps_day
    cs_join = (p.filter(C("cs").is_not_null()).select("cs", "icao", "t")
               .join(d.select("MVT_ID_mvt", "key", "mvt_t").rename({"key": "cs"}), on="cs", how="inner")
               .filter((C("t") >= C("mvt_t") - 5400) & (C("t") <= C("mvt_t") + 900))
               .group_by(["MVT_ID_mvt", "icao"]).len().sort("len", descending=True)
               .unique(subset="MVT_ID_mvt", keep="first"))
    icao_of = dict(zip(cs_join["MVT_ID_mvt"].to_list(), cs_join["icao"].to_list()))
    g = p.filter(C("ground")).select("icao", "t", "gs", "lat", "lon").sort(["icao", "t"])
    a = p.filter(~C("ground")).select("icao", "t")
    tr_g = {k[0]: (v["t"].to_numpy(), v["gs"].fill_null(0).to_numpy(), v["lat"].to_numpy(), v["lon"].to_numpy())
            for k, v in g.partition_by("icao", as_dict=True).items()}
    tr_a = {k[0]: v["t"].to_numpy() for k, v in a.partition_by("icao", as_dict=True).items()}

    rows = []
    for mid, mt, plan in zip(d["MVT_ID_mvt"].to_list(), d["mvt_t"].to_list(), d["plan"].to_list()):
        ic = icao_of.get(mid)
        has_icao = ic is not None
        last_full_t = np.nan
        gap_last_ground = np.nan
        last_lat = last_lon = float("nan")
        n_win_raw = n_win_trunc = 0
        moves = False
        move_s = first_s = last_s = float("nan")
        airborne_after = False
        if has_icao and ic in tr_g:
            t, gs, la, lo = tr_g[ic]
            before = t <= mt + 60
            if before.any():
                j = np.flatnonzero(before)[-1]
                last_full_t, last_lat, last_lon = float(t[j]), float(la[j]), float(lo[j])
                gap_last_ground = mt - last_full_t
            k = np.flatnonzero((t >= mt - 5400) & (t <= mt + 60))
            n_win_raw = int(len(k))
            if len(k):
                tt, gg = t[k], gs[k]
                brk = np.flatnonzero(np.diff(tt) > 1200)
                if len(brk):
                    tt, gg = tt[brk[-1] + 1:], gg[brk[-1] + 1:]
                n_win_trunc = int(len(tt))
                if len(tt):
                    first_s, last_s = float(mt - tt[0]), float(mt - tt[-1])
                    mv = np.flatnonzero(gg >= 5)
                    moves = len(mv) > 0
                    if moves:
                        move_s = float(mt - tt[mv[0]])
        if has_icao and ic in tr_a:
            ta = tr_a[ic]
            airborne_after = bool(np.any((ta > mt) & (ta <= mt + 3600)))
        rows.append((mid, plan, has_icao, last_full_t, gap_last_ground, last_lat, last_lon,
                     n_win_raw, n_win_trunc, moves, airborne_after, first_s, last_s, move_s))
    return pl.DataFrame(rows, schema=["MVT_ID_mvt", "plan", "has_icao", "last_full_t", "gap_last_ground_s",
                                       "last_lat", "last_lon", "n_win_raw", "n_win_trunc", "moves_ge5kt",
                                       "airborne_after", "first_s", "last_s", "txnm_move_s"],
                        orient="row", infer_schema_length=None)


def run_year(census_path, is_2025):
    census = pl.read_parquet(census_path).filter((C("apt") == APT) & (C("category") == "points_no_liftoff"))
    ids = census.select("MVT_ID_mvt")
    if is_2025:
        ids = ids.with_columns(C("MVT_ID_mvt").cast(pl.Float64))
    src = "data/raw/training_*.parquet" if is_2025 else "data/raw/ranking.parquet"
    uni = load_universe(src, ids, with_truth=is_2025)
    days = sorted(uni["day"].unique().to_list())
    n_trunc = args.days_2025 if is_2025 else args.days_2026
    if n_trunc:
        days = days[:n_trunc]
    log(f"{'2025' if is_2025 else '2026'}: {uni.height:,} points_no_liftoff EDDF rows across {len(days)} days")
    out_parts, txnm_parts = [], []
    for i, day in enumerate(days):
        dl = uni.filter(C("day") == day)
        if dl.height == 0:
            continue
        pts = (pl.scan_parquet(f"data/adsb/points/{day}.parquet").filter(C("apt") == APT)
               .select("icao", "callsign", "t", "alt", "gs", "lat", "lon").collect())
        out_parts.append(classify_day(dl.select("MVT_ID_mvt", "key", "mvt_t", "plan"), pts))
        txnm_parts.append(taxinm_match(dl.select("MVT_ID_mvt", C("key").alias("callsign"), "mvt_t"), pts, APT))
        if (i + 1) % 20 == 0:
            log(f"  {'2025' if is_2025 else '2026'} {i + 1}/{len(days)} days")
    diag = pl.concat(out_parts)
    txnm = pl.concat(txnm_parts).select("MVT_ID_mvt", "txnm_floor").rename({"txnm_floor": "txnm_floor_official"})
    out = diag.join(txnm, on="MVT_ID_mvt", how="left").join(uni.select(["MVT_ID_mvt", "day"] + (["y"] if is_2025 else [])),
                                                              on="MVT_ID_mvt", how="left")
    # sanity: our re-derived txnm_move_s must equal the official function's txnm_floor (both = txnm_move_s)
    both_finite = out.filter(C("txnm_move_s").is_not_null() & C("txnm_floor_official").is_not_null())
    if both_finite.height:
        maxdiff = float((both_finite["txnm_move_s"] - both_finite["txnm_floor_official"]).abs().max())
        n_mismatch = int((out["txnm_move_s"].is_not_null() != out["txnm_floor_official"].is_not_null()).sum())
        log(f"  identity check vs official taxinm_match: max|diff| on {both_finite.height:,} both-finite rows = {maxdiff}, "
            f"finite-mismatch rows = {n_mismatch}")
    # location classification for rows with a last ground fix
    feats = []
    for la, lo in zip(out["last_lat"].to_list(), out["last_lon"].to_list()):
        label, dist = nearest_feature(la, lo)
        feats.append((label, dist))
    out = out.with_columns(pl.Series("loc_label", [f[0] for f in feats]),
                            pl.Series("loc_dist_m", [f[1] for f in feats]))
    tag = "2025" if is_2025 else "2026"
    out.write_parquet(f"data/interim/agent_eddf_diag_{tag}.parquet")
    log(f"{tag}: wrote data/interim/agent_eddf_diag_{tag}.parquet ({out.height:,} rows)")
    return out


d26 = run_year("data/interim/agent_coverage_census_2026.parquet", is_2025=False)
d25 = run_year("data/interim/agent_coverage_census_2025.parquet", is_2025=True)


def bucket(df):
    return (df.with_columns(
        pl.when(~C("has_icao")).then(pl.lit("no_icao_in_window"))
          .when(C("n_win_raw") == 0).then(pl.lit("icao_no_ground_fix_in_window"))
          .when(C("n_win_trunc") == 0).then(pl.lit("ground_fix_dropped_by_1200s_break"))
          .when(~C("moves_ge5kt")).then(pl.lit("ground_fix_present_never_ge5kt"))
          .otherwise(pl.lit("recovered_by_txnm")).alias("bucket")))


for tag, df in (("2026", d26), ("2025", d25)):
    b = bucket(df)
    log(f"\n=== {tag} bucket counts (n={b.height:,}) ===")
    print(b.group_by("bucket").len().sort("len", descending=True))
    print(f"  plan share: {b['plan'].mean():.1%}")
    print("  by bucket, location of the LAST ground fix seen that day before MVT+60s (any time, not just the "
          "[MVT-5400,MVT+60] window taxinm itself uses):")
    for buck in b["bucket"].unique().to_list():
        sub = b.filter((C("bucket") == buck) & (C("loc_label") != "none"))
        if sub.height:
            print(f"    {buck}:")
            print(sub.group_by("loc_label").len().sort("len", descending=True).head(8))
    print("  by bucket, gap from that last ground fix to MVT (seconds; NEGATIVE gap is impossible by "
          "construction; a LARGE gap means the fix sits before the taxinm window even starts):")
    for buck in b["bucket"].unique().to_list():
        sub = b.filter(C("bucket") == buck)
        g = sub["gap_last_ground_s"].drop_nulls()
        if g.len():
            print(f"    {buck}: n_with_fix={g.len()}/{sub.height} mean={g.mean():.0f}s median={g.median():.0f}s "
                  f"p10={g.quantile(0.1):.0f}s p90={g.quantile(0.9):.0f}s "
                  f">5400s(outside window)={(g > 5400).sum()} ({(g > 5400).sum() / g.len():.0%})")
    print(f"  airborne fix seen within 1h after MVT, by bucket (aircraft genuinely departs, just unmatched):")
    for buck in b["bucket"].unique().to_list():
        sub = b.filter(C("bucket") == buck)
        print(f"    {buck}: {sub['airborne_after'].mean():.1%} of {sub.height:,}")
log("\nALL DONE")
