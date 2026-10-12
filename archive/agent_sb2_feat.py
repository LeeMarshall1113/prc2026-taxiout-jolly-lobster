"""experiment sb2 feat (docs/LAB-NOTES.md): label-free per-row features for the small bundle.

    python archive/agent_sb2_feat.py osm    -> data/interim/agent_sb2_osm_{2025,2026}.parquet   (C6: osm_dc, osm_dff)
    python archive/agent_sb2_feat.py occ    -> data/interim/agent_sb2_occ_{2025,2026}.parquet   (C7: occ_push)
    python archive/agent_sb2_feat.py hdow   -> data/interim/agent_sb2_hdow_{2025,2026}.parquet  (C2: apt, hour, dow)

2025 rows = data/interim/agent_offblock_rows_c238.parquet (the re-stack's fold rows); 2026 rows =
agent_offblock_rows_2026.parquet. No label is read: the 2026 side reads only MVT_ID / ADEP / MVT_TIME_UTC from
data/raw/ranking.parquet; ADS-B points are data/adsb/points/<day>.parquet.
"""
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from prc.adsb_match import ELEVATION_FT, metres

C = pl.col
I = "data/interim/"
ID = "MVT_ID_mvt"
T0 = time.time()
ROWS = {"2025": I + "agent_offblock_rows_c238.parquet", "2026": I + "agent_offblock_rows_2026.parquet"}
FF = {"2025": I + "agent_backx_ff_2025.parquet", "2026": I + "agent_backx_ff_2026.parquet"}
PTS = Path("data/adsb/points")


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def osm():
    o = (pl.read_parquet("data/external/osm/stands.parquet")
         .with_columns(C("ref").str.strip_chars().str.to_uppercase().alias("ref"))
         .filter(C("ref").is_not_null() & (C("ref") != ""))
         .group_by("apt", "ref").agg(C("lat").median().alias("o_lat"), C("lon").median().alias("o_lon")))
    for yr in ("2025", "2026"):
        r = pl.read_parquet(ROWS[yr], columns=[ID, "apt", "stand"]).with_columns(C(ID).cast(pl.Float64))
        f = (pl.read_parquet(FF[yr], columns=[ID, "st_lat", "st_lon", "ff_lat", "ff_lon"]).with_columns(C(ID).cast(pl.Float64))
             .unique(subset=ID, keep="first"))
        j = (r.with_columns(C("stand").cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("ref"))
             .join(o, on=["apt", "ref"], how="left", maintain_order="left").join(f, on=ID, how="left", maintain_order="left"))
        g = lambda c: j[c].cast(pl.Float64).fill_null(np.nan).to_numpy()  # noqa: E731
        ola, olo = g("o_lat"), g("o_lon")
        ok = np.isfinite(ola)
        dc = np.full(j.height, np.nan)
        dff = np.full(j.height, np.nan)
        for out, la, lo in ((dc, g("st_lat"), g("st_lon")), (dff, g("ff_lat"), g("ff_lon"))):
            m = ok & np.isfinite(la)
            out[m] = [metres(a, b, c, d) for a, b, c, d in zip(la[m], lo[m], ola[m], olo[m])]
        res = pl.DataFrame({ID: j[ID], "osm_dc": dc, "osm_dff": dff})
        assert res[ID].n_unique() == res.height == r.height
        res.write_parquet(I + f"agent_sb2_osm_{yr}.parquet")
        apt = np.asarray(j["apt"].to_list())
        log(f"osm {yr}: {res.height:,} rows; OSM stand found {ok.mean():.3f}; osm_dc finite {np.isfinite(dc).mean():.3f}; "
            f"osm_dff finite {np.isfinite(dff).mean():.3f}; by airport OSM-found: "
            + " ".join(f"{a} {ok[apt == a].mean():.2f}" for a in sorted(set(apt))))


def _presence(day, cache):
    """unique (apt, icao, minute) of ground fixes in one day file (prc.adsb_match ground definition)."""
    if day in cache:
        return cache[day]
    p = PTS / f"{day}.parquet"
    if not p.exists():
        cache[day] = None
        return None
    elev = C("apt").replace_strict(ELEVATION_FT, default=None, return_dtype=pl.Float64)
    q = (pl.scan_parquet(p).select("apt", "icao", "t", "alt", "gs").filter(C("icao").is_not_null())
         .with_columns(((C("alt") == -1) | ((C("alt") != -1) & ((C("alt").cast(pl.Float64) - elev).abs() <= 800.0)
                                             & (C("gs") <= 40.0)).fill_null(False)).alias("ground"))
         .filter(C("ground")).select("apt", "icao", (C("t") // 60).cast(pl.Int64).alias("mn")).unique().collect())
    cache[day] = q
    return q


def occ():
    for yr in ("2025", "2026"):
        r = (pl.read_parquet(ROWS[yr], columns=[ID, "day", "apt", "mvt_t", "prod_t", "e_start"]).with_columns(C(ID).cast(pl.Float64)))
        a = r.select(pl.coalesce(C("prod_t").cast(pl.Float64).fill_nan(None), C("e_start").cast(pl.Float64).fill_nan(None),
                                 C("mvt_t").cast(pl.Float64) - 900.0).alias("a"))["a"].to_numpy()
        r = r.with_columns(pl.Series("amn", np.floor(a / 60.0).astype(np.int64)))
        cache, outs = {}, []
        days = sorted(set(r["day"].cast(pl.Utf8).to_list()))
        nmiss = 0
        for i, d in enumerate(days):
            d0 = date.fromisoformat(d)
            parts = [_presence(str(d0 + timedelta(days=k)), cache) for k in (-1, 0, 1)]
            if parts[1] is None:
                nmiss += 1
            parts = [x for x in parts if x is not None]
            rd = r.filter(C("day").cast(pl.Utf8) == d).select(ID, "apt", "amn")
            if not parts:
                outs.append(rd.select(ID, pl.lit(None, dtype=pl.Float64).alias("occ_push")))
                continue
            pres = pl.concat(parts).unique()
            win = (pres.with_columns(pl.int_ranges(C("mn") - 5, C("mn") + 6).alias("m2")).explode("m2")
                   .select("apt", "icao", "m2").unique().group_by("apt", "m2").agg(pl.len().cast(pl.Float64).alias("occ_push")))
            outs.append(rd.join(win.rename({"m2": "amn"}), on=["apt", "amn"], how="left", maintain_order="left")
                        .with_columns(C("occ_push").fill_null(0.0)).select(ID, "occ_push"))
            cache.pop(str(d0 + timedelta(days=-1)), None)
            if i % 40 == 0:
                log(f"occ {yr}: day {i + 1}/{len(days)} {d}")
        res = pl.concat(outs)
        assert res[ID].n_unique() == res.height == r.height, (res.height, r.height)
        res.write_parquet(I + f"agent_sb2_occ_{yr}.parquet")
        j = r.select(ID, "apt").join(res, on=ID, how="left")
        log(f"occ {yr}: {res.height:,} rows over {len(days)} days ({nmiss} without a points file -> null); "
            f"median occ_push by airport: " + " ".join(
                f"{x} {v:.0f}" for x, v in j.group_by("apt").agg(C("occ_push").median()).sort("apt").iter_rows()))


def hdow():
    import glob
    for yr in ("2025", "2026"):
        if yr == "2025":
            srcs = sorted(glob.glob("data/raw/training_2025-*.parquet"))
        else:
            srcs = ["data/raw/ranking.parquet"]
        f = pl.concat([pl.scan_parquet(s).filter(C("PHASE_mvt") == "DEP")
                       .select(C(ID).cast(pl.Float64), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("t"))
                       .collect() for s in srcs])
        f = (f.unique(subset=ID, keep="first")
             .select(ID, "apt", C("t").dt.hour().cast(pl.Int32).alias("hour"), C("t").dt.weekday().cast(pl.Int32).alias("dow")))
        f.write_parquet(I + f"agent_sb2_hdow_{yr}.parquet")
        log(f"hdow {yr}: {f.height:,} departures; hours {f['hour'].min()}-{f['hour'].max()}, dow {f['dow'].min()}-{f['dow'].max()}")


if __name__ == "__main__":
    {"osm": osm, "occ": occ, "hdow": hdow}[sys.argv[1]]()
    log("done")
