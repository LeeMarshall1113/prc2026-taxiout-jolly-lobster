"""Stand-map distance and apron-occupancy columns of the dump rows (stage 20, parts g and h).

Derived from osm() and occ() of archive/agent_sb2_feat.py (2026 side only). osm: distance between the mapped stand centroid, the
first ground fix and the OpenStreetMap stand position. occ: aircraft on the ground at the airport in the minutes around the
off-block estimate (ground presence from the points day files of the day and its neighbours). Changes: dump files and output
are arguments. Label-free. Run from the repo root.

    python final/_dump_osm_occ.py osm|occ --rows FILE_rows --ff FILE_ff --out FILE
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
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("part", choices=["osm", "occ"])
ap.add_argument("--rows", required=True)
ap.add_argument("--ff", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
ROWS = {"2026": args.rows}
FF = {"2026": args.ff}
PTS = Path("data/adsb/points")


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def osm():
    o = (pl.read_parquet("data/external/osm/stands.parquet")
         .with_columns(C("ref").str.strip_chars().str.to_uppercase().alias("ref"))
         .filter(C("ref").is_not_null() & (C("ref") != ""))
         .group_by("apt", "ref").agg(C("lat").median().alias("o_lat"), C("lon").median().alias("o_lon")))
    for yr in ("2026",):
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
        res.write_parquet(args.out)
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
    for yr in ("2026",):
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
        res.write_parquet(args.out)
        j = r.select(ID, "apt").join(res, on=ID, how="left")
        log(f"occ {yr}: {res.height:,} rows over {len(days)} days ({nmiss} without a points file -> null); "
            f"median occ_push by airport: " + " ".join(
                f"{x} {v:.0f}" for x, v in j.group_by("apt").agg(C("occ_push").median()).sort("apt").iter_rows()))


if __name__ == "__main__":
    {"osm": osm, "occ": occ}[args.part]()
    log("done")
