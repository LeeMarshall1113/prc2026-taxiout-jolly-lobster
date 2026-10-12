"""Taxiway and runway geometry from OpenStreetMap, for the ten airports. Run 2026-09-20.

    .venv/Scripts/python.exe archive/fetch_osm_ways.py

The model has STAND_mvt and RUNWAY_mvt as categoricals and no physical geometry anywhere: there is
no stand-to-runway taxi distance in the codebase, and prc/features.py says so itself above
stand_runway_pair_n ("a crude proxy for apron layout and taxi distance, which we have no geometry
for"). This fetches the ways needed to build the taxiway graph and compute that distance.

Overpass, same two endpoints and the same ten boxes as archive/fetch_osm_stands.py, way + full node
geometry. archive/osm_ways_recon.py counted 5,676 ways and 49,675 nodes, about 4 MB of JSON.
ODbL 1.0, attribution "(c) OpenStreetMap contributors" -- the licence class already declared here.

Writes data/external/osm/ways.parquet   (apt, way_id, aeroway, ref, name, seq, lat, lon)
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")

import polars as pl

from prc.adsb import BOXES

ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
UA = "prc-taxiout-2026 research (contact via the repository page)"
OUT = Path("data/external/osm/ways.parquet")
AEROWAY = '^(taxiway|taxiway_link|runway|runway_link)$'


def fetch(query):
    last = None
    for base in ENDPOINTS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    base, data=urllib.parse.urlencode({"data": query}).encode(),
                    headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read().decode())
            except Exception as exc:
                last = exc
                time.sleep(15 * (attempt + 1))
    raise SystemExit(f"Overpass failed: {last}")


rows, total_bytes = [], 0
for code, la_lo, la_hi, lo_lo, lo_hi in BOXES:
    q = (f'[out:json][timeout:300];way["aeroway"~"{AEROWAY}"]'
         f"({la_lo},{lo_lo},{la_hi},{lo_hi});out geom tags;")
    d = fetch(q)
    total_bytes += len(json.dumps(d))
    nw = nn = 0
    for el in d.get("elements", []):
        if el.get("type") != "way":
            continue
        t = el.get("tags", {})
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        nw += 1
        for i, g in enumerate(geom):
            rows.append((code, int(el["id"]), t.get("aeroway"), t.get("ref"), t.get("name"),
                         i, float(g["lat"]), float(g["lon"])))
            nn += 1
    print(f"{code}: {nw:,} ways, {nn:,} vertices", flush=True)
    time.sleep(3)

frame = pl.DataFrame(rows, schema=["apt", "way_id", "aeroway", "ref", "name", "seq", "lat", "lon"],
                     orient="row")
OUT.parent.mkdir(parents=True, exist_ok=True)
frame.write_parquet(OUT)
print(f"\nwrote {OUT}: {frame.height:,} vertices over {frame['way_id'].n_unique():,} ways, "
      f"{total_bytes / 1e6:.1f} MB of JSON fetched, file {OUT.stat().st_size / 1e6:.2f} MB")
print(frame.group_by("apt").agg(pl.col("way_id").n_unique().alias("ways"), pl.len().alias("vertices"),
                                (pl.col("aeroway") == "runway").sum().alias("runway_vertices")).sort("apt"))
