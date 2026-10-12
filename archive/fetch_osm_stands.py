"""Stand positions from OpenStreetMap, for the ten airports. Downloaded 2026-09-17.

    .venv/Scripts/python.exe archive/fetch_osm_stands.py

Our stand map is built from ADS-B: an arrival's fix at its labelled on-block time says where its
stand is. Stands no tracked arrival ever reached have no position, and every departure off them is
lost -- Madrid 19,394 matched departures, Frankfurt 7,008, Zurich 5,749. OSM tags stands as nodes
or ways with `aeroway=parking_position` (and, in some countries, `aeroway=gate`), most carrying a
`ref` that should be the stand name the airport uses.

Overpass API, one small query per airport inside the same box prc.adsb uses, node + way + relation
with `out center`. A few hundred kB in total; nothing else is downloaded. ODbL 1.0, attribution
"(c) OpenStreetMap contributors" -- the licence class the organisers already accepted for adsb.lol.

Writes data/external/osm/stands.parquet: apt, kind, ref, lat, lon.
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

ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
OUT = Path("data/external/osm/stands.parquet")


def fetch(query: str) -> dict:
    last = None
    for base in ENDPOINTS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(base, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": "prc-taxiout-2026 research (contact via the repository page)"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read().decode())
            except Exception as exc:  # 429s and 504s are routine on the public endpoints
                last = exc
                time.sleep(20 * (attempt + 1))
    raise SystemExit(f"Overpass failed: {last}")


rows, total_bytes = [], 0
for code, lat_lo, lat_hi, lon_lo, lon_hi in BOXES:
    q = (f"[out:json][timeout:120];("
         f'node["aeroway"~"^(parking_position|gate)$"]({lat_lo},{lon_lo},{lat_hi},{lon_hi});'
         f'way["aeroway"~"^(parking_position|gate)$"]({lat_lo},{lon_lo},{lat_hi},{lon_hi});'
         f");out center tags;")
    data = fetch(q)
    total_bytes += len(json.dumps(data))
    n = 0
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        la = el.get("lat", (el.get("center") or {}).get("lat"))
        lo = el.get("lon", (el.get("center") or {}).get("lon"))
        if la is None or lo is None:
            continue
        rows.append((code, tags.get("aeroway"), tags.get("ref"), float(la), float(lo)))
        n += 1
    print(f"{code}: {n} stand features", flush=True)
    time.sleep(3)

frame = pl.DataFrame(rows, schema=["apt", "kind", "ref", "lat", "lon"], orient="row")
OUT.parent.mkdir(parents=True, exist_ok=True)
frame.write_parquet(OUT)
print(f"\nwrote {OUT}: {frame.height:,} features, {total_bytes / 1e6:.1f} MB of JSON fetched")
print(frame.group_by("apt").agg(pl.len().alias("features"), pl.col("ref").is_not_null().sum().alias("with_ref"),
                                pl.col("ref").n_unique().alias("distinct_refs")).sort("apt"))
