"""NOT RUN -- drafted only, so "what I would need" is a concrete,
reviewable artifact rather than a description. This script
performs a network fetch and is not to be executed before the size check below
has been done; archive/fetch_osm_stands.py records the download of 2026-09-17
for the stand/gate points already in this repo.

WHY THIS EXISTS. data/external/osm/stands.parquet (built by fetch_osm_stands.py)
holds ONLY point positions for aeroway=parking_position|gate. It has never
included taxiway or runway geometry -- read that script's Overpass query and it
queries exactly two tags, node and way. There is no taxiway graph anywhere in
this repo to snap a stand or a runway onto, and prc/features.py says as much in
its own comment above stand_runway_pair_n: "a crude proxy for apron layout and
taxi distance, which we have no geometry for." This script is the missing half.

WHAT IT WOULD FETCH. Ways tagged aeroway=taxiway, taxiway_link or runway, WITH
their full node geometry (Overpass "out body; >; out skel qt;" -- way tags/
membership, then every member node's lat/lon), over the same ten bounding boxes
archive/fetch_osm_stands.py already uses (prc.adsb.BOXES, so no new box is introduced). Source: the same Overpass endpoints already in this repo
(overpass-api.de, overpass.kumi.systems). Licence: ODbL 1.0, same class already
accepted for adsb.lol and for stands.parquet -- attribution "(c) OpenStreetMap
contributors".

SIZE. Not measured -- that is the point of asking first. The comparable point-
only fetch (parking_position/gate nodes, all ten airports) was 0.8 MB of JSON
(docs/LAB-NOTES.md, section on stand positions). A
ways-with-geometry fetch is necessarily bigger: a taxiway network has many more
constituent nodes than an apron has stands, and three of our ten airports
(LFPG, EGLL, LTFM) are among the largest taxiway systems in Europe. A rough
analyst's estimate is on the order of 5-25 MB total JSON across all ten
airports -- a guess, not a measurement, and this project's own convention
(see the OPDI entries in the same note, "HEAD only, nothing downloaded" /
"Sample checked") is to confirm a size before pulling, not after.

RECOMMENDED FIRST STEP, costing a few KB, not MB: run the count-only query
below (COUNT_QUERY) per airport BEFORE this fetch, which returns element
counts with no geometry payload, and multiply by ~150-250 bytes/way and
~90-120 bytes/node (Overpass JSON's per-element overhead) to get a real size
estimate before running the full pull.

WHAT IT WOULD WRITE. data/external/osm/taxiway_ways.parquet, one row per
(apt, way_id, aeroway, ref, seq, node_id, lat, lon) -- "long" format so a
consumer can reconstruct each way's polyline in order and detect shared node
ids across ways (that is where a taxiway graph gets its edges: two ways
sharing a node id are connected there). Runway thresholds would come out of
the same file (aeroway=runway ways), NOT as a separate fetch -- a runway's
`ref` tag is normally both ends at once (e.g. "07L/25R"), so resolving which
end matches a given RUNWAY_mvt value ("25L") needs the way's own two endpoint
coordinates and its bearing, matched to the numeric heading each designator
implies (25 -> approximately 250 degrees) -- an algorithm to write once real
geometry is in hand to test it against, not before.

    .venv/Scripts/python.exe archive/agent_osmgeo_fetch_ways.py   # DO NOT RUN before the size check
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
OUT = Path("data/external/osm/taxiway_ways.parquet")
UA = {"User-Agent": "prc-taxiout-2026 research (contact via the repository page)"}


def count_query(lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float) -> str:
    """Near-zero-byte reconnaissance: element counts only, no geometry. Run this
    FIRST, for all ten airports, before the real fetch below."""
    return (f"[out:json][timeout:60];"
            f'way["aeroway"~"^(taxiway|taxiway_link|runway)$"]({lat_lo},{lon_lo},{lat_hi},{lon_hi});'
            f"out count;")


def geometry_query(lat_lo: float, lat_hi: float, lon_lo: float, lon_hi: float) -> str:
    """The real fetch. out body; >; out skel qt; is the standard Overpass idiom
    for "these ways, plus every node they reference, with coordinates"."""
    return (f"[out:json][timeout:180];"
            f'way["aeroway"~"^(taxiway|taxiway_link|runway)$"]({lat_lo},{lon_lo},{lat_hi},{lon_hi});'
            f"out body;>;out skel qt;")


def fetch(query: str) -> dict:
    last = None
    for base in ENDPOINTS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(base, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers=UA)
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read().decode())
            except Exception as exc:  # 429s and 504s are routine on the public endpoints
                last = exc
                time.sleep(20 * (attempt + 1))
    raise SystemExit(f"Overpass failed: {last}")


def main() -> None:
    raise SystemExit(
        "Refusing to run: this script downloads OSM way geometry and has not been "
        "run. See this file's module docstring for what it would fetch, from "
        "where, under which licence, and the size estimate / recommended recon step. "
        "Remove this guard only after the size check has been done."
    )

    rows, total_bytes = [], 0
    for code, lat_lo, lat_hi, lon_lo, lon_hi in BOXES:
        data = fetch(geometry_query(lat_lo, lat_hi, lon_lo, lon_hi))
        total_bytes += len(json.dumps(data))
        nodes = {el["id"]: (el["lat"], el["lon"]) for el in data.get("elements", []) if el.get("type") == "node"}
        n_ways = 0
        for el in data.get("elements", []):
            if el.get("type") != "way":
                continue
            tags = el.get("tags", {})
            for seq, nid in enumerate(el.get("nodes", [])):
                if nid in nodes:
                    lat, lon = nodes[nid]
                    rows.append((code, el["id"], tags.get("aeroway"), tags.get("ref"), seq, nid, lat, lon))
            n_ways += 1
        print(f"{code}: {n_ways} ways, {len(nodes):,} nodes", flush=True)
        time.sleep(3)

    frame = pl.DataFrame(rows, schema=["apt", "way_id", "aeroway", "ref", "seq", "node_id", "lat", "lon"],
                          orient="row")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(OUT)
    print(f"\nwrote {OUT}: {frame.height:,} way-nodes, {total_bytes / 1e6:.1f} MB of JSON fetched")


if __name__ == "__main__":
    main()
