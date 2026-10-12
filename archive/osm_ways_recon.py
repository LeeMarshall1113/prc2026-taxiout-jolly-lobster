"""Recon only: how many taxiway/runway ways and nodes are there, and how big is the real pull?

Overpass `out count;` returns just a tally, costs almost nothing, and pins the download size before
anything is fetched for real. Same endpoints and the same ten boxes the stand fetch already uses.
ODbL 1.0, attribution "(c) OpenStreetMap contributors" -- the licence class already declared for
this entry. Downloads no geometry.
"""
import json
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, ".")
from prc.adsb import BOXES

ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
UA = "prc-taxiout-2026 research (contact via the repository page)"


def fetch(query):
    last = None
    for base in ENDPOINTS:
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    base, data=urllib.parse.urlencode({"data": query}).encode(),
                    headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=180) as r:
                    return json.loads(r.read().decode())
            except Exception as exc:
                last = exc
                time.sleep(15 * (attempt + 1))
    raise SystemExit(f"Overpass failed: {last}")


AEROWAY = '^(taxiway|taxiway_link|runway|runway_link)$'
tot_w = tot_n = 0
print(f"{'apt':<6}{'ways':>8}{'nodes':>10}{'est MB':>9}")
for code, la_lo, la_hi, lo_lo, lo_hi in BOXES:
    q = (f'[out:json][timeout:180];way["aeroway"~"{AEROWAY}"]'
         f"({la_lo},{lo_lo},{la_hi},{lo_hi});out count;>;out count;")
    d = fetch(q)
    tags = [e.get("tags", {}) for e in d.get("elements", []) if e.get("type") == "count"]
    ways = int(tags[0].get("ways", tags[0].get("total", 0))) if tags else 0
    nodes = int(tags[-1].get("nodes", 0)) if len(tags) > 1 else 0
    # a node in out:json geometry is ~60 bytes, a way header with its node list ~40 + 12 per node
    mb = (nodes * 60 + ways * 200) / 1e6
    tot_w += ways
    tot_n += nodes
    print(f"{code:<6}{ways:>8,}{nodes:>10,}{mb:>9.2f}")
    time.sleep(3)
print(f"\ntotal: {tot_w:,} ways, {tot_n:,} nodes, "
      f"estimated {(tot_n * 60 + tot_w * 200) / 1e6:.1f} MB of JSON across all ten airports")
