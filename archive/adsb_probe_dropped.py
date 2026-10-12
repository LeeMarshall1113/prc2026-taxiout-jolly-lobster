"""Does the adsb.lol archive hold airport ground traffic that our extractor throws away?

    .venv/Scripts/python.exe archive/adsb_probe_dropped.py 2026-07-13

prc.adsb.parse_trace keeps a point inside an airport box only if its altitude is "ground" or
a number up to 3,000 ft. Points with NO altitude are dropped -- and positions computed by
multilateration for transponders that do not broadcast ADS-B surface messages often carry
none. Paris and Istanbul show almost no ground fixes in our extraction; this streams one day
through the same reader and counts, per airport, every in-box point by altitude category
(ground / numeric <= 3,000 ft / numeric above / none) and by source type (trace field 9:
adsb_icao, mlat, tisb_..., adsr_..., other), with slow (< 40 kt) counts for the no-altitude
points. Those dropped points are saved to data/adsb/probe/<date>_dropped.parquet for a look at
whether they trace taxi routes. Nothing in data/adsb/points is touched.
"""
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

import polars as pl

import prc.adsb as A

date = sys.argv[1] if len(sys.argv) > 1 else "2026-07-13"
counts = defaultdict(int)
dropped = []


def counting_parse(raw: bytes, rows: list) -> int:
    import gzip
    import json
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    obj = json.loads(raw)
    trace = obj.get("trace") or []
    if not trace:
        return 0
    base = float(obj.get("timestamp") or 0.0)
    icao, actype = obj.get("icao"), obj.get("t")
    cs = None
    for pt in trace:
        if len(pt) > 8 and isinstance(pt[8], dict):
            fl = pt[8].get("flight")
            if fl:
                cs = fl.strip() or cs
        la, lo = pt[1], pt[2]
        if la is None or lo is None or not (A.LAT_MIN <= la <= A.LAT_MAX and A.LON_MIN <= lo <= A.LON_MAX):
            continue
        apt = A._airport(la, lo)
        if apt is None:
            continue
        alt = pt[3]
        if alt == "ground":
            cat = "ground"
        elif alt is None:
            cat = "none"
        elif isinstance(alt, (int, float)):
            cat = "low" if alt <= A.LOW_ALT_FT else "high"
        else:
            cat = "other"
        src = pt[9] if len(pt) > 9 and isinstance(pt[9], str) else "unknown"
        src = "mlat" if src.startswith("mlat") else ("adsb" if src.startswith("adsb") else src.split("_")[0])
        gs = pt[4] if len(pt) > 4 else None
        counts[(apt, cat, src)] += 1
        if cat == "none":
            counts[(apt, "none_slow" if (gs is not None and gs < 40) else "none_fast_or_unknown", src)] += 1
            dropped.append((apt, icao, cs, actype, base + pt[0], la, lo, gs, src))
    return 0


A.parse_trace = counting_parse  # stream_day looks the name up at call time
t0 = time.time()
_, stats = A.stream_day(date)
print(f"{date}: streamed in {time.time() - t0:.0f}s; reader stats {stats}")
apts = sorted({k[0] for k in counts})
for apt in apts:
    parts = {cat: sum(v for (a, c, s), v in counts.items() if a == apt and c == cat)
             for cat in ("ground", "low", "high", "none", "none_slow")}
    by_src = defaultdict(int)
    for (a, c, s), v in counts.items():
        if a == apt and c == "none":
            by_src[s] += v
    print(f"  {apt}: ground {parts['ground']:>8,}  numeric<=3000 {parts['low']:>8,}  numeric>3000 {parts['high']:>8,}  "
          f"NO ALTITUDE {parts['none']:>8,} (slow {parts['none_slow']:,})  no-altitude sources "
          + ", ".join(f"{s} {v:,}" for s, v in sorted(by_src.items(), key=lambda kv: -kv[1])))
out = Path("data/adsb/probe")
out.mkdir(parents=True, exist_ok=True)
pl.DataFrame(dropped, schema=["apt", "icao", "callsign", "actype", "t", "lat", "lon", "gs", "src"], orient="row").write_parquet(
    out / f"{date}_dropped.parquet")
print(f"saved {len(dropped):,} dropped points to {out / (date + '_dropped.parquet')}")
