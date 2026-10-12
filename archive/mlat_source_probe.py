"""Are Rome's and Paris's ground fixes MLAT rather than true ADS-B? A cheap probe of one 2026 day.

    .venv/Scripts/python.exe archive/mlat_source_probe.py --date 2026-07-15 --gb 1.0

THE HYPOTHESIS (from the literature review, 2026-09-19). readsb's trace_full format carries a
POSITION SOURCE at index 9 of every point -- adsb_icao, adsb_icao_nt, mlat, tisb_*, mode_s, other.
MLAT positions are triangulated from time differences across several receivers and are least
accurate exactly where receiver density is thin. Section 28 closed the Rome line because the
ADS-B stand estimator is WORSE than the model there (669s / 445s against the model's 367s) and at
Paris (1,381s against 178s) -- the two thinnest-coverage airports -- and every fix tried since was
about COVERAGE. None asked which fixes are trustworthy. If Rome's ground fixes are mostly MLAT, that
is a candidate cause, and an ADS-B-only estimator could beat the model there.

prc/adsb.py keeps indices 0-6 and reads 8 for the callsign, but DROPS index 9, so no extracted day
carries it. This streams part of one day's archive with its own parse (reusing prc.adsb's streaming
helpers read-only -- that module is imported by the running extraction's workers and is NOT edited)
and tallies the source of every ground and low fix per airport. Traces are filed by the LAST two hex
digits of the ICAO address, so a prefix of the archive is a quasi-random sample of aircraft.

READ IT AS: EHAM (where the estimator works, MAD 57s) is the control. If LIRF/LFPG ground fixes are
MLAT-heavy and EHAM's are not, the hypothesis earns a real test -- which would need every day
re-extracted with index 9 kept, a decision for the team. If Rome's ground fixes are true ADS-B, the idea
is dead for the cost of this probe.
"""
import argparse
import collections
import gzip
import io
import json
import sys
import tarfile

sys.path.insert(0, ".")

from prc.adsb import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, LOW_ALT_FT, RangeStream, Remote, _airport, release_parts, \
    trace_section_offset

ap = argparse.ArgumentParser()
ap.add_argument("--date", default="2026-07-15")
ap.add_argument("--gb", type=float, default=1.0)
args = ap.parse_args()

remote = Remote(release_parts(args.date))
off, _, skipped = trace_section_offset(remote)
print(f"{args.date}: {remote.total / 1e9:.2f} GB release; traces from {off / 1e9:.2f} GB; reading up to {args.gb:.1f} GB")
reader = io.BufferedReader(RangeStream(remote, off, limit=int(args.gb * 1e9)), buffer_size=8 * 1024 * 1024)

# (airport, ground|low) -> Counter(source); and per airport the set of aircraft by dominant source
fixes = collections.defaultdict(collections.Counter)
by_ac = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
traces = 0
try:
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for m in tar:
            if not m.isfile() or "trace_full" not in m.name:
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            raw = f.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            traces += 1
            icao = obj.get("icao")
            for pt in obj.get("trace") or []:
                la, lo = pt[1], pt[2]
                if la is None or lo is None or not (LAT_MIN <= la <= LAT_MAX and LON_MIN <= lo <= LON_MAX):
                    continue
                alt = pt[3]
                ground = alt == "ground"
                if not ground and (alt is None or not isinstance(alt, (int, float)) or alt > LOW_ALT_FT):
                    continue
                apt = _airport(la, lo)
                if apt is None:
                    continue
                src = pt[9] if len(pt) > 9 and pt[9] else "(absent)"
                kind = "ground" if ground else "low"
                fixes[(apt, kind)][src] += 1
                if ground:
                    by_ac[apt][icao][src] += 1
except (tarfile.ReadError, EOFError):
    pass

print(f"read {traces:,} traces\n")
print("GROUND fixes by position source (the ones the stand estimator anchors on)")
print(f"{'apt':5} {'fixes':>8} {'aircraft':>9} | {'adsb':>7} {'mlat':>7} {'tisb':>6} {'other':>7}")
for apt in sorted({a for a, _ in fixes}):
    c = fixes[(apt, "ground")]
    n = sum(c.values())
    if not n:
        continue
    adsb = sum(v for k, v in c.items() if str(k).startswith("adsb"))
    mlat = c.get("mlat", 0)
    tisb = sum(v for k, v in c.items() if str(k).startswith("tisb"))
    print(f"{apt:5} {n:8,} {len(by_ac[apt]):9,} | {adsb / n:6.1%} {mlat / n:6.1%} {tisb / n:5.1%} "
          f"{(n - adsb - mlat - tisb) / n:6.1%}")
print("\nLOW airborne fixes (< 3,000 ft) by source, for comparison")
for apt in sorted({a for a, _ in fixes}):
    c = fixes[(apt, "low")]
    n = sum(c.values())
    if n:
        print(f"  {apt}: {n:8,} fixes, mlat {c.get('mlat', 0) / n:6.1%}")
all_src = collections.Counter()
for c in fixes.values():
    all_src.update(c)
print(f"\ndistinct source values seen: {dict(all_src.most_common(8))}")
