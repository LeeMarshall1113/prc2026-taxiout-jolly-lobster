"""experiment hidden-fields: COPY of prc/adsb.py's streaming path that keeps EVERY per-point field
instead of prc.adsb.COLS's subset, for the one-day probe in docs/LAB-NOTES.md.

    .venv/Scripts/python.exe archive/agent_hidden_extract.py --date 2025-07-16 [--out data/adsb/hidden]

WHY A COPY, NOT AN EDIT. prc/adsb.py is imported live by the running extraction pipeline
(prc/adsb_match.py, archive/agent_moredata_extract.py and every agent_adsb*/agent_np*
script). This file only reads prc.adsb's remote-streaming helpers (Remote, RangeStream,
trace_section_offset, release_parts, _airport, the airport boxes) and monkeypatches its
OWN process-local copy of prc.adsb.parse_trace/to_frame the same way archive/adsb_probe_dropped.py
already does for its dropped-field probe -- prc/adsb.py itself is never modified.

WHAT prc.adsb DROPS that this keeps (readsb trace_full point format: [dt, lat, lon, alt, gs,
track, flags, vrate, aircraft-dict|null, source, geom_alt, geom_vrate, ias, roll], confirmed
against archive/mlat_source_probe.py's use of index 9 and this file's own probe run):
  - vrate (index 7): vertical rate, baro-derived when the aircraft-dict's rate type is absent
  - source (index 9): adsb_icao / adsb_icao_nt / mlat / tisb_* / mode_s / other -- the position
    provenance. prc.adsb keeps this for nothing; archive/mlat_source_probe.py showed it matters
    for Rome/Paris ground coverage.
  - geom_alt, geom_vrate, ias, roll (10-13): present far less often; kept when there.
  - the sparse aircraft-dict (index 8): prc.adsb reads ONLY its "flight" key (carried forward
    as callsign). Everything else in it -- squawk, emergency, category, nav_modes,
    nav_altitude_mcp, nav_heading, nav_qnh, alert, spi, messages, seen, rssi, nic, rc, version,
    dbFlags, gva, sda, nac_p, nac_v, sil, sil_type -- is dropped. This script keeps the WHOLE
    dict as a JSON string column (ac_json) rather than guessing which keys matter: the dict's
    keyset differs by receiver/version and only appears on a fraction of points (when a value
    changes), so a fixed column list would silently drop keys this probe exists to find.

FILTER (relaxed vs prc.adsb.parse_trace, documented in docs/LAB-NOTES.md #1):
  same ten-airport lat/lon box (prc.adsb.BOXES); on altitude, KEEPS one more case than
  prc.adsb: alt is None (no altitude at all) is now KEPT, not dropped. adsb_probe_dropped.py's
  2026-07-13 probe found these no-altitude points are disproportionately MLAT and disproportionately
  slow (< 40 kt) -- i.e. plausibly ground MLAT fixes with no reported altitude, exactly the
  "ground coverage for aircraft without ADS-B out" population step 2 asks about. Numeric
  altitude keeps prc.adsb's <= 3,000 ft cutoff (bounds size; step 1's own request is "near the
  airports", which this cutoff already serves for the climb/descent segment used to anchor
  lift-off). This is the ONLY filter change; the box, the "ground" string case, and the <=3000ft
  numeric case are unchanged from prc.adsb.parse_trace.

Same release_parts patch as archive/agent_moredata_extract.py (prod-0 vs prod-0tmp, the five
unsplit-.tar January days) applied at import time, harmless for a July day but keeps this
script correct if pointed at any other date later.

Output: one file, <date>.parquet, under --out (default data/adsb/hidden/points). Nothing else
touched; data/adsb/points (the production extraction) is not read or written.
"""
import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")

import polars as pl

import prc.adsb as A


# ---- release_parts patch (copy of archive/agent_moredata_extract.py's fix) ----------------
def _parts(date: str, suffix: str):
    tag = f"v{date.replace('-', '.')}-planes-readsb-{suffix}"
    api = f"https://api.github.com/repos/adsblol/globe_history_{date[:4]}/releases/tags/{tag}"
    req = urllib.request.Request(api, headers={**A.UA, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            rel = json.load(r)
    except urllib.error.HTTPError as exc:
        return f"{tag}: HTTP {exc.code}"
    parts = sorted((a["name"], a["browser_download_url"], int(a["size"])) for a in rel["assets"]
                   if a["name"].startswith(tag + ".tar.") or a["name"] == tag + ".tar")
    return [(url, size) for _, url, size in parts] if parts else f"{tag}: no tar assets"


def release_parts(date: str) -> list[tuple[str, int]]:
    got = [_parts(date, s) for s in ("prod-0", "prod-0tmp")]
    found = [p for p in got if isinstance(p, list)]
    if found:
        return max(found, key=lambda p: sum(s for _, s in p))
    raise RuntimeError("; ".join(got))


A.release_parts = release_parts


# ---- full-field parse, monkeypatched into prc.adsb the way adsb_probe_dropped.py does -----
COLS = ("apt", "icao", "callsign", "reg", "actype", "t", "lat", "lon", "alt_raw", "alt_num",
        "gs", "track", "flags", "vrate", "source", "geom_alt", "geom_vrate", "ias", "roll", "ac_json")


def parse_trace_full(raw: bytes, rows: list) -> int:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    obj = json.loads(raw)
    trace = obj.get("trace") or []
    if not trace:
        return 0
    hit = False
    for pt in trace[::25]:
        la, lo = pt[1], pt[2]
        if la is not None and lo is not None and A.LAT_MIN <= la <= A.LAT_MAX and A.LON_MIN <= lo <= A.LON_MAX:
            hit = True
            break
    if not hit:
        la, lo = trace[-1][1], trace[-1][2]
        if la is None or lo is None or not (A.LAT_MIN <= la <= A.LAT_MAX and A.LON_MIN <= lo <= A.LON_MAX):
            return 0
    base = float(obj.get("timestamp") or 0.0)
    icao, reg, actype = obj.get("icao"), obj.get("r"), obj.get("t")
    cs, kept = None, 0
    for pt in trace:
        ac = pt[8] if len(pt) > 8 and isinstance(pt[8], dict) else None
        if ac:
            fl = ac.get("flight")
            if fl:
                cs = fl.strip() or cs
        la, lo = pt[1], pt[2]
        if la is None or lo is None or not (A.LAT_MIN <= la <= A.LAT_MAX and A.LON_MIN <= lo <= A.LON_MAX):
            continue
        alt = pt[3]
        ground = alt == "ground"
        no_alt = alt is None
        numeric_low = isinstance(alt, (int, float)) and alt <= A.LOW_ALT_FT
        if not (ground or no_alt or numeric_low):
            continue
        apt = A._airport(la, lo)
        if apt is None:
            continue
        rows.append((
            apt, icao, cs, reg, actype, base + pt[0], la, lo,
            "ground" if ground else ("none" if no_alt else None),
            None if ground or no_alt else float(alt),
            pt[4] if len(pt) > 4 else None,
            pt[5] if len(pt) > 5 else None,
            pt[6] if len(pt) > 6 else None,
            pt[7] if len(pt) > 7 else None,
            pt[9] if len(pt) > 9 else None,
            pt[10] if len(pt) > 10 else None,
            pt[11] if len(pt) > 11 else None,
            pt[12] if len(pt) > 12 else None,
            pt[13] if len(pt) > 13 else None,
            json.dumps(ac) if ac else None,
        ))
        kept += 1
    return kept


def to_frame_full(rows: list) -> pl.DataFrame:
    schema = {"apt": pl.Utf8, "icao": pl.Utf8, "callsign": pl.Utf8, "reg": pl.Utf8, "actype": pl.Utf8,
              "t": pl.Float64, "lat": pl.Float32, "lon": pl.Float32, "alt_raw": pl.Utf8, "alt_num": pl.Float32,
              "gs": pl.Float32, "track": pl.Float32, "flags": pl.Int32, "vrate": pl.Float32, "source": pl.Utf8,
              "geom_alt": pl.Float32, "geom_vrate": pl.Float32, "ias": pl.Float32, "roll": pl.Float32,
              "ac_json": pl.Utf8}
    if not rows:
        return pl.DataFrame(schema=schema)
    cols = list(zip(*rows))
    data = {}
    for name, series in zip(COLS, cols):
        dt = schema[name]
        if dt in (pl.Float32, pl.Float64):
            series = [None if v is None else float(v) for v in series]
        elif dt == pl.Int32:
            series = [None if v is None else int(v) for v in series]
        data[name] = pl.Series(series, dtype=dt)
    return pl.DataFrame(data)


A.parse_trace = parse_trace_full
A.to_frame = to_frame_full


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", default="data/adsb/hidden/points")
    ap.add_argument("--mb", type=float, default=0, help="probe-limit MB; 0 = full day")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.date}.parquet"
    if target.exists():
        print(f"{args.date}: exists, skipped -> {target}")
        return
    t0 = time.time()
    frame, s = A.stream_day(args.date, limit_bytes=(int(args.mb * 1e6) or None), log=lambda m: print(m, flush=True))
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(target)
    print(f"{args.date}: {frame.height:,} points from {s['traces']:,} traces, {s['bytes'] / 1e9:.2f} GB streamed in "
          f"{(time.time() - t0) / 60:.1f} min ({s['mb_per_s']:.0f} MB/s), parse {s['parse_s'] / 60:.1f} min, "
          f"{s['errors']} errors -> {target} ({target.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
