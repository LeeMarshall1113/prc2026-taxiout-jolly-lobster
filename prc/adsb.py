"""Open ADS-B ground movements around the ten airports, from the adsb.lol archive.

    python -m prc.adsb probe   --date 2026-01-09 [--mb 60]
    python -m prc.adsb extract --dates 2025-01-06,2025-01-10 [--workers 3] [--out data/adsb/points]

DATA. adsb.lol's historical archive: one GitHub release per day in
github.com/adsblol/globe_history_2025 and globe_history_2026, tag
v{YYYY.MM.DD}-planes-readsb-prod-0, a plain tar split into <=2,000,000,000-byte parts
({tag}.tar.aa, .ab, ...). Open data, attribution adsb.lol; the project's own pages give
ODbL 1.0 in one place and CC0 in another -- both open licences. Acquisition was staged
from 2026-09-16: a probe, then ~11 days of 2025 as calibration days, then the 2026 scored
months.

Written for this repository. The approach is informed by a public write-up by another
participant (GPLv3; see the README credits). No code from it is used.

HOW A DAY IS READ -- never stored.
  A day's tar opens with ./heatmap/ members (~0.5 GB of map-replay data, nothing
  per-flight), then ./traces/<xx>/trace_full_<icao>.json, one per aircraft. Tar headers are
  walked with 512-byte HTTP range requests to find where the trace section starts; that
  section is streamed from there straight into tarfile's stream reader. A dropped
  connection resumes at the exact byte. Peak disk is the output parquet.

WHAT IS KEPT.
  Each trace is a list of points [dt, lat, lon, alt ("ground" | feet | null), gs, track,
  flags, vrate, aircraft-dict | null, ...] relative to a base timestamp. The callsign lives
  in the aircraft dict only when it changes, so it is carried forward. A point is kept if
  it lies inside one of the ten airport boxes AND is on the ground or below 3,000 ft -- the
  ground segment plus enough of the climb to find lift-off, which is how a departure is
  anchored (its inbound leg often carries the same callsign and sits on the same stand).
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

UA = {"User-Agent": "prc-taxiout-2026 research (adsb.lol open data)"}
PART_LIMIT = 2_000_000_000
LOW_ALT_FT = 3000

# Aerodrome reference points (deg). Boxes are generous: Schiphol's Polderbaan and Madrid's
# northern runways sit 5-6 km from the terminals.
ARP = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
DLAT, DLON = 0.12, 0.18
BOXES = [(code, la - DLAT, la + DLAT, lo - DLON, lo + DLON) for code, (la, lo) in ARP.items()]
LAT_MIN = min(b[1] for b in BOXES)
LAT_MAX = max(b[2] for b in BOXES)
LON_MIN = min(b[3] for b in BOXES)
LON_MAX = max(b[4] for b in BOXES)


# ---------------------------------------------------------------------------------------
# Remote access
# ---------------------------------------------------------------------------------------
def release_parts(date: str) -> list[tuple[str, int]]:
    """[(url, size)] for a day's prod-0 release, in part order."""
    year = date[:4]
    tag = f"v{date.replace('-', '.')}-planes-readsb-prod-0"
    api = f"https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
    req = urllib.request.Request(api, headers={**UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = json.load(r)
    parts = sorted((a["name"], a["browser_download_url"], int(a["size"])) for a in rel["assets"]
                   if a["name"].startswith(tag + ".tar."))
    if not parts:  # a small day is published as one unsplit file ({tag}.tar), e.g. 2026-06-11
        parts = [(a["name"], a["browser_download_url"], int(a["size"])) for a in rel["assets"] if a["name"] == tag + ".tar"]
    if not parts:
        raise RuntimeError(f"{tag}: no .tar.* assets")
    return [(url, size) for _, url, size in parts]


class Remote:
    """A split tar addressed as one byte range. Reads go through HTTP Range requests."""

    def __init__(self, parts: list[tuple[str, int]]):
        self.parts = parts
        self.total = sum(s for _, s in parts)
        self.starts = np.cumsum([0] + [s for _, s in parts])[:-1].tolist()

    def locate(self, offset: int) -> tuple[int, int]:
        for i in range(len(self.parts) - 1, -1, -1):
            if offset >= self.starts[i]:
                return i, offset - self.starts[i]
        raise ValueError(offset)

    def open_at(self, offset: int):
        i, local = self.locate(offset)
        url, size = self.parts[i]
        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={local}-{size - 1}"})
        resp = urllib.request.urlopen(req, timeout=120)
        if resp.status != 206 and local:
            raise RuntimeError(f"range request not honoured (HTTP {resp.status})")
        return resp, i, size - local

    def read_range(self, offset: int, length: int) -> bytes:
        out = bytearray()
        while len(out) < length:
            i, local = self.locate(offset + len(out))
            url, size = self.parts[i]
            end = min(size - 1, local + (length - len(out)) - 1)
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={local}-{end}"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        out += r.read()
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
        return bytes(out)


class RangeStream(io.RawIOBase):
    """Sequential reader from `offset` to the end of the split tar (or `limit` bytes).

    Survives dropped connections: on any read error it reopens at the exact byte it had
    reached. Crosses part boundaries transparently.
    """

    def __init__(self, remote: Remote, offset: int, limit: int | None = None):
        self.remote, self.pos, self.limit = remote, offset, limit
        self.start = offset
        self.resp, self.part, self.left = None, None, 0
        self.bytes_read = 0

    def readable(self):
        return True

    def _open(self):
        if self.resp is not None:
            try:
                self.resp.close()
            except Exception:
                pass
        for attempt in range(8):
            try:
                self.resp, self.part, self.left = self.remote.open_at(self.pos)
                return
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(3 * (attempt + 1))

    def readinto(self, b):
        if self.pos >= self.remote.total:
            return 0
        if self.limit is not None and self.bytes_read >= self.limit:
            return 0
        want = len(b)
        if self.limit is not None:
            want = min(want, self.limit - self.bytes_read)
        for attempt in range(8):
            try:
                if self.resp is None or self.left == 0:
                    self._open()
                chunk = self.resp.read(min(want, self.left))
                if not chunk:
                    raise IOError("empty read before part end")
                n = len(chunk)
                b[:n] = chunk
                self.pos += n
                self.left -= n
                self.bytes_read += n
                if self.left == 0:
                    self.resp.close()
                    self.resp = None
                return n
            except Exception:
                if attempt == 7:
                    raise
                self.resp = None
                time.sleep(3 * (attempt + 1))
        return 0


def _blocks(size: int) -> int:
    return (size + 511) // 512 * 512


def trace_section_offset(remote: Remote) -> tuple[int, int, int]:
    """Walk tar headers with range requests to the first member under traces/.

    Returns (offset, members walked, bytes skipped). Extension headers (pax, GNU long
    name) are kept attached to the member that follows them.
    """
    off, walked, pending = 0, 0, None
    ext = (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK)
    while off < remote.total:
        hdr = remote.read_range(off, 512)
        if hdr.count(0) == 512:
            raise RuntimeError("reached the tar end marker without finding traces/")
        info = tarfile.TarInfo.frombuf(hdr, "utf-8", "surrogateescape")
        walked += 1
        if info.type in ext:
            pending = off if pending is None else pending
            off += 512 + _blocks(info.size)
            continue
        if "traces" in info.name:
            start = pending if pending is not None else off
            return start, walked, start
        pending = None
        off += 512 + _blocks(info.size)
    raise RuntimeError("no traces/ section found")


# ---------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------
def _airport(lat: float, lon: float):
    for code, a, b, c, d in BOXES:
        if a <= lat <= b and c <= lon <= d:
            return code
    return None


COLS = ("apt", "icao", "callsign", "reg", "actype", "t", "lat", "lon", "alt", "gs", "track", "flags")


def parse_trace(raw: bytes, rows: list) -> int:
    """Append kept points of one trace_full JSON to `rows`. Returns points kept."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    obj = json.loads(raw)
    trace = obj.get("trace") or []
    if not trace:
        return 0
    # Cheap reject: an aircraft that is ever on the ground or low at one of our airports
    # leaves many consecutive points there, so a stride sample cannot miss it.
    hit = False
    for pt in trace[::25]:
        la, lo = pt[1], pt[2]
        if la is not None and lo is not None and LAT_MIN <= la <= LAT_MAX and LON_MIN <= lo <= LON_MAX:
            hit = True
            break
    if not hit:
        la, lo = trace[-1][1], trace[-1][2]
        if la is None or lo is None or not (LAT_MIN <= la <= LAT_MAX and LON_MIN <= lo <= LON_MAX):
            return 0
    base = float(obj.get("timestamp") or 0.0)
    icao, reg, actype = obj.get("icao"), obj.get("r"), obj.get("t")
    cs, kept = None, 0
    for pt in trace:
        if len(pt) > 8 and isinstance(pt[8], dict):
            fl = pt[8].get("flight")
            if fl:
                cs = fl.strip() or cs
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
        rows.append((apt, icao, cs, reg, actype, base + pt[0], la, lo, -1 if ground else int(alt),
                     pt[4] if len(pt) > 4 else None, pt[5] if len(pt) > 5 else None,
                     pt[6] if len(pt) > 6 else None))
        kept += 1
    return kept


def to_frame(rows: list) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={"apt": pl.Utf8, "icao": pl.Utf8, "callsign": pl.Utf8, "reg": pl.Utf8,
                                    "actype": pl.Utf8, "t": pl.Float64, "lat": pl.Float32, "lon": pl.Float32,
                                    "alt": pl.Int32, "gs": pl.Float32, "track": pl.Float32, "flags": pl.Int32})
    cols = list(zip(*rows))
    return pl.DataFrame({
        "apt": pl.Series(cols[0], dtype=pl.Utf8), "icao": pl.Series(cols[1], dtype=pl.Utf8),
        "callsign": pl.Series(cols[2], dtype=pl.Utf8), "reg": pl.Series(cols[3], dtype=pl.Utf8),
        "actype": pl.Series(cols[4], dtype=pl.Utf8), "t": pl.Series(cols[5], dtype=pl.Float64),
        "lat": pl.Series(cols[6], dtype=pl.Float32), "lon": pl.Series(cols[7], dtype=pl.Float32),
        "alt": pl.Series(cols[8], dtype=pl.Int32),
        "gs": pl.Series([None if v is None else float(v) for v in cols[9]], dtype=pl.Float32),
        "track": pl.Series([None if v is None else float(v) for v in cols[10]], dtype=pl.Float32),
        "flags": pl.Series([None if v is None else int(v) for v in cols[11]], dtype=pl.Int32),
    })


def stream_day(date: str, limit_bytes: int | None = None, log=print) -> tuple[pl.DataFrame, dict]:
    """Stream one day's trace section, returning kept points and run statistics."""
    t0 = time.time()
    remote = Remote(release_parts(date))
    off, walked, skipped = trace_section_offset(remote)
    t_walk = time.time() - t0
    log(f"{date}: {remote.total / 1e9:.2f} GB in {len(remote.parts)} parts; traces start at "
        f"{off / 1e9:.3f} GB ({walked} headers walked in {t_walk:.0f}s, {skipped / 1e9:.2f} GB never downloaded)")
    stream = RangeStream(remote, off, limit=limit_bytes)
    reader = io.BufferedReader(stream, buffer_size=8 * 1024 * 1024)
    rows, stats = [], {"traces": 0, "members": 0, "kept": 0, "parse_s": 0.0, "errors": 0}
    t1 = time.time()
    last = t1
    try:
        in_traces = False
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for m in tar:
                stats["members"] += 1
                # The traces/ subtree is contiguous. Archives differ in whether the
                # ~0.5 GB heatmap/ block comes before it (skipped by the header walk) or
                # after it (skipped here), so stop reading the moment the subtree ends.
                if "traces" in m.name:
                    in_traces = True
                elif in_traces:
                    stats["stopped_after_traces"] = m.name
                    break
                if not m.isfile() or "trace_full" not in m.name:
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
                p0 = time.time()
                try:
                    stats["kept"] += parse_trace(raw, rows)
                except Exception:
                    stats["errors"] += 1
                stats["parse_s"] += time.time() - p0
                stats["traces"] += 1
                if time.time() - last > 60:
                    last = time.time()
                    log(f"  {date}: {stream.bytes_read / 1e9:.2f} GB read, {stats['traces']:,} traces, "
                        f"{stats['kept']:,} points kept")
    except (tarfile.ReadError, EOFError) as exc:
        if limit_bytes is None:
            raise
        stats["truncated"] = f"stopped at the probe limit ({type(exc).__name__})"
    elapsed = time.time() - t1
    stats.update(bytes=stream.bytes_read, elapsed_s=elapsed, walk_s=t_walk,
                 mb_per_s=stream.bytes_read / 1e6 / max(elapsed, 1e-9), skipped_bytes=skipped)
    return to_frame(rows), stats


# ---------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------
def cmd_probe(args):
    frame, s = stream_day(args.date, limit_bytes=int(args.mb * 1e6))
    print(f"\nPROBE {args.date}: read {s['bytes'] / 1e6:.0f} MB in {s['elapsed_s']:.0f}s "
          f"({s['mb_per_s']:.1f} MB/s incl. parsing); {s['members']:,} members, {s['traces']:,} traces, "
          f"parse {s['parse_s']:.1f}s, errors {s['errors']}{'; ' + s['truncated'] if 'truncated' in s else ''}")
    print(f"  kept {frame.height:,} points")
    if frame.height:
        by = frame.group_by("apt").agg(pl.len().alias("points"), pl.col("icao").n_unique().alias("aircraft"),
                                       (pl.col("alt") == -1).mean().alias("ground_share"),
                                       pl.col("callsign").is_not_null().mean().alias("callsign_fill"))
        print(by.sort("points", descending=True))
        print("  sample rows:")
        print(frame.filter(pl.col("alt") == -1).head(5))
    per_trace = s["parse_s"] / max(s["traces"], 1)
    print(f"  parse cost {per_trace * 1e3:.2f} ms/trace; network+tar {s['elapsed_s'] - s['parse_s']:.0f}s")


def _extract_one(date: str, out_dir: str) -> str:
    target = Path(out_dir) / f"{date}.parquet"
    if target.exists():
        return f"{date}: exists, skipped"
    frame, s = stream_day(date, log=lambda msg: print(msg, flush=True))
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(target)
    return (f"{date}: {frame.height:,} points from {s['traces']:,} traces, {s['bytes'] / 1e9:.2f} GB in "
            f"{s['elapsed_s'] / 60:.1f} min ({s['mb_per_s']:.0f} MB/s), parse {s['parse_s'] / 60:.1f} min, "
            f"{s['errors']} errors, {s['skipped_bytes'] / 1e9:.2f} GB skipped")


def cmd_extract(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    stop = Path(args.stop_file) if args.stop_file else None
    print(f"extracting {len(dates)} days with {args.workers} workers -> {out}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        queue = list(dates)
        while queue or pending:
            while queue and len(pending) < args.workers:
                if stop is not None and stop.exists():
                    print("stop requested: no new days will start", flush=True)
                    queue.clear()
                    break
                d = queue.pop(0)
                pending[pool.submit(_extract_one, d, str(out))] = d
            if not pending:
                break
            done = next(as_completed(pending))
            d = pending.pop(done)
            try:
                print(done.result(), flush=True)
            except Exception as exc:
                print(f"{d}: FAILED -- {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--date", required=True)
    p.add_argument("--mb", type=float, default=60)
    e = sub.add_parser("extract")
    e.add_argument("--dates", required=True)
    e.add_argument("--workers", type=int, default=3)
    e.add_argument("--out", default="data/adsb/points")
    e.add_argument("--stop-file", default="")
    args = ap.parse_args()
    {"probe": cmd_probe, "extract": cmd_extract}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
