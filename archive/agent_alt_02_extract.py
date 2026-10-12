"""Extract airport-box points from an ALTERNATE adsb.lol daily replica (tag agent_alt, 2026-09-26).

    python archive/agent_alt_02_extract.py --dates 2026-07-08 [--replica auto|staging-0|prod-0tmp|staging-0tmp]
                                           [--workers 2] [--out data/adsb/points_alt]

Same streaming/parsing as prc/adsb.py (Remote, RangeStream, trace_section_offset, parse_trace, to_frame are
imported read-only; prc/ is not modified) -- only the release tag differs. prc/adsb.py hard-codes
`-planes-readsb-prod-0`; here the tag is chosen per day from data/interim/agent_alt_releases.csv
(written by archive/agent_alt_00_meta.py): --replica auto = the first of staging-0, staging-0tmp, prod-0tmp
that has tar parts. Nothing is written to disk but the output parquet (the tar is streamed, never stored),
so there are no raw tars to clean up. Writes <out>/<date>.parquet and appends <out>/manifest.csv.
Never writes into data/adsb/points/.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, ".")
from prc.adsb import UA, Remote, RangeStream, trace_section_offset, parse_trace, to_frame  # noqa: E402

ALT_ORDER = ("staging-0", "staging-0tmp", "prod-0tmp")
REL_CSV = Path("data/interim/agent_alt_releases.csv")
PROTECTED = Path("data/adsb/points").resolve()


def pick_tag(date: str, replica: str) -> str:
    rel = {}
    with open(REL_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["date"] == date and int(r["n_parts"]) > 0:
                rel[r["replica"]] = r["tag"]
    order = ALT_ORDER if replica == "auto" else (replica,)
    for rep in order:
        if rep in rel:
            return rel[rep]
    raise RuntimeError(f"{date}: no alternate replica with tar parts (have {sorted(rel)})")


def release_parts(tag: str) -> list[tuple[str, int]]:
    year = tag[1:5]
    api = f"https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
    req = urllib.request.Request(api, headers={**UA, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        rel = json.load(r)
    parts = sorted((a["name"], a["browser_download_url"], int(a["size"])) for a in rel["assets"]
                   if a["name"].startswith(tag + ".tar."))
    if not parts:
        raise RuntimeError(f"{tag}: no .tar.* assets")
    return [(url, size) for _, url, size in parts]


def stream_tag(tag: str, log=print):
    """prc.adsb.stream_day's logic, for an explicit tag."""
    t0 = time.time()
    remote = Remote(release_parts(tag))
    off, walked, _ = trace_section_offset(remote)
    log(f"{tag}: {remote.total / 1e9:.2f} GB in {len(remote.parts)} parts; traces start at {off / 1e9:.3f} GB "
        f"({walked} headers walked in {time.time() - t0:.0f}s)")
    stream = RangeStream(remote, off)
    reader = io.BufferedReader(stream, buffer_size=8 * 1024 * 1024)
    rows, st = [], {"traces": 0, "members": 0, "kept": 0, "errors": 0}
    t1 = last = time.time()
    in_traces = False
    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for m in tar:
            st["members"] += 1
            if "traces" in m.name:
                in_traces = True
            elif in_traces:
                break
            if not m.isfile() or "trace_full" not in m.name:
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            raw = f.read()
            try:
                st["kept"] += parse_trace(raw, rows)
            except Exception:
                st["errors"] += 1
            st["traces"] += 1
            if time.time() - last > 60:
                last = time.time()
                log(f"  {tag}: {stream.bytes_read / 1e9:.2f} GB read, {st['traces']:,} traces, {st['kept']:,} kept")
    st.update(bytes=stream.bytes_read, elapsed_s=time.time() - t1)
    return to_frame(rows), st


def extract_one(date: str, replica: str, out_dir: str) -> dict:
    target = Path(out_dir) / f"{date}.parquet"
    if target.resolve().parent == PROTECTED:
        raise RuntimeError("refusing to write into data/adsb/points/")
    tag = pick_tag(date, replica)
    if target.exists():
        return {"date": date, "tag": tag, "status": "exists"}
    frame, s = stream_tag(tag, log=lambda msg: print(msg, flush=True))
    tmp = target.with_suffix(".tmp")
    frame.write_parquet(tmp)
    tmp.replace(target)
    return {"date": date, "tag": tag, "status": "ok", "points": frame.height, "traces": s["traces"],
            "gb": round(s["bytes"] / 1e9, 3), "minutes": round(s["elapsed_s"] / 60, 1), "errors": s["errors"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--replica", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="data/adsb/points_alt")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dates = [d.strip() for d in a.dates.split(",") if d.strip()]
    man = out / "manifest.csv"
    fields = ["date", "tag", "status", "points", "traces", "gb", "minutes", "errors"]
    new = not man.exists()
    print(f"extracting {len(dates)} days, replica={a.replica}, {a.workers} workers -> {out}", flush=True)
    fails = 0
    with ProcessPoolExecutor(max_workers=a.workers) as pool, open(man, "a", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=fields)
        if new:
            w.writeheader()
        futs = {pool.submit(extract_one, d, a.replica, str(out)): d for d in dates}
        for fu in as_completed(futs):
            d = futs[fu]
            try:
                r = fu.result()
            except Exception as exc:
                fails += 1
                r = {"date": d, "tag": "", "status": f"FAILED {type(exc).__name__}: {exc}"[:200]}
            print(r, flush=True)
            w.writerow(r)
            mf.flush()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
