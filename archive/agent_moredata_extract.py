"""experiment moredata: prc.adsb extract, with the two release-naming cases prc.adsb.release_parts misses.

    .venv/Scripts/python.exe archive/agent_moredata_extract.py check   --dates 2025-01-01,2025-06-05,...   # metadata only
    .venv/Scripts/python.exe archive/agent_moredata_extract.py extract --dates ... [--workers 3] [--out data/adsb/points]

Found 2026-09-23 from the GitHub release listing of adsblol/globe_history_2025 (metadata only):
  1. 2025-01-01, -01-11, -01-18, -01-19, -01-21 are published as ONE unsplit asset '<tag>.tar' (1.68-2.00 GB).
     release_parts keeps only names starting '<tag>.tar.' -> "no .tar.* assets" (logs/extract*.log). These are the five
     January gaps.
  2. 2025-05-28 .. 2025-06-10 exist ONLY under the tag suffix 'prod-0tmp' (split .tar.aa/.ab as usual, ~3.2-3.3 GB).
     release_parts asks for 'prod-0' -> HTTP 404, which the process pool then reports as "cannot pickle
     'BufferedReader'" (the HTTPError carries its socket). These are the four May gaps AND the first ten June days.
  3. 2025-06-11: prod-0 is a PARTIAL day (0.73 GB, one unsplit .tar); prod-0tmp has the full day (3.49 GB).
  4. 2025-12-31 has no release at all (prod, staging or tmp).
This wrapper replaces prc.adsb.release_parts with one that reads prod-0 AND prod-0tmp, keeps the larger, accepts '<tag>.tar' as a
single part; everything else is prc.adsb unchanged (same parser, same kept fields, same output). The patch is applied at
import time so the spawned worker processes (which re-import this module as __mp_main__) get it too.
"""
import json
import sys
import urllib.error
import urllib.request

sys.path.insert(0, ".")
import prc.adsb as A


def _parts(date: str, suffix: str):
    """[(url, size)] for one tag, or an error string (never raises)."""
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
    """Try prod-0 and prod-0tmp; the LARGER release wins. On 2025-06-11 prod-0 is a 0.73 GB partial day (one unsplit
    .tar) while prod-0tmp holds the full 3.49 GB day. Raises a plain (picklable) RuntimeError when neither exists."""
    got = [_parts(date, s) for s in ("prod-0", "prod-0tmp")]
    found = [p for p in got if isinstance(p, list)]
    if found:
        return max(found, key=lambda p: sum(s for _, s in p))
    raise RuntimeError("; ".join(got))


A.release_parts = release_parts

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        dates = sys.argv[sys.argv.index("--dates") + 1].split(",")
        for d in dates:
            try:
                parts = release_parts(d)
                print(f"{d}: {len(parts)} part(s), {sum(s for _, s in parts) / 1e9:.2f} GB  {parts[0][0].rsplit('/', 1)[-1]}")
            except Exception as exc:
                print(f"{d}: NOT AVAILABLE -- {exc}")
    else:
        sys.argv = [sys.argv[0]] + sys.argv[1:]
        A.main()
