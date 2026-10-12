"""Gates and actions for archive/ship_v20.sh. Each subcommand exits 0 on pass, 1 on fail.

    sanity  OLD NEW          NEW is a plausible successor of OLD: same IDs, corr >= 0.95,
                             median |diff| <= 150s, means within 40s. check_submission.py
                             only prints its verdict, and a broken build can pass the
                             template check while predicting garbage.
    changed OLD NEW LO HI    the number of rows that differ is within [LO, HI].
    upload  PATH             put PATH in the team bucket (prc.data.upload_submission, which
                             handles credentials itself; nothing here reads them).
    poll    FILENAME MINUTES wait for FILENAME to be scored on the public board, print it.
"""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, ".")
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def pair(old, new):
    a = pl.read_parquet(old).rename({T: "old"})
    b = pl.read_parquet(new).rename({T: "new"})
    j = a.join(b, on=ID, how="inner")
    if not (a.height == b.height == j.height):
        print(f"row sets differ: {a.height:,} / {b.height:,} / joined {j.height:,}")
        sys.exit(1)
    return j["old"].to_numpy().astype(float), j["new"].to_numpy().astype(float)


cmd = sys.argv[1]
if cmd == "sanity":
    o, n = pair(sys.argv[2], sys.argv[3])
    corr = float(np.corrcoef(o, n)[0, 1])
    med = float(np.median(np.abs(n - o)))
    dmean = float(n.mean() - o.mean())
    ok = corr >= 0.95 and med <= 150 and abs(dmean) <= 40
    print(f"sanity {'PASS' if ok else 'FAIL'}: corr {corr:.4f} (>=0.95), median |diff| {med:.1f}s (<=150), "
          f"mean shift {dmean:+.1f}s (+/-40); rows changed {int((o != n).sum()):,}")
    sys.exit(0 if ok else 1)

elif cmd == "changed":
    o, n = pair(sys.argv[2], sys.argv[3])
    k, lo, hi = int((o != n).sum()), int(sys.argv[4]), int(sys.argv[5])
    ok = lo <= k <= hi
    print(f"changed {'PASS' if ok else 'FAIL'}: {k} rows differ (allowed {lo}-{hi})")
    sys.exit(0 if ok else 1)

elif cmd == "upload":
    from prc.data import upload_submission
    upload_submission(Path(sys.argv[2]))
    sys.exit(0)

elif cmd == "poll":
    from prc.leaderboard import fetch_all
    name, deadline = sys.argv[2], time.time() + 60 * int(sys.argv[3])
    while time.time() < deadline:
        try:
            subs = fetch_all()
        except Exception as exc:  # the API is flaky; keep polling
            print(f"poll error: {exc}")
            time.sleep(120)
            continue
        mine = [s for s in subs if s.team_name == "jolly-lobster" and s.filename.endswith(name)]
        if mine:
            s = mine[0]
            ours = sorted((x.score for x in subs if x.team_name == "jolly-lobster"))
            best = {}
            for x in subs:
                best[x.team_name] = min(best.get(x.team_name, 1e9), x.score)
            rank = 1 + sum(1 for v in best.values() if v < best["jolly-lobster"])
            print(f"SCORED {s.filename}: {s.score:.4f} at {s.processed_at} (used_pairs {s.used_pairs:,}); "
                  f"team best {ours[0]:.4f}, rank {rank} of {len(best)}")
            sys.exit(0)
        time.sleep(300)
    print(f"{name} not scored within {sys.argv[3]} minutes")
    sys.exit(1)

else:
    print(f"unknown subcommand {cmd}")
    sys.exit(2)
