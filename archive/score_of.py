"""Print SCORED <file> <score> (delta vs prior) and the top 4, if the file has been scored. Read-only."""
import sys

from prc.leaderboard import fetch_all, team_bests

fname, prior = sys.argv[1], float(sys.argv[2])
subs = fetch_all()
v = [s for s in subs if s.team_name == "jolly-lobster" and s.filename == fname]
if v:
    tb = team_bests(subs)
    rank = next(i for i, (t, _, _) in enumerate(tb, 1) if t == "jolly-lobster")
    print(f"SCORED {fname} {v[0].score:.4f} (prior {prior:.4f}, delta {v[0].score - prior:+.4f}s) rank {rank}")
    for i, (t, sc, c) in enumerate(tb[:4], 1):
        print(f"  {i}. {t:20} {sc:.4f}")
