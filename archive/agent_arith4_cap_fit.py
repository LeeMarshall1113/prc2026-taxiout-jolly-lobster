"""arith4 capacity test (2026-10-06). The round-three candidate arm fitted with more model capacity; nothing else changes (same columns, folds, seeds).
Variants, fixed before any scoring (env ARITH4_VAR): nr4000 = round cap 4,000 instead of 1,500 (stored fits reach 1,498, i.e. some stop at the cap);
slow = learning rate 0.015, round cap 8,000; big = 1,023 leaves, min 100 rows per leaf, round cap 4,000. Early stopping unchanged.
usage: ARITH4_VAR=<variant> ARITH_OUT=<folder for this variant> python archive/agent_arith4_cap_fit.py ARM:lomo:MONTH:SEED[,...]
Baseline for the comparison: the same arm at the standard settings (data/interim/agent_arith3_out/)."""
import os
import sys
sys.path.insert(0, "archive")
import agent_arith3_02_fit as F3   # noqa: E402
F = F3.F
V = os.environ["ARITH4_VAR"]
if V == "nr4000":
    F.NR = 4000
elif V == "slow":
    F.NR = 8000
    _p = F.params
    F.params = lambda seed, reduced: dict(_p(seed, reduced), learning_rate=0.015)
elif V == "big":
    F.NR = 4000
    F.CFG.update(num_leaves=1023, min_data_in_leaf=100)
else:
    raise SystemExit("unknown variant " + V)
F.P("variant", V, "NR", F.NR, "params", {k: v for k, v in F.params(0, 0).items() if k in ("learning_rate", "num_leaves", "min_data_in_leaf")})
if __name__ == "__main__":
    for j in sys.argv[1].split(","):
        p = j.split(":")
        assert p[1] == "lomo"
        F3.lomo3(p[0], int(p[2]), int(p[3]))
