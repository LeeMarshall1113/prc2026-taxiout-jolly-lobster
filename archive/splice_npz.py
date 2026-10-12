"""Splice two stage-1 blend files by airport: treatment rows at the named airports, control rows elsewhere.

    python archive/splice_npz.py CONTROL.npz TREAT.npz AIRPORTS OUT.npz

Serving equivalent: run the ADS-B stage twice and take the treatment's predictions only at those airports -- used when
a lever helps its own airport but disturbs others through the shared floor-stack training pool.
"""
import sys

import numpy as np

fc, ft, apts, out = sys.argv[1:5]
c, t = np.load(fc, allow_pickle=True), np.load(ft, allow_pickle=True)
oc, ot = np.argsort(c["ids"], kind="stable"), np.argsort(t["ids"], kind="stable")
assert (c["ids"][oc] == t["ids"][ot]).all()
m = np.isin(c["apts"][oc].astype(str), apts.split(","))
res = {k: c[k][oc] for k in c.keys()}
for k in ("both_guard",):
    res[k] = np.where(m, t[k][ot], c[k][oc])
np.savez(out, **res)
print(f"spliced {m.sum():,} rows at {apts} from {ft} into {fc} -> {out}")
