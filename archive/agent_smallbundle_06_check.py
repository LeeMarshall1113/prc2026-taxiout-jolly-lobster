"""experiment smallbundle 06: build checks (docs/LAB-NOTES.md).
    python archive/agent_smallbundle_06_check.py CANDIDATE REFERENCE
ids equal, row count equal, schema equal, corr > 0.99, |mean shift| < 5 s, no negatives, no nulls; rows moved by airport.
"""
import sys

import numpy as np
import polars as pl

C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
cand, ref = sys.argv[1], sys.argv[2]
a = pl.read_parquet(cand)
b = pl.read_parquet(ref)
ok = True


def chk(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {info}")


print(f"CHECK {cand} vs {ref}")
chk("rows 344,841", a.height == 344_841 and b.height == 344_841, f"({a.height:,} / {b.height:,})")
chk("schema equal", a.schema == b.schema, str(a.schema))
chk("ids unique", a[ID].n_unique() == a.height)
a, b = a.sort(ID), b.sort(ID)
chk("ids equal", a[ID].equals(b[ID]))
x, y = a[T].cast(pl.Float64).to_numpy(), b[T].cast(pl.Float64).to_numpy()
chk("no nulls", a[T].null_count() == 0)
chk("no negatives", (x >= 0).all(), f"(min {x.min():.0f})")
cc = float(np.corrcoef(x, y)[0, 1])
chk("corr > 0.99", cc > 0.99, f"({cc:.6f})")
ms = float((x - y).mean())
chk("|mean shift| < 5 s", abs(ms) < 5, f"({ms:+.3f} s)")
d = x - y
mv = d != 0
print(f"  moved {int(mv.sum()):,} rows; rms dp (moved) {float(np.sqrt((d[mv] ** 2).mean())) if mv.any() else 0:.1f} s; "
      f"max |dp| {float(np.abs(d).max()):.0f} s; sum dp^2/row {float((d ** 2).sum()) / len(d):.1f}")
apt = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP").select(C(ID).cast(pl.Float64), C("ADEP_mvt").alias("apt"))
       .collect())
j = a.select(C(ID).cast(pl.Float64)).with_columns(pl.Series("d", d)).join(apt, on=ID, how="left")
print(j.group_by("apt").agg(n=pl.len(), moved=(C("d") != 0).sum(), mean_dp=C("d").mean().round(2),
                            rms_dp=(C("d") ** 2).mean().sqrt().round(1)).sort("apt"))
print("ALL CHECKS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
