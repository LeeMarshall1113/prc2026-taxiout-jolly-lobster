"""Compare January/July rebuilt from the final file (tags J5jj, J7jj) with the shipped lineage on the same rows.

1. Re-stack level: J7jj_restack against the stored composite of the four production pieces (EGLL a, LEMD b, LFPG/LIRF c,
   EDDF/EDDM/EHAM/LEBL/LSZH d), per airport-month: mean difference and rms difference. The January Munich shift of the stored
   pieces (apply_eddm_january) is not in the rebuilt chain, so EDDM January rows with that shift are listed separately.
2. Stage level: J5jj_v105_s1 against the stored v105_s1 (total), and the stored realisations 1 and 2 against each other
   (single-realisation pair rms). Two independent means of ten realisations differ by pair_rms / sqrt(10). The part of the
   total that is not realisation noise is attributed to the re-drawn take-off jitter: sqrt(total^2 - noise^2).

    python final/26_jj_compare.py
"""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
I = "data/interim/"
PROD = {"a": I + "agent_restack5_x_cap_evt_trk_bx_big_final.parquet", "b": I + "agent_restack5_x_cap_evt_trk_seq_bx_big_final.parquet",
        "c": I + "agent_restack5_b_cap_evt_trk_seq_big_final.parquet", "d": I + "agent_stagecap_9s_bx_big_final.parquet"}
COMP = {"EGLL": "a", "LEMD": "b", "LFPG": "c", "LIRF": "c", "EDDF": "d", "EDDM": "d", "EHAM": "d", "LEBL": "d", "LSZH": "d"}


def rd(p, n):
    return pl.read_parquet(p).select(C(ID).cast(pl.Float64), C(T).cast(pl.Float64).alias(n))


dep = (pl.scan_parquet("data/raw/final_ranking.parquet").filter((C("PHASE_mvt") == "DEP") & C("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
       .select(C(ID).cast(pl.Float64), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("m")).collect())


def stats(d, a, b):
    return (d.group_by("apt", "m").agg((C(a) - C(b)).mean().round(2).alias("mean_diff"), ((C(a) - C(b)) ** 2).mean().sqrt().round(2).alias("rms_diff"),
                                       pl.len().alias("rows")).sort("apt", "m"))


# 1. re-stack level
new = rd("data/final/J7jj_restack.parquet", "new")
parts = []
for pc, p in PROD.items():
    f = rd(p, "old").join(dep.filter(C("apt").is_in([a for a, q in COMP.items() if q == pc])), on=ID)
    parts.append(f)
old = pl.concat(parts)
d = old.join(new, on=ID)
print("re-stack level, J7jj against the stored composite (seconds):")
with pl.Config(tbl_rows=100):
    print(stats(d, "new", "old"))
    ed = d.filter((C("apt") == "EDDM") & (C("m") == 1))
    print("EDDM January rows differing by exactly -38 (stored Munich shift):", int(((ed["new"] - ed["old"]).round(0) == -38).sum()), "of", ed.height)

# 2. stage level
tot_new = rd("data/final/J5jj_v105_s1.parquet", "new")
tot_old = rd(I + "v105_s1.parquet", "old")
r1, r2 = rd(I + "v105_adsb_1.parquet", "r1"), rd(I + "v105_adsb_2.parquet", "r2")
s = dep.join(tot_new, on=ID).join(tot_old, on=ID).join(r1, on=ID).join(r2, on=ID)
t = (s.group_by("apt", "m").agg(((C("new") - C("old")) ** 2).mean().sqrt().alias("total_rms"), ((C("r1") - C("r2")) ** 2).mean().sqrt().alias("pair_rms"))
     .with_columns((C("pair_rms") / 10 ** 0.5).alias("noise_rms"))
     .with_columns((C("total_rms") ** 2 - C("noise_rms") ** 2).clip(0, None).sqrt().alias("jitter_part_rms"))
     .select("apt", "m", C("total_rms").round(2), C("pair_rms").round(2), C("noise_rms").round(2), C("jitter_part_rms").round(2)).sort("apt", "m"))
print("stage level (stream 105 mean of ten, J5jj against stored v105_s1; pair = two single realisations of the stored run):")
with pl.Config(tbl_rows=100):
    print(t)
