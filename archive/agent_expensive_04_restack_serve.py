"""experiment expensive 04: SERVE the re-stack on a 2026 ADS-B-stage output (docs/LAB-NOTES.md).

    python archive/agent_expensive_04_restack_serve.py --train-npz data/interim/modal/TX_S.npz \
        --train-rows data/interim/agent_offblock_rows_c238.parquet \
        --rows-2026 data/interim/agent_offblock_rows_2026.parquet \
        --in data/interim/v<V>_s1.parquet --out data/interim/v<V>_s1rs.parquet [--feats est] [--skip LIRF]

Fits ONE re-stack on every training row (the fold npz's leave-one-day-out `both_guard` as fin, the 2025 dump's
columns), then moves each 2026 row found in --rows-2026 by the clipped prediction, fin = the row's TAXITIME_SEC_mvt in
--in. --in must be the ADS-B stage output BEFORE the post-chain (merged_chain.sh's ${P}_s1 / ${P}_gatebag_s1), because
fin in training is the stage-1 evaluator's both_guard, which is also pre-post-chain. Rows not in --rows-2026 are copied.
Prints the 2026 footprint (sum dp^2 by airport) for archive/agent_lens_09_predict.py. Never uploads anything.
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_expensive_restack import features, fit, load_rows, predict

C = pl.col
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
ap = argparse.ArgumentParser()
ap.add_argument("--train-npz", default="data/interim/modal/TX_S.npz")
ap.add_argument("--blend", default="both_guard")
ap.add_argument("--train-rows", required=True)
ap.add_argument("--rows-2026", default="data/interim/agent_offblock_rows_2026.parquet")
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--feats", choices=["est", "est_v2"], default="est")
ap.add_argument("--skip", default="")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
T0 = time.time()

z = np.load(a.train_npz, allow_pickle=True)
pos_of = {k: i for i, k in enumerate(z["ids"].astype(float))}
tr = load_rows(a.train_rows)
pos = np.array([pos_of.get(m, -1) for m in tr["MVT_ID_mvt"].to_list()])
tr = tr.filter(pl.Series(pos >= 0))
pos = pos[pos >= 0]
fin_tr = z[a.blend][pos].astype(float)
m = fit(features(tr, fin_tr, a.feats), z["y"][pos].astype(float) - fin_tr, a.seed)
print(f"trained on {tr.height:,} rows / {tr['day'].n_unique()} days ({time.time() - T0:.0f}s)")

sub = pl.read_parquet(a.inp)
assert sub[ID].n_unique() == sub.height
r26 = load_rows(a.rows_2026)
j = r26.join(sub.select(C(ID).cast(pl.Float64), C(T).cast(pl.Float64).alias("_fin")), on=ID, how="inner")
skip = np.isin(np.asarray(j["apt"].to_list()), [s for s in a.skip.split(",") if s])
fin = j["_fin"].to_numpy()
add = np.where(skip, 0.0, predict(m, features(j, fin, a.feats)))
new = np.maximum(fin + add, 0.0)
upd = pl.DataFrame({ID: j[ID], "_new": new})
out = (sub.with_columns(C(ID).cast(pl.Float64).alias("_k")).join(upd.rename({ID: "_k"}), on="_k", how="left")
       .with_columns(pl.when(C("_new").is_not_null()).then(C("_new").round(0)).otherwise(C(T).cast(pl.Float64))
                     .cast(sub[T].dtype).alias(T)).select(sub.columns))
assert out.height == sub.height
out.write_parquet(a.out)
dp = out[T].cast(pl.Float64).to_numpy() - sub[T].cast(pl.Float64).to_numpy()
print(f"2026 rows in the dump {r26.height:,}; matched to --in {j.height:,}; moved {int((dp != 0).sum()):,}; "
      f"mean dp {dp.mean():+.2f} s; sum dp^2 {float((dp ** 2).sum()):.3e}")
aa = np.asarray(j["apt"].to_list())
print("footprint by airport (rows, mean dp, sum dp^2): " + "; ".join(
    f"{x} {int((aa == x).sum())} {float(add[aa == x].mean()):+.1f} {float((add[aa == x] ** 2).sum()):.2e}" for x in sorted(set(aa))))
print(f"-> {a.out} ({time.time() - T0:.0f}s)")
