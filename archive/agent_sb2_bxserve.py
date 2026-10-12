"""experiment sb2 serve: COPY of archive/agent_restack5_bxserve.py (docs/LAB-NOTES.md) importing the sb2
re-stack module (tokens mw, lw, unc, osm, occ): osm/occ columns from --osm-*/--occ-* parquets, sample weights from
scripts/agent_sb2_common.weights on the training rows, unc_w = 10-fold OOF widths on the training rows (cached per
train npz) and full-data quantile heads applied to the 2026 rows with THIS stream's fin. Without the new tokens it is
the original bit for bit. Original docstring follows.

experiment restack5 bxserve: COPY of archive/agent_backx_04_serve.py (docs/LAB-NOTES.md) using
archive/agent_restack5_bxrestack.py (token big) and adding the nbr token (2026 columns built from --s0 and --in exactly as
archive/agent_restack4_serve.py / agent_neighbour_07_restack_serve.py). Without big/nbr it is agent_backx_04_serve.py.
Original docstring follows.

experiment backx 04: COPY of archive/agent_smallbundle_04_serve.py (docs/LAB-NOTES.md) using archive/agent_backx_restack.py
(recipe token `bx`, columns from --bx-train (2025, out-of-fold) / --bx-2026 (fitted on all 2025, label-free)).
Without `bx` in the recipe this is agent_smallbundle_04_serve.py. Original docstring follows.

experiment smallbundle 04: COPY of archive/agent_restack3_04_serve.py (docs/LAB-NOTES.md) using
archive/agent_smallbundle_restack.py (recipe token `seq`, columns from --seq-train/--seq-2026) and --revert-lirf: rows at
LIRF with a plan (AOBT_3_flt in data/raw/ranking.parquet) and fin in [1200, 2400) s get NO re-stack correction (member M3).
With neither option this is agent_restack3_04_serve.py. Original docstring follows.

experiment restack3 04: COPY of archive/agent_expensive_04_restack_serve.py adding --recipe (docs/LAB-NOTES.md,
archive/agent_restack3_restack.py). --recipe '' (default) runs the ORIGINAL implementation (scripts/agent_expensive_restack)
unchanged, so the default path is identical to the original script. A non-empty recipe needs --feats est_v2 and uses
--evt-train/--trk-train (2025 columns) and --evt-2026/--trk-2026 (2026 columns). Original docstring follows.

experiment expensive 04: SERVE the re-stack on a 2026 ADS-B-stage output (docs/LAB-NOTES.md).

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
import archive.agent_sb2_restack as R3
from archive.agent_sb2_common import weights as sb2_weights
from archive.agent_neighbour_nbr2026 import build as build_nbr2026

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
ap.add_argument("--recipe", default="")
ap.add_argument("--evt-train", default="data/interim/agent_restack3_evt_2025.parquet")
ap.add_argument("--trk-train", default="data/interim/agent_restack3_trk_2025.parquet")
ap.add_argument("--evt-2026", default="data/interim/agent_restack3_evt_2026.parquet")
ap.add_argument("--trk-2026", default="data/interim/agent_restack3_trk_2026.parquet")
ap.add_argument("--seq-train", default="data/interim/agent_smallbundle_seq_2025.parquet")
ap.add_argument("--seq-2026", default="data/interim/agent_smallbundle_seq_2026.parquet")
ap.add_argument("--revert-lirf", action="store_true")
ap.add_argument("--bx-train", default="data/interim/agent_backx_bx_2025.parquet")
ap.add_argument("--bx-2026", default="data/interim/agent_backx_bx_2026.parquet")
ap.add_argument("--nbr-train", default="data/interim/agent_neighbour_nbr_2025.parquet")
ap.add_argument("--s0", default="")
ap.add_argument("--nbr-2026-out", default="")
ap.add_argument("--osm-train", default="data/interim/agent_sb2_osm_2025.parquet")
ap.add_argument("--osm-2026", default="data/interim/agent_sb2_osm_2026.parquet")
ap.add_argument("--occ-train", default="data/interim/agent_sb2_occ_2025.parquet")
ap.add_argument("--occ-2026", default="data/interim/agent_sb2_occ_2026.parquet")
ap.add_argument("--unc-cache", default="")
ap.add_argument("--nov-train", default="data/interim/agent_nov_train.parquet")
ap.add_argument("--nov-2026", default="data/interim/agent_nov_2026.parquet")
a = ap.parse_args()
if R3.parse(a.recipe):
    assert a.feats == "est_v2", "a recipe extends est_v2"
    _rc = a.recipe
    load_rows = None  # noqa: F811  (rebound below per year)
    features = lambda r, fin, feats: R3.features(r, fin, _rc)  # noqa: E731,F811
    fit = lambda X, y, seed: R3.fit(X, y, _rc, seed, w=_SB2W)  # noqa: E731,F811
    predict = R3.predict  # noqa: F811
    load_train = lambda p: R3.load_rows(p, _rc, a.evt_train, a.trk_train, a.seq_train, a.nbr_train, a.bx_train, extra={"osm": a.osm_train, "occ": a.occ_train})  # noqa: E731
    _nb26 = None
    if "nbr" in R3.parse(a.recipe):
        assert a.s0 and a.nbr_2026_out, "recipe nbr needs --s0 and --nbr-2026-out"
        build_nbr2026(a.s0, a.inp).write_parquet(a.nbr_2026_out)
        _nb26 = a.nbr_2026_out
    load_2026 = lambda p: R3.load_rows(p, _rc, a.evt_2026, a.trk_2026, a.seq_2026, _nb26, a.bx_2026, extra={"osm": a.osm_2026, "occ": a.occ_2026})  # noqa: E731
else:
    load_train = load_2026 = load_rows
print(f"recipe '{a.recipe}'")
T0 = time.time()
_SB2T = R3.parse(a.recipe)
_SB2W = None

z = np.load(a.train_npz, allow_pickle=True)
pos_of = {k: i for i, k in enumerate(z["ids"].astype(float))}
tr = load_train(a.train_rows)
pos = np.array([pos_of.get(m, -1) for m in tr["MVT_ID_mvt"].to_list()])
tr = tr.filter(pl.Series(pos >= 0))
pos = pos[pos >= 0]
fin_tr = z[a.blend][pos].astype(float)
if "unc" in _SB2T:
    import os as _os
    import archive.agent_sb2_unc as U
    _Xq = U.R.features(tr, fin_tr, "evt,trk")
    _tq = z["y"][pos].astype(float) - fin_tr
    _uc = a.unc_cache or ("data/interim/agent_sb2_unc_serve_" + _os.path.basename(a.train_npz).replace(".npz", "") + ".npz")
    if _os.path.exists(_uc) and (np.load(_uc)["ids"] == tr["MVT_ID_mvt"].to_numpy()).all():
        _wtr = np.load(_uc)["w"]
    else:
        _wtr = U.oof_widths(_Xq, _tq, np.asarray(tr["day"].to_list()))
        np.savez(_uc, ids=tr["MVT_ID_mvt"].to_numpy(), w=_wtr)
    _HEADS = U.full_heads(_Xq, _tq)
    del _Xq, _tq
    tr = tr.with_columns(pl.Series("unc_w", _wtr))
    print(f"unc: training-row OOF widths mean {np.nanmean(_wtr):.1f} s ({time.time() - T0:.0f}s)")
if "nov" in _SB2T:
    _nv = pl.read_parquet(a.nov_train).select(C("MVT_ID_mvt").cast(pl.Float64), C("nov").cast(pl.Float64).alias("nov_w"))
    tr = tr.join(_nv, on="MVT_ID_mvt", how="left", maintain_order="left")
    print(f"nov: training rows with a value {tr['nov_w'].is_not_null().mean():.3f}")
_SB2W = sb2_weights(tr, _SB2T)
m = fit(features(tr, fin_tr, a.feats), z["y"][pos].astype(float) - fin_tr, a.seed)
print(f"trained on {tr.height:,} rows / {tr['day'].n_unique()} days ({time.time() - T0:.0f}s)")

sub = pl.read_parquet(a.inp)
assert sub[ID].n_unique() == sub.height
r26 = load_2026(a.rows_2026)
j = r26.join(sub.select(C(ID).cast(pl.Float64), C(T).cast(pl.Float64).alias("_fin")), on=ID, how="inner")
if "unc" in _SB2T:
    j = j.with_columns(pl.Series("unc_w", U.width(_HEADS, U.R.features(j, j["_fin"].to_numpy(), "evt,trk"))))
if "nov" in _SB2T:
    j = j.join(pl.read_parquet(a.nov_2026).select(C("MVT_ID_mvt").cast(pl.Float64), C("nov").cast(pl.Float64).alias("nov_w")),
               on="MVT_ID_mvt", how="left", maintain_order="left")
skip = np.isin(np.asarray(j["apt"].to_list()), [s for s in a.skip.split(",") if s])
fin = j["_fin"].to_numpy()
add = np.where(skip, 0.0, predict(m, features(j, fin, a.feats)))
if a.revert_lirf:
    pl_ = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
           .select(C(ID).cast(pl.Float64), C("AOBT_3_flt").is_not_null().alias("_plan")).collect().unique(subset=ID))
    plan = j.select(ID).join(pl_, on=ID, how="left", maintain_order="left")["_plan"].fill_null(False).to_numpy()
    cell = (np.asarray(j["apt"].to_list()) == "LIRF") & plan & (fin >= 1200) & (fin < 2400)
    add = np.where(cell, 0.0, add)
    print(f"revert-lirf: {int(cell.sum()):,} LIRF plan rows with fin in [1200,2400) get no re-stack correction")
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
