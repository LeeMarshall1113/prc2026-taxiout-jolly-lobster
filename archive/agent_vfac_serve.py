"""experiment vfac serve (generated, see archive/agent_vfac_gen.py; variant hook archive/agent_vfac_models.py; writes ADD arrays per stream/variant to one npz)

experiment sb2 serve: COPY of archive/agent_restack5_serve.py (docs/LAB-NOTES.md) importing the sb2
re-stack module (tokens mw, lw, unc, osm, occ): osm/occ columns from --osm-*/--occ-* parquets, sample weights from
scripts/agent_sb2_common.weights on the training rows, unc_w = 10-fold OOF widths on the training rows (cached per
train npz) and full-data quantile heads applied to the 2026 rows with THIS stream's fin. Without the new tokens it is
the original bit for bit. Original docstring follows.

experiment restack5 serve: COPY of archive/agent_restack4_serve.py (docs/LAB-NOTES.md) adding ONLY --seeds
S1,S2,...: the LightGBM correction is the mean over those seeds' fits (the seed bag); with --blend-cb the CatBoost half uses
the first seed. Without --seeds it is agent_restack4_serve.py exactly. Original docstring follows.

experiment restack4 serve: COPY of archive/agent_neighbour_07_restack_serve.py (docs/LAB-NOTES.md) using
archive/agent_restack4_restack.py: the recipe may contain seq / seqm (--seq-train / --seq-2026), nbr (2026 columns built
here from --s0 and --in, as the original), arr (--arr-train / --arr-2026), big; --blend-cb additionally fits the CatBoost
re-stack on the same matrix (recipe minus cap/big, plus cb) and moves each row by 0.5 * LightGBM + 0.5 * CatBoost (V4).
With a recipe of the original tokens and no --blend-cb this is agent_neighbour_07_restack_serve.py (and, without nbr,
agent_restack3_04_serve.py / agent_smallbundle_04_serve.py). Original docstring follows.

experiment expensive 04: SERVE the re-stack on a 2026 ADS-B-stage output (docs/LAB-NOTES.md).
Fits ONE re-stack on every training row (the fold npz's leave-one-day-out `both_guard` as fin, the 2025 dump's
columns), then moves each 2026 row found in --rows-2026 by the clipped prediction, fin = the row's TAXITIME_SEC_mvt in
--in. --in must be the ADS-B stage output BEFORE the post-chain. Rows not in --rows-2026 are copied.
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_expensive_restack import features, fit, load_rows, predict
import archive.agent_sb2_restack4 as R3
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
ap.add_argument("--nbr-train", default="data/interim/agent_neighbour_nbr_2025.parquet")
ap.add_argument("--seq-train", default="data/interim/agent_smallbundle_seq_2025.parquet")
ap.add_argument("--seq-2026", default="data/interim/agent_smallbundle_seq_2026.parquet")
ap.add_argument("--arr-train", default="data/interim/agent_restack4_arr_2025.parquet")
ap.add_argument("--arr-2026", default="data/interim/agent_restack4_arr_2026.parquet")
ap.add_argument("--s0", default="")
ap.add_argument("--nbr-2026-out", default="")
ap.add_argument("--blend-cb", action="store_true")
ap.add_argument("--seeds", default="")
ap.add_argument("--osm-train", default="data/interim/agent_sb2_osm_2025.parquet")
ap.add_argument("--osm-2026", default="data/interim/agent_sb2_osm_2026.parquet")
ap.add_argument("--occ-train", default="data/interim/agent_sb2_occ_2025.parquet")
ap.add_argument("--occ-2026", default="data/interim/agent_sb2_occ_2026.parquet")
ap.add_argument("--unc-cache", default="")
ap.add_argument("--variants", default="ctrl")
ap.add_argument("--nov-train", default="data/interim/agent_nov_train.parquet")
ap.add_argument("--nov-2026", default="data/interim/agent_nov_2026.parquet")
a = ap.parse_args()
SEEDS = [int(s) for s in a.seeds.split(",") if s != ""] or [a.seed]
fit_cb = None
if R3.parse(a.recipe):
    assert a.feats == "est_v2", "a recipe extends est_v2"
    _rc = a.recipe
    load_rows = None  # noqa: F811  (rebound below per year)
    features = lambda r, fin, feats: R3.features(r, fin, _rc)  # noqa: E731,F811
    fit = lambda X, y, seed: R3.fit(X, y, _rc, seed, w=_SB2W)  # noqa: E731,F811
    predict = R3.predict  # noqa: F811
    load_train = lambda p: R3.load_rows(p, _rc, a.evt_train, a.trk_train, a.nbr_train, a.seq_train, a.arr_train, extra={"osm": a.osm_train, "occ": a.occ_train})  # noqa: E731
    _nb26 = None
    if "nbr" in R3.parse(a.recipe):
        assert a.s0 and a.nbr_2026_out, "recipe nbr needs --s0 and --nbr-2026-out"
        build_nbr2026(a.s0, a.inp).write_parquet(a.nbr_2026_out)
        _nb26 = a.nbr_2026_out
    load_2026 = lambda p: R3.load_rows(p, _rc, a.evt_2026, a.trk_2026, _nb26, a.seq_2026, a.arr_2026, extra={"osm": a.osm_2026, "occ": a.occ_2026})  # noqa: E731
    if a.blend_cb:
        _cbr = ",".join(t for t in _rc.split(",") if t not in ("cap", "big")) + ",cb"
        fit_cb = lambda X, y, seed: R3.fit(X, y, _cbr, seed)  # noqa: E731
else:
    assert not a.blend_cb, "--blend-cb needs a recipe"
    load_train = load_2026 = load_rows
print(f"recipe '{a.recipe}'{' + CatBoost 50/50 blend' if fit_cb else ''}")
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
assert not {"unc", "nov"} & _SB2T, "unc/nov not supported by the vfac hook"
import archive.agent_vfac_models as V
VN = a.variants.split(",")
JS = []
for _inp in a.inp.split(","):
    sub = pl.read_parquet(_inp)
    assert sub[ID].n_unique() == sub.height
    r26 = load_2026(a.rows_2026)
    JS.append(r26.join(sub.select(C(ID).cast(pl.Float64), C(T).cast(pl.Float64).alias("_fin")), on=ID, how="inner"))
OUT = {}
for si, j in enumerate(JS):
    OUT[f"ids_{si}"] = j[ID].to_numpy(); OUT[f"apt_{si}"] = np.asarray(j["apt"].to_list()); OUT[f"fin_{si}"] = j["_fin"].to_numpy()
ytr_ = z["y"][pos].astype(float) - fin_tr
fin_all = np.concatenate([j["_fin"].to_numpy() for j in JS])
for vn in VN:
    rec = V.alt_recipe(vn, a.recipe)
    Xtr_ = R3.features(tr, fin_tr, rec)
    Xte_ = np.vstack([R3.features(j, j["_fin"].to_numpy(), rec) for j in JS])
    ad = V.fit_predict(vn, rec, Xtr_, ytr_, fin_tr, Xte_, fin_all, a.seed, R3)
    del Xtr_, Xte_
    o = 0
    for si, j in enumerate(JS):
        OUT[f"add_{vn}_{si}"] = ad[o:o + j.height]; o += j.height
    print(f"variant {vn} fitted+served ({time.time() - T0:.0f}s)", flush=True)
    np.savez(a.out + ".tmp.npz", **OUT); import os as _o; _o.replace(a.out + ".tmp.npz", a.out)
print(f"vfac serve {a.recipe} variants {VN} streams {len(JS)} -> {a.out} ({time.time() - T0:.0f}s)")
