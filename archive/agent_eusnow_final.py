"""experiment eusnow (COPY of agent_ltreg_final.py; --snow-apts eus|ltfm: snow columns at LTFM+4 airports, or LTFM only = control):
experiment ltreg: COPY of archive/agent_kg_ltfm_final.py + the LTFM SNOW_FEATURES (docs/LAB-NOTES.md serving).
Only --ltfm-seq on is meaningful here; the extra 5 columns ride with the SEQ columns everywhere.
Original: prc.final with the LTFM take-off-sequence family -- COPY-BASED, prc/ untouched (tag "kg-ltfm", 2026-09-25).

    python archive/agent_kg_ltfm_final.py --ltfm-seq on  --out preds_final_treat.parquet <prc.final flags>
    python archive/agent_kg_ltfm_final.py --ltfm-seq off --out preds_final_ctrl.parquet  <prc.final flags>

Serving twin of archive/agent_ltfm_crossval.py, for the LTFM-only delta (treat - ctrl) on the ranking set:
  * feature list = FEATURES minus --basedrop (prc.final.DROP is REPLACED by it), exactly crossval's `--drop` rule, so
    the final model sees the same 80 (ctrl) / 99 (treat) columns as the fold arms. No --keep, no --drop-extra.
  * training rows: SEQ features for 2025 month m use T fitted on 2025 months != m (leave-own-month-out), as the fold
    wrapper does for a training month when nothing is held out.
  * ranking rows: T fitted on ALL 2025 LTFM LOBT rows; features built on ranking.parquet's LTFM departures + arrivals
    in ONE frame (Jan+Jul 2026), level columns demeaned per calendar month over the frame (mode on).
  * bulk model only (NOPLAN_DROP extended), NaN at the other nine airports.
Writes --out with MVT_ID_mvt, ADEP_mvt, prepost (routed, pre regime correction) and final (post regime correction,
float, before clip/rint). NEVER pass --upload (refused here).
"""
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_ltfm_seqfeat import SEQ_FEATURES as _SEQ, build as seq_build, fit_T  # noqa: E402
from agent_ltreg_feat import SNOW_FEATURES  # noqa: E402
from agent_eusnow_feat import snow_features_apts, ALL_APTS, NEW_APTS  # noqa: E402
SEQ_FEATURES = list(_SEQ) + list(SNOW_FEATURES)

argv = list(sys.argv[1:])
if "--upload" in argv:
    raise SystemExit("--upload is refused in this wrapper")


def _pop(flag):
    i = argv.index(flag)
    v = argv[i + 1]
    del argv[i:i + 2]
    return v


MODE = _pop("--ltfm-seq")
SNOW_APTS = ALL_APTS if _pop("--snow-apts") == "eus" else ["LTFM"]   # eus = LTFM+EHAM+LSZH+EDDM+EDDF; ltfm = v117 control
SEED_OFF = int(_pop("--seed-offset")) if "--seed-offset" in argv else 0
if SEED_OFF:
    # single-seed serving kernels: shift ONLY the bulk CatBoost seed (prc.final builds it with verbose=250 and
    # random_seed=1113 + seed*977); the no-plan specialist (crossval._fit_one_noplan, verbose=False) keeps its seeds.
    import catboost as _cb
    _CBR = _cb.CatBoostRegressor

    class _ShiftedCBR(_CBR):
        def __init__(self, *a, **k):
            if k.get("verbose") == 250 and "random_seed" in k:
                k["random_seed"] = k["random_seed"] + SEED_OFF * 977
                print(f"[ltreg final] bulk CatBoost seed -> {k['random_seed']} (offset {SEED_OFF})", flush=True)
            super().__init__(*a, **k)
    _cb.CatBoostRegressor = _ShiftedCBR
OUT = _pop("--out")
BASEDROP = [c for c in _pop("--basedrop").split(",") if c]
assert MODE in ("on", "raw", "off"), MODE
ON, DEMEAN = MODE != "off", MODE == "on"
sys.argv = [sys.argv[0]] + argv

import prc.crossval as cv  # noqa: E402
import prc.features as F  # noqa: E402
import prc.final as fin  # noqa: E402  (import-time registry check runs on the ORIGINAL lists)
from prc.config import RAW_DIR  # noqa: E402

unknown = set(BASEDROP) - set(F.FEATURES)
assert not unknown, unknown
fin.DROP[:] = BASEDROP                       # final's feats = FEATURES - DROP == crossval --drop BASEDROP

frame0 = fin.load_training_features()        # BEFORE the lists are extended: the cache check reads FEATURES
_orig_build = fin.build

if ON:
    COLS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt",
            "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt"]
    ltfm = (pl.col("ADEP_mvt") == "LTFM") | (pl.col("ADES_mvt") == "LTFM")
    raw = pl.concat([pl.scan_parquet(f).select(COLS).filter(ltfm).collect()
                     for f in sorted(RAW_DIR.glob("training_2025-*.parquet"))])
    DEPS = raw.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM")).with_columns(
        _m=pl.col("MVT_TIME_UTC_mvt").dt.month(), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    ARRS = raw.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LTFM"))
    LAB = DEPS.filter(pl.col("LOBT_flt").is_not_null())
    parts = []
    for m in range(1, 13):
        T = fit_T(LAB.filter(pl.col("_m") != m))
        f = seq_build(DEPS, ARRS, T, demean=DEMEAN).join(DEPS.select("MVT_ID_mvt", "_m"), on="MVT_ID_mvt").filter(pl.col("_m") == m)
        parts.append(f.drop("_m"))
    SF_TRAIN = pl.concat(parts)
    _SD = pl.concat([pl.scan_parquet(f).select("MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt")
                     .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(SNOW_APTS)).collect()
                     for f in sorted(RAW_DIR.glob("training_2025-*.parquet"))])
    XS_TRAIN = snow_features_apts(_SD, RAW_DIR, SNOW_APTS)
    assert SF_TRAIN["MVT_ID_mvt"].n_unique() == SF_TRAIN.height == DEPS.height
    r26 = pl.scan_parquet(RAW_DIR / "ranking.parquet").select(COLS).filter(ltfm).collect()
    D26 = r26.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM"))
    A26 = r26.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LTFM"))
    SF_RANK = seq_build(D26, A26, fit_T(LAB), demean=DEMEAN)
    _R = pl.scan_parquet(RAW_DIR / "ranking.parquet").select("MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt")         .filter((pl.col("PHASE_mvt") == "DEP") & pl.col("ADEP_mvt").is_in(SNOW_APTS)).collect()
    XS_RANK = snow_features_apts(_R, RAW_DIR, SNOW_APTS)
    for c in SNOW_FEATURES:
        print(f'[ltreg final] {c}: train nonzero {int((XS_TRAIN[c].fill_nan(0).fill_null(0) != 0).sum()):,}  rank nonzero {int((XS_RANK[c].fill_nan(0).fill_null(0) != 0).sum()):,}')
    print(f"[ltfm-seq final] {MODE.upper()} (demean={DEMEAN}): train {SF_TRAIN.height:,} LTFM deps; "
          f"ranking {SF_RANK.height:,} LTFM deps, {A26.height:,} arrivals; +{len(SEQ_FEATURES)} bulk features")
    for lst in (F.NUMERIC, F.FEATURES):
        lst.extend(SEQ_FEATURES)
    assert fin.FEATURES is F.FEATURES and cv.FEATURES is F.FEATURES
    cv.NOPLAN_DROP.extend(SEQ_FEATURES)

    train_fr = frame0.join(SF_TRAIN, on="MVT_ID_mvt", how="left", maintain_order="left").join(XS_TRAIN, on="MVT_ID_mvt", how="left", maintain_order="left")
    assert train_fr.height == frame0.height
    print(f"[ltfm-seq final] training rows with features: {train_fr.filter(pl.col('sq_win_n').is_not_null()).height:,}")
    fin.load_training_features = lambda *a, **k: train_fr

    def build(raw_frame, *a, **k):
        # features.build selects FEATURES; hide the SEQ columns while it runs (they are joined on below)
        for lst in (F.NUMERIC, F.FEATURES):
            assert lst[-len(SEQ_FEATURES):] == SEQ_FEATURES
            del lst[-len(SEQ_FEATURES):]
        try:
            fr = _orig_build(raw_frame, *a, **k)
        finally:
            for lst in (F.NUMERIC, F.FEATURES):
                lst.extend(SEQ_FEATURES)
        out = fr.join(SF_RANK, on="MVT_ID_mvt", how="left", maintain_order="left").join(XS_RANK, on="MVT_ID_mvt", how="left", maintain_order="left")
        assert out.height == fr.height
        n_cov = out.filter(pl.col("sq_win_n").is_not_null()).height
        print(f"[ltfm-seq final] ranking rows with features: {n_cov:,} (LTFM rows {out.filter(pl.col('ADEP_mvt') == 'LTFM').height:,})")
        return out

    fin.build = build
else:
    print("[ltfm-seq final] OFF: plain prc.final on FEATURES minus basedrop (control)")
    fin.load_training_features = lambda *a, **k: frame0

# capture the post-regime-correction float predictions (the transform is identity; its input is `bagged`)
_CAP = {}
_orig_rt = fin.resolve_transform


def resolve_transform(spec):
    fn = _orig_rt(spec)

    def wrapped(x):
        _CAP["final"] = np.asarray(x, dtype=float).copy()
        return fn(x)
    wrapped.__name__ = fn.__name__
    return wrapped


fin.resolve_transform = resolve_transform
n_feat = len([f for f in F.FEATURES if f not in set(BASEDROP)])
print(f"[ltfm-seq final] {n_feat} features")
fin.main()

pre = pl.read_parquet(fin.INTERIM_DIR / "ranking_predictions_prepost.parquet")
apt = pl.read_parquet(RAW_DIR / "ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
out = pre.rename({"pred": "prepost"}).with_columns(final=pl.Series(_CAP["final"]))
out = out.join(apt.select("MVT_ID_mvt", "ADEP_mvt"), on="MVT_ID_mvt", how="left", maintain_order="left")
assert out.height == pre.height and out["ADEP_mvt"].null_count() == 0
out.write_parquet(OUT)
print(f"[ltfm-seq final] wrote {OUT}: {out.height:,} rows, LTFM {out.filter(pl.col('ADEP_mvt') == 'LTFM').height:,}")
print("FINAL-WRAPPER DONE")
