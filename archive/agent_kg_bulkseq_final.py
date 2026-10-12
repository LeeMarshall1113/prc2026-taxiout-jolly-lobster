"""prc.final with the take-off-SEQUENCE family at ALL ten airports -- COPY-BASED, prc/ untouched (tag "kg-bulkseq").

    python archive/agent_kg_bulkseq_final.py --seq on --out preds.parquet --basedrop <BASEDROP> <prc.final flags>

Serving twin of archive/agent_kg_bulkseq_crossval.py, same mechanics as archive/agent_kg_ltfm_final.py:
  * feature list = FEATURES minus --basedrop (prc.final.DROP replaced), = crossval's --drop rule.
  * 2025 training rows: SEQ with T fitted on 2025 months != the row's month (one whole-year build).
  * 2026 ranking rows: T from ALL 2025; one build over ranking.parquet (Jan+Jul in one frame), demeaned per
    (airport, month) over that frame -- apply_seq_residual.py's serve convention.
Writes --out: MVT_ID_mvt, ADEP_mvt, prepost, final (post regime correction, float). --upload refused.
"""
import glob
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_seqpost_feat as SF  # noqa: E402

argv = list(sys.argv[1:])
if "--upload" in argv:
    raise SystemExit("--upload is refused in this wrapper")


def _pop(flag):
    i = argv.index(flag)
    v = argv[i + 1]
    del argv[i:i + 2]
    return v


MODE = _pop("--seq")
OUT = _pop("--out")
BASEDROP = [c for c in _pop("--basedrop").split(",") if c]
assert MODE in ("on", "off"), MODE
sys.argv = [sys.argv[0]] + argv

import prc.crossval as cv  # noqa: E402
import prc.features as F  # noqa: E402
import prc.final as fin  # noqa: E402
from prc.config import RAW_DIR  # noqa: E402

assert not set(BASEDROP) - set(F.FEATURES)
fin.DROP[:] = BASEDROP
frame0 = fin.load_training_features()
_orig_build = fin.build

if MODE == "on":
    SEQ = SF.SEQ
    D25, A25 = SF.load_movements(sorted(glob.glob(str(RAW_DIR / "training_2025-*.parquet"))))
    mon = D25["month"].to_numpy()
    tt = np.empty(D25.height)
    for j in range(1, 13):
        k = mon == j
        tt[k] = SF.apply_T(D25.filter(pl.Series(k)), SF.fit_T(D25.filter(pl.col("month") != j)))
    SF_TRAIN = SF.build(D25, A25, tt, demean=True)
    D26, A26 = SF.load_movements([str(RAW_DIR / "ranking.parquet")])
    SF_RANK = SF.build(D26, A26, SF.apply_T(D26, SF.fit_T(D25)), demean=True)
    print(f"[bulkseq final] train {SF_TRAIN.height:,} deps; ranking {SF_RANK.height:,} deps, {A26.height:,} arrivals; +{len(SEQ)}")
    for lst in (F.NUMERIC, F.FEATURES):
        lst.extend(SEQ)
    assert fin.FEATURES is F.FEATURES and cv.FEATURES is F.FEATURES
    cv.NOPLAN_DROP.extend(SEQ)
    train_fr = frame0.join(SF_TRAIN, on="MVT_ID_mvt", how="left", maintain_order="left")
    assert train_fr.height == frame0.height
    print(f"[bulkseq final] training rows with features: {train_fr.filter(pl.col('sq_win_n').is_not_null()).height:,} of {train_fr.height:,}")
    fin.load_training_features = lambda *a, **k: train_fr

    def build(raw_frame, *a, **k):
        for lst in (F.NUMERIC, F.FEATURES):
            assert lst[-len(SEQ):] == SEQ
            del lst[-len(SEQ):]
        try:
            fr = _orig_build(raw_frame, *a, **k)
        finally:
            for lst in (F.NUMERIC, F.FEATURES):
                lst.extend(SEQ)
        out = fr.join(SF_RANK, on="MVT_ID_mvt", how="left", maintain_order="left")
        assert out.height == fr.height
        print(f"[bulkseq final] ranking rows with features: {out.filter(pl.col('sq_win_n').is_not_null()).height:,} of {out.height:,}")
        return out

    fin.build = build
else:
    print("[bulkseq final] OFF: plain prc.final on FEATURES minus basedrop (control)")
    fin.load_training_features = lambda *a, **k: frame0

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
print(f"[bulkseq final] {len([f for f in F.FEATURES if f not in set(BASEDROP)])} features")
fin.main()

pre = pl.read_parquet(fin.INTERIM_DIR / "ranking_predictions_prepost.parquet")
apt = pl.read_parquet(RAW_DIR / "ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
out = pre.rename({"pred": "prepost"}).with_columns(final=pl.Series(_CAP["final"]))
out = out.join(apt.select("MVT_ID_mvt", "ADEP_mvt"), on="MVT_ID_mvt", how="left", maintain_order="left")
assert out.height == pre.height and out["ADEP_mvt"].null_count() == 0
out.write_parquet(OUT)
print(f"[bulkseq final] wrote {OUT}: {out.height:,} rows")
print("FINAL-WRAPPER DONE")
