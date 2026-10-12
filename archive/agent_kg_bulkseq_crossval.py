"""prc.crossval with the take-off-SEQUENCE family at ALL ten airports -- COPY-BASED, prc/ untouched (tag "kg-bulkseq").

    python archive/agent_kg_bulkseq_crossval.py --seq on  <every normal prc.crossval flag>
    python archive/agent_kg_bulkseq_crossval.py --seq off <every normal prc.crossval flag>   # == plain prc.crossval

Features: scripts/agent_seqpost_feat.SEQ (21 columns; imported, not re-typed), keyed by (airport, runway), level columns
demeaned by (airport, month) over the frame, bulk model only, appended after every existing column.
FOLD HONESTY (seqpost's construction): ONE whole-year build per fold; a row in month j gets T fitted on 2025 months not in
(held U {j}), so test rows' T never sees the test months and no row's T ever contains its own month's labels.
Pre-registered in docs/LAB-NOTES.md Part 2.
"""
import glob
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_seqpost_feat as SF  # noqa: E402

import prc.crossval as cv  # noqa: E402
import prc.features as F  # noqa: E402
from prc.config import RAW_DIR  # noqa: E402
from prc.train import load_training_features  # noqa: E402

argv = list(sys.argv[1:])
if "--seq" not in argv:
    raise SystemExit("--seq on|off is required")
i = argv.index("--seq")
MODE = argv[i + 1]
assert MODE in ("on", "off"), MODE
del argv[i:i + 2]
sys.argv = [sys.argv[0]] + argv

frame0 = load_training_features()          # BEFORE the lists are extended: the cache check reads FEATURES
cv.load_training_features = lambda: frame0

if MODE == "on":
    DEP, ARR = SF.load_movements(sorted(glob.glob(str(RAW_DIR / "training_2025-*.parquet"))))
    DMON = DEP["month"].to_numpy()
    print(f"[bulkseq] ON: {DEP.height:,} departures, {ARR.height:,} arrivals, all airports; +{len(SF.SEQ)} bulk features")
    for lst in (F.NUMERIC, F.FEATURES):
        lst.extend(SF.SEQ)
    assert cv.FEATURES is F.FEATURES
    cv.NOPLAN_DROP.extend(SF.SEQ)
    _orig_run_fold = cv.run_fold
    _T = {}

    def T_excl(ex):
        ex = tuple(sorted(set(ex)))
        if ex not in _T:
            _T[ex] = SF.fit_T(DEP.filter(~pl.col("month").is_in(list(ex))))
        return _T[ex]

    def run_fold(frame, held, feats, cat_idx, args):
        tt = np.empty(DEP.height)
        for j in range(1, 13):
            k = DMON == j
            tt[k] = SF.apply_T(DEP.filter(pl.Series(k)), T_excl(set(held) | {j}))
        sf = SF.build(DEP, ARR, tt, demean=True)
        assert sf["MVT_ID_mvt"].n_unique() == sf.height == DEP.height
        fr = frame.join(sf, on="MVT_ID_mvt", how="left", maintain_order="left")
        assert fr.height == frame.height
        n_cov = fr.filter(pl.col("sq_win_n").is_not_null()).height
        print(f"[bulkseq] fold {held}: features on {n_cov:,} of {fr.height:,} training-frame rows")
        return _orig_run_fold(fr, held, feats, cat_idx, args)

    cv.run_fold = run_fold
else:
    print("[bulkseq] OFF: plain prc.crossval (control)")

cv.main()
