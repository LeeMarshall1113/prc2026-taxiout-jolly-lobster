"""prc.crossval with the LTFM take-off-sequence family added -- COPY-BASED, prc/ untouched.

    python archive/agent_ltfm_crossval.py --ltfm-seq on  <every normal prc.crossval flag>   # level columns month-demeaned
    python archive/agent_ltfm_crossval.py --ltfm-seq raw <every normal prc.crossval flag>   # level columns raw
    python archive/agent_ltfm_crossval.py --ltfm-seq off <every normal prc.crossval flag>   # control, bit-for-bit prc.crossval

What it adds (bulk model only; NaN at the other nine airports; appended AFTER every existing column so no index moves):
scripts/agent_ltfm_seqfeat.SEQ_FEATURES (19 columns) -- per-runway take-off order vs planned order (overtakes,
isotonic queue-entry), sequence-neighbour slips vs LOBT + T(stand, runway), saturated-runway busy periods, and
arrivals landing per runway group / departures on other runways in the 15 min before wheels-up.

FOLD HONESTY. T (median taxi per stand x runway) is the only label-derived input. For a fold holding out months H:
test-month rows use T fitted on months not in H; each TRAINING month m uses T fitted on months not in H and not m
(leave-one-month-out), so no row's own label, and no held-out label, ever enters any feature. Serve: T from all 2025.
Everything else reads MVT, LOBT, SCHED, STAND, RUNWAY of departures and MVT, RUNWAY of arrivals -- all in ranking.parquet.
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_ltfm_seqfeat import SEQ_FEATURES, build, fit_T  # noqa: E402

import prc.crossval as cv  # noqa: E402
import prc.features as F  # noqa: E402
from prc.config import RAW_DIR  # noqa: E402
from prc.train import load_training_features  # noqa: E402

argv = list(sys.argv[1:])
if "--ltfm-seq" not in argv:
    raise SystemExit("--ltfm-seq on|off is required (so a control can never be mistaken for a treatment)")
i = argv.index("--ltfm-seq")
MODE = argv[i + 1]
assert MODE in ("on", "raw", "off"), MODE
ON, DEMEAN = MODE != "off", MODE == "on"
del argv[i:i + 2]
sys.argv = [sys.argv[0]] + argv

frame0 = load_training_features()          # BEFORE the lists are extended: the cache check reads FEATURES
cv.load_training_features = lambda: frame0

if ON:
    COLS = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt",
            "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt"]
    raw = pl.concat([pl.scan_parquet(f).select(COLS)
                     .filter((pl.col("ADEP_mvt") == "LTFM") | (pl.col("ADES_mvt") == "LTFM")).collect()
                     for f in sorted(RAW_DIR.glob("training_2025-*.parquet"))])
    DEPS = raw.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM")).with_columns(
        _m=pl.col("MVT_TIME_UTC_mvt").dt.month(), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
    ARRS = raw.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LTFM"))
    LAB = DEPS.filter(pl.col("LOBT_flt").is_not_null())
    print(f"[ltfm-seq] {MODE.upper()} (demean={DEMEAN}): {DEPS.height:,} LTFM departures, {ARRS.height:,} arrivals; +{len(SEQ_FEATURES)} bulk features")
    for lst in (F.NUMERIC, F.FEATURES):
        lst.extend(SEQ_FEATURES)            # in place: prc.crossval holds the same FEATURES list object
    assert cv.FEATURES is F.FEATURES
    cv.NOPLAN_DROP.extend(SEQ_FEATURES)     # bulk model only, like every recent family

    _orig_run_fold = cv.run_fold

    def run_fold(frame, held, feats, cat_idx, args):
        parts = []
        for m in range(1, 13):
            excl = set(held) | ({m} if m not in held else set())
            T = fit_T(LAB.filter(~pl.col("_m").is_in(sorted(excl))))
            f = build(DEPS, ARRS, T, demean=DEMEAN).join(DEPS.select("MVT_ID_mvt", "_m"), on="MVT_ID_mvt").filter(pl.col("_m") == m)
            parts.append(f.drop("_m"))
        sf = pl.concat(parts)
        assert sf["MVT_ID_mvt"].n_unique() == sf.height == DEPS.height
        fr = frame.join(sf, on="MVT_ID_mvt", how="left", maintain_order="left")
        assert fr.height == frame.height
        n_ltfm = fr.filter(pl.col("ADEP_mvt") == "LTFM").height
        n_cov = fr.filter(pl.col("sq_win_n").is_not_null()).height
        print(f"[ltfm-seq] fold {held}: features on {n_cov:,} rows (LTFM rows {n_ltfm:,})")
        return _orig_run_fold(fr, held, feats, cat_idx, args)

    cv.run_fold = run_fold
else:
    print("[ltfm-seq] OFF: plain prc.crossval (control)")

cv.main()
