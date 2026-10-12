"""experiment ltfmm (docs/LAB-NOTES.md, lever L1): the four v94 bulk members (c, p3, r, e) fitted at LTFM ONLY,
with (--ltfm-seq on) or without (off) the 19 LTFM take-off-sequence columns.  NEW FILE; prc/ and every existing script
are imported unedited.

    python archive/agent_ltfmm_fold.py --tag ltfmm_treat_f1 --ltfm-seq on  --only c,p3,r,e --ckpt-dir DIR <V94 flags> --folds 1 --drop BASEDROP
    python archive/agent_ltfmm_fold.py --tag ltfmm_ctrl_f1  --ltfm-seq off --only c,p3,r,e --ckpt-dir DIR <V94 flags> --folds 1 --drop BASEDROP

What it is: archive/agent_ltfm_crossval.py's seq injection (copied, tested path of the v113 refit) wrapped around
archive/agent_bulkr_fold.py's bulkr_run_fold (the v94 member fits), with the frame cut to LTFM rows so every member is fitted on
LTFM rows only (c, p3, r were airport-routed anyway -> identical fits to the production LTFM routes; e, which production fits
globally, becomes an LTFM-only ExtraTrees in BOTH arms, so the ctrl/treat pair is matched).  The output is the per-member
checkpoint <ckpt-dir>/<tag>_<a>_<b>_<member>.npz (ids, pred on the member scale, pre-regime), written by bulkr_run_fold.
The candidate is built downstream: D = mean_m(treat_m - ctrl_m) added to the v94 fold `bagged` (member_delta arithmetic;
archive/agent_ltfmm_judge.py).  Fold honesty = agent_ltfm_crossval.py's (T fitted without the held-out months, and without the
training row's own month).  Serve twin: NOT written here (only if the fold judge passes).
"""
import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_bulkr_fold as bf  # noqa: E402  (module level = imports + defs only; its main() is NOT called)
from agent_ltfm_seqfeat import SEQ_FEATURES, build, fit_T  # noqa: E402

import prc.crossval as cv  # noqa: E402
import prc.features as F  # noqa: E402
from prc.config import RAW_DIR  # noqa: E402
from prc.train import load_training_features  # noqa: E402


def _pop(argv, flag, has_val=True):
    if flag not in argv:
        return None
    i = argv.index(flag)
    v = argv[i + 1] if has_val else True
    del argv[i:i + (2 if has_val else 1)]
    return v


def main():
    argv = list(sys.argv[1:])
    mode = _pop(argv, "--ltfm-seq")
    if mode not in ("on", "off"):
        raise SystemExit("--ltfm-seq on|off is required (a control must never be mistaken for a treatment)")
    only = _pop(argv, "--only")
    ckpt = _pop(argv, "--ckpt-dir")
    smoke = _pop(argv, "--smoke-frac")
    if not only or not ckpt:
        raise SystemExit("--only <members> and --ckpt-dir are required (this script only produces member checkpoints)")
    bf.ONLY = {s for s in only.split(",") if s}
    bf.CKPT = ckpt
    if smoke:
        bf.SMOKE_FRAC = float(smoke)
        bf.patch_smoke(bf.SMOKE_FRAC)
    sys.argv = ["prc.crossval", *argv]

    frame0 = load_training_features()          # BEFORE the lists are extended: the cache check reads FEATURES
    if smoke:
        frame0 = frame0.gather_every(max(int(round(1 / float(smoke))), 1))
    cv.load_training_features = lambda: frame0

    on = mode == "on"
    if on:
        cols = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "RUNWAY_mvt",
                "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt"]
        raw = pl.concat([pl.scan_parquet(f).select(cols)
                         .filter((pl.col("ADEP_mvt") == "LTFM") | (pl.col("ADES_mvt") == "LTFM")).collect()
                         for f in sorted(RAW_DIR.glob("training_2025-*.parquet"))])
        deps = raw.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("ADEP_mvt") == "LTFM")).with_columns(
            _m=pl.col("MVT_TIME_UTC_mvt").dt.month(), y=pl.col("TAXITIME_SEC_mvt").cast(pl.Float64))
        arrs = raw.filter((pl.col("PHASE_mvt") == "ARR") & (pl.col("ADES_mvt") == "LTFM"))
        lab = deps.filter(pl.col("LOBT_flt").is_not_null())
        print(f"[ltfmm] seq ON: {deps.height:,} LTFM departures, {arrs.height:,} arrivals; +{len(SEQ_FEATURES)} features",
              flush=True)
        for lst in (F.NUMERIC, F.FEATURES):
            lst.extend(SEQ_FEATURES)
        assert cv.FEATURES is F.FEATURES
        cv.NOPLAN_DROP.extend(SEQ_FEATURES)
    else:
        print("[ltfmm] seq OFF (control, same LTFM-only member fits)", flush=True)

    def run_fold(frame, held, feats, cat_idx, args):
        if on:
            parts = []
            for m in range(1, 13):
                excl = set(held) | ({m} if m not in held else set())
                t = fit_T(lab.filter(~pl.col("_m").is_in(sorted(excl))))
                f = build(deps, arrs, t, demean=True).join(deps.select("MVT_ID_mvt", "_m"), on="MVT_ID_mvt") \
                    .filter(pl.col("_m") == m)
                parts.append(f.drop("_m"))
            sf = pl.concat(parts)
            assert sf["MVT_ID_mvt"].n_unique() == sf.height == deps.height
            frame = frame.join(sf, on="MVT_ID_mvt", how="left", maintain_order="left")
        frame = frame.filter(pl.col("ADEP_mvt") == "LTFM")       # LTFM-only members
        print(f"[ltfmm] fold {held}: LTFM frame {frame.height:,} rows, {len(feats)} features", flush=True)
        return bf.bulkr_run_fold(frame, held, feats, cat_idx, args)

    cv.run_fold = run_fold
    cv.main()


if __name__ == "__main__":
    main()
