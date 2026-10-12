#!/usr/bin/env bash
# Build v105 RAW = v103 base (AIRCS + NP1 + anchors) + LEVER 1 GENERALIZED (taxi-phase NM-callsign
# matching, no lift-off requirement, at ALL 9 covered airports -- archive/agent_taxinm_features.py /
# agent_merged_apply_taxinm.py). Modelled on archive/build_v103_np.sh (V=105, PRIOR=100).
# docs/LAB-NOTES.md. URGENT per the team -- ships 25 Sep.
#
# Modal-gated 2026-09-23 (team, all 9 airports enabled in the shared floor-stack pool, exactly
# as this build enables them -- same evaluator/flags/days as archive/agent_taxinm_stage1_eval.py):
#   unspliced TX (taxinm at all 9) vs NP1 FAILS only on EHAM (+0.485s) -- otherwise -1.102s all rows
#   pre-registered PER-AIRPORT splices PASS at EDDF, EGLL, LEMD, LSZH (+ LFPG, its own relaxed clause)
#   COMBINED splice over S = {EDDF, EGLL, LEMD, LSZH, LFPG} PASSES every clause:
#     -0.909 all rows, minus-top-50 -0.794, Jan-Mar -0.772, Jul-Sep -0.968, 227/238 days better
#
# THIS SCRIPT WRITES THE RAW BUILD, NOT THE SHIP NUMBER, and it MUST be built with all 9 airports
# enabled (not just S) -- the judged number is for the shared floor-stack pool with all 9 active; a
# build with only S enabled would not be the configuration Modal judged. Its own merged_chain.sh run
# writes to submissions/jolly-lobster_v105.parquet -- do not read that file as the ship number.
#
# THE SPLICE (not this script) produces the ship number: take ONLY rows at S = EDDF,EGLL,LEMD,LSZH,LFPG
# from this raw build and splice them into v103's RAW output (data/interim/v103_raw.parquet -- the
# CONTROL the Modal gate was judged against, not v103's Rome-gate-bagged-and-guarded final, per the
# the team's instruction), with the ALREADY-EXISTING archive/splice_apt_parquet.py:
#
#   .venv/Scripts/python.exe archive/splice_apt_parquet.py \
#       data/interim/v103_raw.parquet submissions/jolly-lobster_v105.parquet \
#       EDDF,EGLL,LEMD,LSZH,LFPG submissions/jolly-lobster_v105.parquet
#
# (CONTROL, TREAT, AIRPORTS, OUT -- OUT overwrites the raw build with the spliced result, same
# CONTROL/TREAT/AIRPORTS/OUT interface as archive/splice_npz.py's Modal-side twin.)
#
# DO NOT RUN this script or the splice before the checks named here are done. Prepared and verified (selftest, flag
# off vs on, 2025-07-16 with a 14-day calib set: flags off reproduces agent_merged_apply_np.py within
# the same reseed-noise band the unmodified script shows on repeats; all-9-on moves every listed
# airport, none silently) -- as of 2026-09-23 prepared only: not launched and not uploaded.
cd "${PRC_ROOT:-$PWD}"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
export CALIB_FILE=data/interim/calib_days_264_octnov.txt
export CALIB_PREDS="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,data/interim/calib2026/preds_oct_f4.parquet,data/interim/calib2026/preds_nov_c.parquet"
export FL_EXTRA="--floor-stack-anchors on --floor-stack-anch-skip LIRF"
export AIRCS="--air-cs-all on --air-cs-skip LIRF --noplan-callsign on --floor-stack-taxinm-apts EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH"
export ADSB=archive/agent_merged_apply_taxinm.py
bash archive/merged_chain.sh data/interim/v94_model.parquet 105 100
