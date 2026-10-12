#!/usr/bin/env bash
# Build v100 = v99 (anchors + 264-day recal + floor-stack anchors, LIRF skipped) + ADS-B airborne-callsign matcher fix
# (LIRF skipped). Matcher 177-day gate PASSED (-0.306 s, -0.265 minus top-50, both halves, EDDF -2.07). NO upload here.
cd "${PRC_ROOT:-$PWD}"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
export CALIB_FILE=data/interim/calib_days_264_octnov.txt
export CALIB_PREDS="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,data/interim/calib2026/preds_oct_f4.parquet,data/interim/calib2026/preds_nov_c.parquet"
export FL_EXTRA="--floor-stack-anchors on --floor-stack-anch-skip LIRF"
export AIRCS="--air-cs-all on --air-cs-skip LIRF --noplan-callsign on"
export ADSB=archive/agent_merged_apply_np.py
bash archive/merged_chain.sh data/interim/v94_model.parquet 103 100
