#!/usr/bin/env bash
# Build v40 = v39's chain with the ADS-B step calibrated on 299 days instead of 177 (April, May,
# October and November 2025 added). Those months need two more fold-prediction files, from different
# suite arms, so the day curve tested the package rather than the day count alone.
# GATED: run_daycurve299.sh must PASS its pre-declared bar (>= 0.10s better on the fixed
# ten-day test set, on all rows AND core) before this may run. Every step except the ADS-B one is a rule already shipped:
#   step 1  ADS-B correction, --floor-stack-feats quality (30 days -0.653s, -0.846s on the serving-equivalent
#           harness; 177-day confirmation must PASS first -- logs/floorq_confirm.log), plus, only if
#           logs/snap177.log confirms it, the floor stack LOWERING except at LIRF+LFPG (30 days -1.230s)
#   step 2  gap clamp;  step 3  LOBT box, edge only           (v34's chain)
#   step 3b NEW: per-airport minimum-taxi floor, capped at the LOBT box edge (pooled -0.055s, all five
#           folds negative; below the -0.10s bar, shipped as a rider on the team's call 2026-09-20)
#   step 4  v35 rules: Rome habitual-copy airlines, hidden-plan lower bound at LIRF+LTFM
#   step 5  v36: hidden-plan upper bound at LIRF+LTFM
#   step 6  NEW: Rome, copy impossible -> the hidden plan's AOBT_3 reading (18 rows; folds -0.517 /
#           -0.063s, pooled -0.139s; archive/hidden_reading_rule.py)
# Builds and checks ONLY. Uploading v40 is a separate step, not done here. Run the ADS-B step under a
# slot (~15 min). --spill-tracks on keeps one calibration day of tracks in memory instead of all 177,
# which takes the build from ~8 GB to ~3 GB; validated on the 2025-09-23 selftest, 219.81 against
# 219.75 in-memory with identical floor counts (LightGBM's own run-to-run noise is larger).
# No backslashes.
set -uo pipefail
REPO="${PRC_ROOT:-$PWD}"
cd "$REPO"; PY=./.venv/Scripts/python.exe
LOG="$REPO/logs/build_v40.log"
MODEL=submissions/jolly-lobster_v90.parquet
BASE=data/interim/v29_base_dayearly.parquet
S0=data/interim/v29_rome_gate_step.parquet
S1=data/interim/v40_adsb_step.parquet
S2=data/interim/v40_clamp_step.parquet
S3=data/interim/v40_box_step.parquet
S35=data/interim/v40_minfloor_step.parquet
S4=data/interim/v40_rules_step.parquet
S5=data/interim/v40_upper_step.parquet
S6=data/interim/v40_reading_step.parquet
V40=submissions/jolly-lobster_v40.parquet
STACK_FEATS="${STACK_FEATS:-quality}"
FLOOR_EXTRA="${FLOOR_EXTRA:-}"   # "--floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG" only if run_snap177.sh confirmed it
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
stop () { say "STOPPED, v40 NOT BUILT: $*"; exit 1; }
FILTER="warning|deprecat|sortedness|join_asof"

grep -q "VERDICT: PASS" logs/daycurve299.log 2>/dev/null || stop "the 299-day day curve has not passed (logs/daycurve299.log)"
[ -f "$MODEL" ] || stop "$MODEL does not exist"
[ -f "$BASE" ] || stop "$BASE missing"
[ -f "$S0" ] || stop "$S0 missing"
[ -f "$V40" ] && stop "$V40 already exists; refusing to overwrite it"
$PY archive/ship_helpers.py changed "$MODEL" "$BASE" 1 10 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "the reused day-early base does not match the model"
$PY archive/ship_helpers.py changed "$BASE" "$S0" 15000 30000 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "the reused Rome gate step is not consistent with the base"
rm -f "$S1" "$S2" "$S3" "$S35" "$S4" "$S5" "$S6"

say "step 1: ADS-B correction with the floor-quality floor stack, main stack features $STACK_FEATS, floor extra [$FLOOR_EXTRA]"
CALIB5="results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,results/preds_s4_ctrl_4_10.parquet,results/preds_c_ctrl_5_11.parquet"
$PY -u archive/adsb_apply.py --in "$S0" --out "$S1" --osm-fill on --floor-stack-feats quality --stack-feats "$STACK_FEATS" --floor-q zero --spill-tracks on --calib-preds "$CALIB5" $FLOOR_EXTRA 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$S1" ] || stop "adsb_apply.py did not write $S1"
$PY archive/ship_helpers.py changed "$S0" "$S1" 3000 300000 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "the ADS-B step changed an unexpected number of rows"
say "step 2: gap clamp"
$PY archive/apply_gap_clamp.py --in "$S1" --out "$S2" 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$S2" ] || stop "apply_gap_clamp.py did not write $S2"
say "step 3: LOBT box, edge only"
$PY archive/apply_lobt_box.py --no-hard-binding --in "$S2" --out "$S3" 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$S3" ] || stop "apply_lobt_box.py did not write $S3"
# S3 is v34's recipe with one flag changed: it must differ from v34 (the flag reached serving) but
# only on ADS-B-matched rows
$PY archive/ship_helpers.py changed submissions/jolly-lobster_v39.parquet "$S3" 1000 300000 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "step 3 is not a plausible sibling of v39 (did the flag reach serving?)"
say "step 3b: per-airport minimum-taxi floor (kept as a rider: pooled -0.055s, every fold negative)"
$PY -u archive/apply_min_floor.py --in "$S3" --out "$S35" 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$S35" ] || stop "apply_min_floor.py did not write $S35"
$PY archive/ship_helpers.py changed "$S3" "$S35" 400 2500 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "the minimum-taxi floor moved an unexpected number of rows"
say "step 4: v35 rules"
$PY -u archive/apply_v35_rules.py --in "$S35" --out "$S4" 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$S4" ] || stop "apply_v35_rules.py did not write $S4"
say "step 5: v36 upper bound"
$PY -u archive/apply_upper_bound.py --in "$S4" --out "$S5" 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$S5" ] || stop "apply_upper_bound.py did not write $S5"
say "step 6: Rome copy-impossible reading"
$PY -u archive/apply_hidden_reading.py --in "$S5" --out "$S6" 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$S6" ] || stop "apply_hidden_reading.py did not write $S6"
say "step 7: Munich January level correction (v39's rider)"
$PY -u archive/apply_eddm_january.py --in "$S6" --out "$V40" 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$V40" ] || stop "apply_eddm_january.py did not write $V40"
$PY archive/ship_helpers.py changed "$S6" "$V40" 5000 9000 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "the Munich rider moved an unexpected number of rows"
$PY archive/ship_helpers.py changed "$S5" "$S6" 10 25 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || { mv "$V40" "$V40.failed"; stop "the reading rule changed an unexpected number of rows"; }
$PY archive/check_submission.py "$BASE" "$V40" > logs/build_v40_check.txt 2>&1
cat logs/build_v40_check.txt >> "$LOG"
grep -q "template checks pass" logs/build_v40_check.txt || { mv "$V40" "$V40.failed"; stop "v40 failed the template checks"; }
$PY archive/ship_helpers.py sanity "$BASE" "$V40" 2>&1 | tee -a "$LOG" | grep -q "sanity PASS" || { mv "$V40" "$V40.failed"; stop "v40 failed the sanity gate"; }
$PY archive/ship_helpers.py changed submissions/jolly-lobster_v39.parquet "$V40" 1000 300000 2>&1 | tee -a "$LOG"
say "v40 BUILT and checked: $V40 -- NOT uploaded; waiting for the team"
