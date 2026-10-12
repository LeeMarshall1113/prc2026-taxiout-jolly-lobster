#!/usr/bin/env bash
# Build v38 = v37's chain plus --floor-q zero: the hard floor raises to lb itself and the floor stack
# owns the whole offset (30-day screen -0.274s; 177-day confirmation gates this build). Every step except the ADS-B one is a rule already shipped:
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
# Builds and checks ONLY. Uploading v38 is a separate step, not done here. Run the ADS-B step under a
# slot (~15 min). --spill-tracks on keeps one calibration day of tracks in memory instead of all 177,
# which takes the build from ~8 GB to ~3 GB; validated on the 2025-09-23 selftest, 219.81 against
# 219.75 in-memory with identical floor counts (LightGBM's own run-to-run noise is larger).
# No backslashes.
set -uo pipefail
REPO="${PRC_ROOT:-$PWD}"
cd "$REPO"; PY=./.venv/Scripts/python.exe
LOG="$REPO/logs/build_v38.log"
MODEL=submissions/jolly-lobster_v90.parquet
BASE=data/interim/v29_base_dayearly.parquet
S0=data/interim/v29_rome_gate_step.parquet
S1=data/interim/v38_adsb_step.parquet
S2=data/interim/v38_clamp_step.parquet
S3=data/interim/v38_box_step.parquet
S35=data/interim/v38_minfloor_step.parquet
S4=data/interim/v38_rules_step.parquet
S5=data/interim/v38_upper_step.parquet
V38=submissions/jolly-lobster_v38.parquet
STACK_FEATS="${STACK_FEATS:-quality}"
FLOOR_EXTRA="${FLOOR_EXTRA:-}"   # "--floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG" only if run_snap177.sh confirmed it
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
say () { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
stop () { say "STOPPED, v38 NOT BUILT: $*"; exit 1; }
FILTER="warning|deprecat|sortedness|join_asof"

grep -q "FLOORQ CONFIRMED" logs/v3_confirm177.log 2>/dev/null || stop "the 177-day confirmation of --floor-q zero has not passed (logs/v3_confirm177.log)"
[ -f "$MODEL" ] || stop "$MODEL does not exist"
[ -f "$BASE" ] || stop "$BASE missing"
[ -f "$S0" ] || stop "$S0 missing"
[ -f "$V38" ] && stop "$V38 already exists; refusing to overwrite it"
$PY archive/ship_helpers.py changed "$MODEL" "$BASE" 1 10 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "the reused day-early base does not match the model"
$PY archive/ship_helpers.py changed "$BASE" "$S0" 15000 30000 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "the reused Rome gate step is not consistent with the base"
rm -f "$S1" "$S2" "$S3" "$S35" "$S4" "$S5"

say "step 1: ADS-B correction with the floor-quality floor stack, main stack features $STACK_FEATS, floor extra [$FLOOR_EXTRA]"
$PY -u archive/adsb_apply.py --in "$S0" --out "$S1" --osm-fill on --floor-stack-feats quality --stack-feats "$STACK_FEATS" --floor-q zero --spill-tracks on $FLOOR_EXTRA 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
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
$PY archive/ship_helpers.py changed submissions/jolly-lobster_v37.parquet "$S3" 1000 300000 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || stop "step 3 is not a plausible sibling of v37 (did the flag reach serving?)"
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
$PY -u archive/apply_hidden_reading.py --in "$S5" --out "$V38" 2>&1 | grep -v -i -E "$FILTER" | tee -a "$LOG"
[ -f "$V38" ] || stop "apply_hidden_reading.py did not write $V38"
$PY archive/ship_helpers.py changed "$S5" "$V38" 10 25 2>&1 | tee -a "$LOG" | grep -q "changed PASS" || { mv "$V38" "$V38.failed"; stop "the reading rule changed an unexpected number of rows"; }
$PY archive/check_submission.py "$BASE" "$V38" > logs/build_v38_check.txt 2>&1
cat logs/build_v38_check.txt >> "$LOG"
grep -q "template checks pass" logs/build_v38_check.txt || { mv "$V38" "$V38.failed"; stop "v38 failed the template checks"; }
$PY archive/ship_helpers.py sanity "$BASE" "$V38" 2>&1 | tee -a "$LOG" | grep -q "sanity PASS" || { mv "$V38" "$V38.failed"; stop "v38 failed the sanity gate"; }
$PY archive/ship_helpers.py changed submissions/jolly-lobster_v37.parquet "$V38" 1000 300000 2>&1 | tee -a "$LOG"
say "v38 BUILT and checked: $V38 -- NOT uploaded; waiting for the team"
