#!/usr/bin/env bash
# Rebuild the six v146 piece files and the v146 file in one step.
#
#   bash archive/build_v146_pieces.sh
#
# The six pieces are read by archive/agent_bundle_v146.py (v146 = v145 + six row-level bounds, 51 rows). Their builders read row
# extracts that live in the git-ignored folder data/interim/local/logic_rows/, so the extracts are written first. Order:
#   1. extracts (inputs of the builders): agent_logic_01_extract.py, agent_logic_05_hiddenplan.py (Rome / Istanbul hidden-plan table),
#      agent_logicB_01_extract.py, agent_logicB_10_hidden.py, agent_logicC_01_extract.py, agent_logicC_02_types.py
#   2. the piece builders: agent_logic_06_build.py (bounds re-applied last, Rome with-plan day-early band),
#      agent_logicB_20_build.py (plan-clock window edge, hidden-plan lower bound), agent_logicC_10_ltfm_noplan_bound.py (Istanbul early
#      bound), agent_logicC_12_madrid_ceiling.py (Madrid one-hour ceiling)
#   3. agent_bundle_v146.py: joins the six pieces onto v145 and prints the rows each piece moves.
#
# Needs: data/raw/ (training_2025-*.parquet, ranking.parquet), data/interim/agent_bundle_v145_final.parquet, and for
# agent_logicB_10_hidden.py data/interim/agent_detrules_frame.parquet and agent_detrules_cs.parquet (docs/REPRODUCE.md section 2).
# Every step runs under archive/capped_run.py --cap-gb 3 (Windows job object); set NOCAP=1 to run without the cap on another system.
# It overwrites data/interim/agent_logic*_final.parquet, data/interim/agent_bundle_v146_final.parquet and data/interim/local/logic_rows/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-./.venv/Scripts/python.exe}"
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)"
mkdir -p data/interim/local/logic_rows/audit
run () {
  echo "== $1"
  if [ "${NOCAP:-0}" = 1 ]; then "$PY" "archive/$1"; else "$PY" archive/capped_run.py --cap-gb "${CAP_GB:-3}" -- "$PY" "archive/$1"; fi
}
# 1. extracts
run agent_logic_01_extract.py
run agent_logic_05_hiddenplan.py
run agent_logicB_01_extract.py
run agent_logicB_10_hidden.py
run agent_logicC_01_extract.py
run agent_logicC_02_types.py
# 2. piece builders
run agent_logic_06_build.py
run agent_logicB_20_build.py
run agent_logicC_10_ltfm_noplan_bound.py
run agent_logicC_12_madrid_ceiling.py
# 3. v146 = v145 + the six pieces (prints "<piece> rows changed N" for each piece, then the total)
run agent_bundle_v146.py
