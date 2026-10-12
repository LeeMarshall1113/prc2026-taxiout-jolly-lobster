#!/usr/bin/env bash
# experiment ltreg: serving build chain inside ONE slot (called by agent_ltreg_run.sh). Base/board from env:
#   LR_BASE (default data/interim/v129_osm_final.parquet), LR_BOARD (default 224.59), LR_OUT (default data/interim/v137_ltfmsnow_final.parquet)
cd "${PRC_ROOT:-$PWD}"
BASE=${LR_BASE:-data/interim/v129_osm_final.parquet}; BOARD=${LR_BOARD:-224.59}; OUT=${LR_OUT:-data/interim/v137_ltfmsnow_final.parquet}
S=archive/agent_ltreg_step.sh
bash $S archive/agent_ltreg_judge.py logs/agent_ltreg_judge_snow_atbuild.log judge snow --partial || exit 1
bash $S archive/agent_ltreg_lensprep.py logs/agent_ltreg_lensprep.log || exit 1
bash $S archive/agent_ltreg_build.py logs/agent_ltreg_build.log build --base $BASE --out $OUT || exit 1
bash $S archive/agent_smallbundle_06_check.py logs/agent_ltreg_check.log $OUT $BASE || exit 1
bash $S archive/agent_lens_09_predict.py logs/agent_ltreg_lens.log --fold-ctrl data/interim/agent_ltreg_lens_ctrl.npz \
  --fold-treat data/interim/agent_ltreg_lens_treat.npz --sub-ctrl $BASE --sub-treat $OUT --board-ctrl $BOARD || exit 1
