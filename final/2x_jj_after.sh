#!/usr/bin/env bash
# Waits for data/final/J7jj_restack.parquet, then runs final/26_jj_compare.py and appends the result to logs/final_progress.log.
cd "${PRC_ROOT:-$PWD}" || exit 1
mkdir data/final/.jjafter 2>/dev/null || exit 0
trap 'rm -r data/final/.jjafter' EXIT
until [ -s data/final/J7jj_restack.parquet ]; do sleep 300; done
sleep 30
bash final/2x_run.sh "January/July comparison with the shipped lineage" final/26_jj_compare.py > logs/final_2x_jj_compare.txt 2>&1
{ echo "progress 2x $(date +%H:%M): Jan/Jul rebuild finished; comparison with the shipped lineage (final/26_jj_compare.py):"; grep -v "^\[\|lease-run\|capped_run\|jobcap\|GRANTED" logs/final_2x_jj_compare.txt; } >> logs/final_progress.log
