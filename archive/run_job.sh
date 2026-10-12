#!/usr/bin/env bash
# Pass-through job wrapper.
# Many wrappers in archive/ were written to run under a local job scheduler that granted memory slots. This stub keeps
# their command lines valid: it accepts the scheduling options, ignores them, and runs the command after "--".
#   archive/run_job.sh [--detach] [--project X] [--min-gb N] [--want-gb N] [--minutes N] [--eta-minutes N] \
#                      [--why TEXT] [--argument TEXT] [--match REGEX] [--vram-gb N] [--preemptible] -- command args...
# Memory is NOT limited here; use archive/capped_run.py --cap-gb N for a cap.
detach=0
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; break ;;
    --detach) detach=1; shift ;;
    --preemptible|--gpu-exclusive) shift ;;
    --project|--min-gb|--want-gb|--minutes|--eta-minutes|--why|--argument|--match|--vram-gb) shift 2 ;;
    *) shift ;;
  esac
done
[ $# -gt 0 ] || { echo "run_job.sh: no command after --" >&2; exit 2; }
if [ "$detach" = 1 ]; then
  mkdir -p logs/detached
  nohup "$@" > "logs/detached/job_$$.log" 2>&1 &
  echo "started pid $! (log logs/detached/job_$$.log)"
  exit 0
fi
exec "$@"
