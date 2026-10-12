"""Run a command under a HARD commit cap (Windows job object). Required for every agent_* script (2026-09-23).

    ./.venv/Scripts/python.exe archive/capped_run.py --cap-gb 3 -- ./.venv/Scripts/python.exe agent_x.py ...

Why: at 06:46Z on 2026-09-23 an unleased experiment script reached 30.4 GB of commit, drove the machine to 95% commit / 0 free
RAM and crashed the desktop app. The job object makes the OS refuse allocations past the cap (the script dies with a
MemoryError) and kills the whole process tree if this wrapper exits. Children spawned later inherit the job.
Exit code: the child's; 99 if the job could not be set up (the command is NOT run uncapped).
"""
from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from ctypes import wintypes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class EXTENDED(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JobObjectExtendedLimitInformation = 9
PROCESS_ALL_ACCESS = 0x1F0FFF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-gb", type=float, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("capped_run: no command", file=sys.stderr)
        return 99
    job = k32.CreateJobObjectW(None, None)
    if not job:
        print(f"capped_run: CreateJobObject failed ({ctypes.get_last_error()}) -- NOT running uncapped", file=sys.stderr)
        return 99
    info = EXTENDED()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    info.JobMemoryLimit = int(a.cap_gb * (1 << 30))
    if not k32.SetInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
        print(f"capped_run: SetInformationJobObject failed ({ctypes.get_last_error()}) -- NOT running uncapped", file=sys.stderr)
        return 99
    p = subprocess.Popen(cmd)
    h = k32.OpenProcess(PROCESS_ALL_ACCESS, False, p.pid)
    if not h or not k32.AssignProcessToJobObject(job, h):
        p.kill()
        print(f"capped_run: could not assign pid {p.pid} to the job ({ctypes.get_last_error()}) -- killed it", file=sys.stderr)
        return 99
    rc = p.wait()
    q = EXTENDED()
    k32.QueryInformationJobObject(job, JobObjectExtendedLimitInformation, ctypes.byref(q), ctypes.sizeof(q), None)
    print(f"capped_run: exit {rc}; peak job commit {q.PeakJobMemoryUsed / (1 << 30):.2f} GB of a {a.cap_gb:g} GB cap",
          file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
