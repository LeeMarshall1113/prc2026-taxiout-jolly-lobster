#!/usr/bin/env python
"""No-op stand-in for the local job scheduler client used by some wrappers in scripts/.

Subcommands used by those wrappers: request (prints a dummy id), wait, heartbeat, release, status.
Every one succeeds immediately; heartbeat never asks the caller to yield.
"""
import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "request":
    print("0")
elif cmd == "status":
    print("no scheduler: jobs run immediately")
sys.exit(0)
