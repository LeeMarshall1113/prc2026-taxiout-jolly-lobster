"""Run a script and record every input file it reads (for shipping the exact inputs to Modal).

    python archive/trace_inputs.py OUT.txt some_eval.py --args ...

Records Python-level opens (audit hook) AND the path arguments of polars/numpy/pandas readers, because polars reads
through Rust and never raises a Python 'open' audit event.
"""
import glob as _glob
import os
import runpy
import sys

out, script, *rest = sys.argv[1:]
seen = set()
ROOT = os.path.abspath(".")


def note(p):
    try:
        if isinstance(p, (list, tuple)):
            for q in p:
                note(q)
            return
        s = os.fspath(p)
    except TypeError:
        return
    if isinstance(s, bytes):
        s = s.decode(errors="ignore")
    for q in (_glob.glob(s) if any(c in s for c in "*?[") else [s]):
        a = os.path.abspath(q)
        if a.startswith(ROOT) and os.path.isfile(a):
            seen.add(os.path.relpath(a, ROOT).replace("\\", "/"))


def hook(ev, args):
    if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1] if len(args) > 1 else "r"
        if mode is None or "r" in str(mode):
            note(args[0])


sys.addaudithook(hook)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def wrap(mod, name):
    f = getattr(mod, name, None)
    if f is None:
        return

    def g(src, *a, **k):
        note(src)
        return f(src, *a, **k)
    setattr(mod, name, g)


for n in ("read_parquet", "scan_parquet", "read_csv", "scan_csv", "read_ipc", "scan_ipc", "read_json", "read_ndjson",
          "scan_ndjson", "read_excel"):
    wrap(pl, n)
wrap(np, "load")
try:
    import pandas as pd
    for n in ("read_parquet", "read_csv"):
        wrap(pd, n)
except ImportError:
    pass

sys.argv = [script, *rest]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    with open(out, "w") as fh:
        fh.write("\n".join(sorted(seen)) + "\n")
