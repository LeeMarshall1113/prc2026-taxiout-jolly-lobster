"""Run a shipped script unchanged, with its hard-coded ranking file replaced by another file.

Several shipped scripts read 'data/raw/ranking.parquet' by name (as a string or as RAW_DIR / 'ranking.parquet'). This runner
maps every polars read of that file to the file given by --ranking and then executes the script as __main__ with the rest of
the command line. Nothing else is changed, so the script's own logic, guards and constants stay as shipped.

    python final/_run_as_ranking.py --ranking data/final/ranking_J5.parquet archive/apply_gap_clamp.py --in A --out B

Run from the repo root.
"""
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
args = sys.argv[1:]
assert args and args[0] == "--ranking" and len(args) >= 3, __doc__
ranking, script, rest = args[1], args[2], args[3:]
ranking = str(Path(ranking).resolve())
sys.path.insert(0, os.getcwd())

import polars as pl  # noqa: E402

TARGET = "data/raw/ranking.parquet"


def _map(src):
    if isinstance(src, (str, os.PathLike)) and str(src).replace("\\", "/").endswith(TARGET):
        return ranking
    return src


_scan, _read = pl.scan_parquet, pl.read_parquet
pl.scan_parquet = lambda source, *a, **k: _scan(_map(source), *a, **k)
pl.read_parquet = lambda source, *a, **k: _read(_map(source), *a, **k)
sys.argv = [script] + rest
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
