"""Assembly with ALL FOUR months built by the final/ stages (no carried months).

The assembly (50_assemble.py) reads fixed file names, with February and June from the J-files and January and July from the stored
lineage. Here every stage output is the concatenation of the February/June file and its jj counterpart for January/July (tag jj:
J5jj, J7jj, J3jj, J8jj, J10jj), written under data/final/_rebuilt_in/ with the fixed names; the assembly is then run with
`--carried ""` and its input directory pointed at that folder. Outputs go to data/final/_rebuilt_out/ (final_{A,B,C}.parquet there) and
are copied to data/final/final_{A,B,C}_rebuilt.parquet. The files final_{A,B,C}.parquet are never touched.

  bash final/5x_run.sh "why" final/50_assemble_rebuilt.py
"""
from __future__ import annotations

import glob
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fcommon import OUT  # noqa: E402

IN = OUT / "_rebuilt_in"
RES = OUT / "_rebuilt_out"


def cat_parquet(a: str, b: str, name: str) -> None:
    x, y = pl.read_parquet(OUT / a), pl.read_parquet(OUT / b)
    assert x.columns == y.columns, (a, b)
    ids = pl.concat([x.select(x.columns[0]), y.select(y.columns[0])])
    assert ids[x.columns[0]].n_unique() == ids.height, f"{a} and {b} share ids"
    pl.concat([x, y]).write_parquet(IN / name)
    print(f"{name}: {x.height:,} + {y.height:,} rows")


def cat_npz(a: Path, b: Path, dst: Path) -> None:
    za, zb = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    keys = [k for k in za.files if za[k].ndim == 1 and za[k].shape == za["ids"].shape]
    np.savez(dst, **{k: np.concatenate([za[k], zb[k]]) for k in keys})


def main() -> None:
    IN.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT / "J2_v94model.parquet", IN / "J2_v94model.parquet")
    cat_parquet("J5_base.parquet", "J5jj_base.parquet", "J5_base.parquet")
    cat_parquet("J5_v105_s1.parquet", "J5jj_v105_s1.parquet", "J5_v105_s1.parquet")
    cat_parquet("J7_restack.parquet", "J7jj_restack.parquet", "J7_restack.parquet")
    cat_parquet("J3_ist.parquet", "J3jj_ist.parquet", "J3_ist.parquet")
    cat_parquet("J8_small.parquet", "J8jj_small.parquet", "J8_small.parquet")
    za, zb = np.load(OUT / "J8_deltas.npz", allow_pickle=True), np.load(OUT / "J8jj_deltas.npz", allow_pickle=True)
    np.savez(IN / "J8_deltas.npz", **{k: np.concatenate([za[k], zb[k]]) for k in za.files})
    for sub in ("arith", "pcd"):
        (IN / "J10_stage" / "out" / sub).mkdir(parents=True, exist_ok=True)
        for f in sorted(glob.glob(str(OUT / "J10_stage" / "out" / sub / "*.npz"))):
            g = OUT / "J10jj_stage" / "out" / sub / Path(f).name
            assert g.exists(), g
            cat_npz(Path(f), g, IN / "J10_stage" / "out" / sub / Path(f).name)
    spec = importlib.util.spec_from_file_location("assemble50", HERE / "50_assemble.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["assemble50"] = mod
    spec.loader.exec_module(mod)
    mod.OUT = IN          # inputs of the joins
    sys.argv = ["50_assemble.py", "--months", "1,2,6,7", "--carried", "", "--out-dir", str(RES), "--allow-missing-ist"]
    mod.main()
    for v in "ABC":
        shutil.copy(RES / f"final_{v}.parquet", OUT / f"final_{v}_rebuilt.parquet")
        print("copied", OUT / f"final_{v}_rebuilt.parquet")


if __name__ == "__main__":
    main()
