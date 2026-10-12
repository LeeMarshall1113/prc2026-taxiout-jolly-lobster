"""January/July corrector terms of the new pipeline (Modal serve fits on the staged final-file inputs, view and variant columns empty) against the
stored shipped terms for the same rows: mean of the 12 stored serve fits (data/interim/agent_arith3_serve/KYQSR-pw-s0-r0_serve_s*.npz for the
seven round-three airports, data/interim/agent_pcd_modal/serve/C4ff7V5_s*.npz for LIRF and LSZH), both at the shipped dose of 50_assemble.py.
Inputs and our own outputs only. Prints per airport-month: rows, mean and rms of the shipped term, mean and rms of (new - shipped), correlation.
    python final/45_jj_corrector_compare.py [--stage data/final/J10jj_stage]
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
ID = "MVT_ID_mvt"
DOSE = {"EDDF": 1.25, "EDDM": 1.25, "EGLL": 1.0, "LEBL": 1.0, "LEMD": 1.0, "LIRF": 0.75, "LSZH": 1.25, "EHAM": 1.25, "LFPG": 0.75}
ARITH = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LFPG", "LEMD"]


def mean_pred(pattern):
    acc, ids, n = None, None, 0
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if ids is None:
            ids = z["ids"].astype(float) if "ids" in z.files else None
        p = z["pred"].astype(float)
        acc = p if acc is None else acc + p
        n += 1
    return ids, acc / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=str(REPO / "data/final/J10jj_stage"))
    a = ap.parse_args()
    st = Path(a.stage)
    si_old = np.load(REPO / "data/interim/agent_bigc13_serve_in.npz", allow_pickle=True)
    ids_old, apt_old = si_old["ids"].astype(float), si_old["apt"].astype(str)
    out = {}
    for kind, new_pat, old_pat in (("arith", st / "out/arith/KYQSR-pw-s0-r0_serve_s*.npz", REPO / "data/interim/agent_arith3_serve/KYQSR-pw-s0-r0_serve_s*.npz"),
                                   ("pcd", st / "out/pcd/C4ff7V5_s*.npz", REPO / "data/interim/agent_pcd_modal/serve/C4ff7V5_s*.npz")):
        i1, p1, n1 = mean_pred(str(new_pat))
        i0, p0, n0 = mean_pred(str(old_pat))
        if i0 is None:
            i0 = ids_old
        out[kind] = (i1, p1, n1, i0, p0, n0)
        print(kind, "new fits", n1, "shipped fits", n0)
    meta = pl.read_parquet(REPO / "data/raw/final_ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    meta = meta.select(pl.col(ID).cast(pl.Float64), pl.col("ADEP_mvt").alias("apt"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month"))
    rows = []
    for kind, apts in (("arith", ARITH), ("pcd", ["LIRF", "LSZH"])):
        i1, p1, _, i0, p0, _ = out[kind]
        d = pl.DataFrame({ID: i1, "new": p1}).join(pl.DataFrame({ID: i0, "old": p0}), on=ID, how="inner").join(meta, on=ID, how="inner")
        d = d.filter(pl.col("apt").is_in(apts)).with_columns((pl.col("apt").replace_strict(DOSE, return_dtype=pl.Float64)).alias("k"))
        d = d.with_columns((pl.col("k") * pl.col("new")).alias("n"), (pl.col("k") * pl.col("old")).alias("o"))
        for (apt, mo), g in d.group_by(["apt", "month"], maintain_order=True):
            n, o = g["n"].to_numpy(), g["o"].to_numpy()
            rows.append((apt, mo, len(g), o.mean(), np.sqrt((o ** 2).mean()), (n - o).mean(), np.sqrt(((n - o) ** 2).mean()), np.corrcoef(n, o)[0, 1]))
    r = pl.DataFrame(rows, schema=["apt", "month", "rows", "shipped_mean", "shipped_rms", "diff_mean", "diff_rms", "corr"], orient="row").sort("apt", "month")
    with pl.Config(tbl_rows=40, float_precision=2):
        print(r)
    r.write_csv(REPO / "data/final/jj_corrector_compare.csv")


if __name__ == "__main__":
    main()
