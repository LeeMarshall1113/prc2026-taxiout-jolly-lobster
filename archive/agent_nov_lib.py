"""experiment nov lib (docs/LAB-NOTES.md): shared helper for the per-outer-fold novelty columns."""
import numpy as np
import polars as pl


def nov_folds(path, ids_float, nfold=10):
    """per-outer-fold novelty columns aligned with ids_float: {k: nov_f<k>} (archive/agent_nov_01_feat.py)."""
    v = pl.read_parquet(path, columns=["MVT_ID_mvt"] + [f"nov_f{k}" for k in range(nfold)]).with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Float64))
    assert v["MVT_ID_mvt"].n_unique() == v.height
    ids_float = np.asarray(ids_float, dtype=float)
    j = pl.DataFrame({"MVT_ID_mvt": ids_float}).join(v, on="MVT_ID_mvt", how="left", maintain_order="left")
    assert j.height == len(ids_float) and (j["MVT_ID_mvt"].to_numpy() == ids_float).all()
    return {k: j[f"nov_f{k}"].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in range(nfold)}
