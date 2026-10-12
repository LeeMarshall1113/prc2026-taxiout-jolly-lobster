"""experiment ltreg: fold npz pair for archive/agent_lens_09_predict.py from data/interim/agent_ltreg_snow_fold.parquet (written by
agent_ltreg_judge.py). Per row, ctrl = mean over the seeds available for that row's fold of the control `bagged`, treat = the
same seeds' splice (snow on LTFM rows, control elsewhere). Prints the fold footprint sum dp^2/row by seed count, for the guard."""
import numpy as np
import polars as pl

f = pl.read_parquet("data/interim/agent_ltreg_snow_fold.parquet")
g = f.group_by("MVT_ID_mvt").agg(pl.col("ADEP_mvt").first(), pl.col("y").first(), pl.col("c").mean(), pl.col("w").mean(),
                                 pl.col("seed").n_unique().alias("ns")).sort("MVT_ID_mvt")
dp2 = ((g["w"] - g["c"]) ** 2).to_numpy()
cen = pl.read_parquet("data/interim/agent_coverage_census_2025.parquet", columns=["MVT_ID_mvt"])
inc = g["MVT_ID_mvt"].is_in(cen["MVT_ID_mvt"].cast(pl.Float64)).to_numpy()
print(f"census 2025 covers {int(inc.sum()):,} of {g.height:,} fold rows (the lens reads only those)")
print(f"  fold sum dp^2/row, all fold rows {dp2.mean():.2f}; census rows only {dp2[inc].mean():.2f}")
print(f"rows {g.height:,}; seeds per row: {g.group_by('ns').len().sort('ns').to_dicts()}")
for ns in sorted(g["ns"].unique().to_list()):
    m = (g["ns"] == ns).to_numpy()
    print(f"  rows with {ns} seed(s): {m.sum():,}  fold sum dp^2/row (all rows of that group) {dp2[m].mean():.2f}")
print(f"  all rows: fold sum dp^2/row {dp2.mean():.2f}")
g = g.filter(pl.Series(inc))
ids = g["MVT_ID_mvt"].to_numpy()
for name, col in (("ctrl", "c"), ("treat", "w")):
    np.savez(f"data/interim/agent_ltreg_lens_{name}.npz", ids=ids, y=g["y"].to_numpy(), both_guard=g[col].to_numpy(),
             apts=g["ADEP_mvt"].to_numpy().astype(object))
print("wrote data/interim/agent_ltreg_lens_{ctrl,treat}.npz")
