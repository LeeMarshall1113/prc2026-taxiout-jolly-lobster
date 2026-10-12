"""Build an upload as: best scored base + one ingredient delta = (ingredient file minus reference file).
argv: out version, base version, ingredient stem, reference stem.   (docs/LAB-NOTES.md 2026-10-03)
v140 = base + (agent_snowsev_hedge_final - agent_basefix_final): the Amsterdam 2-6 January no-plan snow-day hedge at the 2025-fit weight.
v141 = base + (agent_mega_final - agent_mega_no4_final): the main corrector refit at seven airports."""
import sys, polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V, B, ING, REF = sys.argv[1:5]
ing, ref = pl.read_parquet(I + ING + ".parquet"), pl.read_parquet(I + REF + ".parquet"); i = lambda s: s.cast(pl.Int64)  # noqa: E731
base = pl.read_parquet(f"submissions/jolly-lobster_v{B}.parquet")
if not (base[ID] == ref[ID]).all():
    base = base.join(ref.select(ID).with_row_index("o"), on=ID).sort("o").drop("o")
assert (base[ID] == ref[ID]).all() and (ing[ID] == ref[ID]).all()
d = i(ing[T]) - i(ref[T])
out = base.with_columns((i(base[T]) + d).clip(0, None).cast(base[T].dtype).alias(T))
out.write_parquet(I + f"agent_bundle_v{V}_final.parquet"); out.write_parquet(f"submissions/jolly-lobster_v{V}.parquet")
print(f"v{V} on v{B}: delta rows {int((d != 0).sum())} (mean {d.filter(d != 0).mean():+.1f}, min {d.min()}, max {d.max()}), changed vs base {int((out[T] != base[T]).sum())}")
