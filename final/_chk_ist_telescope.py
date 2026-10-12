"""Check on January/July with the stored arm outputs: (1) 43_ist_build with ctrl = the sequence arm reproduces the shipped v137 Istanbul rows
(port check); (2) base + (snow - ctrl) in one step against the shipped two steps (treat - ctrl, then snow - treat), same base."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import polars as pl
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ib", REPO / "final" / "43_ist_build.py")
ib = importlib.util.module_from_spec(spec); spec.loader.exec_module(ib)
ID, T = ib.ID, ib.T
R = REPO / "results"
rk = str(REPO / "data/raw/ranking.parquet")
base = pl.read_parquet(REPO / "data/interim/v122_stage9s_final.parquet")
v137 = pl.read_parquet(REPO / "data/interim/v137_ltfmsnow_final.parquet")
class A: pass
a = A(); a.ctrl = str(R / "preds_ltfmfin_treat_s3.parquet"); a.k0 = str(R / "preds_lrfin_snow_k0.parquet"); a.k12 = str(R / "preds_lrfin_snow_k12.parquet")
a.base = str(REPO / "data/interim/v122_stage9s_final.parquet"); a.months = "1,7"; a.ranking = rk; a.out = str(REPO / "data/final/_chk/ist_port.parquet")
ib.build(a)
mine = pl.read_parquet(a.out)
j = mine.select(ID, T).join(v137.select(ID, pl.col(T).alias("s")), on=ID)
d = (j[T].cast(pl.Float64) - j["s"].cast(pl.Float64)).abs()
print("(1) port vs shipped v137: all rows max abs diff", float(d.max()), "differing rows", int((d > 0).sum()))
# (2) telescoping with the sequence-free control
ctrl = pl.read_parquet(R / "preds_ltfmfin_ctrl_s3.parquet")
treat = pl.read_parquet(R / "preds_ltfmfin_treat_s3.parquet")
lt = (treat["ADEP_mvt"] == "LTFM").to_numpy()
d1 = np.where(lt, treat["final"].to_numpy() - ctrl.join(treat.select(ID), on=ID)["final"].to_numpy(), 0.0)
b = base.select(ID, T).join(pl.DataFrame({ID: treat[ID], "d1": d1, "lt": lt}), on=ID, how="left")
b0 = b[T].cast(pl.Float64).to_numpy(); d1 = b["d1"].fill_null(0).to_numpy(); isl = b["lt"].fill_null(False).to_numpy()
step1 = np.rint(np.maximum(b0 + d1, 0))              # shipped v113 step (rounded)
# step 2 = the stored v137 delta on top of step1 is what the shipped chain did; here the single combined step on b0:
a2 = A(); a2.ctrl = str(R / "preds_ltfmfin_ctrl_s3.parquet"); a2.k0 = a.k0; a2.k12 = a.k12; a2.months = "1,7"; a2.ranking = rk
b0f = base.with_columns(pl.Series(T, step1 - d1 + d1).cast(base[T].dtype)) if False else base
a2.base = a.base; a2.out = str(REPO / "data/final/_chk/ist_onestep.parquet")
ib.build(a2)
one = pl.read_parquet(a2.out)
# two-step: base b0 -> step1 -> (snow - treat) on step1
base1 = base.with_columns(pl.Series(T, step1).cast(base[T].dtype)); base1.write_parquet(REPO / "data/final/_chk/ist_step1.parquet")
a3 = A(); a3.ctrl = a.ctrl; a3.k0 = a.k0; a3.k12 = a.k12; a3.months = "1,7"; a3.ranking = rk
a3.base = str(REPO / "data/final/_chk/ist_step1.parquet"); a3.out = str(REPO / "data/final/_chk/ist_twostep.parquet")
ib.build(a3)
two = pl.read_parquet(a3.out)
j = one.select(ID, pl.col(T).alias("o")).join(two.select(ID, pl.col(T).alias("t")), on=ID)
d = (j["o"].cast(pl.Float64) - j["t"].cast(pl.Float64)).abs()
print("(2) one-step (snow - ctrl) vs two-step (treat - ctrl, then snow - treat), same base: max abs diff", float(d.max()), "differing rows", int((d > 0).sum()), "of", int(isl.sum()), "Istanbul rows")
