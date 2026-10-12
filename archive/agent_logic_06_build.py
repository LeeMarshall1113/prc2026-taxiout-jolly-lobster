"""experiment logic 06: build the candidate files of the logic audit on data/interim/agent_bundle_v145_final.parquet.

    .venv/Scripts/python.exe archive/capped_run.py --cap-gb 3 -- .venv/Scripts/python.exe archive/agent_logic_06_build.py

(a) agent_logic_bounds_final.parquet: three certain bounds re-applied on the final file (each a zero- or near-zero-exception
    implication on 2025, with the 2025 counts given here):
    - schedule-gap clamp, 5,000 <= gap < 50,000: prediction <= floor(gap)                 (2025: 58 of 93,993 above gap + 60)
    - plan-clock window, plan rows: ceil(gl - 3,620) <= prediction <= floor(gl + 3,620)   (2025: 0 of 2,062,577 outside +-3,606)
    - hidden-plan lower bound at Rome / Istanbul (no-plan row, plan recovered from the destination's arrival record, gl < 3,606):
      prediction >= gl + 3,606                                                           (2025: 0 of 30 Rome, 0 of 20 Istanbul)
(b) agent_logic_planband_final.parquet: the one scored Rome PLAN row in the 70,000-100,000 s band (ITY340, 2026-07-17): the band
    rule put 10% on a schedule copy (94,560 s) that the plan window [gl - 3,606, gl + 3,606] = [83,214, 90,426] excludes, and
    90% on a no-plan day-early value; the row is a plan row and the label sits at the Network Manager off-block: value = MVT - AOBT_3
    = 86,340 s (2025 plan rows with gap >= 50,000: y - ga median -27 s, within 600 s on 90.6% of 436; Rome plan rows where a copy
    is impossible: median +118 s, within 600 s on 85.9% of 5,518).
"""
import os
import numpy as np
import polars as pl

C = pl.col
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, ".local", "logic_rows", "audit")
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
BASE = os.path.join(ROOT, "data", "interim", "agent_bundle_v145_final.parquet")
W_BOX = 3620.0
W = 3606.0

base = pl.read_parquet(BASE)
rk = pl.read_parquet(os.path.join(D, "dep2026.parquet"), columns=["id", "apt", "flight", "plan", "gap", "gl", "ga", "sched_d", "mvt_d"])
hid = pl.read_parquet(os.path.join(D, "hidden26.parquet")).filter(C("apt").is_in(["LIRF", "LTFM"]) & C("lower_scope")).select(
    "id", (C("gl") + W).alias("hp_lo"))
j = base.rename({ID: "id", T: "p"}).join(rk, on="id", how="left").join(hid, on="id", how="left")
assert j.height == base.height and j["apt"].null_count() == 0
p = j["p"].to_numpy().astype(float)
gap = j["gap"].to_numpy().astype(float)
gl = j["gl"].fill_null(np.nan).to_numpy().astype(float)
plan = j["plan"].to_numpy() & ~np.isnan(gl)
hp = j["hp_lo"].fill_null(-np.inf).to_numpy().astype(float)
APT, FLT, SD, MD = j["apt"].to_list(), j["flight"].to_list(), j["sched_d"].to_list(), j["mvt_d"].to_list()

# (a) bounds
new = p.copy()
m_clamp = (gap >= 5000) & (gap < 50000) & (new > np.floor(gap))
new[m_clamp] = np.floor(gap[m_clamp])
lo = np.where(plan, np.ceil(gl - W_BOX), -np.inf)
hi = np.where(plan, np.floor(gl + W_BOX), np.inf)
m_box = (new < lo) | (new > hi)
new = np.clip(new, lo, hi)
m_hp = new < hp
new[m_hp] = hp[m_hp]
new = np.clip(new, 0, None)
print(f"(a) bounds: clamp {int(m_clamp.sum())} rows, window {int(m_box.sum())} rows, hidden-plan lower bound {int(m_hp.sum())} rows; "
      f"changed {int((new != p).sum())}; sum of squared moves {float(((new - p) ** 2).sum()):,.0f} s^2")
for name, m in [("clamp", m_clamp), ("window", m_box), ("hidden lower", m_hp)]:
    for i in np.flatnonzero(m):
        print(f"    {name:13s} {APT[i]} {str(FLT[i]):9s} gap {gap[i]:>7,.0f} gl {gl[i] if not np.isnan(gl[i]) else float('nan'):>8,.0f} "
              f"served {p[i]:>7,.0f} -> {new[i]:>7,.0f} ({new[i] - p[i]:+,.0f})")
out_a = base.with_columns(pl.Series(T, np.rint(new)).cast(base.schema[T]))
assert out_a[ID].equals(base[ID]) and (out_a[T] >= 0).all()
out_a.write_parquet(os.path.join(ROOT, "data", "interim", "agent_logic_bounds_final.parquet"))

# (b) plan-row band
m_pb = (j["apt"] == "LIRF").to_numpy() & plan & (gap >= 70000) & (gap < 100000)
new_b = p.copy()
ga = j["ga"].fill_null(np.nan).to_numpy().astype(float)
for i in np.flatnonzero(m_pb):
    lo_i, hi_i = gl[i] - W, gl[i] + W
    val = float(np.clip(ga[i], lo_i, hi_i))
    print(f"(b) planband: {FLT[i]} {SD[i]} -> {MD[i]} gap {gap[i]:,.0f} gl {gl[i]:,.0f} ga {ga[i]:,.0f} "
          f"window [{lo_i:,.0f}, {hi_i:,.0f}] served {p[i]:,.0f} -> {val:,.0f} ({val - p[i]:+,.0f}); copy {gap[i]:,.0f} is "
          f"{'inside' if lo_i <= gap[i] <= hi_i else 'OUTSIDE'} the window")
    new_b[i] = val
print(f"(b) rows changed {int((new_b != p).sum())}; sum of squared moves {float(((new_b - p) ** 2).sum()):,.0f} s^2")
out_b = base.with_columns(pl.Series(T, np.rint(new_b)).cast(base.schema[T]))
assert out_b[ID].equals(base[ID]) and (out_b[T] >= 0).all()
out_b.write_parquet(os.path.join(ROOT, "data", "interim", "agent_logic_planband_final.parquet"))
print("wrote data/interim/agent_logic_bounds_final.parquet and agent_logic_planband_final.parquet")
