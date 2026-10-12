"""Swap or add a blend member on an already-built model output, without refitting CatBoost.

    python archive/member_delta.py --base submissions/jolly-lobster_v92.parquet \
        --members data/interim/members_full.npz --mode replace:lgb_routed \
        --out submissions/jolly-lobster_v94.parquet

The shipped model is blended = (cpred + p3 + lpred) / 3 on flight-plan rows. No-plan rows are
overwritten by a separate model, so they never move here. The members npz comes from
`prc.final --fit-members`, on the same training matrix, and holds `lgb`, a refit of the shipped
LightGBM member (seed 0, same data and threads as v92), plus the candidate members.

    replace:K      blended' = (cpred + p3 + K) / 3            delta = (K - lgb) / 3
    add:K1,K2..    blended' = (3 * blended + sum K) / (3 + n) delta = (sum K - n * blended) / (3 + n)
    set:K1,K2..    blended' = (cpred + p3 + sum K) / (2 + n)  -- the shipped LightGBM is DROPPED and
                   K1..Kn take its place (set:lgb_routed == replace:lgb_routed; set:lgb,K == add:K)
                   delta = (sum K - lgb - (n - 1) * blended) / (2 + n)

For add, blended is the model output itself, which equals the pre-correction blend on every
row except hot-day plan rows. THE REGIME CORRECTION moves with the members. On a hot
airport-day with a positive visible shortfall s, plan rows were shifted by
s = mean_ok(proxy + off - blended). The new shift is s - mean_ok(delta), so the row gets
delta - mean_ok(delta). The unknown s cancels exactly, in both modes. A hot day counts as
shifted when its post-correction shortfall M - mean_ok(final) is ~0, where M is the correction
applied to an all-zero prediction (npz regime_zero). An unshifted hot day gets
delta + max(0, short - mean_ok(delta)), because a member can switch its correction on.
Result: clipped at 0 and rounded, which is what prc.final does after its identity transform.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import polars as pl

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--members", required=True)
    ap.add_argument("--mode", required=True, help="replace:K, add:K1,K2 or set:K1,K2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    m = np.load(a.members, allow_pickle=False)
    base = pl.read_parquet(a.base)
    idx = pl.DataFrame({ID: m["mvt_id"].astype(float), "_i": np.arange(len(m["mvt_id"]))})
    j = base.with_row_index("_o").join(idx, on=ID, how="left").sort("_o")
    if j["_i"].null_count() or j.height != len(m["mvt_id"]):
        sys.exit(f"member rows do not cover the base file: {j['_i'].null_count()} unmatched, "
                 f"{j.height} vs {len(m['mvt_id'])}")
    order = j["_i"].to_numpy()
    final = base[T].to_numpy().astype(float)
    plan = m["has_plan"][order] == 1

    kind, _, spec = a.mode.partition(":")
    names = [s.strip() for s in spec.split(",") if s.strip()]
    for k in names + (["lgb"] if kind in ("replace", "set") else []):
        if k not in m.files:
            sys.exit(f"{k} not in {a.members} (have {m.files})")
    if kind == "replace":
        if len(names) != 1:
            sys.exit("replace takes exactly one member")
        delta = (m[names[0]][order] - m["lgb"][order]) / 3.0
    elif kind == "add":
        n = len(names)
        delta = (sum(m[k][order] for k in names) - n * final) / (3.0 + n)
    elif kind == "set":
        n = len(names)
        delta = (sum(m[k][order] for k in names) - m["lgb"][order] - (n - 1) * final) / (2.0 + n)
    else:
        sys.exit(f"unknown mode {a.mode}")
    delta = np.where(plan, delta, 0.0)

    new = final + delta
    n_shift = n_unshift = 0
    if "hot_plan" in m.files:
        hot = m["hot_plan"][order]
        ok = m["proxy_ok"][order]
        day = m["day_key"][order]
        mz = m["regime_zero"][order]
        for d in np.unique(day[hot]):
            rows = hot & (day == d)
            okr = rows & ok
            if not okr.any():
                continue
            short_final = float(mz[okr].mean() - final[okr].mean())
            md = float(delta[okr].mean())
            if abs(short_final) < 2.0:  # shifted day: the correction re-balances
                new[rows] = final[rows] + delta[rows] - md
                n_shift += 1
            else:                       # unshifted: short < 0; a member may switch it on
                new[rows] = final[rows] + delta[rows] + max(0.0, short_final - md)
                n_unshift += 1
    out = np.rint(np.clip(new, 0.0, None))
    base.with_columns(pl.Series(T, out).cast(base[T].dtype)).write_parquet(a.out)

    d = out - final
    print(f"{a.mode}: {int(plan.sum()):,} plan rows; hot days {n_shift} shifted + {n_unshift} unshifted")
    print(f"  change vs base: mean {d.mean():+.2f}s  RMS {np.sqrt((d ** 2).mean()):.2f}s  "
          f"max |d| {np.abs(d).max():.0f}s  rows moved {int((d != 0).sum()):,}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
