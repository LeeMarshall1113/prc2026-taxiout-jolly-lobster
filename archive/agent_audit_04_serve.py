"""AUDIT STEP 4 -- serve-time safety for the survivors, on ranking.parquet (read-only).

Three failure classes this project has shipped before, each checked rather than asserted:

  1. FRAME COMPOSITION. build() runs per month in training and ONCE over January AND July 2026 at
     serve time. Each survivor is built on the whole ranking file and on each month alone; the
     values (raw, and demeaned within airport-month) must agree on EVERY row. Required: 0 differ.
  2. WITHHELD COLUMNS. The candidate library never loads a departure's BLOCK/TAXITIME, so this
     cannot happen by construction -- but the null rate per airport, 2025 (Jan+Jul) vs 2026, is
     reported against prc.leakcheck's 0.20 tolerance anyway, because a construction that silently
     depends on something the ranking file lacks would show here first.
  3. LEVEL DRIFT. The screened form is demeaned within (airport, month), which is immune to a
     2025 -> 2026 level shift. The raw medians are printed so a drift is visible, not hidden.

Never touches prc-2026-testsets or truthing.parquet.

    PYTHONPATH=. .venv/Scripts/python.exe archive/agent_audit_04_serve.py [--names a,b,c]
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "archive")

import numpy as np
import polars as pl

import agent_audit_cands as AC

C = pl.col
RANK = "data/raw/ranking.parquet"
TOL = 0.20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    both_nan = ~np.isfinite(a) & ~np.isfinite(b)
    close = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) <= 1e-9 * np.maximum(1.0, np.abs(a)))
    return both_nan | close


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default="")
    args = ap.parse_args()
    if args.names:
        names = args.names.split(",")
    else:
        s = pl.read_csv("results/agent_audit_screen.csv", infer_schema_length=0)
        names = s.filter(C("verdict") == "PASS")["name"].to_list()
    log(f"{len(names)} candidates: {', '.join(names)}")
    if not names:
        return

    dep, arr = AC.load_raw(RANK)
    log(f"ranking: {dep.height:,} departures, {arr.height:,} arrivals")
    whole = AC.build(dep, arr, only=set(names))
    g_whole = AC.group_apt_month(dep)
    ids = dep["MVT_ID_mvt"].to_numpy()
    pos = {m: i for i, m in enumerate(ids)}

    print("=" * 100)
    print("1. FRAME COMPOSITION -- whole Jan+Jul frame vs each month built alone (differing rows)")
    diff_raw = {n: 0 for n in names}
    diff_dm = {n: 0 for n in names}
    for month in ("2026-01", "2026-07"):
        lf = pl.scan_parquet(RANK).filter(C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m") == month)
        d_m, a_m = AC.load_raw(lf)
        part = AC.build(d_m, a_m, only=set(names))
        g_m = AC.group_apt_month(d_m)
        loc = np.array([pos[m] for m in d_m["MVT_ID_mvt"].to_numpy()])
        for n in names:
            diff_raw[n] += int((~same(whole[n][loc], part[n])).sum())
            dw = AC.demean(whole[n], g_whole)[loc]
            dp = AC.demean(part[n], g_m)
            diff_dm[n] += int((~same(dw, dp)).sum())
        log(f"   {month}: {d_m.height:,} departures compared")
    for n in names:
        print(f"   {n:<24} raw {diff_raw[n]:>6} differ   demeaned {diff_dm[n]:>6} differ   "
              + ("OK" if diff_raw[n] == 0 and diff_dm[n] == 0 else "FRAME-DEPENDENT"))

    print()
    print("2. NULL RATE per airport, 2025 Jan+Jul (training) vs 2026 (ranking); flag |diff| > 0.20")
    lf25 = pl.scan_parquet("data/raw/training_*.parquet").filter(
        C("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
    d25, a25 = AC.load_raw(lf25)
    f25 = AC.build(d25, a25, only=set(names))
    apt25 = d25["apt"].to_numpy().astype(str)
    apt26 = dep["apt"].to_numpy().astype(str)
    for n in names:
        cells, worst = [], 0.0
        for a in AC.TEN:
            n25 = float(np.mean(~np.isfinite(f25[n][apt25 == a])))
            n26 = float(np.mean(~np.isfinite(whole[n][apt26 == a])))
            worst = max(worst, abs(n26 - n25))
            cells.append(f"{a} {n25:.2f}/{n26:.2f}")
        print(f"   {n:<24} worst |diff| {worst:.3f} {'OK' if worst <= TOL else 'FLAG'}")
        print("      " + "  ".join(cells[:5]))
        print("      " + "  ".join(cells[5:]))

    print()
    print("3. LEVEL -- raw median per airport, 2025 Jan+Jul vs 2026 (the screened form is demeaned)")
    for n in names:
        cells = []
        for a in AC.TEN:
            v25 = f25[n][apt25 == a]
            v26 = whole[n][apt26 == a]
            m25 = np.nanmedian(v25) if np.isfinite(v25).any() else np.nan
            m26 = np.nanmedian(v26) if np.isfinite(v26).any() else np.nan
            cells.append(f"{a} {m25:>7.1f}/{m26:<7.1f}")
        print(f"   {n}")
        print("      " + "  ".join(cells[:5]))
        print("      " + "  ".join(cells[5:]))
    log("done")


if __name__ == "__main__":
    main()
