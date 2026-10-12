"""AUDIT STEP 2 -- screen every pre-declared candidate against ONE bar, and report every result.

THE BAR (pre-registered in docs/LAB-NOTES.md at 2026-09-21 03:35Z, before any
candidate was screened). A candidate PASSES only if all four hold, read on its construction
DEMEANED within (airport, month):

  1. SIGN    partial rho has the same sign on all five folds;
  2. FLOOR   |rho| > 0.019 on EVERY fold -- the established placebo floor of this instrument;
  3. MARGIN  min over folds |rho| >= 5 x max over folds |own placebo rho|;
  4. ROWS    >= 1,000 finite rows on every fold, else unscreenable (a fail).

THE INSTRUMENT is archive/agent_cong_common.py, unchanged: FOLDS, chain, Proj, the partial-rho
arithmetic and positive_control are imported, not copied. What this script chooses is the control
block -- the agent_cong_04 block (congestion counts, all twenty nm_*, weather), the seven late_*
columns, and the row's OWN version of every quantity the candidates aggregate, so a neighbour mean
cannot pass by proxying its own row. Linear combinations of controls already present are left out
(gap_eobt = gap_aobt + dep_delay, lobt_minus_aobt = gap_lobt - gap_aobt, the clk_* OBT differences,
turn_sched_slack = turn_age - gap_sched): they add no column space, only rank deficiency.

Printed beside the bar, INFORMATION ONLY: raw (not demeaned) rho, a winsorised-residual rho (residual
clipped at +/-3,000s, so a handful of monster rows cannot carry a reading), and the candidate's
largest |Pearson corr| with any control column -- the near-duplicate flag.

    PYTHONPATH=. .venv/Scripts/python.exe archive/agent_audit_02_screen.py --calib
    PYTHONPATH=. .venv/Scripts/python.exe archive/agent_audit_02_screen.py --families A,B,C
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "archive")

import numpy as np
import polars as pl

import agent_audit_cands as AC
from agent_cong_02_lateness import CONTROLS as CONG02_CONTROLS
from agent_cong_02_lateness import NM_SHIPPED
from agent_cong_04_joint import CONTROLS as CONG04_CONTROLS
from agent_cong_common import FEATCACHE, FOLDS, ID, PLACEBO_FLOOR, Proj, chain, positive_control
from prc.features import LATE_NEIGHBOUR_FEATURES

C = pl.col
CANDS = Path("data/interim/agent_audit")
OUTCSV = Path("results/agent_audit_screen.csv")

FLOOR = PLACEBO_FLOOR          # 0.019
MARGIN = 5.0
MIN_ROWS = 1000
WINSOR = 3000.0

LATE7 = list(LATE_NEIGHBOUR_FEATURES)
ROWSELF = ["gap_lobt", "clk_arvt3_minus_arvt1", "clk_arvt3_minus_mvt", "mvt_sec_00",
           "turn_age", "turn_taxi_in", "turn_in_delay", "turn_type_match", "turn_op_match",
           "arr_taxi_180min", "arr_slow_180min", "arr_sub_day", "prev_stand_gap", "prev_rwy_gap",
           "wake_heavy_60min", "has_flight_plan"]
CONTROLS = list(dict.fromkeys(CONG04_CONTROLS + LATE7 + ROWSELF))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class AuditInstrument:
    """agent_cong_common.Instrument's arithmetic over the candidate universe, with float32 storage.

    Universe = departures from the ten in the candidate files' row order. Fold rows are placed by
    MVT_ID; base = production's chain on out-of-fold `bagged`; LIRF dropped on the three folds whose
    models predate the Rome gate -- all exactly as Instrument does.
    """

    def __init__(self, control_cols: list[str]):
        ids = pl.read_parquet(CANDS / "cands_2025_A.parquet", columns=[ID])[ID].to_numpy()
        self.ids = ids
        pos = {m: i for i, m in enumerate(ids)}
        meta = (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
                .select(C(ID), C("ADEP_mvt").alias("adep"),
                        C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
                        C("AOBT_3_flt").is_not_null().alias("plan"),
                        (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                        (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"))
                .collect())
        mp = np.array([pos.get(m, -1) for m in meta[ID].to_numpy()])
        keep = mp >= 0
        n = len(ids)
        self.adep = np.empty(n, dtype=object)
        self.adep[mp[keep]] = meta["adep"].to_numpy()[keep]
        self.adep = self.adep.astype(str)
        self.ym = np.empty(n, dtype=object)
        self.ym[mp[keep]] = meta["ym"].to_numpy()[keep]
        self.ym = self.ym.astype(str)
        gs = np.full(n, np.nan)
        gl = np.full(n, np.nan)
        plan = np.zeros(n, dtype=bool)
        gs[mp[keep]] = meta["gs"].fill_null(np.nan).to_numpy()[keep]
        gl[mp[keep]] = meta["gl"].fill_null(np.nan).to_numpy()[keep]
        plan[mp[keep]] = meta["plan"].to_numpy()[keep]
        del meta

        cache = pl.scan_parquet(FEATCACHE).select([C(ID)] + [C(c).cast(pl.Float32) for c in control_cols]).collect()
        cp = np.array([pos.get(m, -1) for m in cache[ID].to_numpy()])
        ck = cp >= 0
        self.ctrl = {}
        for c in control_cols:
            col = np.full(n, np.nan, dtype=np.float32)
            col[cp[ck]] = cache[c].fill_null(np.nan).to_numpy()[ck]
            self.ctrl[c] = col
        del cache
        self.control_cols = list(control_cols)

        self.folds = {}
        for label, path, drop_lirf in FOLDS:
            P = pl.read_parquet(path).select(ID, "y", "bagged")
            loc = np.array([pos.get(m, -1) for m in P[ID].to_numpy()])
            lost = int((loc < 0).sum())
            y = P["y"].to_numpy().astype(float)
            bag = P["bagged"].to_numpy().astype(float)
            okr = loc >= 0
            loc, y, bag = loc[okr], y[okr], bag[okr]
            base = chain(bag, gs[loc], gl[loc], plan[loc])
            if drop_lirf:
                k = self.adep[loc] != "LIRF"
                loc, y, base = loc[k], y[k], base[k]
            self.folds[label] = (loc, y, base)
            log(f"   fold {label}: {len(loc):,} rows ({lost} fold rows outside the universe), "
                f"chained RMSE {np.sqrt(np.mean((y - base) ** 2)):.2f}s")

    def design(self, loc, m, base, cols=None) -> np.ndarray:
        """Identical construction to agent_cong_common.Instrument.design."""
        cols = self.control_cols if cols is None else cols
        k = int(m.sum())
        X = [np.ones(k), base[m], base[m] ** 2 / 1e3]
        for c in cols:
            v = self.ctrl[c][loc][m].astype(np.float64)
            X.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
            if not np.isfinite(v).all():
                X.append((~np.isfinite(v)).astype(float))
        apt = self.adep[loc][m]
        for a in np.unique(apt)[1:]:
            X.append((apt == a).astype(float))
        return np.column_stack(X)


def prho(ra: np.ndarray, rb: np.ndarray) -> float:
    """agent_cong_common.partial_rho's arithmetic on already-projected vectors."""
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.mean(ra * rb) / (sa * sb))


class ProjCache:
    """One Proj per (fold, row mask): candidates sharing a missingness pattern share a factorisation."""

    def __init__(self, inst, cols=None):
        self.inst, self.cols, self.key, self.val = inst, cols, None, None

    def get(self, label, m):
        h = (label, hashlib.sha1(np.packbits(m).tobytes()).hexdigest())
        if h != self.key:
            loc, y, base = self.inst.folds[label]
            X = self.inst.design(loc, m, base, self.cols)
            P = Proj(X)
            r = (y - base)[m]
            self.val = (P, P(r), P(np.clip(r, -WINSOR, WINSOR)))
            self.key = h
        return self.val


def read_column(inst, v_all, cols=None, cache=None):
    """Per-fold partial rho of one universe-aligned column."""
    cache = cache or ProjCache(inst, cols)
    out = []
    for label in inst.folds:
        loc, y, base = inst.folds[label]
        v = v_all[loc].astype(np.float64)
        m = np.isfinite(v)
        if m.sum() < MIN_ROWS:
            out.append(np.nan)
            continue
        P, pr, _ = cache.get(label, m)
        out.append(prho(P(v[m]), pr))
    return out


def calibrate(inst) -> None:
    labels = list(inst.folds)
    print()
    positive_control(inst, lambda label: np.ones(len(inst.folds[label][0]), dtype=bool))
    print()
    print("=" * 110)
    print("CALIBRATION -- what known quantities read as in THIS control block")
    no_nm = [c for c in inst.control_cols if c not in NM_SHIPPED]
    for name in ("nm_rwy_past15_mean", "nm_apt_past15_mean"):
        r = read_column(inst, inst.ctrl[name], cols=no_nm)
        print(f"   SHIPPED {name:<26} (nm removed from controls)        "
              + " ".join(f"{x:+.4f}" for x in r))
    no_late = [c for c in inst.control_cols if c not in LATE7]
    for name in ("late_rwy_past15_mean", "late_apt_past15_mean"):
        r = read_column(inst, inst.ctrl[name], cols=no_late)
        print(f"   KNOWN POSITIVE {name:<21} (late removed, extended controls) "
              + " ".join(f"{x:+.4f}" for x in r))
    base02 = [c for c in CONG02_CONTROLS if c in inst.ctrl]
    r = read_column(inst, inst.ctrl["late_rwy_past15_mean"], cols=base02)
    print(f"   KNOWN POSITIVE late_rwy_past15_mean  (agent_cong_02's own controls)   "
          + " ".join(f"{x:+.4f}" for x in r))
    print("   (agent_cong_02 read the RAW construction at +0.079 +0.080 +0.087 +0.102 +0.088; the cache")
    print("    holds the DEMEANED form, so a close but not identical reading is expected)")
    print(f"   folds: {' '.join(labels)}")


def near_dup(inst, v_all):
    """Largest |Pearson corr| of the candidate with any control column, over the universe."""
    best, who = 0.0, ""
    fv = np.isfinite(v_all)
    for c in inst.control_cols:
        w = inst.ctrl[c]
        m = fv & np.isfinite(w)
        if m.sum() < 1000:
            continue
        a = v_all[m].astype(np.float64)
        b = w[m].astype(np.float64)
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            continue
        r = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))
        if abs(r) > abs(best):
            best, who = r, c
    return best, who


def screen_family(inst, fam: str, rows_out: list) -> None:
    names = [(n, why) for f, n, why in AC.ALL_CANDIDATES if f == fam]
    cols = [ID]
    for n, _ in names:
        cols += [n, f"{n}__plc", f"{n}__raw"]
    df = pl.read_parquet(CANDS / f"cands_2025_{fam}.parquet", columns=cols)
    if not np.array_equal(df[ID].to_numpy(), inst.ids):
        raise SystemExit(f"family {fam}: row order differs from the universe")
    labels = list(inst.folds)
    cache = ProjCache(inst)
    for n, why in names:
        t0 = time.time()
        v_all = df[n].to_numpy()
        p_all = df[f"{n}__plc"].to_numpy()
        w_all = df[f"{n}__raw"].to_numpy()
        rho, plc, raw, win, nrows = [], [], [], [], []
        for label in labels:
            loc, y, base = inst.folds[label]
            v = v_all[loc].astype(np.float64)
            m = np.isfinite(v)
            nrows.append(int(m.sum()))
            if m.sum() < MIN_ROWS:
                rho.append(np.nan); plc.append(np.nan); raw.append(np.nan); win.append(np.nan)
                continue
            P, pr, pw = cache.get(label, m)
            pv = P(v[m])
            rho.append(prho(pv, pr))
            win.append(prho(pv, pw))
            plc.append(prho(P(p_all[loc][m].astype(np.float64)), pr))
            raw.append(prho(P(w_all[loc][m].astype(np.float64)), pr))
        rho_a = np.array(rho)
        ok_rows = min(nrows) >= MIN_ROWS and np.isfinite(rho_a).all()
        sign_ok = ok_rows and (np.all(rho_a > 0) or np.all(rho_a < 0))
        minabs = float(np.nanmin(np.abs(rho_a))) if ok_rows else 0.0
        maxplc = float(np.nanmax(np.abs(plc))) if ok_rows else np.nan
        floor_ok = ok_rows and minabs > FLOOR
        margin = minabs / max(maxplc, 1e-9) if ok_rows else 0.0
        margin_ok = ok_rows and margin >= MARGIN
        verdict = "PASS" if (sign_ok and floor_ok and margin_ok) else "fail"
        why_fail = [] if verdict == "PASS" else [s for s, okk in (("rows", ok_rows), ("sign", sign_ok),
                                                                  ("floor", floor_ok), ("margin", margin_ok)) if not okk]
        dup_r, dup_c = near_dup(inst, v_all)
        cover = float(np.isfinite(v_all).mean())
        row = {"family": fam, "name": n, "mechanism": why, "coverage": round(cover, 4),
               "min_rows": min(nrows), "verdict": verdict, "fails": "+".join(why_fail),
               "min_abs_rho": round(minabs, 4), "max_abs_placebo": round(maxplc, 4),
               "margin": round(margin, 2), "mean_rho": round(float(np.nanmean(rho_a)), 4),
               "mean_rho_raw": round(float(np.nanmean(raw)), 4),
               "mean_rho_winsor": round(float(np.nanmean(win)), 4),
               "dup_corr": round(dup_r, 3), "dup_with": dup_c}
        for label, a, b in zip(labels, rho, plc):
            row[f"rho_{label}"] = round(a, 4) if np.isfinite(a) else None
            row[f"plc_{label}"] = round(b, 4) if np.isfinite(b) else None
        rows_out.append(row)
        print(f"   {fam} {n:<22}" + "".join(f"{x:>+8.4f}" if np.isfinite(x) else f"{'nan':>8}" for x in rho)
              + f"  |plc| {maxplc:.4f}  min|r| {minabs:.4f}  x{margin:>5.1f}  raw {row['mean_rho_raw']:+.4f}"
              f"  win {row['mean_rho_winsor']:+.4f}  cov {cover:.2f}  dup {dup_r:+.2f} {dup_c:<22} "
              f"{verdict} {row['fails']}  ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="")
    ap.add_argument("--calib", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    log(f"instrument: {len(CONTROLS)} control columns (agent_cong_04 {len(CONG04_CONTROLS)} + late 7 + "
        f"row-own {len(ROWSELF)})")
    inst = AuditInstrument(CONTROLS)
    log(f"instrument ready in {time.time() - t0:.0f}s")
    if args.calib:
        calibrate(inst)
    fams = [f for f in args.families.split(",") if f]
    if fams:
        print()
        print("=" * 110)
        print(f"SCREEN. BAR: same sign 5/5, |rho| > {FLOOR} on every fold, min|rho| >= {MARGIN:.0f} x max|placebo|,"
              f" >= {MIN_ROWS} rows/fold")
        print(f"   {'':<24}" + "".join(f"{l:>8}" for l in inst.folds))
        rows = []
        for fam in fams:
            screen_family(inst, fam, rows)
        new = pl.DataFrame(rows)
        if OUTCSV.exists():
            old = pl.read_csv(OUTCSV, infer_schema_length=0)
            old = old.filter(~C("name").is_in(new["name"].to_list()))
            new = pl.concat([old, new.cast({c: pl.Utf8 for c in new.columns})], how="diagonal")
        else:
            new = new.cast({c: pl.Utf8 for c in new.columns})
        new.write_csv(OUTCSV)
        log(f"wrote {OUTCSV} ({new.height} rows)")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
