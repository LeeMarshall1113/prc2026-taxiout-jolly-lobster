"""CANDIDATE B, the FAMILY rather than its best single column.

Why this exists. agent_cong_02 reads one construction at a time, which is the right way to
ask "is there anything here", but it under-states what a model would extract: production would
receive the whole family and the trees would combine them. Its best single column implied
-0.847s pooled against a pre-declared -1.0s bar -- a narrow fail on a reading that is known to
be conservative, so the family version decides whether that fail stands.

Everything is leave-one-fold-out: the coefficients come from the other four folds and are
applied blind to the held one, so nothing here is fitted in sample. Three arms:

  raw        the family as built
  demeaned   each column minus its (airport, month) mean, because the serve-time check found
             LFPG's median neighbour lateness at 574s in 2025 and 231s in 2026 -- a level
             shift at the airport carrying the largest share of squared error. A level-shifted
             feature is a different feature; the demeaned form is immune to that.
  scored     the raw family read on fold (1,7) alone and split by month, because January and
             July are the scored months and the board is what this has to survive.

    PYTHONPATH=. .venv/Scripts/python.exe archive/agent_cong_04_joint.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

from agent_cong_02_lateness import CONTROLS as BASE_CONTROLS
from agent_cong_02_lateness import LATE_HI, LATE_LO, neighbour_means
from prc.weather import FEATURES as WX

# Weather is the obvious confounder left out of agent_cong_02: a regulated, snowy morning makes
# neighbours push back late AND makes taxi slow, so without it the family could be reading the
# weather the model already has. Added here rather than there so the two readings can be compared.
CONTROLS = BASE_CONTROLS + list(WX)
from agent_cong_common import ID, Instrument, Proj, concentration, load_departures

C = pl.col

FAMILY = ["late_apt_past15_mean", "late_apt_fol15_mean", "late_rwy_past15_mean",
          "late_rwy_fol15_mean", "late_apt_past60_mean", "late_rwy_past60_mean",
          "late_apt_past60_own_minus"]


def fit(X: np.ndarray, V: np.ndarray, r: np.ndarray):
    """Coefficients of the residual on the family, both taken orthogonal to the controls."""
    p = Proj(X)
    RV = np.column_stack([p(V[:, j]) for j in range(V.shape[1])])
    beta, *_ = np.linalg.lstsq(RV, p(r), rcond=None)
    return beta


def apply(X: np.ndarray, V: np.ndarray, r: np.ndarray, beta: np.ndarray):
    p = Proj(X)
    RV = np.column_stack([p(V[:, j]) for j in range(V.shape[1])])
    r2 = r - RV @ beta
    d = r ** 2 - r2 ** 2
    return float(np.sqrt(np.mean(r2 ** 2)) - np.sqrt(np.mean(r ** 2))), d


def main() -> None:
    dep = load_departures()
    t = dep["t"].to_numpy().astype(np.int64)
    adep = dep["adep"].to_numpy().astype(str)
    ym = dep["ym"].to_numpy().astype(str)
    rwy = dep["rwy"].to_numpy().astype(str)
    a = dep["t_aobt"].fill_null(-1).to_numpy().astype(np.float64)
    e = dep["t_eobt"].fill_null(-1).to_numpy().astype(np.float64)
    late = np.where((a > 0) & (e > 0), a - e, np.nan)
    valid = np.isfinite(late) & (late >= LATE_LO) & (late <= LATE_HI)
    key_apt = np.char.add(adep, np.char.add("|", ym))
    key_rwy = np.char.add(np.char.add(key_apt, "|"), rwy)
    proxy = np.where(valid, late, 0.0)

    feats = {}
    for gname, key in (("apt", key_apt), ("rwy", key_rwy)):
        for nm, v in neighbour_means(t, key, proxy, valid).items():
            feats[f"late_{gname}_{nm}"] = v
    feats["late_apt_past60_own_minus"] = np.where(valid, late, np.nan) - feats["late_apt_past60_mean"]

    dem = {}
    for nm in FAMILY:
        v = feats[nm].copy()
        for k in np.unique(key_apt):
            idx = np.flatnonzero((key_apt == k) & np.isfinite(v))
            if len(idx):
                v[idx] -= v[idx].mean()
        dem[nm] = v

    inst = Instrument(dep, CONTROLS)
    labels = list(inst.folds)

    for tag, src in (("raw", feats), ("demeaned by (airport, month)", dem)):
        print("=" * 96)
        print(f"FAMILY of {len(FAMILY)} columns, {tag}; coefficients from the OTHER four folds")
        mats, rs, Xs = {}, {}, {}
        for label in labels:
            loc, y, base = inst.folds[label]
            V = np.column_stack([src[nm][loc] for nm in FAMILY])
            m = np.isfinite(V).all(axis=1)
            mats[label] = V[m]
            rs[label] = (y - base)[m]
            Xs[label] = inst.design(loc, m, base)
        betas = {lb: fit(Xs[lb], mats[lb], rs[lb]) for lb in labels}
        tot_r, tot_d = [], []
        print(f"   {'fold':<7}{'rows':>10}{'honest':>10}{'-top1':>9}{'-top5':>9}"
              f"{'-top20':>9}{'-top100':>9}")
        for label in labels:
            b = np.mean([betas[k] for k in labels if k != label], axis=0)
            d_out, deltas = apply(Xs[label], mats[label], rs[label], b)
            con = concentration(rs[label], deltas)
            tot_r.append(rs[label])
            tot_d.append(deltas)
            print(f"   {label:<7}{len(rs[label]):>10,}{d_out:>+10.4f}"
                  + "".join(f"{con[k]:>+9.4f}" for k in (1, 5, 20, 100)))
        r, d = np.concatenate(tot_r), np.concatenate(tot_d)
        pooled = float(np.sqrt(np.mean(np.maximum(r ** 2 - d, 0))) - np.sqrt(np.mean(r ** 2)))
        con = concentration(r, d)
        print(f"   {'POOLED':<7}{len(r):>10,}{pooled:>+10.4f}"
              + "".join(f"{con[k]:>+9.4f}" for k in (1, 5, 20, 100)))
        print()

    print("=" * 96)
    print("THE SCORED MONTHS: fold (1,7) split by month, raw family, coefficients from the")
    print("   other four folds. January and July 2026 are what the board scores.")
    loc, y, base = inst.folds["1_7"]
    V = np.column_stack([feats[nm][loc] for nm in FAMILY])
    m = np.isfinite(V).all(axis=1)
    X = inst.design(loc, m, base)
    r = (y - base)[m]
    mats, rs, Xs = {}, {}, {}
    for label in labels:
        l2, y2, b2 = inst.folds[label]
        V2 = np.column_stack([feats[nm][l2] for nm in FAMILY])
        m2 = np.isfinite(V2).all(axis=1)
        mats[label], rs[label], Xs[label] = V2[m2], (y2 - b2)[m2], inst.design(l2, m2, b2)
    b = np.mean([fit(Xs[k], mats[k], rs[k]) for k in labels if k != "1_7"], axis=0)
    mon = np.array([s[5:7] for s in inst.ym[loc][m]])
    d_all, deltas = apply(X, V[m], r, b)
    print(f"   fold (1,7) all rows  n={m.sum():,}  {d_all:+.4f}s")
    for mm in ("01", "07"):
        sel = mon == mm
        rr, dd = r[sel], deltas[sel]
        got = float(np.sqrt(np.mean(np.maximum(rr ** 2 - dd, 0))) - np.sqrt(np.mean(rr ** 2)))
        con = concentration(rr, dd)
        print(f"   month {mm}  n={int(sel.sum()):>8,}  {got:>+9.4f}"
              + "".join(f"  -top{k} {con[k]:+.4f}" for k in (20, 100)))

    print()
    print("SIZING. The project's measured exchange rate for a BULK-MODEL change is 0.38x of the")
    print("   fold figure (docs/LAB-NOTES.md), so a pooled fold gain of g is about 0.38*g on the")
    print("   board. This is a LINEAR reading of a family a gradient-boosted model would use")
    print("   nonlinearly, against controls entered linearly -- it is not a fold test and does")
    print("   not replace one.")


if __name__ == "__main__":
    main()
