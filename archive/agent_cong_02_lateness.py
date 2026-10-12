"""CANDIDATE B: the neighbourhood mean of off-block LATENESS, not of taxi.

WHAT IS ABSENT. prc/features.py `_nm_neighbours` aggregates MVT - AOBT_3 -- a taxi proxy --
over 15- and 60-minute windows before and after each departure, by airport and by airport +
runway, and ships twenty columns of it. `dep_delay` = AOBT_3 - EOBT_1 exists per row and is
never aggregated over anybody. Pushback punctuality and surface taxi are different signals:
the first reflects turnaround, stand and ground-handling stress, the second the taxiway
system. Same windowing code, different proxy column.

WHY IT IS NOT THE DEAD QUEUE FAMILY. The closed family is "any feature that counts events
between off-block and wheels-up", which is a re-reading of gap_aobt. This counts nothing
inside this flight's interval: it is OTHER flights' AOBT_3 against their OWN EOBT_1, in a
fixed wall-clock window. It also cannot be the dead off-block-anchored congestion family --
those were anchored at this row's AOBT_3; this is anchored at MVT, like the shipped family.

KNOWN TAINT, checked explicitly below. AOBT_3 is NM's *planned* value at LEBL, LEMD, EDDM and
LTFM, so at those four `dep_delay` may be a plan artefact rather than a measurement. Reported
per airport, and the four are called out.

BARS: the pre-declared set in archive/agent_cong_00_size.py. B1 |partial rho| > 0.019 with the
same sign on (1,7) and (2,8); B2 placebo under half of it; B3 implied pooled RMSE gain >= 1.0s;
B4 concentration at top 1/5/20/100.

    PYTHONPATH=. .venv/Scripts/python.exe archive/agent_cong_02_lateness.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

from agent_cong_common import (ID, Instrument, PLACEBO_FLOOR, Proj, concentration,
                               linear_gain, load_departures, partial_rho, positive_control)

C = pl.col

WINDOWS = (900, 3600)
# dep_delay is a signed lateness, so unlike the taxi proxy it is not clipped at 0. The bound
# keeps a handful of multi-day plan artefacts out of a MEAN; it is the same guard
# _nm_neighbours applies at 7,200s, widened and made two-sided.
LATE_LO, LATE_HI = -3600.0, 10800.0
PLANNED_AOBT = {"LEBL", "LEMD", "EDDM", "LTFM"}

# EVERY shipped nm_* column is a control, not a subset. The candidate is the same windowing
# code on a different proxy, so leaving even one cell of the shipped family out would let the
# candidate re-read it. The rest of the shipped congestion block is here for the same reason.
NM_SHIPPED = ["nm_apt_past15_mean", "nm_apt_past15_n", "nm_apt_fol15_mean", "nm_apt_fol15_n",
              "nm_apt_past60_mean", "nm_apt_past60_n", "nm_apt_fol60_mean", "nm_apt_fol60_n",
              "nm_rwy_past15_mean", "nm_rwy_past15_n", "nm_rwy_fol15_mean", "nm_rwy_fol15_n",
              "nm_rwy_past60_mean", "nm_rwy_past60_n", "nm_rwy_fol60_mean", "nm_rwy_fol60_n",
              "nm_rwy_past60_own_minus", "nm_rwy_fol60_own_minus", "nm_rwy_next_wait",
              "nm_rwy_next_proxy"]
CONTROLS = ["gap_aobt", "gap_sched", "dep_delay", "dep_30min", "dep_60min", "dep_rwy_30min",
            "dep_rwy_headway", "sched_demand_30min", "arr_30min", "arr_60min",
            "arr_taxi_60min", "arr_slow_60min"] + NM_SHIPPED


def neighbour_means(t, key, proxy, valid, windows=WINDOWS):
    """Mean of `proxy` over OTHER rows sharing `key`, in fixed wall-clock windows.

    The arithmetic is _nm_neighbours': per key, sort by time, prefix-sum the proxy, then
    searchsorted the window edges. The row itself and same-second ties are excluded from the
    past window by taking side='left' at the anchor, and from the forward window by 'right',
    exactly as the shipped code does.
    """
    n = len(t)
    out = {}
    for w in windows:
        for side in ("past", "fol"):
            out[f"{side}{w // 60}_mean"] = np.full(n, np.nan)
            out[f"{side}{w // 60}_n"] = np.full(n, np.nan)
    order = np.argsort(key, kind="stable")
    sk = key[order]
    edges = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    for a, b in zip(edges, np.r_[edges[1:], len(sk)]):
        q = order[a:b]
        s = q[valid[q]]
        if len(s) == 0:
            continue
        so = s[np.argsort(t[s], kind="stable")]
        st = t[so].astype(np.float64)
        cs = np.concatenate([[0.0], np.cumsum(proxy[so])])
        qt = t[q].astype(np.float64)
        for w in windows:
            for side in ("past", "fol"):
                if side == "past":
                    lo, hi = np.searchsorted(st, qt - w, "left"), np.searchsorted(st, qt, "left")
                else:
                    lo, hi = np.searchsorted(st, qt, "right"), np.searchsorted(st, qt + w, "right")
                cnt = (hi - lo).astype(float)
                out[f"{side}{w // 60}_n"][q] = cnt
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"{side}{w // 60}_mean"][q] = np.where(
                        cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1.0), np.nan)
    return out


def main() -> None:
    dep = load_departures()
    n = dep.height
    t = dep["t"].to_numpy().astype(np.int64)
    adep = dep["adep"].to_numpy().astype(str)
    ym = dep["ym"].to_numpy().astype(str)
    rwy = dep["rwy"].to_numpy().astype(str)
    t_aobt = dep["t_aobt"].fill_null(-1).to_numpy().astype(np.float64)
    t_eobt = dep["t_eobt"].fill_null(-1).to_numpy().astype(np.float64)

    late = np.where((t_aobt > 0) & (t_eobt > 0), t_aobt - t_eobt, np.nan)
    valid = np.isfinite(late) & (late >= LATE_LO) & (late <= LATE_HI)
    print(f"dep_delay over {n:,} 2025 departures: finite {np.isfinite(late).mean():.3f}, "
          f"in [{LATE_LO:.0f},{LATE_HI:.0f}] {valid.mean():.3f}")
    q = np.nanpercentile(late[np.isfinite(late)], [1, 10, 50, 90, 99])
    print(f"   percentiles 1/10/50/90/99: {' '.join(f'{x:,.0f}' for x in q)}s   "
          f"exactly zero {np.mean(late[np.isfinite(late)] == 0):.3f}")
    print("   share exactly zero by airport (AOBT_3 planned rather than actual shows here):")
    for a in sorted(set(adep)):
        m = (adep == a) & np.isfinite(late)
        tag = " <- documented as PLANNED" if a in PLANNED_AOBT else ""
        print(f"      {a}  n={int(m.sum()):>7,}  zero={np.mean(late[m] == 0):.3f}  "
              f"median={np.nanmedian(late[m]):>7,.0f}s  sd={np.nanstd(late[m]):>7,.0f}s{tag}")

    key_apt = np.char.add(adep, np.char.add("|", ym))
    key_rwy = np.char.add(np.char.add(key_apt, "|"), rwy)
    proxy = np.where(valid, late, 0.0)

    feats: dict[str, np.ndarray] = {}
    for gname, key in (("apt", key_apt), ("rwy", key_rwy)):
        for name, v in neighbour_means(t, key, proxy, valid).items():
            feats[f"late_{gname}_{name}"] = v
    # the row's own lateness against its neighbourhood -- the shipped family's own_minus shape
    feats["late_apt_past60_own_minus"] = np.where(valid, late, np.nan) - feats["late_apt_past60_mean"]
    feats["late_apt_span60"] = feats["late_apt_fol60_mean"] - feats["late_apt_past60_mean"]

    # PLACEBO: the same construction read at a DIFFERENT flight's time in the same airport and
    # month. Level and marginal distribution are preserved; only the time alignment is broken.
    # The missingness PATTERN is held fixed -- only finite values are shuffled among the rows
    # that already had one -- so the placebo cannot read as signal through "this row has no
    # value", and it is measurable on exactly the rows the real construction is measured on.
    rng = np.random.default_rng(20260920)
    placebo: dict[str, np.ndarray] = {}
    for name, v in feats.items():
        pv = v.copy()
        for k in np.unique(key_apt):
            idx = np.flatnonzero((key_apt == k) & np.isfinite(v))
            if len(idx) > 1:
                pv[idx] = v[idx][rng.permutation(len(idx))]
        placebo[name] = pv

    inst = Instrument(dep, CONTROLS)

    def eligible(label):
        loc, y, base = inst.folds[label]
        return np.ones(len(loc), dtype=bool)

    print()
    positive_control(inst, eligible)

    print()
    print("=" * 100)
    print("CALIBRATION OF THE BASELINE -- what does a feature PRODUCTION ALREADY SHIPS read as?")
    print("   The five fold prediction files predate the NM neighbour family's adoption, so the")
    print("   chained residual may not have it priced in. If the SHIPPED nm_* columns read as")
    print("   large here, then any reading of the candidate is partly 'this baseline lacks the")
    print("   neighbour family', not 'lateness is new information'. Controls: everything except")
    print("   the nm_* block, so the shipped columns are read the same way the candidate is.")
    bare = Instrument(dep, CONTROLS)              # loads every column ...
    bare.control_cols = [c for c in CONTROLS if c not in NM_SHIPPED]   # ... but uses fewer
    for name in ("nm_rwy_past15_mean", "nm_apt_past15_mean", "nm_rwy_past60_mean"):
        rhos = []
        for label in bare.folds:
            loc, y, base = bare.folds[label]
            m = np.isfinite(bare.ctrl[name][loc])
            X = Proj(bare.design(loc, m, base))
            rhos.append(partial_rho(X, bare.ctrl[name][loc][m], (y - base)[m]))
        print(f"   SHIPPED {name:<24}" + " ".join(f"{x:+.4f}" for x in rhos))
    del bare

    print()
    print("=" * 100)
    print("B1/B2: partial rho against the chained residual, per fold, and the placebo beside it")
    print(f"        noise floor from the runway-queue longshot: |rho| <= {PLACEBO_FLOOR}")
    labels = list(inst.folds)
    header = "   " + f"{'construction':<30}" + "".join(f"{l:>9}" for l in labels) + f"{'|placebo|':>11}"
    print(header)
    best = None
    for name in feats:
        rhos, prhos = [], []
        for label in labels:
            loc, y, base = inst.folds[label]
            m = np.isfinite(feats[name][loc])
            if m.sum() < 1000:
                rhos.append(np.nan)
                prhos.append(np.nan)
                continue
            X = Proj(inst.design(loc, m, base))
            r = (y - base)[m]
            rhos.append(partial_rho(X, feats[name][loc][m], r))
            prhos.append(partial_rho(X, placebo[name][loc][m], r))
        pl_abs = np.nanmax(np.abs(prhos))
        print("   " + f"{name:<30}" + "".join(f"{x:>+9.4f}" for x in rhos) + f"{pl_abs:>11.4f}")
        score = np.nanmean(np.abs(rhos))
        if best is None or score > best[1]:
            best = (name, score, rhos, pl_abs)

    print(f"\n   strongest by mean |rho|: {best[0]}  mean |rho| {best[1]:.4f}  "
          f"worst placebo {best[3]:.4f}")

    print()
    print("=" * 100)
    print("B3/B4: implied RMSE change from a LINEAR use of the strongest construction")
    print("   beta fitted in sample is a CEILING; beta fitted on the other folds is the honest read")
    name = best[0]
    v = feats[name]
    betas = {}
    for label in labels:
        loc, y, base = inst.folds[label]
        m = np.isfinite(v[loc])
        X = Proj(inst.design(loc, m, base))
        _, b, _ = linear_gain(X, v[loc][m], (y - base)[m])
        betas[label] = b
    tot_r, tot_d = [], []
    print(f"   {'fold':<6}{'rows':>10}{'beta':>10}{'in-sample':>12}{'honest':>10}"
          f"{'-top1':>9}{'-top5':>9}{'-top20':>9}{'-top100':>9}")
    for label in labels:
        loc, y, base = inst.folds[label]
        m = np.isfinite(v[loc])
        X = Proj(inst.design(loc, m, base))
        r = (y - base)[m]
        d_in, b, _ = linear_gain(X, v[loc][m], r)
        b_out = float(np.mean([betas[k] for k in labels if k != label]))
        d_out, _, deltas = linear_gain(X, v[loc][m], r, beta=b_out)
        con = concentration(r, deltas)
        tot_r.append(r)
        tot_d.append(deltas)
        print(f"   {label:<6}{int(m.sum()):>10,}{b:>10.4f}{d_in:>+12.4f}{d_out:>+10.4f}"
              + "".join(f"{con[k]:>+9.4f}" for k in (1, 5, 20, 100)))
    r = np.concatenate(tot_r)
    d = np.concatenate(tot_d)
    pooled = float(np.sqrt(np.mean(np.maximum(r ** 2 - d, 0))) - np.sqrt(np.mean(r ** 2)))
    con = concentration(r, d)
    print(f"   {'POOLED':<6}{len(r):>10,}{'':>10}{'':>12}{pooled:>+10.4f}"
          + "".join(f"{con[k]:>+9.4f}" for k in (1, 5, 20, 100)))

    print()
    print("=" * 100)
    print("PER AIRPORT, pooled over folds: partial rho of the strongest construction.")
    print("   The four airports where AOBT_3 is NM's PLANNED value are marked.")
    for a in sorted(set(adep)):
        rows, rho = 0, []
        for label in labels:
            loc, y, base = inst.folds[label]
            m = np.isfinite(v[loc]) & (inst.adep[loc] == a)
            if m.sum() < 500:
                continue
            X = Proj(inst.design(loc, m, base))
            rho.append(partial_rho(X, v[loc][m], (y - base)[m]))
            rows += int(m.sum())
        if not rho:
            continue
        tag = "  PLANNED AOBT_3" if a in PLANNED_AOBT else ""
        print(f"   {a}  n={rows:>8,}  partial rho per fold "
              f"{' '.join(f'{x:+.4f}' for x in rho)}   mean {np.mean(rho):+.4f}{tag}")

    print()
    print("=" * 100)
    print("DOSE-RESPONSE, pooled: mean chained residual by decile of the residualised")
    print("   construction. A tree can use a shape a linear rho cannot see.")
    allv, allr = [], []
    for label in labels:
        loc, y, base = inst.folds[label]
        m = np.isfinite(v[loc])
        X = Proj(inst.design(loc, m, base))
        allv.append(X(v[loc][m]))
        allr.append((y - base)[m])
    vv, rr = np.concatenate(allv), np.concatenate(allr)
    cuts = np.quantile(vv, np.linspace(0, 1, 11))
    for i in range(10):
        m = (vv >= cuts[i]) & (vv <= cuts[i + 1] if i == 9 else vv < cuts[i + 1])
        print(f"   decile {i + 1:>2}  n={int(m.sum()):>8,}  mean residual {rr[m].mean():>+9.2f}s  "
              f"median {np.median(rr[m]):>+8.2f}s")


if __name__ == "__main__":
    main()
