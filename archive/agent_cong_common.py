"""Shared instrument for the two congestion-feature probes (A: ADS-B taxi speed, B: neighbour
off-block lateness). Read-only. Nothing here trains a model.

THE INSTRUMENT, and why it is this one. `base` is production's chain (out-of-fold `bagged`
-> clip0 -> gap clamp -> LOBT box) from a model holding every shipped feature, so y - base is
already orthogonalised against everything we ship. A candidate is then residualised against:

  * base and base^2, so "the model already predicted a long taxi" cannot masquerade as signal;
  * airport dummies, so a level difference between airports is not read as a within-airport
    congestion effect (this matters most for A, whose raw speed differs by airfield layout);
  * a caller-chosen block of SHIPPED features from the cached feature frame -- for B that is
    the existing nm_* neighbour family plus the row's own dep_delay, which is exactly the
    "measure incremental value against a model that already has the rest" rule. Raw
    correlation with the target is what let ref_taxi_s read +0.378 and it is never used here.

Reused verbatim from archive/agent_queue_01_probe.py: FOLDS, chain, residualise, partial_rho.
That probe's placebo -- identical arithmetic on a scrambled window -- read |rho| <= 0.019 and
is the noise floor both candidates must clear.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, W = "MVT_ID_mvt", 3606.0

FOLDS = [("1_7", "results/preds_q_gate_1_7.parquet", False),
         ("2_8", "results/preds_q_gate_2_8.parquet", False),
         ("3_9", "results/preds_n3_bulk_3_9.parquet", True),
         ("4_10", "results/preds_s4_ctrl_4_10.parquet", True),
         ("5_11", "results/preds_c_ctrl_5_11.parquet", True)]

PLACEBO_FLOOR = 0.019   # what zero reads as, from the runway-queue longshot
FEATCACHE = "data/interim/train_features.parquet"


def chain(bagged, gs, gl, plan):
    """production's chain, lifted verbatim from archive/screen_rule.py."""
    p = np.clip(bagged, 0, None)
    p = np.where(np.isfinite(gs) & (gs >= 5000) & (gs < 50000) & (p > gs), gs, p)
    box = plan & np.isfinite(gl)
    p = np.where(box, np.clip(p, np.where(box, gl - W, -np.inf), np.where(box, gl + W, np.inf)), p)
    return np.clip(p, 0, None)


def residualise(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


class Proj:
    """One orthonormal basis for a design matrix, reused for every residualisation against it.

    Identical arithmetic to residualise(), but the expensive factorisation happens once
    instead of once per feature -- which matters because the positive control bisects, so a
    naive version runs six hundred least-squares solves on a 330,000-row design.
    """

    def __init__(self, X: np.ndarray):
        self.Q, _ = np.linalg.qr(X)

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return v - self.Q @ (self.Q.T @ v)


def partial_rho(X, a, b) -> float:
    p = X if isinstance(X, Proj) else Proj(X)
    ra, rb = p(a), p(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.mean(ra * rb) / (sa * sb))


def load_departures() -> pl.DataFrame:
    """Every 2025 departure with what the chain, the windows and the callsign map need."""
    return (pl.scan_parquet("data/raw/training_*.parquet")
            .filter(C("PHASE_mvt") == "DEP")
            .select(C(ID),
                    C("ADEP_mvt").alias("adep"),
                    C("RUNWAY_mvt").cast(pl.Utf8).fill_null("?").alias("rwy"),
                    C("FLIGHT_mvt").cast(pl.Utf8).alias("flight"),
                    C("CALLSIGN_flt").cast(pl.Utf8).alias("cs_flt"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"),
                    C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d").alias("date"),
                    C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
                    (C("MVT_TIME_UTC_mvt").dt.weekday() * 24
                     + C("MVT_TIME_UTC_mvt").dt.hour()).alias("how"),
                    C("AOBT_3_flt").dt.epoch("s").alias("t_aobt"),
                    C("EOBT_1_flt").dt.epoch("s").alias("t_eobt"),
                    C("AOBT_3_flt").is_not_null().alias("plan"),
                    (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                    (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"))
            .collect())


class Instrument:
    """Folds, controls and the two readings, over one fixed departure frame."""

    def __init__(self, dep: pl.DataFrame, control_cols: list[str]):
        self.dep = dep
        self.ids = dep[ID].to_numpy()
        self.adep = dep["adep"].to_numpy().astype(str)
        self.ym = dep["ym"].to_numpy().astype(str)
        gs = dep["gs"].fill_null(np.nan).to_numpy().astype(float)
        gl = dep["gl"].fill_null(np.nan).to_numpy().astype(float)
        plan = dep["plan"].fill_null(False).to_numpy()
        pos = {m: i for i, m in enumerate(self.ids)}

        cache = (pl.scan_parquet(FEATCACHE).select([ID] + control_cols).collect())
        cpos = {m: i for i, m in enumerate(cache[ID].to_numpy())}
        order = np.array([cpos.get(m, -1) for m in self.ids])
        self.ctrl = {}
        for c in control_cols:
            v = cache[c].cast(pl.Float64).fill_null(np.nan).to_numpy().astype(float)
            col = np.full(len(self.ids), np.nan)
            ok = order >= 0
            col[ok] = v[order[ok]]
            self.ctrl[c] = col
        self.control_cols = control_cols

        self.folds = {}
        for label, path, drop_lirf in FOLDS:
            P = pl.read_parquet(path).select(ID, "y", "bagged")
            loc = np.array([pos[m] for m in P[ID].to_numpy()])
            y = P["y"].to_numpy().astype(float)
            base = chain(P["bagged"].to_numpy().astype(float), gs[loc], gl[loc], plan[loc])
            if drop_lirf:
                keep = self.adep[loc] != "LIRF"
                loc, y, base = loc[keep], y[keep], base[keep]
            self.folds[label] = (loc, y, base)

    def design(self, loc, m, base) -> np.ndarray:
        """Controls: intercept, the chain's own prediction, airport dummies, shipped features.

        A control column that is null is passed as zero WITH a missing flag, so a row is never
        silently dropped and 'this feature is absent here' can itself be absorbed.
        """
        n = int(m.sum())
        X = [np.ones(n), base[m], base[m] ** 2 / 1e3]
        for c in self.control_cols:
            v = self.ctrl[c][loc][m]
            X.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
            if not np.isfinite(v).all():
                X.append((~np.isfinite(v)).astype(float))
        apt = self.adep[loc][m]
        for a in np.unique(apt)[1:]:
            X.append((apt == a).astype(float))
        return np.column_stack(X)


def linear_gain(X, v, r, beta: float | None = None):
    """RMSE change from using v linearly on top of the controls, plus the concentration check.

    Returns (delta_rmse, beta, deltas) where deltas[i] = r_i^2 - r'_i^2, the squared error this
    row gives up. beta=None fits in sample (a ceiling); passing a beta fitted elsewhere is the
    honest read.
    """
    p = X if isinstance(X, Proj) else Proj(X)
    rv = p(v)
    denom = float(np.dot(rv, rv))
    if denom < 1e-9:
        return 0.0, 0.0, np.zeros_like(r)
    b = float(np.dot(rv, p(r)) / denom) if beta is None else beta
    r2 = r - b * rv
    d = r ** 2 - r2 ** 2
    return float(np.sqrt(np.mean(r2 ** 2)) - np.sqrt(np.mean(r ** 2))), b, d


def concentration(r, d, ks=(1, 5, 20, 100)) -> dict[int, float]:
    """RMSE change after deleting the k rows the feature helps most. Same rule as screen_rule."""
    out = {}
    order = np.argsort(-d)
    for k in ks:
        keep = np.ones(len(r), dtype=bool)
        keep[order[:k]] = False
        r2 = r[keep] ** 2 - d[keep]
        out[k] = float(np.sqrt(np.mean(np.maximum(r2, 0))) - np.sqrt(np.mean(r[keep] ** 2)))
    return out


def positive_control(inst: Instrument, eligible, targets=(-1.0, -0.25), seed=20260920) -> None:
    """Inject a synthetic signal of KNOWN size and confirm the instrument returns it.

    Without this, 'we measured nothing' and 'our instrument measures nothing' read the same --
    the failure mode this project has logged twenty-eight times.
    """
    rng = np.random.default_rng(seed)
    print("POSITIVE CONTROL -- a synthetic signal sized to an exact RMSE gain, then measured")
    for target in targets:
        got, rhos = [], []
        for label in inst.folds:
            loc, y, base = inst.folds[label]
            m = eligible(label)
            if m.sum() < 1000:
                continue
            X = Proj(inst.design(loc, m, base))
            r = (y - base)[m]
            z = X(rng.standard_normal(int(m.sum())))
            z /= max(z.std(), 1e-9)
            # bisect the injected amplitude c so the TRUE RMSE gain from regressing r + c*z
            # on z is exactly `target`, then run the identical reading on it
            rmse = np.sqrt(np.mean(r ** 2))
            lo, hi = 0.0, 10 * rmse
            for _ in range(60):
                c = 0.5 * (lo + hi)
                r2 = r + c * z
                d, _, _ = linear_gain(X, z, r2)
                if d < target:
                    hi = c
                else:
                    lo = c
            r2 = r + c * z
            d, _, _ = linear_gain(X, z, r2)
            got.append(d)
            rhos.append(partial_rho(X, z, r2))
        print(f"   target {target:+.3f}s -> recovered per fold "
              f"{' '.join(f'{x:+.3f}' for x in got)}   partial rho "
              f"{' '.join(f'{x:+.3f}' for x in rhos)}")
