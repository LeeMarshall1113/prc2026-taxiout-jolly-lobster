"""Published IFR departure count as a second equation for the Rome record-type step (used by final/32_published_daily.py --count-equation).

Design and gate: docs/LAB-NOTES.md.

At Rome the published daily count of IFR departures (FLT_DEP_IFR_2 of the EUROCONTROL daily airport file) equals the number of departure rows filed under the
UTC date of the off-block stamp (take-off minus label). A no-plan row whose schedule gap is long is either a schedule copy (stamp = schedule, filed under the
schedule's date) or a real late departure (stamp = take-off minus an ordinary taxi). When those two dates differ by one day (the schedule is in the evening and the
take-off after midnight) the count of the two days separates the two types; when the dates agree the count is blind.

The step multiplies the minutes weight of every assignment of types to a day's rows by the likelihood of the published counts, in one exact forward-backward pass
over consecutive days. State of a date = the copy / real pattern of its crossing rows and the flip variable of the midnight boundary. Count noise:

    published(d) - ours(d) = t_d - t_(d+1) + e_d

t_b is the net number of fixed rows whose true date differs from their served date at boundary b (prior: near-midnight hazard computed per boundary from the rows'
served stamps and the empirical served-stamp error, plus the background rates below); e_d is an isolated day extra. Dates with no usable count (last day of a
calendar month, no figure published, day dropped by the minutes step's own gates) carry no factor.

Functions here are pure numpy; nothing reads a label of the scored months. The error sample comes from the 2025 training labels.
"""
from __future__ import annotations

import datetime
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EURO = REPO / "data" / "external" / "eurocontrol"
C = pl.col

MARGIN_S = 1200.0         # a real stamp this close to a UTC midnight has an uncertain date: the row is count-blind
MAX_INF = 6               # crossing rows per day kept as variables (more are left at their served stamp)
BG_FWD, BG_REV = 0.009, 0.012   # background flip rates per boundary (docs/LAB-NOTES.md)
ETA = 0.003               # isolated day extra, floor of one in a year
HAZ_WINDOW = 10800.0      # rows within this many seconds of a boundary can flip
AMBIG_FLIP = 0.5          # flip probability of a fixed row whose copy and real dates differ
PAD_DAYS = 2              # quiet days kept at both ends of the chain


@dataclass
class CountEq:
    """Everything the count factor needs. `ifr` maps a UTC day index (days since 1970-01-01) to the published IFR count."""
    ifr: dict
    errs: np.ndarray                  # sorted sample of served-stamp error (served value minus label, seconds) of ordinary rows
    bg_fwd: float = BG_FWD
    bg_rev: float = BG_REV
    eta: float = ETA
    log: list = field(default_factory=list)


def load_ifr(apt: str = "LIRF") -> dict:
    """Published IFR departures per UTC day for every year on disk: {day index: count}."""
    out: dict = {}
    for f in sorted(EURO.glob("all_pre_departure_delays_20*.csv")):
        d = pl.read_csv(f, infer_schema_length=0).filter(C("APT_ICAO") == apt)
        d = d.select(C("FLT_DATE").str.slice(0, 10).str.to_date().dt.epoch("d").alias("x"), C("FLT_DEP_IFR_2").cast(pl.Float64, strict=False).alias("n"))
        for x, n in zip(d["x"].to_list(), d["n"].to_list()):
            if n is not None and np.isfinite(n):
                out[int(x)] = float(n)
    return out


def error_sample(err: np.ndarray, n: int = 4001) -> np.ndarray:
    """Quantile summary (sorted) of a served-stamp error sample."""
    q = np.linspace(0.0, 1.0, n)
    return np.quantile(np.asarray(err, float), q)


def month_last_day(x: int) -> bool:
    d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(x))
    return (d + datetime.timedelta(days=1)).day == 1


def _cdf(errs: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.searchsorted(errs, v, side="left") / len(errs)


def boundary_priors(ce: CountEq, served_stamp: np.ndarray, extra: list[tuple[int, float, float]], lo: int, hi: int) -> dict:
    """Per boundary b in [lo, hi+1] (midnight starting UTC day b): (P(t=-1), P(t=0), P(t=+1)).

    served_stamp: stamps (epoch s) of the fixed rows. Forward flip = truth after the boundary while served before it: needs error >= distance.
    extra: (boundary, forward probability, reverse probability) of fixed rows whose copy and real dates differ.
    """
    out = {}
    s = np.sort(served_stamp)
    for b in range(lo, hi + 2):
        m = b * 86400.0
        a, c = np.searchsorted(s, m - HAZ_WINDOW), np.searchsorted(s, m)
        e_hi = np.searchsorted(s, m + HAZ_WINDOW)
        before, after = s[a:c], s[c:e_hi]
        lam_f = float((1.0 - _cdf(ce.errs, m - before)).sum()) if len(before) else 0.0
        lam_r = float(_cdf(ce.errs, -(after - m)).sum()) if len(after) else 0.0
        pf, pr = 1.0 - np.exp(-(lam_f + ce.bg_fwd)), 1.0 - np.exp(-(lam_r + ce.bg_rev))
        out[b] = [pf, pr]
    for b, ef, er in extra:
        if b in out:
            pf, pr = out[b]
            out[b] = [1.0 - (1.0 - pf) * (1.0 - ef), 1.0 - (1.0 - pr) * (1.0 - er)]
    pri = {}
    for b, (pf, pr) in out.items():
        p_plus, p_minus = pf * (1.0 - pr), pr * (1.0 - pf)
        pri[b] = np.array([p_minus, 1.0 - p_plus - p_minus, p_plus])
    return pri


def pattern_posteriors(ce: CountEq, entries: list[dict], fixed_stamp: np.ndarray, extra: list[tuple[int, float, float]], avail: set, note: list[str]) -> dict:
    """Posterior over the copy / real pattern of the crossing rows of every day, given minutes weights and counts.

    entries: one dict per enumerated day with keys day (int), m (number of crossing rows, <= MAX_INF), W (array of 2**m minutes weights summed over the
             day's other rows, any scale; bit i of the pattern index = crossing row i is a copy).
    fixed_stamp: epoch stamps of every row that is not a crossing variable. avail: set of day indices with a usable count.
    Returns {day: posterior array over the 2**m patterns}; days without crossing rows are absent.
    """
    inf_days = {e["day"]: e for e in entries if e["m"] > 0}
    if not inf_days:
        return {}
    lo, hi = min(inf_days) - 1 - PAD_DAYS, max(inf_days) + PAD_DAYS
    Fx = {}
    fx_days = np.floor(fixed_stamp / 86400.0).astype(np.int64)
    u, c = np.unique(fx_days[(fx_days >= lo - 1) & (fx_days <= hi + 1)], return_counts=True)
    Fx = dict(zip(u.tolist(), c.tolist()))
    pri = boundary_priors(ce, fixed_stamp, extra, lo, hi)
    pe = np.array([ce.eta / 2.0, 1.0 - ce.eta, ce.eta / 2.0])
    TV = np.array([-1, 0, 1])

    def Pe(x):
        x = np.rint(x).astype(int)
        return pe[np.clip(x + 1, 0, 2)] * (np.abs(x) <= 1) + 1e-9

    def npat(x):
        return 1 << inf_days[x]["m"] if x in inf_days else 1

    pop = {x: np.array([bin(i).count("1") for i in range(1 << e["m"])]) for x, e in inf_days.items()}

    def Wd(x):
        if x not in inf_days:
            return np.ones(1)
        w = np.asarray(inf_days[x]["W"], float)
        return w / w.sum()

    def trans(x):
        """array [c, t, c', t'] of the weight of moving from date x's state to date x+1's state (includes the entering weights of x+1)."""
        P0, P1 = npat(x), npat(x + 1)
        w1 = Wd(x + 1)
        pt1 = pri[x + 1]
        real_x = (inf_days[x]["m"] - pop[x]) if x in inf_days else np.zeros(1)
        cop_next = pop[x + 1] if (x + 1) in inf_days else np.zeros(1)
        if x in avail and x in ce.ifr:
            r = ce.ifr[x] - Fx.get(x, 0) - real_x[:, None] - cop_next[None, :]            # [c, c']
            d = r[:, None, :, None] - (TV[None, :, None, None] - TV[None, None, None, :])   # [c, t, c', t'] residual left to e
            L = Pe(d)
        else:
            L = np.ones((P0, 3, P1, 3))
        return L * w1[None, None, :, None] * pt1[None, None, None, :]

    days = list(range(lo, hi + 1))
    alpha = {lo: (Wd(lo)[:, None] * pri[lo][None, :])}
    scale = {}
    for x in days:
        T_ = trans(x)
        a = np.einsum("ct,ctdu->du", alpha[x], T_)
        s = a.sum()
        if not np.isfinite(s) or s <= 0:
            note.append("Rome count: chain likelihood vanished, count factor skipped")
            return {}
        alpha[x + 1] = a / s
        scale[x] = s
    beta = {hi + 1: np.ones((npat(hi + 1), 3))}
    for x in reversed(days):
        T_ = trans(x)
        b = np.einsum("ctdu,du->ct", T_, beta[x + 1])
        beta[x] = b / b.sum()
    out = {}
    for x in inf_days:
        g = (alpha[x] * beta[x]).sum(axis=1)
        out[x] = g / g.sum()
    return out


def crossing_flags(Te: np.ndarray, Se: np.ndarray, taxi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For rows with take-off Te, schedule Se (epoch s) and ordinary taxi: (real date index, copy date index, crossing & away from midnight)."""
    real_stamp = Te - taxi
    dr = np.floor(real_stamp / 86400.0).astype(np.int64)
    dc = np.floor(Se / 86400.0).astype(np.int64)
    tod = np.mod(real_stamp, 86400.0)
    ok = (dc == dr - 1) & (tod >= MARGIN_S) & (tod <= 86400.0 - MARGIN_S)
    return dr, dc, ok


def served_error_2025(cache: Path | None = None) -> np.ndarray:
    """Quantile summary of the served-stamp error (held-out chain value minus label) of ordinary Rome rows in the eight 2025 frame months.

    Reads the 2025 training labels (training data) and the stored held-out chain values data/interim/agent_eb136_tbl.parquet, like final/32 gate_from_2025.
    Cached in data/final/_gen/rome_count_err_2025.npy.
    """
    cache = cache or (REPO / "data" / "final" / "_gen" / "rome_count_err_2025.npy")
    if cache.exists():
        return np.load(cache)
    files = sorted((REPO / "data" / "raw").glob("training_*.parquet"))
    raw = (pl.scan_parquet([str(x) for x in files]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF"))
           .select(C("MVT_ID_mvt").alias("id"), (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("gs")).collect())
    eb = (pl.scan_parquet(REPO / "data" / "interim" / "agent_eb136_tbl.parquet").filter(C("apt") == "LIRF").select("id", "mon", "y", "pred").collect())
    d = eb.join(raw, on="id", how="inner").filter(C("gs").abs() < 5400)
    q = error_sample((d["pred"] - d["y"]).to_numpy())
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, q)
    return q
