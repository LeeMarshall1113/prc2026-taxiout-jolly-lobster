"""experiment adsbq: extra track anchors for the ADS-B second stage. ONE implementation, imported by
archive/agent_adsbq_apply.py for both its calibration rows and its serving rows (a flag and its
implementation living in different files is the bug this project has shipped eight times).

Label-free: everything comes from the row's own ground track, the stand map and the matched run.
Measured in archive/agent_adsbq_03_combine.py / _04_chain.py (docs/LAB-NOTES.md).

ANCHOR_COLS, in order:
  ci_20 .. ci_300   MVT - t(interpolated crossing of the R-m circle round the stand centroid), taking the LAST
                    fix inside R in the 3 h lookback and the next fix (which must be further out and follow
                    within stand_next_s); gated against the stand/appear estimate `taxi` so that a crossing
                    from some other visit to the stand (the inbound arrival) is dropped
  appear_raw        MVT - start of the lift-off run, gated the same way
  gp_20, gp_150     gap from the last fix inside 20 / 150 m to the next fix (time-from-last-fix)
  t_in150           time from the first to the last fix inside 150 m in the lookback (visit length)
  dens              fixes within +-300 s of the last in-150 m fix
  start_d           distance of the run's first fix from the stand centroid
The stack sees ci_R - p and appear_raw - p (p = the prediction), like x - p.
"""
import numpy as np

from prc.adsb_match import metres

RADII = (20, 30, 50, 75, 100, 150, 300)
GATE = {20: (-60, 600), 30: (-60, 600), 50: (-60, 600), 75: (-60, 600), 100: (-60, 600), 150: (-120, 120), 300: (-900, 60)}
APPEAR_GATE = (-60, 1800)
ANCHOR_COLS = tuple(f"ci_{R}" for R in RADII) + ("appear_raw", "gp_20", "gp_150", "t_in150", "dens", "start_d")
REL_COLS = tuple(f"ci_{R}" for R in RADII) + ("appear_raw",)   # fed to the stack as value - p


def anchors(r: dict, cmap: dict, tracks: dict, g, taxi) -> dict:
    """ANCHOR_COLS for one matched departure; NaN where not available. `taxi` = estimate()'s taxi (or None)."""
    nan = float("nan")
    out = dict.fromkeys(ANCHOR_COLS, nan)
    if r.get("start_t") is not None:
        out["appear_raw"] = r["mvt_t"] - r["start_t"]
    c = cmap.get((r["apt"], r.get("stand")))
    if c is not None and r.get("start_lat") is not None:
        out["start_d"] = float(metres(np.array([r["start_lat"]]), np.array([r["start_lon"]]), c[0], c[1])[0])
    tr = tracks.get((r["apt"], r["icao"]))
    if c is not None and tr is not None:
        t, la, lo, gs = tr
        win = np.flatnonzero((t <= r["liftoff_t"]) & (t >= r["liftoff_t"] - g.lookback_s))
        if len(win):
            d = metres(la[win], lo[win], c[0], c[1])
            for R in RADII:
                ins = np.flatnonzero(d <= R)
                if not len(ins):
                    continue
                k = ins.max()
                j = win[k]
                if j + 1 < len(t):
                    gap = t[j + 1] - t[j]
                    if R in (20, 150):
                        out[f"gp_{R}"] = float(gap)
                    dn = float(metres(np.array([la[j + 1]]), np.array([lo[j + 1]]), c[0], c[1])[0])
                    if dn > d[k] and gap <= g.stand_next_s:
                        f = min(max((R - d[k]) / (dn - d[k]), 0.0), 1.0)
                        out[f"ci_{R}"] = r["mvt_t"] - (t[j] + f * gap)
                if R == 150:
                    near = win[ins]
                    out["t_in150"] = float(t[j] - t[near.min()])
                    out["dens"] = float(len(np.flatnonzero((t >= t[j] - 300) & (t <= t[j] + 300))))
    # gates: a crossing inconsistent with the row's own estimate belongs to another visit to the stand
    for R in RADII:
        v = out[f"ci_{R}"]
        if taxi is None or not np.isfinite(v) or not (GATE[R][0] <= v - taxi <= GATE[R][1]):
            out[f"ci_{R}"] = nan
    v = out["appear_raw"]
    if taxi is None or not np.isfinite(v) or not (APPEAR_GATE[0] <= v - taxi <= APPEAR_GATE[1]):
        out["appear_raw"] = nan
    return out


def stack_columns(anc: dict, p: float) -> list:
    """The values appended to the second-stage feature vector, in ANCHOR_COLS order."""
    return [anc[k] - p if k in REL_COLS else anc[k] for k in ANCHOR_COLS]
