"""istarith lib (experiment "istarith", 2026-10-05): the new columns for the Istanbul (LTFM) views corrector. No labels of 2026 read.
Group A "level distance" (22 columns): for 11 of the corrector's own 20 keys, current prediction minus the key's out-of-month label mean, and that divided by the key's
  out-of-month label sd (floor 30 s). The mean and sd are the corrector's OWN key-statistic columns (the y block of agent_lks_te.KS), so the statistic is out-of-month exactly as it builds its own.
Group E "within-cell level" (8 columns, label-free): prediction minus the (shrunk, out-of-month) mean prediction of its stand x runway cell, runway x hour-bucket cell, hour-of-day cell,
  each plain and divided by the cell sd of the prediction (floor 30 s); plus prediction minus the mean prediction of the same-day same-runway departures, plain and divided by sd (needs >= 5 rows, else missing).
Current prediction = the out-of-fold prediction the corrector already receives (fold: 'ref'; serve: the served Istanbul base before the corrector)."""
import numpy as np

KN = ['op', 'csp', 'csfn', 'ades', 'cty', 'reg', 'stand', 'area', 'pier', 'type', 'wk', 'mkt', 'rwy_stand', 'rwy_area', 'op_hb', 'ty_stand', 'op_stand', 'op_rwy', 'rwy_hb', 'op_ty']
AKEYS = ['op', 'ades', 'type', 'stand', 'rwy_stand', 'op_hb', 'op_rwy', 'rwy_hb', 'ty_stand', 'op_stand', 'op_ty']
ANAMES = [f"{p}_{k}" for k in AKEYS for p in ("lvl", "lvlz")]
ENAMES = ["e_rs_dev", "e_rs_z", "e_rh_dev", "e_rh_z", "e_h_dev", "e_h_z", "e_day_dev", "e_day_z"]
NAMES = ANAMES + ENAMES
PRIOR = 30.0


def group_a(pv, Sy):
    """Sy = y block of the corrector's key statistics (N, 20*6): per key mean, med, sd, cnt, p90, tail."""
    out = []
    for k in AKEYS:
        j = KN.index(k) * 6
        d = pv - Sy[:, j]; out += [d, d / np.maximum(Sy[:, j + 2], 30.0)]
    return np.column_stack(out).astype(np.float32)


def _cell(code, pv_t, mi_t, hold_t, pv_pool, mi_pool, use, nm, G):
    """target rows: month-exclusive cell mean/sd of pool prediction (rows not held exclude their own month; held/serve rows use the whole pool)."""
    idx = code[0][use] * nm + mi_pool[use]
    S = np.bincount(idx, weights=pv_pool[use], minlength=G * nm).reshape(G, nm); Q = np.bincount(idx, weights=pv_pool[use] ** 2, minlength=G * nm).reshape(G, nm)
    N = np.bincount(idx, minlength=G * nm).reshape(G, nm).astype(float)
    St, Qt, Nt = S.sum(1), Q.sum(1), N.sum(1)
    g = code[1]; s = St[g].copy(); q = Qt[g].copy(); n = Nt[g].copy()
    own = ~hold_t
    if mi_t is not None and own.any():
        m = mi_t[own]; s[own] -= S[g[own], m]; q[own] -= Q[g[own], m]; n[own] -= N[g[own], m]
    mu = pv_pool[use].mean(); m2 = (pv_pool[use] ** 2).mean()
    mean = (s + PRIOR * mu) / (n + PRIOR); var = (q + PRIOR * m2) / (n + PRIOR) - mean ** 2
    sd = np.sqrt(np.maximum(var, 0)); dev = pv_t - mean
    return dev, dev / np.maximum(sd, 30.0)


def group_e(pv_t, hold_t, mi_t, pv_pool, mi_pool, use, nm, cell_codes_pool, cell_codes_t, day_code_t):
    """cell_codes_*: list of 3 integer arrays (stand x runway, runway x hour bucket, hour). pool = fold rows (real predictions); t = target rows (fold rows or serve rows)."""
    out = []
    for cp, ct in zip(cell_codes_pool, cell_codes_t):
        G = int(max(cp.max(), ct.max())) + 1
        d, z = _cell((cp, ct), pv_t, mi_t, hold_t, pv_pool, mi_pool, use, nm, G); out += [d, z]
    u, inv = np.unique(day_code_t, return_inverse=True)
    n = np.bincount(inv).astype(float); s = np.bincount(inv, weights=pv_t); q = np.bincount(inv, weights=pv_t ** 2)
    mean = s / n; sd = np.sqrt(np.maximum(q / n - mean ** 2, 0)); dev = pv_t - mean[inv]
    ok = n[inv] >= 5
    out += [np.where(ok, dev, np.nan), np.where(ok, dev / np.maximum(sd[inv], 30.0), np.nan)]
    return np.column_stack(out).astype(np.float32)


def winsor(M, tr, lo=0.5, hi=99.5):
    """clip every column at its lo/hi percentiles over the training rows (NaN ignored)."""
    out = M.copy()
    for j in range(M.shape[1]):
        c = M[tr, j]; c = c[np.isfinite(c)]
        if len(c) == 0: continue
        a, b = np.percentile(c, [lo, hi]); out[:, j] = np.clip(M[:, j], a, b)
    return out
