"""istnext lib (experiment "istnext", 2026-10-06): the position block E + K + Y + Q + S + R (43 columns), built by the FROZEN round-three builder (agent_arith3_02_cols.py)
for either population (nine airports, or Istanbul with its own cells). No 2026 label; no row's own label enters its columns.
D must hold: pv, y, mon, apt, hour, RUNWAY_mvt, is_weekend, Kf (int matrix), KN (names incl. op_apt, op_stand, ty_rwy, st_rwy, cty), Dpcd (>= 4 plan-clock columns d_SCHED, d_LOBT, d_IOBT, d_EOBT), t (take-off, epoch s)."""
import numpy as np
import agent_arith3_02_cols as C3

E_NAMES = ["E_SR_dev", "E_SR_z", "E_AH_dev", "E_AH_z"]
POS_GROUPS = ["K", "Y", "Q", "S", "R"]
POS_NAMES = E_NAMES + sum([C3.GROUPS3[g] for g in POS_GROUPS], [])
Z9 = ["E_SR_z", "E_AH_z", "K_OA_z", "K_OSTD_z", "K_TR_z", "K_RHW_z", "K_CR_z", "Y_day_z", "Y_win60_z"]   # D3: the nine standardised positions


def cells_sr_ah(D):
    ac = np.unique(D["apt"], return_inverse=True)[1]
    return np.unique(D["Kf"][:, D["KN"].index("st_rwy")], return_inverse=True)[1], C3.factor(ac, D["hour"])


def position_block(D, use, pool_mask, day=None):
    """(N, 43) float32, names. use = pool rows (bool); pool_mask = bool per month index of the pool months (rows of those months leave out their own month)."""
    mon = D["mon"]; months = sorted(set(mon.tolist())); nm = len(months); MI = np.searchsorted(months, mon)
    pv = D["pv"].astype(np.float64)
    out = []
    for G in cells_sr_ah(D):
        m, sd, n, _ = C3.loo_cell(G, pv, use, MI, nm, pool_mask)
        out += [(pv - m).astype(np.float32), ((pv - m) / np.maximum(sd, 30.0)).astype(np.float32)]
    if day is None:
        day = C3.day_block(D)
    B, nms = C3.build3(D, POS_GROUPS, use, pool_mask, np.zeros((len(pv), 7), np.float32), {"st_rwy": 0}, day=day)
    assert E_NAMES + nms == POS_NAMES
    return np.column_stack(out + [B]).astype(np.float32), POS_NAMES


def limits(M, rows):
    lo, hi = np.nanpercentile(M[rows], [0.5, 99.5], axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)
