"""arith4 step X (experiment arith4, 2026-10-06): the FROZEN round-4 column list = the prediction-arithmetic trick applied SYSTEMATICALLY to every one of the corrector's
16 key-statistic keys (docs/LAB-NOTES.md section 3, written before any scoring). Rounds two and three hand-picked 7 keys (A), 5 cells (K) and 3 cells (Q);
this list covers what they left: for all 16 keys of the Te2 block (ades, cty, op_apt, op_stand, op_hb, op_rwy, st_rwy, st_hb, ty_st, mkt, wk, op_ades, op_ty, ty_rwy, ades_hb, st_op_rwy).
Groups (letters for the arm string):
  T "t-statistic" (16): the key's out-of-month residual mean over its standard error, mean_k / (sd_k / sqrt(cnt_k + 30)) (reliability-weighted level; the trees hold mean, sd, cnt separately).
  L "level" (18): pv minus the key's out-of-month label mean, and over max(sd_k, 30), for the 9 keys round two did NOT cover (ades, cty, op_hb, ty_st, mkt, wk, op_ades, op_ty, ades_hb).
  W "within-cell level" (24): pv minus the out-of-month mean pv of the key's cell, and over the cell sd of pv (floor 30), for the 12 keys neither E (st_rwy, apt-hour) nor K (op_apt, op_stand, ty_rwy, ...) covered:
      ades, cty, op_hb, op_rwy, st_hb, ty_st, mkt, wk, op_ades, op_ty, ades_hb, st_op_rwy.
  D "cross-key second differences" (30): for the 15 keys other than st_rwy: ymean_k minus ymean_st_rwy (does this key's level sit above what the stand x runway cell implies) and mean_k minus mean_st_rwy (the same on the residual).
Total 88. Every column is a pure function of the row's own inputs (pv, keys) and out-of-month statistics built the corrector's way (s112 for the fold; the pv cell statistics exclude the row's own month).
Winsorised at the 0.5/99.5 percentiles of the pooled non-held rows (every seventh row), as rounds two and three. Shuffled-block control: the block permuted jointly within airport-month (draw = seed)."""
import numpy as np

KEYS16 = ["ades", "cty", "op_apt", "op_stand", "op_hb", "op_rwy", "st_rwy", "st_hb", "ty_st", "mkt", "wk", "op_ades", "op_ty", "ty_rwy", "ades_hb", "st_op_rwy"]
L_KEYS = ["ades", "cty", "op_hb", "ty_st", "mkt", "wk", "op_ades", "op_ty", "ades_hb"]                                         # not in round-2 group A
W_KEYS = ["ades", "cty", "op_hb", "op_rwy", "st_hb", "ty_st", "mkt", "wk", "op_ades", "op_ty", "ades_hb", "st_op_rwy"]        # not in round-2 E or round-3 K
D_KEYS = [k for k in KEYS16 if k != "st_rwy"]
GROUPS4 = {"T": ["T_t_" + k for k in KEYS16],
           "L": sum([["L_pv-ym_" + k, "L_z_" + k] for k in L_KEYS], []),
           "W": sum([["W_pvdev_" + k, "W_pvz_" + k] for k in W_KEYS], []),
           "D": sum([["D_ym_%s-st_rwy" % k, "D_mean_%s-st_rwy" % k] for k in D_KEYS], [])}
# s112 layout (agent_bigc_te.STATS): 0 mean (residual), 1 med, 2 sd, 3 cnt, 4 p90, 5 tail, 6 ymean (label)


def build4(groups, pv, Kf, KI, s112, mon, months, use, pool_mask):
    """(N, ncols) float32 in fold row order and the names. pv float32 (N,), Kf (N,16) int64 keys, KI name->column, s112 (N,112) this fold's key statistics,
    use = rows in the pool (non-held), pool_mask = per-month flag of pool membership (own month excluded for pool rows)."""
    pv = pv.astype(np.float64)
    st = lambda k, j: s112[:, KI[k] * 7 + j].astype(np.float64)   # noqa: E731
    nm = len(months)
    MI = np.searchsorted(months, mon)
    out, names = [], []
    for g in groups:
        if g == "T":
            for k in KEYS16:
                out.append(st(k, 0) / (np.maximum(st(k, 2), 30.0) / np.sqrt(st(k, 3) + 30.0)))
        elif g == "L":
            for k in L_KEYS:
                d = pv - st(k, 6)
                out += [d, d / np.maximum(st(k, 2), 30.0)]
        elif g == "W":
            for k in W_KEYS:
                G = np.unique(Kf[:, KI[k]], return_inverse=True)[1]
                ng = G.max() + 1
                idx = G * nm + MI
                S = np.bincount(idx[use], weights=pv[use], minlength=ng * nm).reshape(ng, nm)
                Q = np.bincount(idx[use], weights=pv[use] ** 2, minlength=ng * nm).reshape(ng, nm)
                Nn = np.bincount(idx[use], minlength=ng * nm).reshape(ng, nm).astype(np.float64)
                inpool = pool_mask[MI]
                s_ = S.sum(1)[G] - np.where(inpool, S[G, MI], 0.0)
                q = Q.sum(1)[G] - np.where(inpool, Q[G, MI], 0.0)
                n = Nn.sum(1)[G] - np.where(inpool, Nn[G, MI], 0.0)
                mean = np.where(n > 20, s_ / np.maximum(n, 1), np.nan)
                sd = np.sqrt(np.maximum(q / np.maximum(n, 1) - (s_ / np.maximum(n, 1)) ** 2, 0))
                d = pv - mean
                out += [d, d / np.maximum(sd, 30.0)]
        elif g == "D":
            for k in D_KEYS:
                out += [st(k, 6) - st("st_rwy", 6), st(k, 0) - st("st_rwy", 0)]
        else:
            raise ValueError(g)
        names += GROUPS4[g]
    return np.column_stack([np.asarray(o, np.float32) for o in out]), names
