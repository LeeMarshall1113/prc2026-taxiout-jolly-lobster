"""experiment lv: ONE view builder for the fold and the serve twin (docs/LAB-NOTES.md). Importable, numpy only.

make_views(ref, m=None, q=None, t=None, groups="AQC") -> (matrix float32, names)
  m: (n,4) ctrl members c,p3,r,e (member scale, pre-regime)   q: (n,3) q10,q50,q90   t: (n,4) seq-ON members; ref: (n,) served-analogue prediction.
  A: m_k - mean(m) (4), sd(m), mean(m) - ref                    -> 6
  Q: q90-q10, skew = (q90+q10-2*q50)/max(q90-q10,1), q50-ref     -> 3
  C: d_k = t_k - m_k (4), mean(d), sd(d)                         -> 6
Fold loaders read results/ltfmm/*.npz (ids, pred); serve loaders read the same names with held 99_98."""
import os

import numpy as np

MEM = ["c", "p3", "r", "e"]
QS = ["q10", "q50", "q90"]
FOLD_OF = {1: ("f1", "1_7"), 7: ("f1", "1_7"), 2: ("f2", "2_8"), 8: ("f2", "2_8"), 3: ("f3", "3_9"), 9: ("f3", "3_9"),
           4: ("f4", "4_10"), 10: ("f4", "4_10"), 5: ("f5", "5_11"), 11: ("f5", "5_11"), 6: ("f6", "6_12"), 12: ("f6", "6_12")}


def _ck(d, arm, f, tag, k):
    p = f"{d}/ltfmm_{arm}_{f}_{tag}_{k}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return dict(zip(z["ids"].astype("float64"), z["pred"].astype("float64")))


def load_fold(fid, monf, arm, kinds, d="results/ltfmm"):
    """(len(fid), len(kinds)) predictions of the fold member/quantile `kinds` for arm ctrl|treat; NaN where the checkpoint does not exist."""
    out = np.full((len(fid), len(kinds)), np.nan)
    for mo in np.unique(monf):
        if int(mo) not in FOLD_OF:
            continue
        f, tag = FOLD_OF[int(mo)]
        rows = np.where(monf == mo)[0]
        for j, k in enumerate(kinds):
            dd = _ck(d, arm, f, tag, k)
            if dd is not None:
                out[rows, j] = [dd.get(x, np.nan) for x in fid[rows]]
    return out


def load_serve(sid, arm, kinds, d="results/ltfmm"):
    out = np.full((len(sid), len(kinds)), np.nan)
    for j, k in enumerate(kinds):
        dd = _ck(d, arm, "srv", "99_98", k)
        if dd is not None:
            out[:, j] = [dd.get(x, np.nan) for x in sid]
    return out


EXTRA = {"E": ["q05", "q25", "q75", "q95"], "H": ["cbS", "cbD", "dart", "xgb", "l1"], "S": ["cs1", "cs2", "cs3", "cs4"], "B": ["nowx", "nosch"]}


def make_views(ref, m=None, q=None, t=None, groups="AQC", ex=None):
    """ex: dict kind -> (n,) predictions for the extra kinds (E quantiles, H hyper-param members, S seeds of c, B ablations)."""
    cols, names = [], []
    if "A" in groups:
        mm = m.mean(1)
        cols += [m - mm[:, None], m.std(1)[:, None], (mm - ref)[:, None]]
        names += [f"va_{k}" for k in MEM] + ["va_sd", "va_mean_ref"]
    if "Q" in groups:
        sp = q[:, 2] - q[:, 0]
        cols += [sp[:, None], ((q[:, 2] + q[:, 0] - 2 * q[:, 1]) / np.maximum(sp, 1.0))[:, None], (q[:, 1] - ref)[:, None]]
        names += ["vq_spread", "vq_skew", "vq_q50_ref"]
    if "C" in groups:
        d = t - m
        cols += [d, d.mean(1)[:, None], d.std(1)[:, None]]
        names += [f"vc_{k}" for k in MEM] + ["vc_mean", "vc_sd"]
    if "E" in groups:   # wider quantile family; needs q (q10,q50,q90) too
        q05, q25, q75, q95 = (ex[k] for k in EXTRA["E"]); q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
        sp90, iqr = np.maximum(q95 - q05, 1.0), np.maximum(q75 - q25, 1.0)
        cols += [np.column_stack([q95 - q05, q75 - q25, (q95 + q05 - 2 * q50) / sp90, (q75 + q25 - 2 * q50) / iqr, q25 - ref, q75 - ref, q95 - q90, q10 - q05])]
        names += ["ve_sp90", "ve_iqr", "ve_skew90", "ve_skewiqr", "ve_q25_ref", "ve_q75_ref", "ve_up", "ve_lo"]
    if "H" in groups:
        mm = m.mean(1); h = np.column_stack([ex[k] for k in EXTRA["H"]])
        cols += [h - mm[:, None], h.std(1)[:, None], (h.mean(1) - mm)[:, None]]
        names += [f"vh_{k}" for k in EXTRA["H"]] + ["vh_sd", "vh_mean_m"]
    if "S" in groups:
        sd5 = np.column_stack([m[:, 0]] + [ex[k] for k in EXTRA["S"]])
        cols += [sd5.std(1)[:, None], (sd5.mean(1) - m[:, 0])[:, None]]
        names += ["vs_sd", "vs_mean_c"]
    if "B" in groups:
        mm = m.mean(1)
        cols += [np.column_stack([ex["nowx"] - mm, ex["nosch"] - mm, ex["nowx"] - ex["nosch"]])]
        names += ["vb_nowx", "vb_nosch", "vb_wx_sch"]
    if "R" in groups:   # stand-handover columns (agent_lv_07_handover.py), NaN where no re-occupation within 2 h
        cols += [ex["R"]]
        names += ["r_next", "r_bind", "r_gap", "r_cnt", "a_c1", "a_c2"]
    if "W" in groups:   # runway-configuration columns (agent_lv_08_rwycfg.py)
        cols += [ex["W"]]
        names += ["w_n", "w_dom", "w_cfg35", "w_since", "w_s_36", "w_s_35L", "w_s_18", "w_s_17R", "w_s_34L", "w_s_35R", "w_own"]
    FL = ["d_south60", "d_south120", "d_asouth60", "d_asouth120", "d_chg_flow2h", "d_chg_rwy2h", "d_own_h-1", "d_tot_h-1", "d_own_h", "d_tot_h", "d_own_h+1", "d_tot_h+1", "p_strip", "p_south", "p_side", "p_rare"]
    if "P" in groups:   # runway physical encodings
        cols += [ex["F"][:, 12:16]]; names += FL[12:16]
    if "D" in groups:   # day-state columns
        cols += [ex["F"][:, 0:12]]; names += FL[0:12]
    if "T" in groups:   # arrival-to-departure transfer (agent_lv_09_arrT.py)
        cols += [ex["T"]]; names += [f"t{j}" for j in range(ex["T"].shape[1])]
    return np.hstack(cols).astype(np.float32), names
