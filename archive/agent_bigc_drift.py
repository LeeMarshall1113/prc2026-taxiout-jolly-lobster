"""label-free drift signal: a key's mean of our own prediction (pv) and of the ADS-B-implied span within the ROW'S OWN MONTH minus the same key's mean in the OTHER months (shrunk n/(n+20)).
Uses no labels. fold: month = row month, others = the other 7 fold months. serve: month = 2026 month of the row, others = all 8 fold months."""
import numpy as np
SH = 20.0
def _gm(g, G, v):
    ok = np.isfinite(v); s = np.bincount(g[ok], weights=v[ok], minlength=G); n = np.bincount(g[ok], minlength=G).astype(float); return s, n
def drift_fold(keys, pv, span, mon):
    N, K = keys.shape; out = np.zeros((N, 2 * K), np.float32); months = sorted(set(mon.tolist()))
    for k in range(K):
        u, g = np.unique(keys[:, k], return_inverse=True); G = len(u)
        for vi, v in enumerate((pv, span)):
            s_all, n_all = _gm(g, G, v)
            for m in months:
                sel = np.where(mon == m)[0]; s_m, n_m = _gm(g[sel], G, v[sel])
                s_o, n_o = s_all - s_m, n_all - n_m; gg = g[sel]
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = (s_m[gg] / n_m[gg] - s_o[gg] / n_o[gg]) * (n_m[gg] / (n_m[gg] + SH))
                out[sel, 2 * k + vi] = np.where(np.isfinite(d), d, 0.0)
    return out
def drift_serve(keys_f, pv_f, span_f, keys_s, pv_s, span_s, mon_s):
    K = keys_f.shape[1]; out = np.zeros((len(keys_s), 2 * K), np.float32)
    for k in range(K):
        u = np.unique(keys_f[:, k]); gf = np.searchsorted(u, keys_f[:, k]); G = len(u)
        pos = np.clip(np.searchsorted(u, keys_s[:, k]), 0, G - 1); hit = u[pos] == keys_s[:, k]
        for vi, (vf, vs) in enumerate(((pv_f, pv_s), (span_f, span_s))):
            s_o, n_o = _gm(gf, G, vf)
            for m in np.unique(mon_s):
                sel = np.where((mon_s == m) & hit)[0]
                if len(sel) == 0: continue
                uu, gs = np.unique(pos[sel], return_inverse=True); s_m, n_m = _gm(gs, len(uu), vs[sel])
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = (s_m[gs] / n_m[gs] - s_o[pos[sel]] / n_o[pos[sel]]) * (n_m[gs] / (n_m[gs] + SH))
                out[sel, 2 * k + vi] = np.where(np.isfinite(d), d, 0.0)
    return out
