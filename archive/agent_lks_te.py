"""LTFM key-stat engine (experiment lks, docs/LAB-NOTES.md). Out-of-month, smoothed to the pool prior, unseen key -> prior.
fold(held): row of month j != held uses months not in {held, j}; row of held month uses months != held; held=None -> leave-own-month-out (final fit).
serve(): pool = all months."""
import numpy as np
PRIOR = 30.0
STATS = ["mean", "med", "sd", "cnt", "p90", "tail"]
EY = np.array([0, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1350, 1500, 1800, 2200, 3000, 6000, 1e5])
ER = np.array([-1e5, -900, -600, -400, -300, -200, -150, -100, -60, -30, -10, 0, 10, 30, 60, 100, 150, 200, 300, 400, 600, 900, 1e5])
TARGETS = {"y": (EY, 1200.0), "r": (ER, 300.0)}   # tail = share of value > thr (y > 20 min; residual > +300 s)

def _q(H, q, E):
    NB = len(E) - 1; cum = np.cumsum(H, 1); tgt = q * cum[:, -1]; idx = np.minimum((cum < tgt[:, None]).sum(1), NB - 1)
    ar = np.arange(len(H)); prev = np.where(idx > 0, cum[ar, np.maximum(idx - 1, 0)], 0.0); h = H[ar, idx]
    fr = np.where(h > 0, (tgt - prev) / np.maximum(h, 1e-9), 0.5)
    lo = np.maximum(E[idx], -2000.0); hi = np.minimum(E[idx + 1], 6000.0)
    return lo + fr * (hi - lo)

class KS:
    def __init__(self, keys):          # keys (N, K) int64, N = all rows ever featured (fold + serve) so groups share ids
        self.K = keys.shape[1]; self.g = []; self.G = []
        for k in range(self.K):
            u, g = np.unique(keys[:, k], return_inverse=True); self.g.append(g); self.G.append(len(u))
    def _feat(self, rows, tab, own_sub, mi_rows, E, thr, k, pri):
        Hm, S, Q, Nn, T = tab; g = self.g[k][rows]; NB = len(E) - 1
        Ht, St, Qt, Nt, Tt = Hm.sum(1), S.sum(1), Q.sum(1), Nn.sum(1), T.sum(1)
        out = np.empty((len(rows), 6), np.float32)
        for c0 in range(0, len(rows), 100000):
            sl = slice(c0, c0 + 100000); gg = g[sl]
            if own_sub is not None:
                m = mi_rows[sl]; H = Ht[gg] - Hm[gg, m]; s = St[gg] - S[gg, m]; q = Qt[gg] - Q[gg, m]; n = Nt[gg] - Nn[gg, m]; t = Tt[gg] - T[gg, m]
            else:
                H, s, q, n, t = Ht[gg], St[gg], Qt[gg], Nt[gg], Tt[gg]
            pm = mi_rows[sl] if own_sub is not None else np.full(len(gg), len(pri["mu"]) - 1)
            Hs = H + pri["gh"][pm] * PRIOR; mean = (s + PRIOR * pri["mu"][pm]) / (n + PRIOR)
            var = (q + PRIOR * pri["m2"][pm]) / (n + PRIOR) - mean ** 2
            out[sl] = np.column_stack([mean, _q(Hs, .5, E), np.sqrt(np.maximum(var, 0)), n, _q(Hs, .9, E), (t + PRIOR * pri["tl"][pm]) / (n + PRIOR)])
        return out
    def build(self, rows_pool_idx, rows, tgt, v, mi, nm, held, serve_rows=None):
        """rows: indices (into keys) of labelled rows with value v (len=len(rows)) and month idx mi. returns features (len(rows), K*6) [+ serve (len(serve_rows), K*6)]"""
        E, thr = TARGETS[tgt]; NB = len(E) - 1
        hs = np.atleast_1d(held) if held is not None else np.array([], int)   # held = month INDEX or list of indices
        use = ~np.isin(mi, hs)
        b = np.clip(np.searchsorted(E, v, side="right") - 1, 0, NB - 1)
        # priors are month-exclusive too (row of month j: pool minus j; held/serve rows: whole pool) -> index nm = no exclusion
        gh = np.zeros((nm + 1, NB)); mu = np.zeros(nm + 1); m2 = np.zeros(nm + 1); tl = np.zeros(nm + 1)
        for m in range(nm + 1):
            u = use & (mi != m) if m < nm else use
            if u.sum() == 0: u = use
            h_ = np.bincount(b[u], minlength=NB).astype(float); gh[m] = h_ / h_.sum(); mu[m] = v[u].mean(); m2[m] = (v[u] ** 2).mean(); tl[m] = (v[u] > thr).mean()
        pri = dict(gh=gh, mu=mu, m2=m2, tl=tl)
        out = np.empty((len(rows), self.K * 6), np.float32); outs = None if serve_rows is None else np.empty((len(serve_rows), self.K * 6), np.float32)
        for k in range(self.K):
            g = self.g[k][rows]; G = self.G[k]; idx = g[use] * nm + mi[use]
            Hm = np.bincount(idx * NB + b[use], minlength=G * nm * NB).reshape(G, nm, NB).astype(np.float32)
            S = np.bincount(idx, weights=v[use], minlength=G * nm).reshape(G, nm); Q = np.bincount(idx, weights=v[use] ** 2, minlength=G * nm).reshape(G, nm)
            Nn = np.bincount(idx, minlength=G * nm).reshape(G, nm).astype(float); T = np.bincount(idx, weights=(v[use] > thr).astype(float), minlength=G * nm).reshape(G, nm)
            tab = (Hm, S, Q, Nn, T); ar = np.arange(len(rows))
            if len(hs) == 0:
                out[:, k * 6:(k + 1) * 6] = self._feat(rows, tab, True, mi, E, thr, k, pri)
            else:
                o = ~np.isin(mi, hs)
                out[o, k * 6:(k + 1) * 6] = self._feat(rows[o], tab, True, mi[o], E, thr, k, pri)
                out[~o, k * 6:(k + 1) * 6] = self._feat(rows[~o], tab, None, None, E, thr, k, pri)
            if serve_rows is not None:
                outs[:, k * 6:(k + 1) * 6] = self._feat(serve_rows, tab, None, None, E, thr, k, pri)
        return (out, outs) if serve_rows is not None else out
def names(keynames, tgt): return [f"ks_{tgt}_{k}_{s}" for k in keynames for s in STATS]
