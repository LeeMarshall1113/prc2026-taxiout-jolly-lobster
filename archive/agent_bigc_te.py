"""out-of-month target statistics for the bigc corrector (ladder step 3). Keys = apt-specific operator / callsign prefix / stand / aircraft type codes.
Fold: for held month h, rows of month j != h use stats from months not in {h, j}; rows of h use stats from all months != h (never their own month, never the held month's labels)."""
import numpy as np
KEYCOLS = ["AIRCRAFT_OPERATOR_flt", "callsign_op", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
PRIOR = 30.0
def keys_from(X, cols):
    ap = X[:, cols.index("ADEP_mvt")].astype(np.int64)
    return np.column_stack([ap * 1_000_000 + X[:, cols.index(c)].astype(np.int64) + 1 for c in KEYCOLS])
def te_fold(keys, r, mon, held, months):
    N, K = keys.shape; out = np.full((N, K), np.nan, np.float32); mi = np.searchsorted(months, mon); hi = months.index(held); trm = np.array([m != held for m in months])
    for k in range(K):
        _, g = np.unique(keys[:, k], return_inverse=True); G = g.max() + 1; msk = mi != hi
        S = np.bincount(g[msk] * len(months) + mi[msk], weights=r[msk], minlength=G * len(months)).reshape(G, -1)
        n = np.bincount(g[msk] * len(months) + mi[msk], minlength=G * len(months)).reshape(G, -1).astype(float)
        St, Nt = S.sum(1), n.sum(1)
        own = mi != hi; res = np.empty(N)
        res[own] = (St[g[own]] - S[g[own], mi[own]]) / (Nt[g[own]] - n[g[own], mi[own]] + PRIOR)
        res[~own] = St[g[~own]] / (Nt[g[~own]] + PRIOR); out[:, k] = res
    return out
def te_serve(keys_fold, r, keys_serve):
    out = np.zeros(keys_serve.shape, np.float32)
    for k in range(keys_fold.shape[1]):
        u, g = np.unique(keys_fold[:, k], return_inverse=True); S = np.bincount(g, weights=r); n = np.bincount(g).astype(float)
        pos = np.searchsorted(u, keys_serve[:, k]); pos = np.clip(pos, 0, len(u) - 1); hit = u[pos] == keys_serve[:, k]
        out[:, k] = np.where(hit, S[pos] / (n[pos] + PRIOR), 0.0)
    return out
def te_loo(keys, r, mon, months):
    """final fit: every row's stat from all OTHER months (own month excluded)."""
    N, K = keys.shape; out = np.empty((N, K), np.float32); mi = np.searchsorted(months, mon)
    for k in range(K):
        _, g = np.unique(keys[:, k], return_inverse=True); G = g.max() + 1
        S = np.bincount(g * len(months) + mi, weights=r, minlength=G * len(months)).reshape(G, -1); n = np.bincount(g * len(months) + mi, minlength=G * len(months)).reshape(G, -1).astype(float)
        out[:, k] = (S.sum(1)[g] - S[g, mi]) / (n.sum(1)[g] - n[g, mi] + PRIOR)
    return out

# ---------------------------------------------------------------- v2 (Y): rich out-of-month key statistics
E_ = np.array([-900, -600, -400, -300, -200, -150, -100, -60, -30, -10, 0, 10, 30, 60, 100, 150, 200, 300, 400, 600, 900.0]); NB = len(E_) - 1
STATS = ["mean", "med", "sd", "cnt", "p90", "tail", "ymean"]
def _bin(r): return np.clip(np.searchsorted(E_, r, side="right") - 1, 0, NB - 1)
def _q(H, q):
    cum = np.cumsum(H, 1); tgt = q * cum[:, -1]; idx = np.minimum((cum < tgt[:, None]).sum(1), NB - 1)
    prev = np.where(idx > 0, cum[np.arange(len(H)), np.maximum(idx - 1, 0)], 0.0); h = H[np.arange(len(H)), idx]
    fr = np.where(h > 0, (tgt - prev) / np.maximum(h, 1e-9), 0.5); return E_[idx] + fr * (E_[idx + 1] - E_[idx])
def _feat(H, S, Q, Nn, SY, ymu, gvar, prior):
    """H (n,NB) counts, S,Q,Nn,SY (n,) sums; returns (n,7)"""
    gh = _GH[None, :] * prior; Hs = H + gh
    mean = S / (Nn + prior); var = (Q + prior * gvar) / (Nn + prior) - mean ** 2
    tail = (Hs[:, 0] + Hs[:, 1] + Hs[:, -1]) / Hs.sum(1)
    ym = (SY + prior * ymu) / (Nn + prior)
    return np.column_stack([mean, _q(Hs, 0.5), np.sqrt(np.maximum(var, 0)), Nn, _q(Hs, 0.9), tail, ym]).astype(np.float32)
_GH = None
class Te2:
    PRIOR = 30.0
    def __init__(self, keys):
        self.K = keys.shape[1]; self.u = []; self.g = []
        for k in range(self.K):
            u, g = np.unique(keys[:, k], return_inverse=True); self.u.append(u); self.g.append(g)
    def _tables(self, k, r, y, mi, nm, use):
        global _GH
        g = self.g[k]; G = len(self.u[k]); b = _bin(r)
        idx = (g[use] * nm + mi[use]); Hm = np.bincount(idx * NB + b[use], minlength=G * nm * NB).reshape(G, nm, NB).astype(np.float32)
        S = np.bincount(idx, weights=r[use], minlength=G * nm).reshape(G, nm); Q = np.bincount(idx, weights=r[use] ** 2, minlength=G * nm).reshape(G, nm)
        Nn = np.bincount(idx, minlength=G * nm).reshape(G, nm).astype(float); SY = np.bincount(idx, weights=y[use], minlength=G * nm).reshape(G, nm)
        return Hm, S, Q, Nn, SY
    def fold(self, r, y, ymu_row, mon, held, months):
        """features (N, K*7). held=None -> leave-own-month-out for every row (final fit)."""
        global _GH
        nm = len(months); mi = np.searchsorted(months, mon); N = len(r); hi = months.index(held) if held is not None else -1
        use = mi != hi; gh = np.bincount(_bin(r[use]), minlength=NB).astype(float); _GH = gh / gh.sum(); gvar = r[use].var()
        out = np.empty((N, self.K * 7), np.float32); own = use  # rows whose own month is in the pool
        for k in range(self.K):
            g = self.g[k]; Hm, S, Q, Nn, SY = self._tables(k, r, y, mi, nm, use)
            Ht, St, Qt, Nt, SYt = Hm.sum(1), S.sum(1), Q.sum(1), Nn.sum(1), SY.sum(1)
            for sel, sub in ((own, True), (~own, False)):
                ii = np.where(sel)[0]
                if len(ii) == 0: continue
                for c0 in range(0, len(ii), 200000):
                    c = ii[c0:c0 + 200000]; gg = g[c]
                    if sub: out[c, k * 7:(k + 1) * 7] = _feat(Ht[gg] - Hm[gg, mi[c]], St[gg] - S[gg, mi[c]], Qt[gg] - Q[gg, mi[c]], Nt[gg] - Nn[gg, mi[c]], SYt[gg] - SY[gg, mi[c]], ymu_row[c], gvar, self.PRIOR)
                    else: out[c, k * 7:(k + 1) * 7] = _feat(Ht[gg], St[gg], Qt[gg], Nt[gg], SYt[gg], ymu_row[c], gvar, self.PRIOR)
        return out
    def serve(self, r, y, mon, months, keys_s, ymu_s):
        global _GH
        nm = len(months); mi = np.searchsorted(months, mon); use = np.ones(len(r), bool); gh = np.bincount(_bin(r), minlength=NB).astype(float); _GH = gh / gh.sum(); gvar = r.var()
        out = np.empty((len(keys_s), self.K * 7), np.float32)
        for k in range(self.K):
            Hm, S, Q, Nn, SY = self._tables(k, r, y, mi, nm, use); Ht, St, Qt, Nt, SYt = Hm.sum(1), S.sum(1), Q.sum(1), Nn.sum(1), SY.sum(1)
            pos = np.clip(np.searchsorted(self.u[k], keys_s[:, k]), 0, len(self.u[k]) - 1); hit = self.u[k][pos] == keys_s[:, k]
            z = lambda a: np.where(hit, a[pos], 0.0)
            Hh = np.where(hit[:, None], Ht[pos], 0.0)
            out[:, k * 7:(k + 1) * 7] = _feat(Hh, z(St), z(Qt), z(Nt), z(SYt), ymu_s, gvar, self.PRIOR)
        return out
