"""LOCAL WIRING CHECKS ONLY (ARITH4_MMAP=1): makes numpy.load hand out memory-mapped .npy copies (data/interim/agent_arith4_mmap/, written by agent_arith4_00_decomp.py) for the
three big matrices of the corrector inputs (agent_bigc13_fold.npz X, agent_bigc12_vfac_clean5.npz Xf, agent_bigc12_extra.npz Xf) so the library chain's import stays under a 4 GB cap
(the libraries' import takes five column views of the 963 MB fold matrix at once, about 5 GB of commit, which is fine in the 20 GB cloud container and not on the shared machine).
Never set in the cloud; the fits there read the npz members as always."""
import os
import numpy as np

_load0 = np.load
MM = os.environ.get("ARITH_I", "data/interim/") + "agent_arith4_mmap/"
MAP = {"agent_bigc13_fold.npz": ("X", "fold_X.npy"), "agent_bigc12_vfac_clean5.npz": ("Xf", "vf_Xf.npy"), "agent_bigc12_extra.npz": ("Xf", "ex_Xf.npy")}


class _Z:
    def __init__(self, npz, key, path):
        self._z, self._k, self._p = npz, key, path
        self.files = npz.files

    def __getitem__(self, k):
        if k == self._k:
            return _load0(self._p, mmap_mode="r")
        return self._z[k]


def load(path, *a, **kw):
    base = os.path.basename(str(path))
    if base in MAP and os.path.exists(MM + MAP[base][1]):
        return _Z(_load0(path, *a, **kw), MAP[base][0], MM + MAP[base][1])
    return _load0(path, *a, **kw)


np.load = load
