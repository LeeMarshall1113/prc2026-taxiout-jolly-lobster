"""Merge members_full.npz (round 1) and members_full2.npz (round 2) and add the derived bags."""
import numpy as np

a, b = np.load("data/interim/members_full.npz"), np.load("data/interim/members_full2.npz")
assert (a["mvt_id"] == b["mvt_id"]).all() and np.allclose(a["base_rank"], b["base_rank"])
out = {k: a[k] for k in a.files}
out.update({k: b[k] for k in b.files if k.startswith("lgb_")})
out["et_bag5"] = np.mean([out["lgb_extratrees"]] + [out[f"lgb_et_s{k}"] for k in range(1, 5)], axis=0)
out["etr_bag2"] = np.mean([out["lgb_etr_s0"], out["lgb_etr_s1"]], axis=0)
np.savez_compressed("data/interim/members_v95.npz", **out)
print("merged", sorted(k for k in out if k.startswith(("lgb", "et"))))
