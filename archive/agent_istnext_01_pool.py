"""istnext 01 (design D2, docs/LAB-NOTES.md section 2): the pooled response model. Fitted on the 1.72M NINE-AIRPORT fold rows only (no Istanbul row, no 2026 row):
target = clip(label - prediction, +-900); inputs = the 43 position columns E + K + Y + Q + S + R (agent_istnext_lib.position_block; statistics leave out the row's own month), winsorised at the
0.5 / 99.5 percentiles of every seventh row. LightGBM 63 leaves, 300 per leaf, L2 20, rate 0.05, 400 rounds fixed, bagging 0.5, feature fraction 0.7, seed 0.
usage: python archive/agent_istnext_01_pool.py JOB1,JOB2,...   JOB = m<k> (fit without calendar month k) | ha (fit on months 1-6) | hb (fit on months 7-12) | all
Inputs under ARITH_I (default data/interim/): agent_bigc13_fold.npz, agent_bigc13_keys2.npz, agent_pcd_cols.npz, agent_bigc12_vfac_clean5.npz, agent_yearfp_names.json, agent_arith3_time.npz.
Outputs under ARITH_OUT (default data/interim/agent_istnext_pool/): one file per fit <JOB>.txt (LightGBM model) + limits.npz (clip limits, names)."""
import os
import sys
import time
import numpy as np
import lightgbm as lgb

sys.path.insert(0, "archive")
import agent_arith3_02_cols as C3  # noqa: E402
import agent_istnext_lib as LB  # noqa: E402

I = os.environ.get("ARITH_I", "data/interim/"); OUT = os.environ.get("ARITH_OUT", I + "agent_istnext_pool/"); os.makedirs(OUT, exist_ok=True)
C3.I = I
THREADS = int(os.environ.get("BIGC_THREADS", 6)); NR = int(os.environ.get("ISTNEXT_NR", 400))
t0 = time.time()
def P(*a): print("[%5.0fs]" % (time.time() - t0), *a, flush=True)
jobs = [j for j in sys.argv[1].split(",") if not os.path.exists(OUT + j + ".txt")]
if not jobs and os.path.exists(OUT + "limits.npz"):
    P("nothing to do"); sys.exit(0)
D = C3.load_inputs(); D.pop("vf", None)
N = len(D["y"]); mon = D["mon"]; months = sorted(set(mon.tolist())); P("rows", N, "months", months)
B, names = LB.position_block(D, np.ones(N, bool), np.ones(len(months), bool)); P("position block", B.shape)
lo, hi = LB.limits(B, np.arange(N)[::7]); B = np.clip(B, lo, hi)
np.savez(OUT + "limits.npz", lo=lo, hi=hi, names=np.array(names), nan_share=np.isnan(B).mean(0))
r = np.clip(D["y"] - D["pv"], -900, 900).astype(np.float32)
PR = dict(objective="l2", learning_rate=0.05, num_leaves=63, min_data_in_leaf=300, lambda_l2=20.0, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.7, max_bin=63, num_threads=THREADS, verbose=-1, seed=0)
for j in jobs:
    if j == "all": tr = np.ones(N, bool)
    elif j == "ha": tr = mon <= 6
    elif j == "hb": tr = mon >= 7
    else: tr = mon != int(j[1:])
    b = lgb.train(PR, lgb.Dataset(B[tr], r[tr], params=dict(max_bin=63, verbose=-1)), NR)
    msg = ""
    if (~tr).any():
        p = b.predict(B[~tr]); msg = "held-out rows %d: RMSE of the clipped residual %.3f -> %.3f" % ((~tr).sum(), float(np.sqrt((r[~tr] ** 2).mean())), float(np.sqrt(((r[~tr] - p) ** 2).mean())))
    b.save_model(OUT + j + ".txt.tmp"); os.replace(OUT + j + ".txt.tmp", OUT + j + ".txt"); P("fit", j, "train rows", int(tr.sum()), msg)
P("done")
