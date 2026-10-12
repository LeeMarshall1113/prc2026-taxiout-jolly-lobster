"""LTFM variant views (docs/LAB-NOTES.md): FOLD and SERVE twins of the LTFM-only members c,p3,r,e (seq off = ctrl, seq on = treat) and of the
LightGBM quantile views q10/q50/q90, ONE app, ONE code path (archive/agent_lv_members.py; serve = the same bulkr member fit with the 2026 LTFM ranking rows as
the test month 99).  Image/volumes/constants/flags = agent_ltfmm_app.py (L1).  Existing L1 checkpoints on prc-stage1:/ltfmm/ckpt (ctrl+treat f1-3) are
reused by tag; per-member checkpoints + vol.commit make `--detach` + spawn safe.

    $SYS -m modal run modal/agent_lv_app.py::smoke_serve            # serve twin, seq on, 5% rows, 30 iters (~$0.1): SMOKE PASS or stop
    $SYS -m modal run --detach modal/agent_lv_app.py::lv            # 7 arms (below), spawned
    $SYS modal/agent_lv_app.py fetch                                # /vol/ltfmm/ckpt/*.npz (+q, srv, f4) -> results/ltfmm/, logs -> logs/modal/
Arms (tag, mode, seq, fold, quant):  ltfmm_ctrl_f1/f2/f3 (quantiles only, members reloaded), ltfmm_ctrl_f4 (4,10: members + quantiles),
ltfmm_treat_f4 (members), ltfmm_ctrl_srv (serve: members + quantiles), ltfmm_treat_srv (serve: members).   ~$2 total (L1: 6 arms x 4 members = $1.3).
"""
import os
import sys

import modal

PINS = ["catboost==1.2.10", "polars==1.44.1", "numpy==2.5.2", "pyarrow", "scikit-learn==1.9.0", "lightgbm==4.7.0", "pandas"]
IGN = ["**/__pycache__/**", "**/*.pyc"]
image = (modal.Image.debian_slim(python_version="3.13").pip_install(*PINS)
         .add_local_dir("prc", remote_path="/repo/prc", ignore=IGN)
         .add_local_file("archive/agent_bulkr_fold.py", "/repo/archive/agent_bulkr_fold.py")
         .add_local_file("archive/agent_ltfm_seqfeat.py", "/repo/archive/agent_ltfm_seqfeat.py")
         .add_local_file("archive/agent_lv_members.py", "/repo/archive/agent_lv_members.py"))
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
vol = modal.Volume.from_name("prc-stage1", create_if_missing=True)
app = modal.App("prc-lv")

BASEDROP = ("airport_plan,arr_slow_180min,arr_slow_60min,arr_sub_day,arr_sub_stand,arr_taxi_180min,arr_taxi_trend,cfg_new,"
            "cfg_shift,clk_aobt_minus_sched,clk_arvt3_minus_aobt,clk_arvt3_minus_arvt1,clk_arvt3_minus_mvt,"
            "clk_eobt_minus_lobt,clk_iobt_minus_lobt,clk_lobt_minus_sched,clk_sec_aobt,clk_sec_mvt,clk_sec_sched,doy,"
            "lobt_minus_aobt,month,mvt_sec_00,prev_rwy_gap,prev_stand_gap,prev_stand_headway,ref_level,ref_taxi_s,"
            "turn_in_delay,turn_rwy_match,turn_sched_slack,wake_heavy_60min,wake_prev_heavy,wake_sep_req")
V94 = ("--seeds 1 --depth 8 --lr 0.08 --l2 3.0 --threads 8 --baseline blend_apt --regime-correction 1.0 --residual "
       "--bulk-plan-only --noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF --bulk-routes ALL "
       "--blend-cat-depth 10 --blend-lightgbm 0.70 --lgb-routes --iterations 2500 --noplan-iterations 600 "
       "--noplan-seeds 20").split()
FEATHASH = "ef679677ea3afa0a"   # == modal/agent_bulkr_bulk.py (prc-data cache already uploaded by the bulkr experiment)
N_BASE, N_SEQ = 80, 19
CK = "/vol/ltfmm/ckpt"
LOGS = "/vol/ltfmm/logs"


def _env():
    return {**os.environ, "PYTHONPATH": "/repo", "PYTHONIOENCODING": "utf-8",
            "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage",
            **{k: "8" for k in ("OMP_NUM_THREADS", "POLARS_MAX_THREADS")}}


@app.function(image=image, volumes={"/repo/data": data_vol, "/vol": vol}, cpu=8.0, memory=24576, timeout=60 * 60 * 3)
def run_arm(tag: str, seq: str, fold: str, expect: int, mode: str = "fold", quant: str = "", extra: list | None = None) -> dict:
    import re
    import subprocess
    import time
    vol.reload()
    os.chdir("/repo")
    os.makedirs("/repo/results", exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    h = open("/repo/data/interim/train_features.featurehash").read().strip()
    assert h == FEATHASH, f"prc-data feature cache hash {h} != {FEATHASH}"
    cmd = [sys.executable, "-u", "-W", "ignore", "archive/agent_lv_members.py", "--mode", mode, "--tag", tag, "--ltfm-seq", seq, *(["--quant", quant] if quant else []),
           "--only", "c,p3,r,e", "--ckpt-dir", CK, *V94, "--folds", fold, "--drop", BASEDROP, *(extra or [])]
    t0 = time.time()
    log = f"{LOGS}/{tag}.log"
    with open(log, "a") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=_env())
        fh.write(f"\nMODAL rc {rc} wall {time.time() - t0:.0f}s\n")
    txt = open(log).read()
    m = re.findall(r"\[" + re.escape(tag) + r"\] (\d+) features", txt) or re.findall(r"LTFM frame [\d,]+ rows, (\d+) features", txt)
    got = int(m[-1]) if m else None
    cks = sorted(f for f in os.listdir(CK) if f.startswith(tag + "_") and f.endswith(".npz"))
    vol.commit()
    out = {"tag": tag, "rc": rc, "seconds": round(time.time() - t0), "features": got, "ckpts": cks, "tail": txt[-3000:]}
    if got != expect:
        out["rc"] = out["rc"] or 90
        out["tail"] = f"FEATURE COUNT MISMATCH: expected {expect}, built {got}\n" + out["tail"]
    return out


@app.local_entrypoint()
def smoke_serve():
    r = run_arm.remote("lvsmoke_srv", "on", "1", N_BASE + N_SEQ, "serve", "q10,q50,q90",
                       ["--iterations", "30", "--noplan-iterations", "30", "--smoke-frac", "0.05"])
    print(r["tail"])
    ok = r["rc"] == 0 and r["features"] == N_BASE + N_SEQ and len(r["ckpts"]) >= 4
    print(f"rc={r['rc']} features={r['features']} ckpts={r['ckpts']} {r['seconds']}s")
    print("SMOKE PASS" if ok else "SMOKE FAIL")


@app.local_entrypoint()
def lv():
    """PRE-REGISTERED arms (docs/LAB-NOTES.md): fold month 10 (fold 4), quantiles at folds 1-4, serve twins ctrl + treat."""
    Q = "q10,q50,q90"
    arms = [("ltfmm_ctrl_f1", "off", "1", N_BASE, "fold", Q), ("ltfmm_ctrl_f2", "off", "2", N_BASE, "fold", Q),
            ("ltfmm_ctrl_f3", "off", "3", N_BASE, "fold", Q), ("ltfmm_ctrl_f4", "off", "4", N_BASE, "fold", Q),
            ("ltfmm_treat_f4", "on", "4", N_BASE + N_SEQ, "fold", ""),
            ("ltfmm_ctrl_srv", "off", "1", N_BASE, "serve", Q), ("ltfmm_treat_srv", "on", "1", N_BASE + N_SEQ, "serve", "")]
    for tag, seq, fold, n, mode, q in arms:
        call = run_arm.spawn(tag, seq, fold, n, mode, q)
        print(f"spawned {tag}: {call.object_id}")


def fetch():
    os.makedirs("results/ltfmm", exist_ok=True)
    os.makedirs("logs/modal", exist_ok=True)
    for sub, dst, ext in (("ltfmm/ckpt", "results/ltfmm", ".npz"), ("ltfmm/logs", "logs/modal", ".log")):
        for e in vol.listdir(sub):
            name = os.path.basename(e.path)
            if not name.endswith(ext) or ".tmp." in name:
                continue
            with open(f"{dst}/{name}", "wb") as fh:
                for chunk in vol.read_file(e.path):
                    fh.write(chunk)
            print("fetched", f"{dst}/{name}")


if __name__ == "__main__":
    {"fetch": fetch}[sys.argv[1]]()
