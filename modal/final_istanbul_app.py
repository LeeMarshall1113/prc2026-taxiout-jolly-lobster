"""Final phase, Istanbul arms on rented CPU (route B). NOTHING HERE IS LAUNCHED BY THE SCRIPTS; the operator launches.

Three arms, each one container, run in parallel, same platform and same --threads 4 as the shipped Kaggle runs (the delta is a
PAIRED difference of CatBoost fits, so all arms must share platform and thread count):
    ctrl     final/ist_kg_ltfm_final.py --ltfm-seq off  --seeds 3          shipped Kaggle time 4.2 h
    snow_k0  final/ist_ltreg_final.py   --ltfm-seq on   --seeds 1          2.25 h
    snow_k12 final/ist_ltreg_final.py   --ltfm-seq on   --seeds 2 --seed-offset 1   3.1 h
(the sequence-feature "treat" arm of the shipped chain is NOT run: the shipped steps telescope, see final/43_ist_build.py.)
Volume prc-data (mounted at /repo/data) already holds data/raw/training_2025-*, weather and the training-feature cache whose
features.py hash (ef679677ea3afa0a) equals the current prc/features.py. Only the two final_* files are added, under
raw/ (the shipped ranking.parquet is not touched).

    $SYS modal/final_istanbul_app.py upload            # final_ranking.parquet (86 MB) + final_submitting.parquet -> prc-data:/raw/
    $SYS -m modal run --detach modal/final_istanbul_app.py::launch                  # the three arms
    $SYS modal/final_istanbul_app.py status | fetch     # fetch -> data/final/ist/preds_<arm>.parquet
Estimate: 5 CPU cores and 16 GB per container (peak 11-12 GB in the shipped runs), wall about 4.2 h (the control arm), 9.6
container-hours in all, about 3.5 USD (0.047 USD per core-hour + 0.008 per GB-hour; not measured). Timeout 6 h per container.
"""
import os
import sys

import modal

PINS = ["catboost==1.2.10", "polars==1.44.1", "numpy==2.5.2", "pyarrow", "scikit-learn", "lightgbm"]
IGN = ["**/__pycache__/**", "**/*.pyc"]
image = (modal.Image.debian_slim(python_version="3.13").pip_install(*PINS)
         .add_local_dir("prc", remote_path="/repo/prc", ignore=IGN)
         .add_local_file("final/10_base.py", "/repo/final/10_base.py")
         .add_local_file("final/ist_kg_ltfm_final.py", "/repo/final/ist_kg_ltfm_final.py")
         .add_local_file("final/ist_ltreg_final.py", "/repo/final/ist_ltreg_final.py")
         .add_local_file("archive/agent_ltfm_seqfeat.py", "/repo/archive/agent_ltfm_seqfeat.py")
         .add_local_file("archive/agent_ltreg_feat.py", "/repo/archive/agent_ltreg_feat.py")
         .add_local_file("archive/agent_bulkr_fold.py", "/repo/archive/agent_bulkr_fold.py")
         .add_local_file("final/ist_lv_members.py", "/repo/final/ist_lv_members.py"))
vol = modal.Volume.from_name("prc-data", create_if_missing=True)
app = modal.App("prc-final-ist")

BASEDROP = ("airport_plan,arr_slow_180min,arr_slow_60min,arr_sub_day,arr_sub_stand,arr_taxi_180min,arr_taxi_trend,cfg_new,"
            "cfg_shift,clk_aobt_minus_sched,clk_arvt3_minus_aobt,clk_arvt3_minus_arvt1,clk_arvt3_minus_mvt,"
            "clk_eobt_minus_lobt,clk_iobt_minus_lobt,clk_lobt_minus_sched,clk_sec_aobt,clk_sec_mvt,clk_sec_sched,doy,"
            "lobt_minus_aobt,month,mvt_sec_00,prev_rwy_gap,prev_stand_gap,prev_stand_headway,ref_level,ref_taxi_s,"
            "turn_in_delay,turn_rwy_match,turn_sched_slack,wake_heavy_60min,wake_prev_heavy,wake_sep_req")
# FINAL_COMMON of archive/agent_kg_ltfm_kaggle.py without --seeds
COMMON = ("--iterations 2500 --depth 8 --lr 0.08 --l2 3.0 --threads 4 --baseline blend_apt --noplan-iterations 300 --noplan-seeds 5 "
          "--regime-correction 1.0 --residual --bulk-plan-only --noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF").split()
ARMS = {  # tag: (script, ltfm-seq, extra wrapper/base flags)
    "ctrl": ("final/ist_kg_ltfm_final.py", "off", ["--seeds", "3"]),
    "snow_k0": ("final/ist_ltreg_final.py", "on", ["--seeds", "1"]),
    "snow_k12": ("final/ist_ltreg_final.py", "on", ["--seeds", "2"]),   # + --seed-offset 1 (wrapper flag, placed first below)
}
OUTD = "/repo/data/ist_out"


@app.function(image=image, volumes={"/repo/data": vol}, cpu=5.0, memory=16384, timeout=60 * 60 * 6)
def run_arm(tag: str, months: str = "2,6") -> dict:
    import subprocess
    import time
    script, seq, extra = ARMS[tag]
    os.chdir("/repo")
    os.makedirs(OUTD, exist_ok=True)
    pre = ["--seed-offset", "1"] if tag == "snow_k12" else []
    env = {**os.environ, "PYTHONPATH": "/repo", "PYTHONIOENCODING": "utf-8", "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage",
           "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4", "FINAL_RANKING": "/repo/data/raw/final_ranking.parquet", "FINAL_MONTHS": months}
    cmd = [sys.executable, "-u", script, "--ltfm-seq", seq, *pre, "--out", f"{OUTD}/preds_{tag}.parquet", "--basedrop", BASEDROP, *COMMON, *extra,
           "--ranking", "/repo/data/raw/final_ranking.parquet", "--template", "/repo/data/raw/final_submitting.parquet", "--months", months,
           "--out-dir", OUTD, "--tag", f"ist_{tag}"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    open(f"{OUTD}/{tag}.log", "w").write((p.stdout or "") + (p.stderr or ""))
    vol.commit()
    return {"tag": tag, "rc": p.returncode, "seconds": round(time.time() - t0), "tail": ((p.stdout or "")[-1500:] + (p.stderr or "")[-1500:])}


@app.local_entrypoint()
def launch(arms: str = "ctrl,snow_k0,snow_k12", months: str = "2,6"):
    for a in arms.split(","):
        c = run_arm.spawn(a, months)
        print("spawned", a, c.object_id)


V94 = ("--seeds 1 --depth 8 --lr 0.08 --l2 3.0 --threads 8 --baseline blend_apt --regime-correction 1.0 --residual "
       "--bulk-plan-only --noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF --bulk-routes ALL "
       "--blend-cat-depth 10 --blend-lightgbm 0.70 --lgb-routes --iterations 2500 --noplan-iterations 600 "
       "--noplan-seeds 20").split()   # as modal/agent_lv_app.py


@app.function(image=image, volumes={"/repo/data": vol}, cpu=8.0, memory=24576, timeout=60 * 60 * 3)
def run_lv(tag: str, seq: str, quant: str, months: str = "2,6") -> dict:
    """Views members, SERVE twin (all 2025 Istanbul rows -> the served Istanbul rows): modal/agent_lv_app.py run_arm(mode=serve)."""
    import subprocess
    import time
    os.chdir("/repo")
    ck = "/repo/data/ist_out/lv"
    os.makedirs(ck, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "/repo", "PYTHONIOENCODING": "utf-8", "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage",
           "OMP_NUM_THREADS": "8", "POLARS_MAX_THREADS": "8", "FINAL_RANKING": "/repo/data/raw/final_ranking.parquet", "FINAL_MONTHS": months}
    cmd = [sys.executable, "-u", "-W", "ignore", "final/ist_lv_members.py", "--mode", "serve", "--tag", tag, "--ltfm-seq", seq,
           *(["--quant", quant] if quant else []), "--only", "c,p3,r,e", "--ckpt-dir", ck, *V94, "--folds", "1", "--drop", BASEDROP]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    open(f"{ck}/{tag}.log", "w").write((p.stdout or "") + (p.stderr or ""))
    vol.commit()
    return {"tag": tag, "rc": p.returncode, "seconds": round(time.time() - t0), "tail": (p.stdout or "")[-1500:] + (p.stderr or "")[-1500:]}


@app.local_entrypoint()
def launch_lv(months: str = "2,6"):
    """The two serve arms of the views members (about 1 USD). Run after `upload`; independent of the three bulk arms."""
    for tag, seq, q in (("ltfmm_ctrl_srv", "off", "q05,q10,q25,q50,q75,q90,q95,cbS,cbD,dart,xgb,l1,cs1,cs2,cs3,cs4,nowx,nosch"), ("ltfmm_treat_srv", "on", "")):
        c = run_lv.spawn(tag, seq, q, months)
        print("spawned", tag, c.object_id)


def upload():
    with vol.batch_upload(force=True) as up:
        for n in ("final_ranking.parquet", "final_submitting.parquet"):
            up.put_file(f"data/raw/{n}", f"/raw/{n}")
            print("queued", n)
    print("upload done")


def status():
    try:
        for e in vol.listdir("ist_out"):
            print(e.path)
    except Exception as ex:
        print("nothing yet", ex)


def fetch():
    for sub, dst in (("ist_out", "data/final/ist"), ("ist_out/lv", "data/final/ist/lv")):   # lv checkpoints: FINAL_LV_DIR=data/final/ist/lv
        os.makedirs(dst, exist_ok=True)
        try:
            es = vol.listdir(sub)
        except Exception:
            es = []
        for e in es:
            n = os.path.basename(e.path)
            if n.endswith((".parquet", ".npz", ".log")) and not os.path.exists(f"{dst}/{n}"):
                open(f"{dst}/{n}", "wb").write(b"".join(vol.read_file(e.path)))
                print("fetched", n)


if __name__ == "__main__":
    {"upload": upload, "status": status, "fetch": fetch}[sys.argv[1]]()
