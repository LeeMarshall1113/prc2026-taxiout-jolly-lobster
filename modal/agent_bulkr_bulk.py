"""BULKREFRESH (docs/LAB-NOTES.md) STEP 2 input: the v94-configuration bulk OOF per fold, on Modal.

One container per (fold, member) -- c (CatBoost d8 routed), p3 (CatBoost d10 routed), r (routed LightGBM), e (extra-trees
LightGBM) -- each running archive/agent_bulkr_fold.py --only <member> --ckpt-dir /vol/bulkr/ckpt and committing its
checkpoint on completion. PREEMPTION-SAFE: a restarted container loads a committed checkpoint instead of refitting, and a
finished member is never refitted. A tiny watcher per fold (cpu 0.25) polls the volume and, once all four members are
committed, spawns ONE combine container (no-plan model + v94 blend + regime correction), which writes
/vol/results/preds_bulkr_f<k>_<a>_<b>.parquet (the stage apps read /vol/results directly).

Volumes: prc-data (mounted read-only at /repo/data: raw training files + the feature cache, as app.py) and
prc-stage1 (/vol: checkpoints, logs, results). Threads 8 on Modal (Kaggle folds use 4; different folds, not a paired
comparison -- disclosed).

    python modal/agent_bulkr_bulk.py upload                       # feature cache + hash + raw files -> prc-data if absent
    modal run --detach modal/agent_bulkr_bulk.py::launch --folds 2,4,5          # 4 members x folds + a watcher per fold
    python modal/agent_bulkr_bulk.py fetch                        # /vol/results/preds_bulkr_* + /vol/bulkr/logs -> local
Cost (s0.4 of the note): ~4.1 container-hours per fold at cpu 8 / 24 GiB (~$0.57/h) + combine ~$0.1 -> ~$2.4 per fold.
"""
import os
import sys

import modal

PINS = ["catboost==1.2.10", "polars==1.44.1", "numpy==2.5.2", "pyarrow", "scikit-learn==1.9.0", "lightgbm==4.7.0", "pandas"]
IGN = ["**/__pycache__/**", "**/*.pyc"]
image = (modal.Image.debian_slim(python_version="3.13").pip_install(*PINS)
         .add_local_dir("prc", remote_path="/repo/prc", ignore=IGN)
         .add_local_file("archive/agent_bulkr_fold.py", "/repo/archive/agent_bulkr_fold.py"))
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
vol = modal.Volume.from_name("prc-stage1", create_if_missing=True)
app = modal.App("prc-bulkr-bulk")

BASEDROP = ("airport_plan,arr_slow_180min,arr_slow_60min,arr_sub_day,arr_sub_stand,arr_taxi_180min,arr_taxi_trend,cfg_new,"
            "cfg_shift,clk_aobt_minus_sched,clk_arvt3_minus_aobt,clk_arvt3_minus_arvt1,clk_arvt3_minus_mvt,"
            "clk_eobt_minus_lobt,clk_iobt_minus_lobt,clk_lobt_minus_sched,clk_sec_aobt,clk_sec_mvt,clk_sec_sched,doy,"
            "lobt_minus_aobt,month,mvt_sec_00,prev_rwy_gap,prev_stand_gap,prev_stand_headway,ref_level,ref_taxi_s,"
            "turn_in_delay,turn_rwy_match,turn_sched_slack,wake_heavy_60min,wake_prev_heavy,wake_sep_req")
V94 = ("--seeds 1 --depth 8 --lr 0.08 --l2 3.0 --threads 8 --baseline blend_apt --regime-correction 1.0 --residual "
       "--bulk-plan-only --noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF --bulk-routes ALL "
       "--blend-cat-depth 10 --blend-lightgbm 0.70 --lgb-routes --iterations 2500 --noplan-iterations 600 "
       "--noplan-seeds 20").split()
FOLDS = {"1": (1, 7), "2": (2, 8), "3": (3, 9), "4": (4, 10), "5": (5, 11)}
MEMBERS = ("c", "p3", "r", "e")
FEATHASH = "ef679677ea3afa0a"   # sha256(prc/features.py)[:16] == data/interim/train_features.featurehash (2026-09-27)
CK = "/vol/bulkr/ckpt"


def _env():
    return {**os.environ, "PYTHONPATH": "/repo", "PYTHONIOENCODING": "utf-8",
            "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage",
            **{k: "8" for k in ("OMP_NUM_THREADS", "POLARS_MAX_THREADS")}}


def _cmd(fold, extra):
    return [sys.executable, "-u", "-W", "ignore", "archive/agent_bulkr_fold.py", "--tag", f"bulkr_f{fold}", *V94,
            "--folds", fold, "--drop", BASEDROP, "--ckpt-dir", CK, *extra]


def _final(fold):
    a, b = FOLDS[fold]
    return f"/vol/results/preds_bulkr_f{fold}_{a}_{b}.parquet"


def _ck(fold, m):
    a, b = FOLDS[fold]
    return f"{CK}/bulkr_f{fold}_{a}_{b}_{m}.npz"


def _run(cmd, log):
    import subprocess
    import time
    os.chdir("/repo")
    os.makedirs("/repo/results", exist_ok=True)
    os.makedirs("/vol/bulkr/logs", exist_ok=True)
    h = open("/repo/data/interim/train_features.featurehash").read().strip()
    assert h == FEATHASH, f"prc-data feature cache hash {h} != {FEATHASH}: run `python modal/agent_bulkr_bulk.py upload`"
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=_env())
        fh.write(f"\nMODAL rc {rc} wall {time.time() - t0:.0f}s\n")
    return rc, time.time() - t0


@app.function(image=image, volumes={"/repo/data": data_vol, "/vol": vol}, cpu=8.0, memory=24576, timeout=60 * 60 * 4)
def run_member(fold: str, member: str) -> str:
    vol.reload()
    if os.path.exists(_ck(fold, member)):
        return f"f{fold} {member}: already committed"
    rc, wall = _run(_cmd(fold, ["--only", member]), f"/vol/bulkr/logs/bulkr_f{fold}_{member}.log")
    vol.commit()
    return f"f{fold} {member}: rc {rc} wall {wall:.0f}s"


@app.function(image=image, volumes={"/repo/data": data_vol, "/vol": vol}, cpu=4.0, memory=24576, timeout=60 * 90)
def run_combine(fold: str) -> str:
    import shutil
    vol.reload()
    if os.path.exists(_final(fold)):
        return f"f{fold} combine: already committed"
    missing = [m for m in MEMBERS if not os.path.exists(_ck(fold, m))]
    assert not missing, f"f{fold}: members missing {missing}"
    rc, wall = _run(_cmd(fold, ["--skip-members", "l"]), f"/vol/bulkr/logs/bulkr_f{fold}_combine.log")
    if rc == 0:
        a, b = FOLDS[fold]
        os.makedirs("/vol/results", exist_ok=True)
        shutil.copy(f"/repo/results/preds_bulkr_f{fold}_{a}_{b}.parquet", _final(fold) + ".tmp")
        os.replace(_final(fold) + ".tmp", _final(fold))
    vol.commit()
    return f"f{fold} combine: rc {rc} wall {wall:.0f}s"


@app.function(image=image, volumes={"/vol": vol}, cpu=0.25, memory=512, timeout=60 * 60 * 6)
def watch(fold: str) -> str:
    """Polls every 2 min; spawns the combine once all four members are committed (a marker stops double spawns)."""
    import time
    marker = f"/vol/bulkr/combine_f{fold}.spawned"
    for _ in range(180):
        vol.reload()
        if os.path.exists(_final(fold)):
            return f"f{fold}: final present"
        if all(os.path.exists(_ck(fold, m)) for m in MEMBERS):
            if not os.path.exists(marker) or time.time() - os.path.getmtime(marker) > 2 * 3600:
                os.makedirs(os.path.dirname(marker), exist_ok=True)
                open(marker, "w").write(str(time.time()))
                vol.commit()
                call = run_combine.spawn(fold)
                return f"f{fold}: combine spawned {call.object_id}"
        time.sleep(120)
    return f"f{fold}: watcher gave up after 6 h"


@app.local_entrypoint()
def launch(folds: str = "2,4,5", members: str = "c,p3,r,e"):
    for f in folds.split(","):
        for m in members.split(","):
            call = run_member.spawn(f, m)
            print(f"spawned f{f} {m}: {call.object_id}")
        w = watch.spawn(f)
        print(f"spawned watcher f{f}: {w.object_id}")


@app.local_entrypoint()
def combine(folds: str = "3,4,5"):
    """Fallback when a watcher was preempted without restart: spawn the combine for every listed fold whose four member
    checkpoints are committed and whose final is missing (run_combine is itself idempotent). Client-side listdir
    (reads the committed state; vol.reload() only works inside a remote function)."""
    have = {e.path.split("/")[-1] for e in vol.listdir("bulkr/ckpt")}
    fins = {e.path.split("/")[-1] for e in vol.listdir("results")}
    for f in folds.split(","):
        a, b = FOLDS[f]
        miss = [m for m in MEMBERS if f"bulkr_f{f}_{a}_{b}_{m}.npz" not in have]
        if os.path.basename(_final(f)) in fins:
            print(f"f{f}: final present")
        elif miss:
            print(f"f{f}: members missing {miss} -- not spawning")
        else:
            print(f"f{f}: combine spawned {run_combine.spawn(f).object_id}")


def upload():
    """Idempotent: the feature cache + hash and the raw training files, into prc-data (the layout app.py uses)."""
    want = ["interim/train_features.parquet", "interim/train_features.featurehash"] + \
        sorted(f"raw/{p}" for p in os.listdir("data/raw") if p.endswith(".parquet")) + \
        sorted(f"raw/weather/{p}" for p in os.listdir("data/raw/weather"))
    have_hash = None
    try:
        have_hash = b"".join(data_vol.read_file("interim/train_features.featurehash")).decode().strip()
    except Exception:
        pass
    have = set()
    for sub in ("raw", "raw/weather", "interim"):
        try:
            have |= {e.path.lstrip("/") for e in data_vol.listdir(sub)}
        except Exception:
            pass
    todo = [f for f in want if f not in have or (f.startswith("interim/train_features") and have_hash != FEATHASH)]
    print(f"prc-data cache hash {have_hash} (want {FEATHASH}); {len(todo)} files to upload "
          f"({sum(os.path.getsize('data/' + f) for f in todo) / 1e9:.2f} GB)")
    with data_vol.batch_upload(force=True) as b:
        for f in todo:
            b.put_file("data/" + f, "/" + f)
    print("upload done")


def upload_results(paths):
    """Kaggle-made fold files (f1, f3) -> /vol/results so the stage apps see every fold in one place."""
    with vol.batch_upload(force=True) as b:
        for p in paths:
            b.put_file(p, "/results/" + os.path.basename(p))
            print("uploaded", p)


def fetch():
    os.makedirs("logs/modal", exist_ok=True)
    for sub, dst in (("results", "results"), ("bulkr/logs", "logs/modal")):
        for e in vol.listdir(sub):
            name = os.path.basename(e.path)
            if not name.startswith(("preds_bulkr", "bulkr_")):
                continue
            with open(f"{dst}/{name}", "wb") as fh:
                for chunk in vol.read_file(e.path):
                    fh.write(chunk)
            print("fetched", f"{dst}/{name}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "upload_results":
        upload_results(sys.argv[2:])
    else:
        {"upload": upload, "fetch": fetch}[cmd]()
