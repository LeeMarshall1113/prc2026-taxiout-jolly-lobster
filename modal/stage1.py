"""Run 238-day ADS-B stage-1 evaluation arms on Modal, all at once (Modal budget about $5, 2026-09-23).

WHY. Each arm holds 7 GB for ~35 min locally, so five arms would queue for ~3 h. The arms
are independent CPU jobs; Modal runs them in parallel for ~$0.25 each.

COMPARABILITY RULE. Every verdict pairs a treatment with a control run HERE (never a local npz): thread count and
library build can move a paired delta (CatBoost 8 vs 4 threads moved one 0.9 s). The Q arm doubles as the platform
check against local data/interim/trackfeat_c238_Q.npz.

Inputs are exactly the files archive/trace_inputs.py recorded for these evaluators (logs/trace/all.txt) plus the
June/December days and fold-6 predictions for the JD arm.

    python modal/stage1.py upload                       # push inputs to the prc-stage1 volume (idempotent)
    modal run --detach modal/stage1.py::launch --arms Q,R1,...   # spawn arms; returns at once
    python modal/stage1.py fetch                        # pull out/*.npz + logs into data/interim/modal/
"""
import os
import sys

import modal

VOL = "prc-stage1"
PINS = ["polars==1.44.1", "numpy==2.5.2", "lightgbm==4.7.0", "scikit-learn==1.9.0", "scipy", "pandas", "pyarrow"]
IGN = ["**/__pycache__/**", "**/*.pyc"]
image = (modal.Image.debian_slim(python_version="3.13").pip_install(*PINS)
         .add_local_dir("prc", remote_path="/repo/prc", ignore=IGN)
         .add_local_dir("archive", remote_path="/repo/scripts", ignore=IGN))
vol = modal.Volume.from_name(VOL, create_if_missing=True)
app = modal.App("prc-stage1")

SHIP = ("--stack gbm --stack-feats quality_anch --stack-skip-airports LIRF --floor on --floor-stack on "
        "--floor-stack-snapshot on --floor-q zero --floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG "
        "--osm-fill on --spill-tracks on --blends both_guard --floor-stack-feats quality_anch --floor-stack-anch-skip LIRF")
AIRCS = "--air-cs-all on --air-cs-skip LIRF"
PREDS = ("results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,"
         "data/interim/calib2026/preds_oct_f4.parquet,data/interim/calib2026/preds_nov_c.parquet")
F6 = "results/preds_late2500_treat_f6_6_12.parquet"
TF, RD, NP, LF = ("archive/agent_trackfeat_stage1_eval.py", "archive/agent_riders_stage1_eval.py",
                  "archive/agent_noplan_stage1_eval.py", "archive/agent_lfpgnewstand_stage1_eval.py")
# tag: (evaluator, extra flags, day set)  -- day set "base" = the 264-day calibration list, "jd" = base + Jun/Dec 2025
ARMS = {
    "Q":     (TF, "", "base"),                                   # control for R1 and JD; platform check vs local Q
    "R1":    (RD, "--kind-off LIRF:appear", "base"),
    "JD":    (TF, "", "jd"),                                     # + June/December calibration days, fold-6 preds
    "AIRCS": (NP, AIRCS, "base"),                                # control for NP1 (= v100 matcher)
    "NP1":   (NP, AIRCS + " --noplan-callsign on", "base"),
    "Qa":    (LF, AIRCS, "base"),                                # control for Ta (LFPG evaluator's own copy)
    "Ta":    (LF, AIRCS + " --floor-stack-lfpgnm on", "base"),
}
NP1 = AIRCS + " --noplan-callsign on"   # production matcher as of v103 (board 234.972): the control for new arms


def _days():
    base = open("data/interim/calib_days_264_octnov.txt").read().strip()
    jd = [d for d in open("data/interim/adsb_junedec_days.txt").read().strip().split(",")
          if os.path.exists(f"data/adsb/points/{d}.parquet")]
    return base, ",".join(sorted(set(base.split(",")) | set(jd)))


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=14336, timeout=60 * 100)
def run_arm(tag: str, script: str, flags: str, days: str, rep: str, preds: str) -> str:
    import subprocess
    import time
    os.chdir("/repo")
    for d in ("data", "results"):
        if not os.path.exists(d):
            os.symlink(f"/vol/{d}", d)
    os.makedirs("/vol/out", exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/repo", PYTHONIOENCODING="utf-8",
               POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage",
               **{k: "8" for k in ("OMP_NUM_THREADS", "POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                   "NUMEXPR_MAX_THREADS", "RAYON_NUM_THREADS")})
    cmd = [sys.executable, "-u", "-W", "ignore", script, *SHIP.split(), *flags.split(), "--preds", preds,
           "--days", days, "--report-days", rep, "--save-blends", f"/vol/out/{tag}.npz"]
    t0 = time.time()
    with open(f"/vol/out/{tag}.log", "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        fh.write(f"\nMODAL rc {rc} wall {time.time() - t0:.0f}s\n")
    vol.commit()
    return f"{tag} rc {rc} wall {time.time() - t0:.0f}s"


@app.local_entrypoint()
def launch(arms: str = "Q", smoke: int = 0, spec: str = ""):
    """--spec FILE.json: {"TAG": ["evaluator.py", "extra flags", "base"], ...} adds ad-hoc arms to ARMS."""
    if spec:
        import json
        ARMS.update({k: tuple(v) for k, v in json.load(open(spec)).items()})
    base, jd = _days()
    rep = ",".join(d for d in base.split(",") if not d.startswith("2025-04"))
    for tag in arms.split(","):
        script, flags, dset = ARMS[tag]
        days = jd if dset == "jd" else base
        preds = PREDS + ("," + F6 if dset == "jd" else "")
        r = rep
        if smoke:
            days = r = ",".join(rep.split(",")[:: max(len(rep.split(",")) // smoke, 1)][:smoke])
            tag = f"smoke_{tag}"
        call = run_arm.spawn(tag, script, flags, days, r, preds)
        print(f"spawned {tag}: {call.object_id}  days {len(days.split(','))} report {len(r.split(','))}")


def upload():
    base, jd = _days()
    files = [ln.strip() for ln in open("logs/trace/all.txt") if ln.startswith(("data/", "results/"))
             and "adsb/points" not in ln]
    files += [f"data/adsb/points/{d}.parquet" for d in jd.split(",")] + [F6]
    have = set()
    for sub in ("data/raw", "data/external/osm", "data/interim/calib2026", "results", "data/adsb/points"):
        try:
            have |= {e.path.lstrip("/") for e in vol.listdir(sub)}
        except Exception:
            pass
    todo = [f for f in dict.fromkeys(files) if f not in have]
    size = sum(os.path.getsize(f) for f in todo) / 1e9
    print(f"{len(files)} inputs, {len(todo)} to upload ({size:.2f} GB)")
    with vol.batch_upload(force=True) as b:
        for f in todo:
            b.put_file(f, "/" + f)
    print("upload done")


def fetch():
    os.makedirs("data/interim/modal", exist_ok=True)
    for e in vol.listdir("out"):
        name = os.path.basename(e.path)
        with open(f"data/interim/modal/{name}", "wb") as fh:
            for chunk in vol.read_file(e.path):
                fh.write(chunk)
        print("fetched", name)


if __name__ == "__main__":
    {"upload": upload, "fetch": fetch}[sys.argv[1]]()
