"""Final phase: the ADS-B stage realisations of final/21_adsb_stage.py on rented CPU. NOTHING HERE IS LAUNCHED BY THE SCRIPTS: the
launch commands spend money (about 10 USD) and need approval.

Derived from agent_bulkr_serve.py (stage part: same image pins, same volume prc-stage1, same per-job container pattern).
OWN app name (prc-final-adsb) and OWN volume folder /vol/final21; the shared 2025 inputs (/vol/data/raw, /vol/data/adsb/points,
/vol/data/external/osm, /vol/results) are only read, and `upload` adds a file there only when it is missing.

What a job is: one realisation (1 to 10) or the Rome-path run (gb) of one stream (103, 105, 105sc) = 33 jobs, each the command
that final/21_adsb_stage.py runs locally (archive/agent_stagecap_apply_taxinm.py through final/_run_as_ranking.py with the flags
of final/21_adsb_stage.flags()). Outputs have the local names: data/final/J5_v<stream>_adsb_<k>.parquet and
J5_v<stream>_gatebag_s1.parquet. The mean over the ten realisations (J5_v<stream>_s1) is taken by final/21_adsb_stage.py locally,
which skips every output that exists.

Commands, in order (Git Bash, repo root; SYS=/c/Python314/python.exe, the Python that has modal):
    1. the local chain has written the pre-rules (data/final/J5_frame.parquet, J5_s0ship.parquet, J5_gatebag_s0.parquet; this
       happens by itself after data/final/J2_v94model.parquet is complete)  [check: ls data/final/J5_s0ship.parquet]
    2. $SYS modal/final_adsb_stage_app.py sizes                          # what upload would send
    3. $SYS modal/final_adsb_stage_app.py upload                         # idempotent
    4. touch data/final/.cloud21                                         # the local chain then only waits and averages
    5. $SYS -m modal run --detach modal/final_adsb_stage_app.py::launch  # 33 containers
    6. $SYS modal/final_adsb_stage_app.py status                         # outputs on the volume
    7. $SYS modal/final_adsb_stage_app.py fetch                          # -> data/final/ (existing files kept)
Dry run of the container entry point, locally, on a small sample (no cloud):
    .venv/Scripts/python.exe final/_adsb_job.py --frame data/final/_t2x/T5_frame.parquet --inp data/final/_t2x/T5_s0ship.parquet
Per container: 4 cores, 12 GiB (the local serving process peaks a little above 3 GB; the shipped Modal stage jobs used 12 GiB).
"""
import os
import sys

import modal

VOL = "prc-stage1"
PINS = ["polars==1.44.1", "numpy==2.5.2", "lightgbm==4.7.0", "scikit-learn==1.9.0", "scipy", "pandas", "pyarrow"]
IGN = ["**/__pycache__/**", "**/*.pyc", "**/*.bak*"]
image = (modal.Image.debian_slim(python_version="3.13").pip_install(*PINS)
         .add_local_dir("prc", remote_path="/repo/prc", ignore=IGN)
         .add_local_dir("archive", remote_path="/repo/scripts", ignore=IGN)
         .add_local_dir("final", remote_path="/repo/final", ignore=IGN))
vol = modal.Volume.from_name(VOL, create_if_missing=True)
app = modal.App("prc-final-adsb")
F21 = "/vol/final21"
CALIB_LOCAL = ["results/preds_n_feat_1_7.parquet", "results/preds_n_feat_2_8.parquet", "results/preds_n3_bulk_3_9.parquet",
               "data/interim/calib2026/preds_oct_f4.parquet", "data/interim/calib2026/preds_nov_c.parquet"]
STREAMS = ["103", "105", "105sc"]
JOBS = [(s, j) for s in STREAMS for j in [str(k) for k in range(1, 11)] + ["gb"]]


sys.path.insert(0, os.environ.get("REPO", "/repo") + "/final")
sys.path.insert(0, "final")
from _adsb_job import build_cmd, env as _env  # noqa: E402


def _links():
    os.chdir("/repo")
    for d in ("data/adsb", "data/external", "data/final", "logs"):
        os.makedirs(d, exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)
    for src, dst in (("/vol/data/raw", "data/raw"), ("/vol/data/adsb/points", "data/adsb/points"),
                     ("/vol/data/external/osm", "data/external/osm"), ("/vol/results", "results"),
                     (F21 + "/calib2026", "data/interim/calib2026"), (F21 + "/calib_days_264_octnov.txt", "data/interim/calib_days_264_octnov.txt")):
        if not os.path.exists(dst):
            os.symlink(src, dst)


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=12288, timeout=60 * 150, retries=0)
def run_job(stream: str, job: str) -> str:
    import shutil
    import subprocess
    import time
    out_name = f"J5_v{stream}_gatebag_s1.parquet" if job == "gb" else f"J5_v{stream}_adsb_{job}.parquet"
    out = f"{F21}/out/{out_name}"
    vol.reload()
    if os.path.exists(out):
        return f"{stream} {job}: already on the volume"
    _links()
    days = open(F21 + "/serve_days.txt").read().strip()
    inp = F21 + ("/J5_gatebag_s0.parquet" if job == "gb" else "/J5_s0ship.parquet")
    tmp = f"/tmp/{out_name}"
    os.makedirs(F21 + "/out", exist_ok=True)
    t0 = time.time()
    with open(f"{F21}/out/{out_name}.log", "w") as fh:
        rc = subprocess.call(build_cmd(stream, job, F21 + "/J5_frame.parquet", inp, tmp, days), stdout=fh, stderr=subprocess.STDOUT, env=_env())
        fh.write(f"\nMODAL rc {rc} wall {time.time() - t0:.0f}s\n")
    if rc == 0:
        shutil.copy(tmp, out + ".tmp")
        os.replace(out + ".tmp", out)
    vol.commit()
    return f"{stream} {job}: rc {rc} wall {time.time() - t0:.0f}s"


@app.local_entrypoint()
def launch():
    for s, j in JOBS:
        print(f"spawned {s} {j}: {run_job.spawn(s, j).object_id}")


# ---------------- local helpers ----------------
def _shared_files():
    """(local, volume path) of inputs that are shared with the other apps' folders and uploaded only when missing."""
    pairs = []
    for f in sorted(os.listdir("data/raw")):
        if f.startswith("training_2025-"):
            pairs.append((f"data/raw/{f}", f"/data/raw/{f}"))
    days = open("data/interim/calib_days_264_octnov.txt").read().strip().split(",")
    pairs += [(f"data/adsb/points/{d}.parquet", f"/data/adsb/points/{d}.parquet") for d in days]
    pairs += [("data/external/osm/stands.parquet", "/data/external/osm/stands.parquet"),
              ("data/external/osm/ways.parquet", "/data/external/osm/ways.parquet")]
    pairs += [(f, "/" + f) for f in CALIB_LOCAL if f.startswith("results/")]
    return pairs


def _own_files():
    pts = sorted(p for p in os.listdir("data/adsb/points") if p[:7] in ("2026-02", "2026-06"))
    pairs = [(f"data/adsb/points/{p}", f"/data/adsb/points/{p}") for p in pts]
    pairs += [("data/final/J5_frame.parquet", "/final21/J5_frame.parquet"), ("data/final/J5_s0ship.parquet", "/final21/J5_s0ship.parquet"),
              ("data/final/J5_gatebag_s0.parquet", "/final21/J5_gatebag_s0.parquet"),
              ("data/interim/calib_days_264_octnov.txt", "/final21/calib_days_264_octnov.txt")]
    pairs += [(f, "/final21/calib2026/" + os.path.basename(f)) for f in CALIB_LOCAL if f.startswith("data/interim/calib2026")]
    return pairs


def _listing(sub):
    try:
        return {e.path.lstrip("/") for e in vol.listdir(sub)}
    except Exception:
        return set()


def sizes():
    have = _listing("data/raw") | _listing("data/adsb/points") | _listing("data/external/osm") | _listing("results")
    sh = [(f, d) for f, d in _shared_files() if os.path.exists(f) and d.lstrip("/") not in have]
    ow = [(f, d) for f, d in _own_files() if os.path.exists(f)]
    miss = [f for f, _ in _shared_files() + _own_files() if not os.path.exists(f)]
    gb = lambda ps: sum(os.path.getsize(f) for f, _ in ps) / 1e9  # noqa: E731
    print(f"shared inputs missing on the volume: {len(sh)} files, {gb(sh):.2f} GB")
    print(f"own inputs (Feb/Jun day files, month frame, pre-rules output, calibration lists): {len(ow)} files, {gb(ow):.2f} GB")
    print("not on this machine yet:", miss)
    return sh, ow


def upload():
    sh, ow = sizes()
    days = sorted({os.path.basename(f)[:-8] for f, _ in ow if f.startswith("data/adsb/points/")})
    with open("/tmp/serve_days.txt" if os.name != "nt" else "serve_days.tmp", "w") as fh:
        fh.write(",".join(days))
    with vol.batch_upload(force=True) as b:
        for f, d in sh + ow:
            b.put_file(f, d)
            print("uploaded", f, "->", d)
        b.put_file("/tmp/serve_days.txt" if os.name != "nt" else "serve_days.tmp", "/final21/serve_days.txt")
    print("upload done")


def status():
    es = [os.path.basename(e.path) for e in vol.listdir("final21/out")] if _listing("final21") else []
    print(len([e for e in es if e.endswith(".parquet")]), "of", len(JOBS), "outputs on the volume")
    for e in sorted(es):
        print("  ", e)


def fetch():
    os.makedirs("data/final", exist_ok=True)
    for e in vol.listdir("final21/out"):
        n = os.path.basename(e.path)
        loc = f"data/final/{n}"
        if n.endswith(".parquet") and not os.path.exists(loc):
            with open(loc + ".part", "wb") as fh:
                for chunk in vol.read_file(e.path):
                    fh.write(chunk)
            os.replace(loc + ".part", loc)
            print("fetched", loc)


if __name__ == "__main__":
    {"sizes": sizes, "upload": upload, "status": status, "fetch": fetch}.get(sys.argv[1], lambda: dryrun(sys.argv[2:]))() \
        if sys.argv[1] != "dryrun" else dryrun(sys.argv[2:])
