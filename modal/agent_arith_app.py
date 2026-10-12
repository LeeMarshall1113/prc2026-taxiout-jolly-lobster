"""agent_arith: run the LOCAL corrector fit scripts (archive/agent_arith_02_fit.py and archive/agent_arith3_02_fit.py) unchanged in the cloud,
one container per arm, so the held-out fits do not wait on the shared machine. Same library versions as the local environment.
OWN app name (prc-arith) and OWN volume folders (/vol/arith/in, /vol/arith/out_arith, /vol/arith/out_arith3): nothing else on the volume is written.
    $SYS modal/agent_arith_app.py upload                                   # the fit inputs (about 3.4 GB) -> prc-stage1:/arith/in
    $SYS -m modal run --detach modal/agent_arith_app.py::launch --script agent_arith_02_fit.py --jobs-file archive/agent_arith_jobs20.txt
    $SYS -m modal run --detach modal/agent_arith_app.py::launch --script agent_arith3_02_fit.py --jobs-file archive/agent_arith3_jobs11.txt
    $SYS modal/agent_arith_app.py status | fetch                           # fetch -> data/interim/agent_arith_out/ and data/interim/agent_arith3_out/ (existing files are kept)
A job string is ARM:mode:month:seed (as the local runner takes them, comma separated); jobs are grouped by ARM, one container each.
"""
import os
import sys

import modal

SCRIPTS = ("agent_arith_02_fit.py", "agent_arith3_02_fit.py", "agent_arith3_02_cols.py", "agent_bigc_te.py", "agent_arith3_04_fit_temp.py", "agent_arith4_cap_fit.py", "agent_arith_14_fit_serve.py", "agent_arith3_05_servecols.py", "agent_arith3_06_fit_serve.py", "agent_arith4_X_cols.py", "agent_arith4_X_fit.py")
image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.2", "lightgbm==4.7.0", "polars==1.44.1")
for _s in SCRIPTS:
    image = image.add_local_file(f"archive/{_s}", f"/repo/scripts/{_s}")
vol = modal.Volume.from_name("prc-stage1", create_if_missing=True)
app = modal.App("prc-arith")
INPUTS = ("agent_yearfp_names.json", "agent_bigc13_fold.npz", "agent_bigc13_keys2.npz", "agent_bigc12_keys.npz", "agent_bigc12_vfac_clean5.npz",
          "agent_bigc12_extra.npz", "agent_bigc12_extra2.npz", "agent_pcd_cols.npz", "agent_aobtflag_flags.parquet", "agent_arith3_time.npz")
OUTDIR = {"agent_arith_02_fit.py": "out_arith", "agent_arith3_02_fit.py": "out_arith3", "agent_arith3_04_fit_temp.py": "out_arith3", "agent_arith4_cap_fit.py": "out_arith4", "agent_arith3_06_fit_serve.py": "out_serve3", "agent_arith4_X_fit.py": "out_arith4"}
LOCAL = {"out_arith": "data/interim/agent_arith_out", "out_arith3": "data/interim/agent_arith3_out", "out_arith4": "data/interim/agent_arith4_out", "out_serve3": "data/interim/agent_arith3_serve"}


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=20480, timeout=60 * 60 * 6, retries=0)
def run_jobs(script: str, jobs: str, sub: str = "", var: str = "") -> dict:
    """runs `python scripts/<script> <jobs>` with data/interim pointing at /vol/arith/in and the output folder on the volume."""
    import subprocess
    import time
    assert script in OUTDIR
    vol.reload()
    t0 = time.time(); out = f"/vol/arith/{OUTDIR[script]}{("_" + sub) if sub else ""}/"
    os.makedirs("/work/data", exist_ok=True); os.makedirs(out, exist_ok=True)
    if not os.path.exists("/work/data/interim"):
        os.symlink("/vol/arith/in", "/work/data/interim")
    if not os.path.exists("/work/scripts"):
        os.symlink("/repo/scripts", "/work/scripts")
    env = dict(os.environ, ARITH_I="/vol/arith/in/", ARITH_OUT=out, BIGC_THREADS="6", OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8")
    if var:
        env["ARITH4_VAR"] = var
    p = subprocess.Popen([sys.executable, "-u", f"archive/{script}", jobs], cwd="/work", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []; last = time.time()
    for line in p.stdout:
        tail.append(line.rstrip()); tail = tail[-40:]; print(line.rstrip(), flush=True)
        if time.time() - last > 240:   # keep finished fits if the container is lost
            vol.commit(); last = time.time()
    rc = p.wait(); vol.commit()
    return {"script": script, "jobs": jobs[:80], "rc": rc, "seconds": round(time.time() - t0), "tail": tail[-6:]}


@app.local_entrypoint()
def launch(script: str, jobs_file: str = "", jobs: str = "", per_container: int = 0, sub: str = "", var: str = ""):
    js = (open(jobs_file).read().strip() if jobs_file else jobs).split(",")
    groups = {}
    for j in js:
        groups.setdefault(j.split(":")[0], []).append(j)
    for arm, g in groups.items():
        chunks = [g] if not per_container else [g[i:i + per_container] for i in range(0, len(g), per_container)]
        for ch in chunks:
            c = run_jobs.spawn(script, ",".join(ch), sub, var); print("spawned", arm, len(ch), "fits", c.object_id)


def upload():
    with vol.batch_upload(force=True) as up:
        for n in INPUTS:
            p = f"data/interim/{n}"
            up.put_file(p, f"/arith/in/{n}"); print("queued", n, round(os.path.getsize(p) / 1e6), "MB")
    print("upload done")


def status():
    for d in ("arith/in", "arith/out_arith", "arith/out_arith3"):
        try:
            es = vol.listdir(d)
        except Exception:
            es = []
        print(d, len(es))
        for e in es:
            print("  ", os.path.basename(e.path))


def fetch():
    for d, loc in LOCAL.items():
        os.makedirs(loc, exist_ok=True)
        try:
            es = vol.listdir(f"arith/{d}")
        except Exception:
            es = []
        for e in es:
            n = os.path.basename(e.path)
            if not n.endswith(".npz") or ".tmp" in n or os.path.exists(f"{loc}/{n}"):
                continue
            open(f"{loc}/{n}", "wb").write(b"".join(vol.read_file(e.path))); print("fetched", d, n)


if __name__ == "__main__":
    {"upload": upload, "status": status, "fetch": fetch}[sys.argv[1]]()
