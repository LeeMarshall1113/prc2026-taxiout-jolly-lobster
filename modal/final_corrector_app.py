"""Final phase: the serve fits of final/41_corrector_serve.py on rented CPU, one container per seed. NOTHING HERE IS LAUNCHED BY THE SCRIPTS:
the commands below spend money (about 2 to 4 USD for 24 fits, as the January/July serve fits did) and need approval.

Derived from modal/agent_arith_app.py (same image pins, same volume prc-stage1). OWN app name (prc-final-corr) and OWN volume folder
/vol/final: nothing of the other apps' folders is read or written.

    $SYS final/41_corrector_serve.py stage --serve-dir data/final/J9_serve --tag J10         # local, builds data/final/J10_stage
    $SYS modal/final_corrector_app.py upload                                                # the stage directory (about 3.4 GB) -> prc-stage1:/final/stage
    $SYS -m modal run --detach modal/final_corrector_app.py::launch --kind arith --seeds 0-11   # 12 containers, about 16 min each
    $SYS -m modal run --detach modal/final_corrector_app.py::launch --kind pcd   --seeds 0-11   # 12 containers
    $SYS modal/final_corrector_app.py status | fetch                                         # fetch -> data/final/J10_stage/out/<kind>/ (existing files kept)
(--kind v5 with a stage built by `41 stage --prefix agent_bigc12` gives the three fits that feed the tail component.)
Each container needs about 10 GB: memory is set to 20 GB. A container fits one seed after the matrices are built (about 7 min).
"""
import os
import sys

import modal

image = modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.2", "lightgbm==4.7.0", "polars==1.44.1")
for _s in ("agent_arith_14_fit_serve.py", "agent_arith3_02_cols.py", "agent_arith3_05_servecols.py", "agent_bigc_te.py"):
    image = image.add_local_file(f"archive/{_s}", f"/repo/scripts/{_s}")
image = image.add_local_file("final/41_corrector_serve.py", "/repo/final/41_corrector_serve.py")
vol = modal.Volume.from_name("prc-stage1", create_if_missing=True)
app = modal.App("prc-final-corr")
_TAG = os.environ.get("FINAL_STAGE_TAG", "J10")
STAGE_LOCAL = f"data/final/{_TAG}_stage"
STAGE_VOL = f"/vol/final/{_TAG}_stage"
KINDS = {"arith": "arith", "pcd": "pcd", "v5": "v5"}


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=20480, timeout=60 * 60 * 4, retries=0)
def run_seed(kind: str, seeds: str, stage_vol: str = "/vol/final/J10_stage") -> dict:
    """One container fits a group of seeds one after another (the matrices are built once per container)."""
    import subprocess
    import time
    assert kind in KINDS
    vol.reload()
    t0 = time.time()
    env = dict(os.environ, BIGC_THREADS="6", OMP_NUM_THREADS="6", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", "/repo/final/41_corrector_serve.py", kind, "--stage", stage_vol, "--seeds", seeds]
    p = subprocess.Popen(cmd, cwd="/repo", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        tail = (tail + [line.rstrip()])[-40:]
        print(line.rstrip(), flush=True)
    rc = p.wait()
    vol.commit()
    return {"kind": kind, "seeds": seeds, "rc": rc, "seconds": round(time.time() - t0), "tail": tail[-5:]}


@app.local_entrypoint()
def launch(kind: str = "arith", seeds: str = "0-11", per: int = 3):
    lo, _, hi = seeds.partition("-")
    s = list(range(int(lo), int(hi or lo) + 1))
    for k in range(0, len(s), per):
        g = s[k:k + per]
        c = run_seed.spawn(kind, f"{g[0]}-{g[-1]}", STAGE_VOL)
        print("spawned", kind, g, c.object_id)


def upload():
    with vol.batch_upload(force=True) as up:
        for f in sorted(os.listdir(STAGE_LOCAL)):
            p = f"{STAGE_LOCAL}/{f}"
            if os.path.isfile(p):
                up.put_file(p, f"/final/{_TAG}_stage/{f}")
                print("queued", f, round(os.path.getsize(p) / 1e6), "MB")
    print("upload done")


def status():
    for k in KINDS:
        try:
            es = vol.listdir(f"final/{_TAG}_stage/out/{k}")
        except Exception:
            es = []
        print(k, len(es), [os.path.basename(e.path) for e in es])


def fetch():
    for k in KINDS:
        loc = f"{STAGE_LOCAL}/out/{k}"
        os.makedirs(loc, exist_ok=True)
        try:
            es = vol.listdir(f"final/{_TAG}_stage/out/{k}")
        except Exception:
            es = []
        for e in es:
            n = os.path.basename(e.path)
            if n.endswith(".npz") and ".tmp" not in n and not os.path.exists(f"{loc}/{n}"):
                open(f"{loc}/{n}", "wb").write(b"".join(vol.read_file(e.path)))
                print("fetched", k, n)


if __name__ == "__main__":
    {"upload": upload, "status": status, "fetch": fetch}[sys.argv[1]]()
