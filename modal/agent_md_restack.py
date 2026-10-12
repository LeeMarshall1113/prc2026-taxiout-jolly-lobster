"""MOREDAYS STEP 2 re-stack arms on Modal (docs/LAB-NOTES.md). Same pattern as modal/stage1.py
(image pins, volume prc-stage1, detached `modal run --detach` on a local_entrypoint that .spawn()s every container).
Preemption-safe: each container is ONE fit, commits its npz as soon as it exists, and a restarted container
(Modal re-runs preempted inputs) returns at once if that npz is already on the volume. One container per (arm, seed, piece): arms A / C, seeds 0-2, pieces bxTX / bxE9f (archive/agent_md_bxnest2.py) and f4TX
(archive/agent_md_fold4b.py) = 18 containers, cpu 4 (the fits use n_jobs=4), 12 GiB. The same argument lists as the
Kaggle driver archive/agent_md_kg2.py.

    python modal/agent_md_restack.py upload          # inputs -> /vol/md2/data/interim/... (idempotent by name)
    modal run --detach modal/agent_md_restack.py::launch --arms A,C --seeds 0,1,2
    python modal/agent_md_restack.py fetch           # /vol/out/md2_* -> data/interim/agent_md_kg2_<ARM><seed>.npz
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
app = modal.App("prc-md-restack")
I = "data/interim/"
FILES = [I + f for f in (
    "modal/TX_S6.npz", "agent_stagecap_CAND_E9f.npz", "agent_offblock_rows_c238.parquet", "agent_restack3_evt_2025.parquet",
    "agent_restack3_trk_2025.parquet", "agent_smallbundle_seq_2025.parquet", "agent_backx_bx_2025.parquet",
    "agent_backx_ff_2025.parquet", "agent_restack5_bxnest_cols.npz", "agent_md_rows_new.parquet", "agent_md_evt_new.parquet",
    "agent_md_trk_new.parquet", "agent_md_seq_new.parquet", "agent_md_ff_new.parquet", "agent_md_TXS6_354.npz",
    "agent_md_E9f_354.npz", "agent_md_TXS6_354_238.npz", "agent_md_E9f_354_238.npz", "agent_sb2_osm_2025.parquet",
    "agent_md_osm_new.parquet")]


def fits(arm):
    tx, e9 = I + "modal/TX_S6.npz", I + "agent_stagecap_CAND_E9f.npz"
    if arm == "C":
        tx, e9 = I + "agent_md_TXS6_354_238.npz", I + "agent_md_E9f_354_238.npz"
    t354, e354 = I + "agent_md_TXS6_354.npz", I + "agent_md_E9f_354.npz"
    d = {"bxTX": ["archive/agent_md_bxnest2.py", "--recipe", "cap,evt,trk,bx,big", "--fin-npz", tx, "--new-fin-npz", t354],
            "bxE9f": ["archive/agent_md_bxnest2.py", "--recipe", "cap,evt,trk,bx,big", "--fin-npz", e9, "--new-fin-npz", e354],
            "f4TX": ["archive/agent_md_fold4b.py", "--recipe", "cap,evt,trk,seq,big", "--fin-npz", tx, "--new-fin-npz", t354]}
    if arm == "D":   # moredays s4.5: arm A + the osm token
        return {k: v + ["--osm"] for k, v in d.items()}
    return d


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=12288, timeout=60 * 55)
def run_fit(arm: str, seed: int, piece: str) -> str:
    import subprocess
    import time

    import numpy as np
    os.chdir("/repo")
    if not os.path.exists("data"):
        os.symlink("/vol/md2/data", "data")
    os.makedirs("/vol/out", exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/repo", PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage",
               OMP_NUM_THREADS="4", POLARS_MAX_THREADS="4")
    tag = f"md2_{arm}{seed}_{piece}"
    tmp = f"/tmp/{tag}.npz"
    vol.reload()
    if os.path.exists(f"/vol/out/{tag}.npz"):   # preemption-safe: a restarted container skips a fit already committed
        return f"{tag} already on the volume"
    cmd = [sys.executable, "-u", "-W", "ignore", *fits(arm)[piece], "--seed", str(seed), "--out", tmp]
    t0 = time.time()
    with open(f"/vol/out/{tag}.log", "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        fh.write(f"\nMODAL rc {rc} wall {time.time() - t0:.0f}s\n")
    if rc == 0:
        z = np.load(tmp, allow_pickle=True)
        np.savez(f"/vol/out/{tag}.npz", ids=z["ids"], both_guard=z["both_guard"])
    vol.commit()
    return f"{tag} rc {rc} wall {time.time() - t0:.0f}s"


@app.local_entrypoint()
def launch(arms: str = "A,C", seeds: str = "0,1,2", pieces: str = "bxTX,bxE9f,f4TX"):
    for arm in arms.split(","):
        for s in seeds.split(","):
            for piece in pieces.split(","):
                call = run_fit.spawn(arm, int(s), piece)
                print(f"spawned {arm}{s} {piece}: {call.object_id}")


def upload():
    have = set()
    for sub in ("md2/data/interim", "md2/data/interim/modal"):
        try:
            have |= {e.path.lstrip("/") for e in vol.listdir(sub)}
        except Exception:
            pass
    todo = [f for f in FILES if "md2/" + f not in have]
    miss = [f for f in todo if not os.path.exists(f)]
    if miss:   # the S1-stage (E9f) fins may land later; upload again when they exist (idempotent)
        print(f"NOT YET LOCAL (skipped; re-run upload later): {miss}")
        todo = [f for f in todo if f not in miss]
    print(f"{len(FILES)} inputs, {len(todo)} to upload ({sum(os.path.getsize(f) for f in todo) / 1e9:.2f} GB)")
    with vol.batch_upload(force=True) as b:
        for f in todo:
            b.put_file(f, "/md2/" + f)
    print("upload done")


def fetch():
    got = {}
    for e in vol.listdir("out"):
        name = os.path.basename(e.path)
        if not name.startswith("md2_"):
            continue
        local = f"data/interim/modal/{name}"
        with open(local, "wb") as fh:
            for chunk in vol.read_file(e.path):
                fh.write(chunk)
        if name.endswith(".npz"):
            key, piece = name[4:-4].split("_", 1)
            got.setdefault(key, {})[piece] = local
    try:
        import numpy as np
    except ImportError:   # system python (the modal client) has no numpy: assemble with `.venv python modal/agent_md_restack.py assemble`
        print("downloaded; run the assemble step with the project venv")
        return
    assemble()


def assemble():
    import glob
    import numpy as np
    got = {}
    for local in glob.glob("data/interim/modal/md2_*.npz"):
        name = os.path.basename(local)
        key, piece = name[4:-4].split("_", 1)
        got.setdefault(key, {})[piece] = local
    for key, d in sorted(got.items()):
        if set(d) == {"bxTX", "bxE9f", "f4TX"}:
            zs = {p: np.load(f) for p, f in d.items()}
            ids = zs["bxTX"]["ids"]
            assert all((z["ids"] == ids).all() for z in zs.values())
            np.savez(f"data/interim/agent_md_kg2_{key}.npz", ids=ids, **{p: z["both_guard"] for p, z in zs.items()})
            print(f"assembled data/interim/agent_md_kg2_{key}.npz")
        else:
            print(f"{key}: have {sorted(d)} only")


if __name__ == "__main__":
    {"upload": upload, "fetch": fetch, "assemble": assemble}[sys.argv[1]]()
