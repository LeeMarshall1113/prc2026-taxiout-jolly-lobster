"""BULKREFRESH STEP 2 re-stacks on Modal (docs/LAB-NOTES.md). COPY of modal/agent_md_restack.py's pattern
(image pins, volume prc-stage1, detached .spawn(), ONE fit per container, commit on completion, a restarted container
returns at once if its npz is committed = preemption-safe).
Chains x seeds 0-2 x pieces = 18 containers (cpu 4, 12 GiB, 55 min timeout), all with the shipped osm token (recipes + osm):
  NEW: archive/agent_sb2_bxnest.py / agent_sb2_fold4.py (UNEDITED, the production osm path) on agent_bulkr_{TXS6,E9f}_T.npz
  PROD: the same on the production fins, seed 0 = Modal reproduction gate vs data/interim/modal_sb2/agent_sb2_*_osm_s0.npz
  SA : archive/agent_bulkr_sb2bxnest.py / agent_bulkr_sb2fold4.py trained on agent_bulkr_{TXS6,E9f}_SAold.npz and APPLIED to
       agent_bulkr_{TXS6,E9f}_SA.npz (--apply-fin-npz) = today's production (old-trained chain fed the served p0)
Pieces: bxTX (recipe cap,evt,trk,bx,big on TXS6; EGLL/LEMD), bxE9f (same on E9f; M), f4TX (cap,evt,trk,seq,big on TXS6; LFPG/LIRF).

    python modal/agent_bulkr_restack.py upload
    modal run --detach modal/agent_bulkr_restack.py::launch --chains NEW,SA --seeds 0,1,2
    python modal/agent_bulkr_restack.py fetch       # -> data/interim/modal/bulkr2_<CHAIN><seed>_<piece>.npz
Cost: ~22 min per fit (moredays precedent) at ~$0.25/h -> 18 x ~$0.1 = ~$1.7; worst case at the timeouts ~$4.
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
app = modal.App("prc-bulkr-restack")
I = "data/interim/"
FILES = [I + f for f in (
    "agent_offblock_rows_c238.parquet", "agent_restack3_evt_2025.parquet", "agent_restack3_trk_2025.parquet",
    "agent_smallbundle_seq_2025.parquet", "agent_backx_bx_2025.parquet", "agent_backx_ff_2025.parquet",
    "agent_restack5_bxnest_cols.npz", "agent_sb2_osm_2025.parquet", "agent_neighbour_nbr_2025.parquet",
    "agent_restack4_arr_2025.parquet", "agent_sb2_occ_2025.parquet", "modal/TX_S6.npz", "agent_stagecap_CAND_E9f.npz",
    *[f"agent_bulkr_{n}_{x}.npz" for n in ("TXS6", "E9f") for x in ("T", "SA", "SAold")])]


def fits(chain):
    """Production osm code path (smallbundle2 C6 = agent_sb2_*_osm fits): recipes + osm, as modal/agent_sb2_app.fit_spec."""
    bx, sq = "cap,evt,trk,bx,big,osm", "cap,evt,trk,seq,big,osm"
    if chain == "NEW":
        tx, e9 = I + "agent_bulkr_TXS6_T.npz", I + "agent_bulkr_E9f_T.npz"
        return {"bxTX": ["archive/agent_sb2_bxnest.py", "--recipe", bx, "--fin-npz", tx],
                "bxE9f": ["archive/agent_sb2_bxnest.py", "--recipe", bx, "--fin-npz", e9],
                "f4TX": ["archive/agent_sb2_fold4.py", "--recipe", sq, "--fin-npz", tx]}
    if chain == "PROD":   # the unedited sb2 scripts on the PRODUCTION fins = must reproduce agent_sb2_*_osm_s<r> (Modal gate)
        tx, e9 = I + "modal/TX_S6.npz", I + "agent_stagecap_CAND_E9f.npz"
        return {"bxTX": ["archive/agent_sb2_bxnest.py", "--recipe", bx, "--fin-npz", tx],
                "bxE9f": ["archive/agent_sb2_bxnest.py", "--recipe", bx, "--fin-npz", e9],
                "f4TX": ["archive/agent_sb2_fold4.py", "--recipe", sq, "--fin-npz", tx]}
    tx, e9 = I + "agent_bulkr_TXS6_SAold.npz", I + "agent_bulkr_E9f_SAold.npz"
    txa, e9a = I + "agent_bulkr_TXS6_SA.npz", I + "agent_bulkr_E9f_SA.npz"
    return {"bxTX": ["archive/agent_bulkr_sb2bxnest.py", "--recipe", bx, "--fin-npz", tx, "--apply-fin-npz", txa],
            "bxE9f": ["archive/agent_bulkr_sb2bxnest.py", "--recipe", bx, "--fin-npz", e9, "--apply-fin-npz", e9a],
            "f4TX": ["archive/agent_bulkr_sb2fold4.py", "--recipe", sq, "--fin-npz", tx, "--apply-fin-npz", txa]}


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=12288, timeout=60 * 55)
def run_fit(chain: str, seed: int, piece: str) -> str:
    import subprocess
    import time

    import numpy as np
    os.chdir("/repo")
    if not os.path.exists("data"):
        os.symlink("/vol/bulkr2/data", "data")
    os.makedirs("/vol/out", exist_ok=True)
    env = dict(os.environ, PYTHONPATH="/repo", PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage",
               OMP_NUM_THREADS="4", POLARS_MAX_THREADS="4")
    tag = f"bulkr2_{chain}{seed}_{piece}"
    tmp = f"/tmp/{tag}.npz"
    vol.reload()
    if os.path.exists(f"/vol/out/{tag}.npz"):
        return f"{tag} already on the volume"
    cmd = [sys.executable, "-u", "-W", "ignore", *fits(chain)[piece], "--seed", str(seed), "--out", tmp]
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
def launch(chains: str = "NEW,SA", seeds: str = "0,1,2", pieces: str = "bxTX,bxE9f,f4TX"):
    for c in chains.split(","):
        for s in seeds.split(","):
            for p in pieces.split(","):
                call = run_fit.spawn(c, int(s), p)
                print(f"spawned {c}{s} {p}: {call.object_id}")


def upload():
    """Idempotent by name, EXCEPT the agent_bulkr_* fins, which are always re-uploaded (they change per stage run)."""
    have = set()
    try:
        have = {e.path.lstrip("/") for e in vol.listdir("bulkr2/data/interim")}
    except Exception:
        pass
    todo = [f for f in FILES if "bulkr2/" + f not in have or "agent_bulkr_" in f]
    miss = [f for f in todo if not os.path.exists(f)]
    if miss:   # the stage fins land later: re-run upload then (idempotent; fins always re-sent)
        print(f"NOT YET LOCAL (skipped; re-run upload later): {miss}")
        todo = [f for f in todo if f not in miss]
    print(f"{len(FILES)} inputs, {len(todo)} to upload ({sum(os.path.getsize(f) for f in todo) / 1e9:.2f} GB)")
    with vol.batch_upload(force=True) as b:
        for f in todo:
            b.put_file(f, "/bulkr2/" + f)
    print("upload done")


def fetch():
    os.makedirs("data/interim/modal", exist_ok=True)
    for e in vol.listdir("out"):
        name = os.path.basename(e.path)
        if name.startswith("bulkr2_"):
            with open(f"data/interim/modal/{name}", "wb") as fh:
                for chunk in vol.read_file(e.path):
                    fh.write(chunk)
            print("fetched", name)


if __name__ == "__main__":
    {"upload": upload, "fetch": fetch}[sys.argv[1]]()
