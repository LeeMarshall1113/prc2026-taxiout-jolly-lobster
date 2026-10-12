"""VARIANT FACTORY on Modal (docs/LAB-NOTES.md). Paired fold + 2026-serve re-stack variants, all fitted in this one
environment by the SAME hook (archive/agent_vfac_models.py): ctrl (= production LightGBM re-stack, same code path) plus cb, xgb, dart, l1,
logt (model classes on the same features) and a_noosm, a_nobx, a_noseq, a_notrk, a_cap (recipe ablations). d_v = add_v - add_ctrl.

Pieces (fold = the etrs/bulkr fold pieces; serve = the same recipe trained on the same fin; 2026 stream = production v105 path):
  bxTX  fold archive/agent_vfac_bxnest.py recipe cap,evt,trk,bx,big,osm fin TX_S6   -> EGLL, LEMD             | serve piece a (bxserve, TX_S6, v105)
  bxE9f fold archive/agent_vfac_bxnest.py same recipe, fin CAND_E9f                 -> EDDF,EDDM,EHAM,LEBL,LSZH | serve piece d (E9f, v105sc)
  f4TX  fold archive/agent_vfac_fold4.py  recipe cap,evt,trk,seq,big,osm fin TX_S6   -> LFPG, LIRF             | serve piece c (serve.py, TX_S6, v105)
(production serves LEMD with piece b = seq,bx; here LEMD is served with the bxTX recipe so that fold and serve are the SAME recipe.)
All inputs are already on volume prc-stage1:/bulkr2/data (uploaded by modal/agent_bulkr_restack.py / agent_bulkr_serve.py): NO upload step.

    SYS=C:/Python314/python.exe     # has modal; run from the repo root
    $SYS -m modal run --detach modal/agent_vfac_app.py::launch_fold --variants ctrl --pieces f4TX            # GATE: ctrl must reproduce agent_etrs_f4TX_s0 (~25 min, ~$0.1)
    $SYS -m modal run --detach modal/agent_vfac_app.py::launch_fold --variants ctrl,xgb,l1,logt,a_noosm     # wave 1 (15 containers)
    $SYS -m modal run --detach modal/agent_vfac_app.py::launch_fold --variants dart,a_nobx,a_noseq,a_notrk,a_cap   # wave 2
    $SYS -m modal run --detach modal/agent_vfac_app.py::launch_fold --variants cb                             # wave 3 (slow; 3 containers)
    $SYS -m modal run --detach modal/agent_vfac_app.py::launch_serve                                          # all variants, 3 containers
    $SYS modal/agent_vfac_app.py status ; $SYS modal/agent_vfac_app.py fetch     # -> data/interim/modal_vfac/
Cost: cpu 4 / 12 GiB ~ $0.25/h. Fold fit ~20-30 min per (piece, variant) except cb (~60-90 min); serve ~5 min per variant per piece.
Wave 1 ~ $2.0, wave 2 ~ $1.5, cb ~ $1, serve ~ $0.7 -> ~ $5 for everything. Per-fold checkpoints are committed every 2 min (a preempted
container resumes at the first missing fold).
"""
import os
import sys

import modal

VOL = "prc-stage1"
PINS = ["polars==1.44.1", "numpy==2.5.2", "lightgbm==4.7.0", "scikit-learn==1.9.0", "scipy", "pandas", "pyarrow",
        "catboost==1.2.10", "xgboost==3.4.1"]
IGN = ["**/__pycache__/**", "**/*.pyc"]
image = (modal.Image.debian_slim(python_version="3.13").pip_install(*PINS)
         .add_local_dir("prc", remote_path="/repo/prc", ignore=IGN)
         .add_local_dir("archive", remote_path="/repo/scripts", ignore=IGN))
vol = modal.Volume.from_name(VOL, create_if_missing=True)
app = modal.App("prc-vfac")
I = "data/interim/"
BX, SQ = "cap,evt,trk,bx,big,osm", "cap,evt,trk,seq,big,osm"
FOLD = {"bxTX": ("archive/agent_vfac_bxnest.py", BX, I + "modal/TX_S6.npz"),
        "bxE9f": ("archive/agent_vfac_bxnest.py", BX, I + "agent_stagecap_CAND_E9f.npz"),
        "f4TX": ("archive/agent_vfac_fold4.py", SQ, I + "modal/TX_S6.npz")}
SERVE = {"a": ("archive/agent_vfac_bxserve.py", I + "modal/TX_S6.npz", BX, "v105"),
         "d": ("archive/agent_vfac_bxserve.py", I + "agent_stagecap_CAND_E9f.npz", BX, "v105sc"),
         "c": ("archive/agent_vfac_serve.py", I + "modal/TX_S6.npz", SQ, "v105")}
ALLV = ["ctrl", "xgb", "l1", "logt", "a_noosm", "dart", "a_nobx", "a_noseq", "a_notrk", "a_cap", "cb"]
# recipe edits (same table as archive/agent_vfac_models.py; duplicated so the local entrypoint needs no repo import)
DROP = {"a_noosm": {"osm", "occ"}, "a_nobx": {"bx"}, "a_noseq": {"seq"}, "a_notrk": {"trk", "evt"}}


def _noop(variant, recipe):
    """True when the variant's recipe edit changes nothing for this piece (e.g. dropping seq from a bx piece)."""
    toks = recipe.split(",")
    if variant in DROP:
        return not (DROP[variant] & set(toks))
    if variant == "a_cap":
        return set(toks) <= {"cap", "big"}
    return False


def _env():
    return dict(os.environ, PYTHONPATH="/repo", PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage",
                OMP_NUM_THREADS="4", POLARS_MAX_THREADS="4")


def _links():
    os.chdir("/repo")
    for d in ("data/adsb", "data/external", "logs"):
        os.makedirs(d, exist_ok=True)
    for src, dst in (("/vol/data/raw", "data/raw"), ("/vol/bulkr2/data/interim", "data/interim")):
        if not os.path.exists(dst):
            os.symlink(src, dst)
    os.makedirs("/vol/out/vfac/ck", exist_ok=True)


def _run(cmd, log):
    """subprocess + a thread that commits the volume every 2 min (per-fold checkpoints survive preemption)."""
    import subprocess
    import threading
    import time
    stop = threading.Event()

    def pump():
        while not stop.wait(120):
            try:
                vol.commit()
            except Exception:
                pass
    threading.Thread(target=pump, daemon=True).start()
    t0 = time.time()
    with open(log, "a") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, env=_env())
        fh.write(f"\nMODAL rc {rc} wall {time.time() - t0:.0f}s\n")
    stop.set()
    return rc, time.time() - t0


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=12288, timeout=60 * 60 * 4, retries=2)
def run_fold(piece: str, variant: str, seed: int = 0) -> str:
    script, recipe, fin = FOLD[piece]
    vol.reload()
    out = f"/vol/out/vfac/fold_{piece}_{variant}_s{seed}.npz"
    if os.path.exists(out):
        return f"{piece} {variant} already done"
    _links()
    cmd = [sys.executable, "-u", "-W", "ignore", script, "--recipe", recipe, "--fin-npz", fin, "--seed", str(seed),
           "--variants", variant, "--ckpt-dir", "/vol/out/vfac/ck", "--tag", f"{piece}_s{seed}", "--out", "/tmp/o.npz"]
    rc, wall = _run(cmd, f"/vol/out/vfac/fold_{piece}_{variant}_s{seed}.log")
    if rc == 0:
        import shutil
        shutil.copy("/tmp/o.npz", out + ".tmp")
        os.replace(out + ".tmp", out)
    vol.commit()
    return f"fold {piece} {variant} rc {rc} wall {wall:.0f}s"


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=12288, timeout=60 * 60 * 3, retries=1)
def run_serve(piece: str, variants: str, seed: int = 0, tag: str = "") -> str:
    script, train, recipe, p105 = SERVE[piece]
    vol.reload()
    out = f"/vol/out/vfac/serve_{piece}{tag}_s{seed}.npz"
    if os.path.exists(out):
        return f"serve {piece} done"
    _links()
    vs = [v for v in variants.split(",") if not _noop(v, recipe)]
    cmd = [sys.executable, "-u", "-W", "ignore", script, "--train-npz", train, "--train-rows", I + "agent_offblock_rows_c238.parquet",
           "--rows-2026", I + "agent_offblock_rows_2026.parquet", "--feats", "est_v2", "--recipe", recipe, "--seed", str(seed),
           "--variants", ",".join(vs), "--in", f"{I}{p105}_s1.parquet,{I}{p105}_gatebag_s1.parquet", "--out", "/tmp/s.npz"]
    rc, wall = _run(cmd, f"/vol/out/vfac/serve_{piece}{tag}_s{seed}.log")
    if rc == 0:
        import shutil
        shutil.copy("/tmp/s.npz", out + ".tmp")
        os.replace(out + ".tmp", out)
    vol.commit()
    return f"serve {piece} rc {rc} wall {wall:.0f}s ({','.join(vs)})"


@app.local_entrypoint()
def launch_fold(variants: str = "ctrl", pieces: str = "bxTX,bxE9f,f4TX", seeds: str = "0"):
    for p in pieces.split(","):
        for v in variants.split(","):
            if _noop(v, FOLD[p][1]):
                print(f"skip {p} {v}: the recipe edit is a no-op for this piece")
                continue
            for s in seeds.split(","):
                print(f"spawned fold {p} {v} s{s}: {run_fold.spawn(p, v, int(s)).object_id}")


@app.local_entrypoint()
def launch_serve(variants: str = ",".join(ALLV), pieces: str = "a,d,c", seed: int = 0, tag: str = ""):
    for p in pieces.split(","):
        print(f"spawned serve {p}: {run_serve.spawn(p, variants, seed, tag).object_id}")


def status():
    for e in sorted(vol.listdir("out/vfac"), key=lambda e: e.path):
        print(e.path)


def fetch():
    os.makedirs("data/interim/modal_vfac", exist_ok=True)
    for e in vol.listdir("out/vfac"):
        n = os.path.basename(e.path)
        if n.endswith((".npz", ".log")) and not os.path.exists("data/interim/modal_vfac/" + n):
            with open("data/interim/modal_vfac/" + n, "wb") as fh:
                for ch in vol.read_file(e.path):
                    fh.write(ch)
            print("fetched", n)


if __name__ == "__main__":
    {"status": status, "fetch": fetch}[sys.argv[1]]()
