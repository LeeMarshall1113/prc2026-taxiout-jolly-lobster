"""STEP 2 of MOREDAYS on Kaggle FREE CPU (docs/LAB-NOTES.md): arms A / C at model seeds 0, 1, 2, one kernel per
(arm, seed), the three re-stack pieces inside (archive/agent_md_bxnest2.py x2, agent_md_fold4b.py). Copy of
archive/agent_md_kaggle.py (the STEP 1 driver); the dataset is a NEW VERSION of <kaggle-user>/prc-md-kg with the STEP 2 files.
Tags: A0 A1 A2 C0 C1 C2 (B* added when its construction is registered).
Original docstring of the STEP 1 driver follows.

Stage / write kernels / status / pull for the MOREDAYS learning curve on Kaggle FREE CPU (docs/LAB-NOTES.md).

    python archive/agent_md_kaggle.py stage                 # flat PRIVATE dataset dir under kaggle-stage/prc-md-kg
    python archive/agent_md_kaggle.py kernels TAG [TAG..]   # kernel dirs (TAG = f100 | f50s0 | f50s1 | f75s0 | f75s1)
    python archive/agent_md_kaggle.py status [TAG..]
    python archive/agent_md_kaggle.py pull TAG [TAG..]      # outputs -> data/interim/agent_md_kg_<TAG>.npz, logs/kg/

Kaggle CLI = SYSTEM python (`python -m kaggle`). Dataset created without --public; kernels is_private, CPU, internet ON only
to pip-pin lightgbm 4.7.0 / polars 1.44.1 (the local versions). Pattern of archive/agent_kg_ltfm_kaggle.py.
Each kernel runs the three re-stack fits of moredays s0.1/s0.2 at one (frac, sub_seed) and saves ids + both_guard of each.
"""
import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
STAGE = pathlib.Path("kaggle-stage")
USER = __import__("os").environ["KAGGLE_USERNAME"]
DS = "prc-md-kg"
PINS = ["lightgbm==4.7.0", "polars==1.44.1"]
SCRIPTS = ["agent_md_bxnest.py", "agent_md_fold4.py", "agent_md_bxnest2.py", "agent_md_fold4b.py", "agent_restack6_restack.py", "agent_restack6_restack4.py",
           "agent_expensive_restack.py", "agent_seqpost_feat.py", "agent_neighbour_feat.py", "agent_restack4_arr.py"]
INTERIM = ["agent_stagecap_CAND_E9f.npz", "agent_offblock_rows_c238.parquet", "agent_restack3_evt_2025.parquet",
           "agent_restack3_trk_2025.parquet", "agent_smallbundle_seq_2025.parquet", "agent_backx_bx_2025.parquet",
           "agent_backx_ff_2025.parquet", "agent_restack5_bxnest_cols.npz",
           "agent_md_rows_new.parquet", "agent_md_evt_new.parquet", "agent_md_trk_new.parquet", "agent_md_seq_new.parquet",
           "agent_md_ff_new.parquet", "agent_md_TXS6_354.npz", "agent_md_E9f_354.npz", "agent_md_TXS6_354_238.npz",
           "agent_md_E9f_354_238.npz", "agent_sb2_osm_2025.parquet", "agent_md_osm_new.parquet"]
ARMS = {f"{a}{r}": (a, r) for a in ("A", "C", "D") for r in (0, 1, 2)}

RUNNER = r'''
import glob, json, os, resource, shutil, subprocess, sys, time
import numpy as np
TAG, ARM, SEED, PINS = @TAG@, @ARM@, @SEED@, @PINS@
W = "/kaggle/working"
T0 = time.time()
LOG = open(f"{W}/{TAG}.log", "w", buffering=1)
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n")
hits = glob.glob("/kaggle/input/**/scripts__agent_md_bxnest2.py", recursive=True)
assert len(hits) == 1, hits
SRC = os.path.dirname(hits[0])
REPO = "/tmp/repo"
for d in ("archive", "data/interim/modal"):
    os.makedirs(f"{REPO}/{d}", exist_ok=True)
for name in sorted(os.listdir(SRC)):
    p = os.path.join(SRC, name)
    if name.startswith("scripts__"):
        shutil.copy(p, f"{REPO}/scripts/{name[9:]}")
    elif name.startswith("interim__modal__"):
        os.symlink(p, f"{REPO}/data/interim/modal/{name[16:]}")
    elif name.startswith("interim__"):
        os.symlink(p, f"{REPO}/data/interim/{name[9:]}")
def ver(m):
    r = subprocess.run([sys.executable, "-c", f"import {m}; print({m}.__version__)"], capture_output=True, text=True)
    return r.stdout.strip() or r.stderr.strip()[-200:]
need = [p for p in PINS if ver(p.split("==")[0]) != p.split("==")[1]]
if need:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], capture_output=True, text=True)
    say("pip install", need, "rc", r.returncode, r.stderr[-800:])
VERS = {m: ver(m) for m in ("lightgbm", "polars", "numpy", "sklearn")}
say("versions:", VERS, "nproc:", os.cpu_count())
env = {**os.environ, "PYTHONPATH": REPO, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "4", "POLARS_MAX_THREADS": "4",
       "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage"}
I = "data/interim/"
TX, E9, T354, E354 = I + "modal__TX_S6.npz", I + "agent_stagecap_CAND_E9f.npz", I + "agent_md_TXS6_354.npz", I + "agent_md_E9f_354.npz"
if ARM == "C":
    TX, E9 = I + "agent_md_TXS6_354_238.npz", I + "agent_md_E9f_354_238.npz"
TX = TX.replace("modal__", "modal/")
FITS = [("bxTX", ["archive/agent_md_bxnest2.py", "--recipe", "cap,evt,trk,bx,big", "--fin-npz", TX, "--new-fin-npz", T354]),
        ("bxE9f", ["archive/agent_md_bxnest2.py", "--recipe", "cap,evt,trk,bx,big", "--fin-npz", E9, "--new-fin-npz", E354]),
        ("f4TX", ["archive/agent_md_fold4b.py", "--recipe", "cap,evt,trk,seq,big", "--fin-npz", TX, "--new-fin-npz", T354])]
if ARM == "D":   # moredays s4.5: arm A + the osm token
    FITS = [(n, args + ["--osm"]) for n, args in FITS]
out = {}
for name, args in FITS:
    o = f"/tmp/{name}.npz"
    cmd = [sys.executable, "-u", "-W", "ignore", *args, "--seed", str(SEED), "--out", o]
    say("RUN", " ".join(cmd))
    t1 = time.time()
    with open(f"{W}/{TAG}_{name}.log", "w") as fh:
        rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
    say(name, "rc", rc, f"{time.time() - t1:.0f}s", open(f"{W}/{TAG}_{name}.log").read().strip().splitlines()[-1][-200:])
    if rc:
        raise SystemExit(rc)
    z = np.load(os.path.join(REPO, o) if not o.startswith("/") else o, allow_pickle=True)
    out["ids"] = z["ids"]
    out[name] = z["both_guard"]
    os.remove(o)
peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 / 1024
np.savez(f"{W}/md_{TAG}.npz", **out)
json.dump({"tag": TAG, "arm": ARM, "seed": SEED, "seconds": round(time.time() - T0), "versions": VERS,
           "peak_rss_child_gb": peak}, open(f"{W}/{TAG}_meta.json", "w"), indent=1)
say(f"DONE {time.time() - T0:.0f}s peak child rss {peak:.2f} GB")
'''


def slug(tag):
    return "prc-md2-" + tag.lower()


def stage():
    d = STAGE / DS
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for s in SCRIPTS:
        shutil.copy(REPO / "archive" / s, d / f"scripts__{s}")
    for n in INTERIM:
        shutil.copy(REPO / "data" / "interim" / n, d / f"interim__{n}")
    shutil.copy(REPO / "data" / "interim" / "modal" / "TX_S6.npz", d / "interim__modal__TX_S6.npz")
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": DS, "id": f"{USER}/{DS}", "licenses": [{"name": "other"}]}, indent=1))
    print(f"staged {len(list(d.iterdir())) - 1} files, {sum(p.stat().st_size for p in d.iterdir()) / 1e6:.0f} MB -> {d}")


def kernels(tags):
    for tag in tags:
        arm, seed = ARMS[tag]
        code = (RUNNER.replace("@TAG@", repr(tag)).replace("@ARM@", repr(arm)).replace("@SEED@", repr(seed))
                .replace("@PINS@", repr(PINS)))
        kd = STAGE / "kernels" / f"md2_{tag}"
        if kd.exists():
            shutil.rmtree(kd)
        kd.mkdir(parents=True)
        (kd / "run.py").write_text(code, encoding="utf-8")
        (kd / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{slug(tag)}", "title": slug(tag), "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{USER}/{DS}"], "competition_sources": [],
            "kernel_sources": []}, indent=1))
        print(f"kernel dir {kd} (arm {arm}, seed {seed})")


def kag(*a):
    r = subprocess.run(["python", "-m", "kaggle", *a], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def status(tags):
    for tag in tags:
        print(f"{tag:<8} {kag('kernels', 'status', f'{USER}/{slug(tag)}')[-160:]}")


def pull(tags):
    for tag in tags:
        out = STAGE / "out" / f"md2_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        print(tag, kag("kernels", "output", f"{USER}/{slug(tag)}", "-p", str(out), "-o")[-300:])
        (REPO / "logs" / "kg").mkdir(parents=True, exist_ok=True)
        for p in out.iterdir():
            if p.name == f"md_{tag}.npz":
                shutil.copy(p, REPO / "data" / "interim" / f"agent_md_kg2_{tag}.npz")
            elif p.suffix in (".log", ".json"):
                shutil.copy(p, REPO / "logs" / "kg" / f"md2_{p.name}")


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"stage": lambda: stage(), "kernels": lambda: kernels(rest),
     "status": lambda: status(rest or list(ARMS)), "pull": lambda: pull(rest)}[cmd]()
