"""SMALLBUNDLE2 fold chain on Modal (docs/LAB-NOTES.md). Modelled on agent_md_stage1.py: same volume
prc-stage1 (repo-relative paths under /vol), same image pins (scipy pinned to the local 1.18.1), --detach + .spawn().

The chain logic is archive/agent_sb2_chain2.py's, unchanged: every fit is an UNEDITED local script (agent_sb2_bxnest.py /
agent_sb2_fold4.py) in its own CPU container (one fit per container, run in parallel per stage); judging is the unedited
archive/agent_sb2_judge.py in the orchestrator container between stages. Stage 0 = a MODAL REPRODUCTION GATE: the three
no-token seed-0 fits are refitted on Modal and must equal the local ones (uploaded) exactly (max |diff| 0.0), because the
paired controls PCTRL_1/2 and the two cached treatment fits were fitted locally; any nonzero diff -> STOP, nothing else
runs (the local slot stays the fallback).

    python modal/agent_sb2_app.py sizes                                   # list the inputs and their size
    python modal/agent_sb2_app.py upload                                  # idempotent: only files not yet on the volume
    modal run --detach modal/agent_sb2_app.py::launch                     # spawns the orchestrator, returns at once
    python modal/agent_sb2_app.py fetch                                   # results -> data/interim/, logs -> logs/
"""
import json
import os
import sys

import modal

VOL = "prc-stage1"
PINS = ["polars==1.44.1", "numpy==2.5.2", "lightgbm==4.7.0", "scikit-learn==1.9.0", "scipy==1.18.1", "pandas", "pyarrow"]
IGN = ["**/__pycache__/**", "**/*.pyc"]
image = (modal.Image.debian_slim(python_version="3.13").pip_install(*PINS)
         .add_local_dir("prc", remote_path="/repo/prc", ignore=IGN)
         .add_local_dir("archive", remote_path="/repo/scripts", ignore=IGN))
vol = modal.Volume.from_name(VOL, create_if_missing=True)
app = modal.App("prc-sb2")
I = "data/interim/"
INPUTS = [  # everything the fits and the judge read (repo-relative)
    I + "modal/TX_S6.npz", I + "agent_stagecap_CAND_E9f.npz", I + "agent_offblock_rows_c238.parquet",
    I + "agent_restack3_evt_2025.parquet", I + "agent_restack3_trk_2025.parquet", I + "agent_smallbundle_seq_2025.parquet",
    I + "agent_neighbour_nbr_2025.parquet", I + "agent_backx_bx_2025.parquet", I + "agent_backx_ff_2025.parquet",
    I + "agent_restack5_bxnest_cols.npz", I + "agent_sb2_osm_2025.parquet", I + "agent_sb2_occ_2025.parquet",
    I + "agent_sb2_unc_TX.npz", I + "agent_sb2_unc_E9f.npz", I + "agent_nov_fold.parquet", I + "agent_nov_2026.parquet",
    I + "agent_offblock_rows_2026.parquet", I + "agent_sb2_hdow_2025.parquet",
    I + "agent_stagecap_RS_E9S_s0.npz", I + "agent_stagecap_RS_E9S_s1.npz", I + "agent_stagecap_RS_E9S_s2.npz",
    I + "agent_blend_oof_seedA.npz", I + "agent_blend_v109analogue.npz", I + "agent_restack5_V9TOP1.npz",
    *[I + f"agent_restack5_bxn_cap_evt_trk_bx_big_s{s}.npz" for s in (0, 1, 2)],
    *[I + f"agent_stagecap_bxn_E9f_s{s}.npz" for s in (0, 1, 2)],
    *[I + f"agent_restack5_f_cap_evt_trk_seq_big_s{s}.npz" for s in (0, 1, 2)],
    I + "agent_sb2_bxTX_none_s0.npz", I + "agent_sb2_bxE9f_none_s0.npz", I + "agent_sb2_sq_none_s0.npz",
    I + "agent_sb2_bxTX_mw_s0.npz", I + "agent_sb2_sq_mw_s0.npz", I + "agent_sb2_verdict.json",
]
TX, E9F = I + "modal/TX_S6.npz", I + "agent_stagecap_CAND_E9f.npz"
BXREC, SQREC = "cap,evt,trk,bx,big", "cap,evt,trk,seq,big"
REPRO_TOL = 1e-9  # seconds; s0.10 amendment (team decision, pre-number): identical trees, float summation order, max 9.1e-13 s
ENV = {"PYTHONPATH": "/repo", "PYTHONIOENCODING": "utf-8", "POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR": "load_as_storage",
       **{k: "4" for k in ("OMP_NUM_THREADS", "POLARS_MAX_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def _cd():
    os.chdir("/repo")
    for d in ("data", "logs"):
        os.makedirs(f"/vol/{d}", exist_ok=True)
        if not os.path.exists(d):
            os.symlink(f"/vol/{d}", d)


def tokstr(toks):
    return "_".join(sorted(toks)) if toks else "none"


def fit_spec(piece, toks, seed, out=None):
    rec = (SQREC if piece == "sq" else BXREC) + "".join("," + t for t in sorted(toks))
    fin = E9F if piece == "bxE9f" else TX
    unc = I + ("agent_sb2_unc_E9f.npz" if piece == "bxE9f" else "agent_sb2_unc_TX.npz")
    script = "archive/agent_sb2_fold4.py" if piece == "sq" else "archive/agent_sb2_bxnest.py"
    out = out or I + f"agent_sb2_{piece}_{tokstr(toks)}_s{seed}.npz"
    return {"name": os.path.basename(out)[:-4], "out": out, "log": f"logs/{os.path.basename(out)[:-4]}.log",
            "args": [script, "--fin-npz", fin, "--recipe", rec, "--seed", str(seed), "--unc-npz", unc, "--out", out[:-4] + "_tmp.npz"]}


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=8192, timeout=60 * 40)
def fit_one(spec: dict) -> str:
    import subprocess
    import time
    _cd()
    vol.reload()
    if os.path.exists(spec["out"]):
        return f"{spec['name']} exists"
    t0 = time.time()
    with open(spec["log"], "w") as fh:
        rc = subprocess.call([sys.executable, "-u", "-W", "ignore"] + spec["args"], stdout=fh, stderr=subprocess.STDOUT,
                             env=dict(os.environ, **ENV))
        fh.write(f"\nMODAL rc {rc} wall {time.time() - t0:.0f}s\n")
    if rc == 0:
        os.replace(spec["args"][-1], spec["out"])
    vol.commit()
    return f"{spec['name']} rc {rc} wall {time.time() - t0:.0f}s"


def _comp_specs(toks, seed):
    toks = set(toks) - {"c2"}
    if not toks:
        return []
    bt = toks - {"lw"}
    ss = [fit_spec("sq", toks, seed)]
    if bt:
        ss += [fit_spec("bxTX", bt, seed), fit_spec("bxE9f", bt, seed)]
    return [s for s in ss if not os.path.exists(s["out"])]


@app.function(image=image, volumes={"/vol": vol}, cpu=2.0, memory=6144, timeout=60 * 240)
def orchestrate() -> str:
    import subprocess
    import time
    import numpy as np
    _cd()
    vol.reload()
    LOGF = "logs/agent_sb2_chain_modal.log"

    # s0.10: NO file may be open during vol.commit()/vol.reload() (the 22:46Z crash: the chain log was held open) --
    # every write is open-append-close, every json/npz read is closed before the next volume operation.
    def write(txt):
        with open(LOGF, "a") as fh:
            fh.write(txt)

    def rjson(p):
        with open(p) as fh:
            return json.load(fh)

    def say(m):
        write(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {m}\n")
        vol.commit()

    def stage(specs, label):
        specs = [s for s in specs if s]
        if specs:
            say(f"STAGE {label}: {len(specs)} fits in parallel containers")
            for r in fit_one.map(specs):
                say("  " + r)
            vol.reload()
            bad = [s["name"] for s in specs if not os.path.exists(s["out"])]
            assert not bad, f"fits failed: {bad}"

    def judge(st):
        vol.reload()
        say(f"JUDGE {st}")
        p = subprocess.run([sys.executable, "-u", "-W", "ignore", "archive/agent_sb2_judge.py", st], capture_output=True, text=True,
                           env=dict(os.environ, **ENV))
        with open(f"logs/agent_sb2_judge_{st}.log", "w") as fh:
            fh.write(p.stdout + p.stderr)
        write(p.stdout + p.stderr)
        vol.commit()
        assert p.returncode == 0, f"judge {st} failed"
        return rjson(I + "agent_sb2_verdict.json")

    # ---- stage 0: Modal reproduction gate (s0.9)
    rep = {pc: fit_spec(pc, set(), 0, out=I + f"agent_sb2_modalrepro_{pc}_s0.npz") for pc in ("bxTX", "bxE9f", "sq")}
    stage([s for s in rep.values() if not os.path.exists(s["out"])], "0 Modal reproduction")
    diffs = {}
    for pc, s in rep.items():
        with np.load(s["out"], allow_pickle=True) as a, np.load(I + f"agent_sb2_{pc}_none_s0.npz", allow_pickle=True) as b:
            diffs[pc] = float(np.abs(a["both_guard"].astype(float) - b["both_guard"].astype(float)).max())
    say(f"MODAL REPRO vs local no-token seed-0 fits: {diffs} (tolerance {REPRO_TOL} s, s0.10)")
    if any(v > REPRO_TOL for v in diffs.values()):
        say("MODAL_REPRO_FAIL -- STOP (s0.9); the local slot is the fallback")
        return "MODAL_REPRO_FAIL"
    v = rjson(I + "agent_sb2_verdict.json")
    comps = ["mw", "c2", "lw", "unc", "osm", "occ"] + (["nov"] if v["nov_sanity"]["ok"] else [])
    # ---- step 1a / 1b / 2, as archive/agent_sb2_chain2.py
    stage([s for c in comps for s in _comp_specs({c}, 0)], "1a seed-0 screens")
    v = judge("screen0")
    surv = v["survivors0"]
    if surv:
        stage([s for c in surv for s in _comp_specs({c}, 1)], "1b seed-1 screens")
        v = judge("screen1")
    IN = v.get("IN", [])
    final, name, comps_final = "FAIL", "", []
    if IN:
        stage([s for sd in (0, 1, 2) for s in _comp_specs(set(IN), sd)], "2 bundle")
        v = judge("bundle")
        if v["bundle"]["pass"]:
            final, name, comps_final = "PASS_BUNDLE", "bundle", IN
        elif len(IN) >= 2:
            fb = [c for c in IN if c != v["worst_IN"]]
            stage([s for sd in (0, 1, 2) for s in _comp_specs(set(fb), sd)], "fallback")
            v = judge("fallback")
            if v["fallback"]["pass"]:
                final, name, comps_final = "PASS_FALLBACK", "fallback", fb
    toks = sorted(c for c in comps_final if c != "c2")
    v = rjson(I + "agent_sb2_verdict.json")
    v["final"], v["final_components"], v["ran_on"] = final, comps_final, "modal"
    with open(I + "agent_sb2_verdict.json", "w") as fh:
        json.dump(v, fh, indent=1, default=float)
    say(f"SB2 VERDICT: {final} {comps_final}")
    write(f"FINAL={final}\nTOKENS={','.join(toks)}\nC2={1 if 'c2' in comps_final else 0}\nNAME={name}\n")
    vol.commit()
    return final


@app.local_entrypoint()
def launch():
    call = orchestrate.spawn()
    print(f"spawned sb2 orchestrator: {call.object_id}")


def sizes():
    tot = 0
    for f in INPUTS:
        s = os.path.getsize(f) if os.path.exists(f) else -1
        tot += max(s, 0)
        print(f"{s / 1e6:9.1f} MB  {f}" if s >= 0 else f"  MISSING    {f}")
    print(f"{tot / 1e9:.2f} GB in {len(INPUTS)} files")


def upload():
    have = set()
    for sub in ("data/interim", "data/interim/modal"):
        try:
            have |= {e.path.lstrip("/") for e in vol.listdir(sub)}
        except Exception:
            pass
    todo = [f for f in INPUTS if f not in have or f.endswith("agent_sb2_verdict.json")]
    print(f"{len(INPUTS)} inputs, {len(todo)} to upload ({sum(os.path.getsize(f) for f in todo) / 1e9:.2f} GB)")
    with vol.batch_upload(force=True) as b:
        for f in todo:
            b.put_file(f, "/" + f)
    print("upload done")


def fetch():
    """fit npzs are NOT needed locally unless a build follows; fetch the verdict, the composites and the logs."""
    want = ("agent_sb2_verdict.json", "agent_sb2_bundle_s0.npz", "agent_sb2_bundle_preC2_s0.npz", "agent_sb2_fallback_s0.npz",
            "agent_sb2_fallback_preC2_s0.npz", "agent_sb2_ctrl_r0.npz")
    for e in vol.listdir("data/interim"):
        n = os.path.basename(e.path)
        if n in want:
            with open(I + n, "wb") as fh:
                for ch in vol.read_file(e.path):
                    fh.write(ch)
            print("fetched", n)
    for e in vol.listdir("logs"):
        n = os.path.basename(e.path)
        if n.startswith("agent_sb2_"):
            with open("logs/" + n, "wb") as fh:
                for ch in vol.read_file(e.path):
                    fh.write(ch)
    print("fetched logs/agent_sb2_* (chain log: logs/agent_sb2_chain_modal.log)")


# ================================================================ C6 serving (s3: the team's model-selection upload of C6)
# The re-stack SERVE steps only (the heavy part: one full-2025 re-stack fit per stage stream), with the unedited
# archive/agent_sb2_bxserve.py / agent_sb2_serve.py and exactly the arguments of scripts/agent_sb2_{bx,}build.sh; the
# post-processing chain, splices, compose, checks and lens run locally (agent_sb2_osm_build.sh).
SERVE_UP = [I + f for f in ("agent_offblock_rows_2026.parquet", "agent_restack3_evt_2026.parquet", "agent_restack3_trk_2026.parquet",
                            "agent_smallbundle_seq_2026.parquet", "agent_backx_bx_2026.parquet", "agent_sb2_osm_2026.parquet",
                            "v103_s1.parquet", "v103_gatebag_s1.parquet", "v105_s1.parquet", "v105_gatebag_s1.parquet",
                            "v105sc_s1.parquet", "v105sc_gatebag_s1.parquet", "v105_s1r5x_est_v2_cap_evt_trk_bx_big.parquet")]
PIECES = {  # piece: (serve script, train npz, recipe, v105-path prefix)
    "a": ("archive/agent_sb2_bxserve.py", TX, "cap,evt,trk,bx,big,osm", "v105"),
    "b": ("archive/agent_sb2_bxserve.py", TX, "cap,evt,trk,seq,bx,big,osm", "v105"),
    "c": ("archive/agent_sb2_serve.py", TX, "cap,evt,trk,seq,big,osm", "v105"),
    "d": ("archive/agent_sb2_bxserve.py", E9F, "cap,evt,trk,bx,big,osm", "v105sc"),
}


def serve_specs():
    out = []
    for pc, (script, train, rec, p105) in PIECES.items():
        for pfx in ("v103", p105):
            for st in ("s1", "gatebag_s1"):
                out.append({"name": f"t{pc}_{pfx}_{st}", "script": script, "train": train, "recipe": rec,
                            "inp": I + f"{pfx}_{st}.parquet", "out": I + f"sb2serve/t{pc}_{pfx}_{st}.parquet"})
    # pipeline check: the same serve path WITHOUT osm on one stream must reproduce production's serve output
    out.append({"name": "check_a_v105_s1", "script": "archive/agent_sb2_bxserve.py", "train": TX, "recipe": "cap,evt,trk,bx,big",
                "inp": I + "v105_s1.parquet", "out": I + "sb2serve/check_a_v105_s1.parquet"})
    return out


@app.function(image=image, volumes={"/vol": vol}, cpu=4.0, memory=8192, timeout=60 * 30)
def serve_one(sp: dict) -> str:
    import subprocess
    import time
    _cd()
    vol.reload()
    if os.path.exists(sp["out"]):
        return f"{sp['name']} exists"
    os.makedirs(I + "sb2serve", exist_ok=True)
    t0 = time.time()
    tmp = sp["out"][:-8] + "_tmp.parquet"
    log = f"logs/agent_sb2_serve_{sp['name']}.log"
    with open(log, "w") as fh:
        rc = subprocess.call([sys.executable, "-u", "-W", "ignore", sp["script"], "--train-npz", sp["train"],
                              "--train-rows", I + "agent_offblock_rows_c238.parquet", "--rows-2026", I + "agent_offblock_rows_2026.parquet",
                              "--feats", "est_v2", "--recipe", sp["recipe"], "--in", sp["inp"], "--out", tmp],
                             stdout=fh, stderr=subprocess.STDOUT, env=dict(os.environ, **ENV))
        fh.write(f"\nMODAL rc {rc} wall {time.time() - t0:.0f}s\n")
    if rc == 0:
        os.replace(tmp, sp["out"])
    vol.commit()
    return f"{sp['name']} rc {rc} wall {time.time() - t0:.0f}s"


@app.local_entrypoint()
def launch_serve():
    for sp in serve_specs():  # detached: one spawned call per serve step; completion = files on the volume
        print(f"spawned {sp['name']}: {serve_one.spawn(sp).object_id}")


def upload_serve():
    with vol.batch_upload(force=True) as b:
        for f in SERVE_UP:
            b.put_file(f, "/" + f)
    print(f"uploaded {len(SERVE_UP)} files ({sum(os.path.getsize(f) for f in SERVE_UP) / 1e6:.0f} MB)")


def fetch_serve():
    os.makedirs(I + "sb2serve", exist_ok=True)
    n = 0
    for e in vol.listdir("data/interim/sb2serve"):
        nm = os.path.basename(e.path)
        if nm.endswith("_tmp.parquet"):
            continue
        with open(I + "sb2serve/" + nm, "wb") as fh:
            for ch in vol.read_file(e.path):
                fh.write(ch)
        n += 1
    for e in vol.listdir("logs"):
        nm = os.path.basename(e.path)
        if nm.startswith("agent_sb2_serve_"):
            with open("logs/" + nm, "wb") as fh:
                for ch in vol.read_file(e.path):
                    fh.write(ch)
    print(f"fetched {n} serve outputs -> {I}sb2serve/")


if __name__ == "__main__":
    {"sizes": sizes, "upload": upload, "fetch": fetch, "upload_serve": upload_serve, "fetch_serve": fetch_serve}[sys.argv[1]]()
