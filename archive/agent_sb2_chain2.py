"""experiment sb2 chain2: orchestrator for docs/LAB-NOTES.md + amendment s0.8 (seed-0-first screens, fits 2 at
a time, C9 novelty). Holds no fold arrays: fits run in a pool of POOL processes; judging is archive/agent_sb2_judge.py, run
alone between fit stages. Resumable (every output is written to a _tmp name and renamed after rc 0); between tasks it checks
JOB_PREEMPT_FILE and exits 10. A task that fails under the parallel load is retried once alone.

    python archive/agent_sb2_chain2.py        (inside a scheduler slot, under archive/capped_run.py)
Prints FINAL=..., TOKENS=..., C2=..., NAME=... for archive/agent_sb2_buildall.sh; writes data/interim/agent_sb2_bundle_spec.json.
"""
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

I = "data/interim/"
PY = sys.executable
POOL = int(os.environ.get("SB2_POOL", "2"))
TX, E9F = I + "modal/TX_S6.npz", I + "agent_stagecap_CAND_E9f.npz"
BXREC, SQREC = "cap,evt,trk,bx,big", "cap,evt,trk,seq,big"
VJ = I + "agent_sb2_verdict.json"
T0 = time.time()
LOCK = threading.Lock()


def say(m):
    with LOCK:
        print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z +{time.time() - T0:5.0f}s] {m}", flush=True)


def preempted():
    f = os.environ.get("JOB_PREEMPT_FILE", "")
    return bool(f and os.path.exists(f))


def tokstr(toks):
    return "_".join(sorted(toks)) if toks else "none"


def fit_task(piece, toks, seed):
    """(name, args, log, out) or None when the output already exists / is a production file."""
    if not toks and seed > 0:
        return None
    out = I + f"agent_sb2_{piece}_{tokstr(toks)}_s{seed}.npz"
    if os.path.exists(out):
        return None
    rec = (SQREC if piece == "sq" else BXREC) + "".join("," + t for t in sorted(toks))
    fin = E9F if piece == "bxE9f" else TX
    unc = I + ("agent_sb2_unc_E9f.npz" if piece == "bxE9f" else "agent_sb2_unc_TX.npz")
    script = "archive/agent_sb2_fold4.py" if piece == "sq" else "archive/agent_sb2_bxnest.py"
    tmp = out[:-4] + "_tmp.npz"
    return (f"{piece}_{tokstr(toks)}_s{seed}", [script, "--fin-npz", fin, "--recipe", rec, "--seed", str(seed), "--unc-npz", unc,
                                                 "--out", tmp], f"logs/agent_sb2_{piece}_{tokstr(toks)}_s{seed}.log", out)


def comp_tasks(toks, seed):
    toks = set(toks) - {"c2"}
    if not toks:
        return []
    bt = toks - {"lw"}
    ts = [fit_task("sq", toks, seed)]
    if bt:  # lw alone refits only the SQ piece
        ts += [fit_task("bxTX", bt, seed), fit_task("bxE9f", bt, seed)]
    return [t for t in ts if t]


def _exec(t):
    name, args, log, out = t
    say(f"START {name}")
    with open(log, "w") as fh:
        rc = subprocess.call([PY, "-u", "-W", "ignore"] + args, stdout=fh, stderr=subprocess.STDOUT)
    if rc == 0 and out:
        tmp = [a for a in args if a.endswith("_tmp.npz")]
        if tmp:
            os.replace(tmp[0], out)
    tail = open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-1:] or [""]
    say(f"END   {name} rc {rc} | {tail[0][:160]}")
    return rc


def pool(tasks, label):
    seen, uniq = set(), []
    for t in tasks:
        if t and t[0] not in seen:
            seen.add(t[0])
            uniq.append(t)
    if not uniq:
        return
    say(f"POOL {label}: {len(uniq)} tasks, {POOL} at a time")
    failed = []
    with ThreadPoolExecutor(POOL) as ex:
        futs = []
        for t in uniq:
            futs.append((t, ex.submit(lambda t=t: 99 if preempted() else _exec(t))))
        for t, f in futs:
            rc = f.result()
            if rc != 0:
                failed.append((t, rc))
    if preempted():
        say("PREEMPT requested -- yielding (finished tasks are cached)")
        print("YIELD", flush=True)
        sys.exit(10)
    for t, rc in failed:
        say(f"RETRY alone {t[0]} (rc {rc} under the parallel load)")
        assert _exec(t) == 0, f"task {t[0]} failed twice"


def judge(stage):
    if preempted():
        print("YIELD", flush=True)
        sys.exit(10)
    say(f"JUDGE {stage}")
    log = f"logs/agent_sb2_judge_{stage}.log"
    with open(log, "w") as fh:
        rc = subprocess.call([PY, "-u", "-W", "ignore", "archive/agent_sb2_judge.py", stage], stdout=fh, stderr=subprocess.STDOUT)
    txt = open(log, encoding="utf-8", errors="replace").read()
    with LOCK:
        print(txt, flush=True)
    assert rc == 0, f"judge {stage} failed rc {rc}"
    return json.load(open(VJ))


def prep_task(name, args, out):
    if os.path.exists(out):
        return None
    tmp = out[:-4] + "_tmp.npz" if out.endswith(".npz") else None
    return (name, args + (["--out", tmp] if tmp else []), f"logs/agent_sb2_{name}.log", out if tmp else None)


# ---------------------------------------------------------------- prep: shared inputs, computed once
for kind in ("hdow", "osm", "occ"):
    if not all(os.path.exists(I + f"agent_sb2_{kind}_{yr}.parquet") for yr in ("2025", "2026")):
        pool([(f"feat_{kind}", ["archive/agent_sb2_feat.py", kind], f"logs/agent_sb2_feat_{kind}.log", None)], f"feat {kind}")
prep = [prep_task("unc_TX", ["archive/agent_sb2_unc.py", "--fin-npz", TX], I + "agent_sb2_unc_TX.npz"),
        prep_task("unc_E9f", ["archive/agent_sb2_unc.py", "--fin-npz", E9F], I + "agent_sb2_unc_E9f.npz")]
if not all(os.path.exists(I + f) for f in ("agent_nov_fold.parquet", "agent_nov_train.parquet", "agent_nov_2026.parquet")):
    prep.append(("nov_feat", ["archive/agent_nov_01_feat.py"], "logs/agent_sb2_nov_feat.log", None))
prep += [fit_task(pc, set(), 0) for pc in ("bxTX", "bxE9f", "sq")]  # reproduction fits (cached from the first slot)
pool(prep, "prep + reproduction")

v = judge("repro")
assert v["repro_ok"], "REPRODUCTION FAILED -- STOP (s0.2)"
comps = ["mw", "c2", "lw", "unc", "osm", "occ"] + (["nov"] if v["nov_sanity"]["ok"] else [])
if not v["nov_sanity"]["ok"]:
    say("C9 novelty FAILED its sanity checks -> dropped before its screen (s0.8)")

# ---------------------------------------------------------------- step 1a: seed 0, all components
pool([t for c in comps for t in comp_tasks({c}, 0)], "step 1a seed-0 screens")
v = judge("screen0")
surv = v["survivors0"]
say(f"seed-0 survivors: {surv or 'none'}")
# ---------------------------------------------------------------- step 1b: seed 1, survivors only
if surv:
    pool([t for c in surv for t in comp_tasks({c}, 1)], "step 1b seed-1 screens")
    v = judge("screen1")
IN = v.get("IN", [])
say(f"STEP 1 IN: {IN or 'none'}")

# ---------------------------------------------------------------- step 2: bundle (+ one fallback)
final, name, comps_final = "FAIL", "", []
if IN:
    pool([t for s in (0, 1, 2) for t in comp_tasks(set(IN), s)], "step 2 bundle fits")
    v = judge("bundle")
    if v["bundle"]["pass"]:
        final, name, comps_final = "PASS_BUNDLE", "bundle", IN
    elif len(IN) >= 2:
        fb = [c for c in IN if c != v["worst_IN"]]
        say(f"FALLBACK (once): without {v['worst_IN']} -> {fb}")
        pool([t for s in (0, 1, 2) for t in comp_tasks(set(fb), s)], "fallback fits")
        v = judge("fallback")
        if v["fallback"]["pass"]:
            final, name, comps_final = "PASS_FALLBACK", "fallback", fb

v = json.load(open(VJ))
v["final"], v["final_components"] = final, comps_final
json.dump(v, open(VJ, "w"), indent=1, default=float)
toks = sorted(c for c in comps_final if c != "c2")
spec = {"final": final, "components": comps_final,
        "bx_tokens": [t for t in toks if t != "lw"], "sq_tokens": toks, "c2_after": "c2" in comps_final,
        "base_recipes": {"bx": BXREC, "sq": SQREC},
        "weights": {"mw": "archive/agent_sb2_common.month_weights: Jan 0.5*0.443/s_Jan, Jul 0.5*0.557/s_Jul, other 0.5/s_rest, "
                          "mean-normalised, shares from the re-stack's own training rows",
                    "lw": "2.5 on LIRF rows, SQ (LFPG/LIRF-serving) fit only"},
        "features": {"unc": "archive/agent_sb2_unc.py (fold: nested per fin base; serving: OOF train + full heads, in serve)",
                     "osm": ["data/interim/agent_sb2_osm_2025.parquet", "data/interim/agent_sb2_osm_2026.parquet"],
                     "occ": ["data/interim/agent_sb2_occ_2025.parquet", "data/interim/agent_sb2_occ_2026.parquet"],
                     "nov": ["data/interim/agent_nov_fold.parquet (nov_f<k> per outer fold)", "data/interim/agent_nov_train.parquet",
                             "data/interim/agent_nov_2026.parquet"],
                     "c2": "archive/agent_sb2_calib.py (fit on the refit composite's residuals, applied after)"},
        "how": "append the tokens to the recipe of archive/agent_sb2_restack.py / agent_sb2_restack4.py (load_rows(extra=...), "
               "fit(w=scripts/agent_sb2_common.weights(rows, tokens)), set unc_w / nov_w per outer fold as agent_sb2_bxnest.py)"}
json.dump(spec, open(I + "agent_sb2_bundle_spec.json", "w"), indent=1)
say(f"SB2 VERDICT: {final} {comps_final}")
print(f"FINAL={final}")
print(f"TOKENS={','.join(toks)}")
print(f"C2={1 if 'c2' in comps_final else 0}")
print(f"NAME={name}")
