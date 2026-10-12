"""experiment restack5 chain: the PRE-REGISTERED fold chain of docs/LAB-NOTES.md, run end to end with no
human choice in between. Resumable: a fit whose npz exists is not refitted; exits 10 at a checkpoint if the scheduler asks the
slot to yield (JOB_PREEMPT_FILE). Clause code (rm / clauses / show / per-airport clause) copied verbatim from
archive/agent_restack4_chain.py (== agent_seqsplice_02_judge.py == agent_expensive_06_judge.py).
Fits go through archive/agent_restack4_fold.py, unedited (stores rs_add / rs_dump).

    python archive/agent_restack5_chain.py      (inside a scheduler slot, under archive/capped_run.py --cap-gb 6)

Writes data/interim/agent_restack5_verdict.json, composite npz files data/interim/agent_restack5_TOP{k}.npz for the
accepted nest steps, and prints STEP=... lines for the build driver.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")
import numpy as np

I = "data/interim/"
PY = sys.executable
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
S0 = ["LEMD", "LFPG", "LIRF"]
R6 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LSZH"]
CTRL = I + "agent_restack4_ctrl.npz"
FOLD = "archive/agent_restack4_fold.py"
TOKORDER = ["cap", "evt", "trk", "seq", "nbr", "big"]
T0 = time.time()
VJ = I + "agent_restack5_verdict.json"
res = json.load(open(VJ)) if os.path.exists(VJ) else {}
PRE = os.environ.get("JOB_PREEMPT_FILE", "")


def say(m):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z +{time.time() - T0:5.0f}s] {m}", flush=True)


def save():
    json.dump(res, open(VJ, "w"), indent=1, default=float)


def checkpoint():
    if PRE and os.path.exists(PRE):
        save()
        say("PREEMPT requested by the scheduler -> stopping at a checkpoint (resumable; exit 10)")
        sys.exit(10)


def run(args, log):
    say("RUN " + " ".join(args) + f"  > {log}")
    with open(log, "w") as fh:
        rc = subprocess.call([PY, "-u"] + args, stdout=fh, stderr=subprocess.STDOUT)
    tail = open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-2:]
    for ln in tail:
        print("    | " + ln, flush=True)
    return rc


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys if k in z.files}


# ---------------------------------------------------------------- control
c0 = load(CTRL)
y, apts, days = c0["y"].astype(float), c0["apts"].astype(str), c0["days"].astype(str)
mon = np.array([d[5:7] for d in days])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])
pq = c0["both_guard"].astype(float)
N = len(y)
rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean())) if m.any() else float("nan")  # noqa: E731


def clauses(pt, base, base_name):
    d = lambda m: rm(pt, m) - rm(base, m)  # noqa: E731
    gain = (base - y) ** 2 - (pt - y) ** 2
    keep = np.ones(N, bool)
    order = np.argsort(-gain)[:50]
    keep[order] = False
    ch = pt != base
    h = [d(m) if (ch & m).any() else 0.0 for m in (H1, H2)]
    per = {a: d(apts == a) for a in AIRPORTS}
    c = {"vs": base_name, "all": d(np.ones(N, bool)), "worst_apt": max(per.values()), "m50": d(keep),
         "h1": h[0], "h2": h[1], "per": per, "rows_changed": int(ch.sum())}
    c["pass"] = c["all"] <= -0.10 and c["worst_apt"] <= 0.10 and c["m50"] < 0 and c["h1"] < 0 and c["h2"] < 0
    return c


def show(name, c):
    print(f"{name} (vs {c['vs']}): all {c['all']:+.4f}  worst airport {c['worst_apt']:+.3f}  minus-top-50 {c['m50']:+.3f}  "
          f"Jan-Mar {c['h1']:+.3f}  Jul-Sep {c['h2']:+.3f}  rows changed {c['rows_changed']:,}  "
          f"4-clause {'PASS' if c['pass'] else 'FAIL'}", flush=True)
    print("   by airport: " + "  ".join(f"{a} {v:+.3f}" for a, v in c["per"].items()), flush=True)


def per_airport_clause(pt, base, domain):
    """seqsplice s0.2 clause, verbatim logic, over the candidate's domain only."""
    gain0 = (base - y) ** 2 - (pt - y) ** 2
    S, per = [], {}
    for a in AIRPORTS:
        if a not in domain:
            continue
        m = apts == a
        d_all = rm(pt, m) - rm(base, m)
        d_h1 = rm(pt, m & H1) - rm(base, m & H1) if (m & H1).any() else float("nan")
        d_h2 = rm(pt, m & H2) - rm(base, m & H2) if (m & H2).any() else float("nan")
        idx_a = np.flatnonzero(m)
        top10 = idx_a[np.argsort(-gain0[idx_a])[:10]]
        keep = m.copy()
        keep[top10] = False
        d_m10 = rm(pt, keep) - rm(base, keep)
        ok1, ok2, ok3 = d_all <= -0.10, (d_h1 < 0 and d_h2 < 0), d_m10 < 0
        per[a] = {"d_all": d_all, "d_h1": d_h1, "d_h2": d_h2, "d_minus_top10": d_m10, "included": bool(ok1 and ok2 and ok3)}
        print(f"  {a}: d_all={d_all:+.3f} (<=-0.10 {ok1})  Jan-Mar={d_h1:+.3f} Jul-Sep={d_h2:+.3f} (both<0 {ok2})  "
              f"minus-top10={d_m10:+.3f} (<0 {ok3})  -> {'IN' if ok1 and ok2 and ok3 else 'OUT'}", flush=True)
        if ok1 and ok2 and ok3:
            S.append(a)
    return S, per


# ---------------------------------------------------------------- fits (registry of existing equivalents, else new)
def canon(toks):
    return ",".join(t for t in TOKORDER if t in toks)


EXIST = {("cap,evt", 0): I + "agent_restack3_CA.npz",
         ("cap,evt,trk,seq", 0): I + "agent_smallbundle_B.npz",
         ("cap,evt,trk,seq", 1): I + "agent_seqsplice_seed1_B.npz",
         ("cap,evt,trk,seq", 2): I + "agent_seqsplice_seed2_B.npz",
         ("cap,evt,trk,nbr", 0): I + "agent_neighbour_B.npz",
         ("cap,evt,trk,nbr", 1): I + "agent_neighbour_B_s1.npz"}
CBDROP = set()


def fitpath(recipe, seed, cb=False):
    """npz of the re-stack fit (recipe, seed); cb=True -> the CatBoost fit on the same matrix. None if the cb guard fired."""
    if not cb and (recipe, seed) in EXIST:
        return EXIST[(recipe, seed)]
    if cb and recipe in CBDROP:
        return None
    tag = recipe.replace(",", "_") + ("_cb" if cb else "") + f"_s{seed}"
    if recipe == "cap,evt,trk" and seed == 0 and not cb:
        tag = "L0"
    out = I + f"agent_restack5_f_{tag}.npz"
    if not os.path.exists(out):
        rc = run([FOLD, "--recipe", recipe + (",cb" if cb else ""), "--seed", str(seed), "--out", out],
                 f"logs/agent_restack5_fold_{tag}.log")
        if rc == 3:
            CBDROP.add(recipe)
            res.setdefault("cb_guard_fired", []).append(recipe)
            save()
            return None
        assert rc == 0, f"fold {tag} failed rc={rc}"
    return out


_cache = {}


def arr(path, key="both_guard"):
    k = (path, key)
    if k not in _cache:
        z = np.load(path, allow_pickle=True)
        o = np.argsort(z["ids"], kind="stable")
        assert (z["ids"][o] == c0["ids"]).all(), f"{path}: id mismatch"
        if key == "add" and "rs_add" not in z.files:
            v = z["both_guard"][o].astype(float) - z["both_guard_prers"][o].astype(float)
        else:
            v = z["rs_add" if key == "add" else key][o].astype(float)
        _cache[k] = v
    return _cache[k]


# ---------------------------------------------------------------- 0.1 checks
if not os.path.exists(I + "agent_restack5_ctrl_chk.npz"):
    assert run(["archive/splice_npz.py", I + "agent_restack3_CAB.npz", I + "agent_smallbundle_B.npz", "LEMD,LFPG,LIRF",
                I + "agent_restack5_ctrl_chk.npz"], "logs/agent_restack5_ctrl_chk.log") == 0
dchk = float(np.abs(arr(I + "agent_restack5_ctrl_chk.npz") - pq).max())
say(f"CTRL rebuild check: max |diff| {dchk}")
assert dchk == 0.0, "CTRL is not the CAB + seq-arm splice -- stop"
cab = arr(I + "agent_restack3_CAB.npz")
cc = clauses(pq, cab, "CAB")
show("SANITY CTRL", cc)
assert abs(cc["all"] - (-0.558)) < 0.0015 and abs(cc["m50"] - (-0.124)) < 0.0015, "CTRL does not reproduce seqsplice s1.3"
res["sanity_ctrl_vs_cab"] = cc
save()

L0 = fitpath("cap,evt,trk", 0)
dI0 = float(np.abs(arr(L0) - cab).max())
say(f"IDENTITY I0 (cap,evt,trk seed 0 through agent_restack4_fold.py) vs agent_restack3_CAB.npz: max |diff| {dI0}")
res["identity_I0"] = dI0
save()
assert dI0 == 0.0, "identity I0 failed -- stop"
fin = arr(L0, "both_guard_prers")
dump = np.load(L0, allow_pickle=True)
dump = dump["rs_dump"][np.argsort(dump["ids"], kind="stable")].astype(bool)
assert (pq[~dump] == fin[~dump]).all(), "CTRL moves non-dump rows?"


# ---------------------------------------------------------------- the generic construction
def base_toks(a):
    return {"cap", "evt", "trk"} | ({"seq"} if a in S0 else set())


MODS = {"ntrk": lambda t: t - {"trk"}, "nbr": lambda t: t | {"nbr"}, "big": lambda t: t | {"big"},
        "cb": lambda t: t, "bag": lambda t: t}


def group_pred(recipe, cb, bag, rep):
    """prediction of one union group at replicate rep (0, 1, 2); None if a CatBoost fit was dropped by its guard."""
    seeds = [3 * rep, 3 * rep + 1, 3 * rep + 2] if bag else [rep]
    paths = [fitpath(recipe, s) for s in seeds]
    if not cb:
        return np.mean([arr(p) for p in paths], axis=0)
    cz = fitpath(recipe, seeds[0], cb=True)
    if cz is None:
        return None
    addL = np.mean([arr(p, "add") for p in paths], axis=0)
    addC = arr(cz, "add")
    out = fin.copy()
    out[dump] = np.maximum(fin[dump] + 0.5 * addL[dump] + 0.5 * addC[dump], 0.0)
    return out


def groups_for(members, SX, X):
    """{(recipe, cb, bag): [airports]} for the airports of SX, applying every member covering the airport plus X."""
    g = {}
    for a in SX:
        toks, cb, bag = base_toks(a), False, False
        for tag, S in members + [(X, SX)]:
            if a in S:
                toks = MODS[tag](toks)
                cb |= tag == "cb"
                bag |= tag == "bag"
        g.setdefault((canon(toks), cb, bag), []).append(a)
    return g


def construct(base_pred, members, SX, X, rep):
    p = base_pred.copy()
    for (rec, cb, bag), aa in groups_for(members, SX, X).items():
        gp = group_pred(rec, cb, bag, rep)
        if gp is None:
            return None
        m = np.isin(apts, aa)
        p[m] = gp[m]
    return p


DOMAIN = {"ntrk": R6, "nbr": R6, "cb": R6, "big": AIRPORTS}
res.setdefault("stage1", {})
save()


def stage1(X):
    """0.3 steps 1-2 on seed 0; returns S_X (or None)."""
    D = DOMAIN[X]
    say(f"== STAGE 1 {X}: domain {D}")
    p = construct(pq, [], D, X, 0)
    if p is None:
        res["stage1"][X] = {"verdict": "DROPPED (CatBoost compute guard)"}
        save()
        return None
    c = clauses(p, pq, "CTRL")
    show(f"{X} whole", c)
    r = {"whole": c, "domain": D}
    S = list(D) if c["pass"] else None
    if S is None:
        say(f"{X} whole FAILS -> per-airport splice fallback over {D}")
        Ssel, per = per_airport_clause(p, pq, D)
        r["per_airport"] = per
        r["fallback_set"] = Ssel
        if Ssel:
            cs = clauses(construct(pq, [], Ssel, X, 0), pq, "CTRL")
            show(f"{X} splice {','.join(Ssel)}", cs)
            r["splice"] = cs
            if cs["pass"]:
                S = Ssel
        else:
            say(f"{X}: empty fallback set")
    r["S_seed0"] = S
    r["d_seed0"] = (r["whole"] if S == list(D) else r.get("splice", {})).get("all") if S else None
    res["stage1"][X] = r
    save()
    say(f"{X} seed 0: {'PASS at ' + ','.join(S) if S else 'FAIL'}")
    return S


def stability(X, S):
    ok = True
    for rep in (1, 2):
        p = construct(pq, [], S, X, rep)
        if p is None:
            res["stage1"][X][f"rep{rep}"] = "DROPPED (CatBoost compute guard)"
            ok = False
            break
        c = clauses(p, pq, "CTRL")
        show(f"{X} replicate {rep} at {','.join(S)}", c)
        res["stage1"][X][f"rep{rep}"] = c
        ok = ok and c["pass"]
        save()
        if not ok:
            break
    res["stage1"][X]["stable"] = ok
    res["stage1"][X]["verdict"] = "PASS" if ok else "FAIL (seed stability)"
    save()
    say(f"{X} 3-seed stability: {'PASS' if ok else 'FAIL'}")
    return ok


seed0 = {}
# ---- A: no new fits (C1 ntrk, C4 nbr)
for X in ("ntrk", "nbr"):
    seed0[X] = stage1(X)

# ---- B: C2 bag, spread gate first (needs cap,evt,trk seeds 1, 2)
say("== C2 spread gate: Q_s = cap,evt,trk seed s at R6 + cap,evt,trk,seq seed s at S0, s = 0, 1, 2")
Qs = [None, None, None]
for s in range(3):
    q = pq.copy()
    for rec, aa in (("cap,evt,trk", R6), ("cap,evt,trk,seq", S0)):
        m = np.isin(apts, aa)
        q[m] = arr(fitpath(rec, s))[m]
    Qs[s] = q
spread = {}
for a in AIRPORTS:
    m = apts == a
    r_ = [rm(q, m) for q in Qs]
    spread[a] = {"rmse_by_seed": r_, "spread": max(r_) - min(r_)}
    print(f"  {a}: RMSE seeds 0/1/2 = {r_[0]:.3f} / {r_[1]:.3f} / {r_[2]:.3f}  spread {max(r_) - min(r_):.3f}", flush=True)
for s in (1, 2):
    c = clauses(Qs[s], pq, "CTRL")
    show(f"(noise reference) Q_{s}", c)
    spread[f"Q{s}_vs_CTRL"] = c
NZ = [a for a in AIRPORTS if spread[a]["spread"] > 0.10]
res["C2_spread"] = spread
res["C2_NZ"] = NZ
save()
say(f"C2 noisy set NZ (spread > 0.10 s) = {NZ or '(empty)'}")
if NZ:
    DOMAIN["bag"] = NZ
    seed0["bag"] = stage1("bag")
else:
    res["stage1"]["bag"] = {"verdict": "NOT RUN (spread gate: no airport > 0.10 s)"}
    seed0["bag"] = None
    save()

# ---- C: C5 big
seed0["big"] = stage1("big")

# ---- D: C3 cb, diversity gate first
say("== C3 diversity gate")
cz = fitpath("cap,evt,trk", 0, cb=True)
if cz is None:
    res["stage1"]["cb"] = {"verdict": "DROPPED (CatBoost compute guard: first fold > 15 min), before any number"}
    seed0["cb"] = None
else:
    mR = dump & np.isin(apts, R6)
    rho = float(np.corrcoef(arr(L0, "add")[mR], arr(cz, "add")[mR])[0, 1])
    res["C3_rho"] = rho
    say(f"DIVERSITY: corr(add_LightGBM, add_CatBoost) over {int(mR.sum()):,} R6 dump rows = {rho:.4f} (gate < 0.95)")
    pcb = pq.copy()
    mm = np.isin(apts, R6)
    pcb[mm] = arr(cz)[mm]
    ci = clauses(pcb, pq, "CTRL")
    show("(info) CatBoost re-stack alone at R6", ci)
    res["C3_cb_alone_info"] = ci
    if rho < 0.95:
        seed0["cb"] = stage1("cb")
    else:
        res["stage1"]["cb"] = {"verdict": f"NOT JUDGED (rho {rho:.4f} >= 0.95)"}
        seed0["cb"] = None
save()

# ---- E: 3-seed stability for the seed-0 passers (cheapest first)
checkpoint()  # a scheduler yield is honoured only at phase boundaries (starved queues flag every preemptible slot at once)
passers = {}
for X in ("ntrk", "nbr", "big", "bag", "cb"):
    S = seed0.get(X)
    if S:
        if stability(X, S):
            passers[X] = (S, res["stage1"][X]["d_seed0"])
    elif X in res["stage1"] and "verdict" not in res["stage1"][X]:
        res["stage1"][X]["verdict"] = "FAIL"
save()
say(f"STAGE 1 passers: {passers or 'none'}")

# ---------------------------------------------------------------- 0.4 stage 2 nesting
checkpoint()
order = sorted(passers, key=lambda k: passers[k][1])
res["stage2_order"] = order
TOP, members, steps = pq.copy(), [], []
U = set()
top_npz_prev = CTRL
for X in order:
    if len(steps) >= 3:
        break
    SX = passers[X][0]
    say(f"== STAGE 2: {X} at {SX} on TOP {[m[0] for m in members] or 'CTRL'} (touched {sorted(U) or '-'})")
    ok, cs = True, {}
    for rep in (0, 1, 2):
        p = construct(TOP, members, SX, X, rep)
        if p is None:
            ok = False
            cs[f"rep{rep}"] = "DROPPED (CatBoost compute guard)"
            break
        c = clauses(p, TOP, "TOP")
        show(f"{X}-on-TOP replicate {rep}", c)
        cs[f"rep{rep}"] = c
        if rep == 0:
            p0 = p
            cc0 = clauses(p, pq, "CTRL")
            show(f"(cumulative) {X}-on-TOP replicate 0", cc0)
            cs["cumulative_vs_CTRL"] = cc0
        ok = ok and c["pass"]
        if not ok:
            break
    cs["pass"] = ok
    res.setdefault("stage2", {})[X] = cs
    save()
    if not ok:
        say(f"{X} FAILS on TOP -> dropped")
        continue
    k = len(steps) + 1
    out = I + f"agent_restack5_TOP{k}.npz"
    zc = np.load(CTRL, allow_pickle=True)
    o = np.argsort(zc["ids"], kind="stable")
    np.savez(out, ids=zc["ids"][o], y=zc["y"][o], days=zc["days"][o], apts=zc["apts"][o], both_guard=p0)
    grp = groups_for(members, SX, X)
    steps.append({"k": k, "tag": X, "airports": SX, "groups": [{"recipe": r, "cb": cb, "bag": bag, "airports": aa}
                                                             for (r, cb, bag), aa in grp.items()],
                  "npz_before": top_npz_prev, "npz_after": out, "d_step": cs["rep0"]["all"],
                  "d_cum": cs["cumulative_vs_CTRL"]["all"]})
    TOP, top_npz_prev = p0, out
    members.append((X, SX))
    U |= set(SX)
    say(f"ACCEPTED step {k}: {X} at {SX}")
res["steps"] = steps
save()
if not steps:
    say("no increment passes -> nothing built")
    print("FINAL=NONE")
    sys.exit(0)
for st in steps:
    print("STEP=" + json.dumps(st))
print(f"FINAL={len(steps)}")
