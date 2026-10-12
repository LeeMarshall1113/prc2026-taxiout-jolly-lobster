"""experiment restack4 chain: the PRE-REGISTERED fold chain of docs/LAB-NOTES.md, run end to end with no
human choice in between. Resumable: a step whose output npz exists is not refitted. Clause code (rm / clauses / show and
the per-airport clause) copied verbatim from archive/agent_seqsplice_02_judge.py (== agent_expensive_06_judge.py).

    python archive/agent_restack4_chain.py            (inside a scheduler slot, under archive/capped_run.py)

Writes data/interim/agent_restack4_verdict.json (every comparison, S1, BEST, TOP, FINAL, seed checks, rho) and prints
FINAL=<tag>|<recipe>|<S1 csv>|<lgb npz>|<cb npz or ->.
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
CTRL = I + "agent_restack4_ctrl.npz"
T0 = time.time()
VJ = I + "agent_restack4_verdict.json"
res = json.load(open(VJ)) if os.path.exists(VJ) else {}


def say(m):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z +{time.time() - T0:5.0f}s] {m}", flush=True)


def save():
    json.dump(res, open(VJ, "w"), indent=1, default=float)


def run(args, log):
    say("RUN " + " ".join(args) + f"  > {log}")
    with open(log, "w") as fh:
        rc = subprocess.call([PY, "-u"] + args, stdout=fh, stderr=subprocess.STDOUT)
    tail = open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-3:]
    for ln in tail:
        print("    | " + ln, flush=True)
    return rc


def fold(tag, recipe, seed=0):
    out = I + f"agent_restack4_{tag}.npz"
    if not os.path.exists(out):
        rc = run(["archive/agent_restack4_fold.py", "--recipe", recipe, "--seed", str(seed), "--out", out],
                 f"logs/agent_restack4_fold_{tag}.log")
        if rc == 3:
            return None
        assert rc == 0, f"fold {tag} failed rc={rc}"
    return out


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys if k in z.files}


# ---------------------------------------------------------------- 0.1 control
if not os.path.exists(CTRL):
    assert run(["archive/splice_npz.py", I + "agent_restack3_CAB.npz", I + "agent_smallbundle_B.npz", "LEMD,LFPG,LIRF", CTRL],
               "logs/agent_restack4_ctrl.log") == 0
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


def per_airport_clause(pt, base):
    """seqsplice s0.2 clause, verbatim logic: d(A) <= -0.10; both halves < 0; minus A's own top-10 helped rows < 0."""
    gain0 = (base - y) ** 2 - (pt - y) ** 2
    S, per = [], {}
    for a in AIRPORTS:
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


def P(npz):
    z = load(npz)
    assert (z["ids"] == c0["ids"]).all(), f"{npz}: id mismatch"
    return z["both_guard"].astype(float)


def sp(p, S):  # the arm evaluated at S (CTRL elsewhere)
    return p if set(S) == set(AIRPORTS) else np.where(np.isin(apts, S), p, pq)


# sanity: CTRL vs CAB reproduces seqsplice s1.3
cab = P(I + "agent_restack3_CAB.npz")
cc = clauses(pq, cab, "CAB")
show("SANITY CTRL", cc)
assert abs(cc["all"] - (-0.558)) < 0.0015 and abs(cc["m50"] - (-0.124)) < 0.0015, "CTRL does not reproduce seqsplice s1.3"
res["sanity_ctrl_vs_cab"] = cc

# ---------------------------------------------------------------- 0.2 identity
for tag, rec, ref in (("I1", "cap,evt,trk,seq", "agent_smallbundle_B.npz"), ("I2", "cap,evt,trk,nbr", "agent_neighbour_B.npz")):
    d = float(np.abs(P(fold(tag, rec)) - P(I + ref)).max())
    say(f"IDENTITY {tag} ({rec}) vs {ref}: max |diff| {d}")
    res[f"identity_{tag}"] = d
    save()
    assert d == 0.0, f"identity {tag} failed -- stop (s0.2)"

# ---------------------------------------------------------------- V1
R1 = "cap,evt,trk,seqm,nbr"
p1 = P(fold("V1", R1))
say("== V1 JOINT vs CTRL")
c = clauses(p1, pq, "CTRL")
show("V1 whole", c)
res["V1_vs_CTRL"] = c
S1 = list(AIRPORTS) if c["pass"] else None
if S1 is None:
    say("V1 whole FAILS -> pre-registered per-airport splice fallback (seqsplice s0.2 clause)")
    Ssel, per = per_airport_clause(p1, pq)
    res["V1_per_airport"] = per
    if Ssel:
        cs = clauses(sp(p1, Ssel), pq, "CTRL")
        show(f"V1 splice {','.join(Ssel)}", cs)
        res["V1S_vs_CTRL"] = cs
        if cs["pass"]:
            S1 = Ssel
res["S1"] = S1
save()
V1ok = S1 is not None
say(f"V1 {'PASS' if V1ok else 'FAIL'}; S1 = {S1}")
S = S1 if V1ok else list(AIRPORTS)   # V2 is still judged when V1 fails (at all airports: no S1 exists)
steps = {"V1": {"recipe": R1, "npz": I + "agent_restack4_V1.npz", "pass": V1ok, "pred": "CTRL"}}

# ---------------------------------------------------------------- ARR columns (label-free)
for yr in ("2025", "2026"):
    f = I + f"agent_restack4_arr_{yr}.parquet"
    if not os.path.exists(f):
        assert run(["archive/agent_restack4_arr.py", "--mode", yr, "--out", f], f"logs/agent_restack4_arr_{yr}.log") == 0

# ---------------------------------------------------------------- V2
R2 = R1 + ",arr"
p2 = P(fold("V2", R2))
say("== V2 (+arr)")
a_ = clauses(sp(p2, S), pq, "CTRL")
b_ = clauses(sp(p2, S), sp(p1, S), "V1")
show("V2", a_)
show("V2", b_)
res["V2_vs_CTRL"], res["V2_vs_V1"] = a_, b_
V2ok = a_["pass"] and b_["pass"]
steps["V2"] = {"recipe": R2, "npz": I + "agent_restack4_V2.npz", "pass": V2ok, "pred": "V1"}
BEST = "V2" if V2ok else ("V1" if V1ok else None)
res["BEST"] = BEST
save()
say(f"V2 {'PASS' if V2ok else 'FAIL'}; BEST(V1,V2) = {BEST}")
if BEST is None:
    res["FINAL"] = None
    save()
    say("nothing passes -> V3/V4 not run, nothing built")
    print("FINAL=NONE")
    sys.exit(0)
if not V1ok:  # V2 passed although V1 failed: V2 is judged at all airports
    S1 = list(AIRPORTS)
    res["S1"] = S1
S = S1

# ---------------------------------------------------------------- V3
R3 = steps[BEST]["recipe"] + ",big"
p3 = P(fold("V3", R3))
pb = P(steps[BEST]["npz"])
say("== V3 (+big)")
a_ = clauses(sp(p3, S), pq, "CTRL")
b_ = clauses(sp(p3, S), sp(pb, S), BEST)
show("V3", a_)
show("V3", b_)
res["V3_vs_CTRL"], res[f"V3_vs_{BEST}"] = a_, b_
V3ok = a_["pass"] and b_["pass"]
steps["V3"] = {"recipe": R3, "npz": I + "agent_restack4_V3.npz", "pass": V3ok, "pred": BEST}
TOP = "V3" if V3ok else BEST
res["TOP"] = TOP
save()
say(f"V3 {'PASS' if V3ok else 'FAIL'}; TOP = {TOP}")

# ---------------------------------------------------------------- V4 (diversity gate first)
feat_rec = ",".join(t for t in steps[TOP]["recipe"].split(",") if t not in ("cap", "big"))


def cb_fold(seed):
    return fold(f"CB_s{seed}", feat_rec + ",cb", seed)


def blend(lgb_npz, cb_npz, out):
    if not os.path.exists(out):
        zl = np.load(lgb_npz, allow_pickle=True)
        zc = np.load(cb_npz, allow_pickle=True)
        assert (zl["ids"] == zc["ids"]).all() and (zl["rs_dump"] == zc["rs_dump"]).all()
        assert (zl["both_guard_prers"] == zc["both_guard_prers"]).all()
        Z = {k: zl[k] for k in zl.files}
        dm = zl["rs_dump"]
        new = zl["both_guard_prers"].astype(float).copy()
        new[dm] = np.maximum(new[dm] + 0.5 * zl["rs_add"][dm] + 0.5 * zc["rs_add"][dm], 0.0)
        Z["both_guard"] = new
        Z["rs_add"] = 0.5 * zl["rs_add"] + 0.5 * zc["rs_add"]
        np.savez(out, **Z)
    return out


steps["V4"] = {"recipe": steps[TOP]["recipe"] + "+cb", "npz": None, "pass": False, "pred": TOP}
cbz = cb_fold(0)
if cbz is None:
    res["V4"] = "dropped: CatBoost compute guard (first fold > 15 min), before any number"
    say(res["V4"])
else:
    zl, zc = np.load(steps[TOP]["npz"], allow_pickle=True), np.load(cbz, allow_pickle=True)
    dm = zl["rs_dump"]
    rho = float(np.corrcoef(zl["rs_add"][dm], zc["rs_add"][dm])[0, 1])
    res["rho_TOP_CB"] = rho
    say(f"DIVERSITY: corr(add_{TOP}, add_CB) over {int(dm.sum()):,} dump rows = {rho:.4f} (gate < 0.95)")
    if rho < 0.95:
        v4 = blend(steps[TOP]["npz"], cbz, I + "agent_restack4_V4.npz")
        p4, pt = P(v4), P(steps[TOP]["npz"])
        say("== V4 (50/50 LightGBM TOP + CatBoost)")
        a_ = clauses(sp(p4, S), pq, "CTRL")
        b_ = clauses(sp(p4, S), sp(pt, S), TOP)
        show("V4", a_)
        show("V4", b_)
        res["V4_vs_CTRL"], res[f"V4_vs_{TOP}"] = a_, b_
        steps["V4"]["npz"] = v4
        steps["V4"]["pass"] = a_["pass"] and b_["pass"]
        say(f"V4 {'PASS' if steps['V4']['pass'] else 'FAIL'}")
    else:
        res["V4"] = f"not judged: rho {rho:.4f} >= 0.95"
        say(res["V4"])
res["steps"] = steps
save()

# ---------------------------------------------------------------- 0.6 seed stability of FINAL (cascade)
order = [s for s in ("V4", "V3", "V2", "V1") if steps.get(s, {}).get("pass")]
# the nest path: FINAL's predecessors in order
path_of = {"V1": ["V1"], "V2": (["V1", "V2"] if V1ok else ["V2"])}
path_of["V3"] = path_of[BEST] + ["V3"]
path_of["V4"] = path_of[TOP] + ["V4"]
FINAL = None
cand = order[0] if order else None
tried = []
while cand is not None:
    tried.append(cand)
    say(f"== seed check of {cand}")
    ok = True
    for s in (1, 2):
        if cand == "V4":
            lg = fold(f"{TOP}_s{s}", steps[TOP]["recipe"], s)
            cz = cb_fold(s)
            if cz is None:
                ok = False
                break
            pz = blend(lg, cz, I + f"agent_restack4_V4_s{s}.npz")
        else:
            pz = fold(f"{cand}_s{s}", steps[cand]["recipe"], s)
        cs = clauses(sp(P(pz), S), pq, "CTRL")
        show(f"{cand} seed{s}", cs)
        res[f"seed_{cand}_s{s}"] = cs
        ok = ok and cs["pass"]
    save()
    if ok:
        FINAL = cand
        break
    prev = path_of[cand][:-1]
    cand = prev[-1] if prev else None
res["FINAL"] = FINAL
res["FINAL_path"] = path_of[FINAL] if FINAL else []
res["seed_tried"] = tried
save()
if FINAL is None:
    say("no step survives the seed check -> nothing built")
    print("FINAL=NONE")
    sys.exit(0)
say(f"FINAL = {FINAL}; nest path {path_of[FINAL]}; S1 = {S1}")
for st in path_of[FINAL]:
    print(f"STEP={st}|{steps[st]['recipe']}|{','.join(S1)}|{steps[st]['npz']}")
print(f"FINAL={FINAL}")
