"""experiment sb2 chain: the PRE-REGISTERED small-bundle test of docs/LAB-NOTES.md run end to end with no human
choice in between. Resumable: every fit npz / feature parquet that exists is reused (written to a .tmp name and renamed
only after rc 0). Between steps it checks JOB_PREEMPT_FILE and exits 10 (yield) with everything cached.
Clause code (rm / clauses / show) copied verbatim from archive/agent_restack6_chain.py.

    python archive/agent_sb2_chain.py      (inside a scheduler slot, under archive/capped_run.py)

Writes data/interim/agent_sb2_verdict.json; prints FINAL=PASS|FAIL, TOKENS=..., C2=0|1 for the build driver.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_sb2_calib import oof as c2_oof

I = "data/interim/"
PY = sys.executable
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
SBX = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LSZH"]
SQ = ["LFPG", "LIRF"]
M = ["EDDF", "EDDM", "EHAM", "LEBL", "LSZH"]
TX, E9F = I + "modal/TX_S6.npz", I + "agent_stagecap_CAND_E9f.npz"
BXREC, SQREC = "cap,evt,trk,bx,big", "cap,evt,trk,seq,big"
COMPONENTS = ["mw", "c2", "lw", "unc", "osm", "occ"]  # C1, C2, C3, C4, C6, C7 (C5 dropped, C8 parked: s0.1)
CNAME = {"mw": "C1 month-mix weight", "c2": "C2 (apt,hour,dow) OOF calibration", "lw": "C3 soft LIRF weight",
         "unc": "C4 uncertainty width", "osm": "C6 OSM stand trust", "occ": "C7 ground occupancy"}
T0 = time.time()
VJ = I + "agent_sb2_verdict.json"
res = json.load(open(VJ)) if os.path.exists(VJ) else {}


def say(m):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z +{time.time() - T0:5.0f}s] {m}", flush=True)


def save():
    json.dump(res, open(VJ, "w"), indent=1, default=float)


def preempt():
    f = os.environ.get("JOB_PREEMPT_FILE", "")
    if f and os.path.exists(f):
        save()
        say("PREEMPT requested by the scheduler -- yielding at a step boundary (everything so far is cached)")
        print("YIELD", flush=True)
        sys.exit(10)


def run(args, log):
    preempt()
    say("RUN " + " ".join(args) + f"  > {log}")
    with open(log, "w") as fh:
        rc = subprocess.call([PY, "-u", "-W", "ignore"] + args, stdout=fh, stderr=subprocess.STDOUT)
    for ln in open(log, encoding="utf-8", errors="replace").read().strip().splitlines()[-2:]:
        print("    | " + ln, flush=True)
    return rc


def load(p, keys=("ids", "y", "days", "apts", "both_guard")):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys if k in z.files}


c0 = load(I + "agent_stagecap_RS_E9S_s0.npz")
y, apts, days = c0["y"].astype(float), c0["apts"].astype(str), c0["days"].astype(str)
mon = np.array([d[5:7] for d in days])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])
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


def P(path):
    z = load(path, ("ids", "both_guard"))
    assert (z["ids"] == c0["ids"]).all(), f"{path}: id mismatch"
    return z["both_guard"].astype(float)


def sp(base, arm, aa):
    return np.where(np.isin(apts, aa), arm, base)


DNBR = P(I + "agent_blend_oof_seedA.npz") - P(I + "agent_blend_v109analogue.npz")
CTRL0 = P(I + "agent_stagecap_RS_E9S_s0.npz") + DNBR
TXP = P(TX)

# ---- label-free features (C2 hour/dow, C6, C7)
for kind in ("hdow", "osm", "occ"):
    if not all(os.path.exists(I + f"agent_sb2_{kind}_{yr}.parquet") for yr in ("2025", "2026")):
        assert run(["archive/agent_sb2_feat.py", kind], f"logs/agent_sb2_feat_{kind}.log") == 0, f"feature {kind} failed"
hd = (pl.DataFrame({"MVT_ID_mvt": c0["ids"].astype(float)})
      .join(pl.read_parquet(I + "agent_sb2_hdow_2025.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left"))
HOUR = hd["hour"].fill_null(-1).to_numpy().astype(int)
DOW = hd["dow"].fill_null(-1).to_numpy().astype(int)
res["hdow_coverage"] = float((HOUR >= 0).mean())
say(f"hour/dow found for {res['hdow_coverage']:.4f} of fold rows")


def tokstr(toks):
    return "_".join(sorted(toks)) if toks else "none"


def fitfile(piece, toks, seed):
    """piece in bxTX, bxE9f, sq; toks = the NEW tokens only. Existing production fits when toks is empty and seed > 0."""
    if not toks and seed > 0:
        return {"bxTX": I + f"agent_restack5_bxn_cap_evt_trk_bx_big_s{seed}.npz",
                "bxE9f": I + f"agent_stagecap_bxn_E9f_s{seed}.npz",
                "sq": I + f"agent_restack5_f_cap_evt_trk_seq_big_s{seed}.npz"}[piece]
    out = I + f"agent_sb2_{piece}_{tokstr(toks)}_s{seed}.npz"
    if os.path.exists(out):
        return out
    rec = (SQREC if piece == "sq" else BXREC) + "".join("," + t for t in sorted(toks))
    fin = E9F if piece == "bxE9f" else TX
    unc = I + ("agent_sb2_unc_E9f.npz" if piece == "bxE9f" else "agent_sb2_unc_TX.npz")
    if "unc" in toks:
        ensure_unc()
    tmp = out[:-4] + "_tmp.npz"
    if piece == "sq":
        args = ["archive/agent_sb2_fold4.py", "--fin-npz", fin, "--recipe", rec, "--seed", str(seed), "--unc-npz", unc,
                "--out", tmp]
    else:
        args = ["archive/agent_sb2_bxnest.py", "--fin-npz", fin, "--recipe", rec, "--seed", str(seed), "--unc-npz", unc,
                "--out", tmp]
    rc = run(args, f"logs/agent_sb2_{piece}_{tokstr(toks)}_s{seed}.log")
    assert rc == 0, f"fit {piece} {rec} s{seed} failed rc={rc}"
    os.replace(tmp, out)
    return out


def ensure_unc():
    for tag, fin in (("TX", TX), ("E9f", E9F)):
        out = I + f"agent_sb2_unc_{tag}.npz"
        if not os.path.exists(out):
            tmp = out[:-4] + "_tmp.npz"
            assert run(["archive/agent_sb2_unc.py", "--fin-npz", fin, "--out", tmp], f"logs/agent_sb2_unc_{tag}.log") == 0
            os.replace(tmp, out)


def compose(toks, seed, c2=False, pre_out=None):
    """re-stack composition (s0.2) with the NEW tokens `toks` (lw only in the SQ fit) + Dnbr, then C2 if asked."""
    bt = set(toks) - {"lw"}
    bx_tx, bx_e9, sq = P(fitfile("bxTX", bt, seed)), P(fitfile("bxE9f", bt, seed)), P(fitfile("sq", set(toks), seed))
    p = sp(sp(sp(TXP, bx_tx, SBX), sq, SQ), bx_e9, M) + DNBR
    if pre_out:  # the pre-C2 composite (C2's serving fit uses it)
        np.savez(pre_out, ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=p)
    if c2:
        p, _, kap = c2_oof(p, y, apts, days, HOUR, DOW)
        res.setdefault("c2_kappas", {})[f"{tokstr(toks)}_s{seed}"] = kap
    return p


# ================================================================ step R: reproduction (s0.2)
if not res.get("repro_ok"):
    ref = {"bxTX": I + "agent_restack5_bxn_cap_evt_trk_bx_big_s0.npz", "bxE9f": I + "agent_stagecap_bxn_E9f_s0.npz",
           "sq": I + "agent_restack5_f_cap_evt_trk_seq_big_s0.npz"}
    rep = {}
    for piece, rp in ref.items():
        rep[piece] = float(np.abs(P(fitfile(piece, set(), 0)) - P(rp)).max())
        say(f"REPRO {piece}: sb2 no-token seed-0 fit vs {rp}: max |diff| {rep[piece]}")
    v9 = sp(sp(TXP, P(ref["bxTX"]), SBX), P(ref["sq"]), SQ)
    rep["V9TOP1_vs_composition"] = float(np.abs(P(I + "agent_restack5_V9TOP1.npz") - v9).max())
    rep["CTRL0_vs_sb2_composition"] = float(np.abs(compose(set(), 0) - CTRL0).max())
    say(f"REPRO V9TOP1 == composition of the production pieces: max |diff| {rep['V9TOP1_vs_composition']}; "
        f"CTRL_0 == sb2 composition + Dnbr: max |diff| {rep['CTRL0_vs_sb2_composition']}")
    res["repro"] = rep
    res["repro_ok"] = all(v == 0.0 for v in rep.values())
    save()
    if not res["repro_ok"]:
        say("REPRODUCTION FAILED -- STOP (s0.2): nothing else runs")
        print("FINAL=FAIL")
        sys.exit(2)
np.savez(I + "agent_sb2_ctrl_r0.npz", ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=CTRL0)

PCTRL = {0: CTRL0}
for r in (1, 2):
    PCTRL[r] = compose(set(), r)
BL2 = {r: P(I + f"agent_stagecap_RS_E9S_s{r}.npz") + DNBR for r in (0, 1, 2)}  # blend2's CTRL_r (context only)
res["pctrl_vs_blend2ctrl_all"] = {r: rm(PCTRL[r], np.ones(N, bool)) - rm(BL2[r], np.ones(N, bool)) for r in (0, 1, 2)}
save()


def treat(comp, seed):
    if comp == "c2":
        p, _, kap = c2_oof(PCTRL[seed], y, apts, days, HOUR, DOW)
        res.setdefault("c2_kappas", {})[f"screen_s{seed}"] = kap
        return p
    return compose({comp}, seed)


# ================================================================ step 1: individual screens, RELAXED gate (s0.4)
res.setdefault("screen", {})
for comp in COMPONENTS:
    if res["screen"].get(comp, {}).get("done"):
        continue
    out = {}
    for s in (0, 1):
        p = treat(comp, s)
        c = clauses(p, PCTRL[s], f"PCTRL_{s}")
        show(f"SCREEN {CNAME[comp]} seed {s}", c)
        c["vs_blend2_ctrl_all"] = rm(p, np.ones(N, bool)) - rm(BL2[s], np.ones(N, bool))
        out[f"seed{s}"] = c
        save()
    out["IN"] = all(out[f"seed{s}"]["all"] <= 0.0 and out[f"seed{s}"]["worst_apt"] <= 0.10 for s in (0, 1))
    out["done"] = True
    res["screen"][comp] = out
    save()
    say(f"SCREEN {CNAME[comp]}: {'IN' if out['IN'] else 'out'} (all d seed0 {out['seed0']['all']:+.4f} / seed1 "
        f"{out['seed1']['all']:+.4f}; worst airport {out['seed0']['worst_apt']:+.3f} / {out['seed1']['worst_apt']:+.3f})")

IN = [c for c in COMPONENTS if res["screen"][c]["IN"]]
res["IN"] = IN
save()
say(f"STEP 1 IN: {IN or 'none'}")


# ================================================================ step 2: bundle, FULL rule (s0.5)
def judge_bundle(name, comps):
    toks = {c for c in comps if c != "c2"}
    use_c2 = "c2" in comps
    out, ok = {"components": comps}, True
    for s in (0, 1, 2):
        pre = I + f"agent_sb2_{name}_preC2_s0.npz" if s == 0 else None
        p = compose(toks, s, c2=use_c2, pre_out=pre) if toks else c2_oof(PCTRL[s], y, apts, days, HOUR, DOW)[0]
        if not toks and s == 0:
            np.savez(I + f"agent_sb2_{name}_preC2_s0.npz", ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"],
                     both_guard=PCTRL[0])
        c = clauses(p, PCTRL[s], f"PCTRL_{s}")
        show(f"{name} seed {s}", c)
        c["vs_blend2_ctrl_all"] = rm(p, np.ones(N, bool)) - rm(BL2[s], np.ones(N, bool))
        out[f"seed{s}"] = c
        ok = ok and c["pass"]
        if s == 0:
            np.savez(I + f"agent_sb2_{name}_s0.npz", ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=p)
            dp = p - CTRL0
            out["fold_dp2_per_row"] = float((dp ** 2).sum() / N)
            out["fold_dp2_by_apt"] = {a: float((dp[apts == a] ** 2).sum()) for a in AIRPORTS}
            out["fold_rows_by_apt"] = {a: int((apts == a).sum()) for a in AIRPORTS}
        save()
    alls = [out[f"seed{s}"]["all"] for s in (0, 1, 2)]
    out["spread_all"] = max(alls) - min(alls)
    out["pass"] = ok
    return out


final, final_comps = "FAIL", []
if IN:
    b = judge_bundle("bundle", IN)
    res["bundle"] = b
    save()
    say(f"BUNDLE {IN}: {'PASS' if b['pass'] else 'FAIL'}")
    if b["pass"]:
        final, final_comps = "PASS_BUNDLE", IN
    elif len(IN) >= 2:
        worst = max(IN, key=lambda c: (res["screen"][c]["seed0"]["all"], res["screen"][c]["seed1"]["all"]))
        fb = [c for c in IN if c != worst]
        say(f"FALLBACK (once): bundle without the worst IN component {worst} -> {fb}")
        f = judge_bundle("fallback", fb)
        f["dropped"] = worst
        res["fallback"] = f
        save()
        if f["pass"]:
            final, final_comps = "PASS_FALLBACK", fb
res["final"] = final
res["final_components"] = final_comps
save()
say(f"SB2 VERDICT: {final} {final_comps}")
print(f"FINAL={final}")
print(f"TOKENS={','.join(sorted(c for c in final_comps if c != 'c2'))}")
print(f"C2={1 if 'c2' in final_comps else 0}")
print(f"NAME={'bundle' if final == 'PASS_BUNDLE' else 'fallback' if final == 'PASS_FALLBACK' else ''}")
