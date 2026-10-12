"""experiment sb2 judge (docs/LAB-NOTES.md + amendment s0.8): the judging half of the chain, run as a
separate process between fit stages so the orchestrator holds no fold arrays while fits run in parallel.
Clause code (rm / clauses / show) copied verbatim from archive/agent_restack6_chain.py; composition as
archive/agent_sb2_chain.py (s0.2). Reads/writes data/interim/agent_sb2_verdict.json.

    python archive/agent_sb2_judge.py repro|screen0|screen1|bundle|fallback
"""
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_sb2_calib import oof as c2_oof

I = "data/interim/"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
SBX = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LSZH"]
SQ = ["LFPG", "LIRF"]
M = ["EDDF", "EDDM", "EHAM", "LEBL", "LSZH"]
TX = I + "modal/TX_S6.npz"
COMPONENTS = ["mw", "c2", "lw", "unc", "osm", "occ", "nov"]  # C1, C2, C3, C4, C6, C7, C9 (C5 dropped, C8 parked)
CNAME = {"mw": "C1 month-mix weight", "c2": "C2 (apt,hour,dow) OOF calibration", "lw": "C3 soft LIRF weight",
         "unc": "C4 uncertainty width", "osm": "C6 OSM stand trust", "occ": "C7 ground occupancy", "nov": "C9 row novelty"}
VJ = I + "agent_sb2_verdict.json"
res = json.load(open(VJ)) if os.path.exists(VJ) else {}


def save():
    json.dump(res, open(VJ, "w"), indent=1, default=float)


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


def tokstr(toks):
    return "_".join(sorted(toks)) if toks else "none"


def fitfile(piece, toks, seed):
    """same naming as archive/agent_sb2_chain2.py; the file must already exist (fits run in the orchestrator's pool)."""
    if not toks and seed > 0:
        return {"bxTX": I + f"agent_restack5_bxn_cap_evt_trk_bx_big_s{seed}.npz",
                "bxE9f": I + f"agent_stagecap_bxn_E9f_s{seed}.npz",
                "sq": I + f"agent_restack5_f_cap_evt_trk_seq_big_s{seed}.npz"}[piece]
    out = I + f"agent_sb2_{piece}_{tokstr(toks)}_s{seed}.npz"
    assert os.path.exists(out), f"missing fit {out}"
    return out


DNBR = P(I + "agent_blend_oof_seedA.npz") - P(I + "agent_blend_v109analogue.npz")
CTRL0 = P(I + "agent_stagecap_RS_E9S_s0.npz") + DNBR
TXP = P(TX)
hd = (pl.DataFrame({"MVT_ID_mvt": c0["ids"].astype(float)})
      .join(pl.read_parquet(I + "agent_sb2_hdow_2025.parquet"), on="MVT_ID_mvt", how="left", maintain_order="left"))
HOUR = hd["hour"].fill_null(-1).to_numpy().astype(int)
DOW = hd["dow"].fill_null(-1).to_numpy().astype(int)
del hd


def compose(toks, seed, c2=False, pre_out=None):
    bt = set(toks) - {"lw"}
    bx_tx, bx_e9, sq = P(fitfile("bxTX", bt, seed)), P(fitfile("bxE9f", bt, seed)), P(fitfile("sq", set(toks), seed))
    p = sp(sp(sp(TXP, bx_tx, SBX), sq, SQ), bx_e9, M) + DNBR
    if pre_out:
        np.savez(pre_out, ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=p)
    if c2:
        p, _, kap = c2_oof(p, y, apts, days, HOUR, DOW)
        res.setdefault("c2_kappas", {})[f"{tokstr(toks)}_s{seed}"] = kap
    return p


def pctrl(r):
    return CTRL0 if r == 0 else compose(set(), r)


def treat(comp, seed):
    if comp == "c2":
        p, _, kap = c2_oof(pctrl(seed), y, apts, days, HOUR, DOW)
        res.setdefault("c2_kappas", {})[f"screen_s{seed}"] = kap
        return p
    return compose({comp}, seed)


def screen(comp, s):
    p = treat(comp, s)
    c = clauses(p, pctrl(s), f"PCTRL_{s}")
    show(f"SCREEN {CNAME[comp]} seed {s}", c)
    bl2 = P(I + f"agent_stagecap_RS_E9S_s{s}.npz") + DNBR
    c["vs_blend2_ctrl_all"] = rm(p, np.ones(N, bool)) - rm(bl2, np.ones(N, bool))
    return c


def nov_sanity():
    """amendment s0.8: C9 is dropped before its screen if its columns fail these label-free checks."""
    f = pl.read_parquet(I + "agent_nov_fold.parquet")
    d25 = pl.read_parquet(I + "agent_offblock_rows_c238.parquet", columns=["MVT_ID_mvt", "apt"]).with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Float64))
    d26 = pl.read_parquet(I + "agent_offblock_rows_2026.parquet", columns=["MVT_ID_mvt", "apt"]).with_columns(
        pl.col("MVT_ID_mvt").cast(pl.Float64))
    j25 = d25.join(f.with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)), on="MVT_ID_mvt", how="left")
    j26 = d26.join(pl.read_parquet(I + "agent_nov_2026.parquet").with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64)),
                   on="MVT_ID_mvt", how="left")
    full = j25["nov_full"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    same_null = min(float((np.isfinite(j25[f"nov_f{k}"].cast(pl.Float64).fill_null(np.nan).to_numpy()) == np.isfinite(full)).mean())
                    for k in range(10))
    v26 = j26["nov"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    out = {"fold_dump_coverage": float(np.isfinite(full).mean()), "y2026_dump_coverage": float(np.isfinite(v26).mean()),
           "min_null_pattern_agreement_fk_vs_full": same_null,
           "fold_median": float(np.nanmedian(full)), "y2026_median": float(np.nanmedian(v26)), "per_airport": {}}
    a25, a26 = np.asarray(j25["apt"].to_list()), np.asarray(j26["apt"].to_list())
    for a in AIRPORTS:
        x, z = full[a25 == a], v26[a26 == a]
        out["per_airport"][a] = {"fold_cov": float(np.isfinite(x).mean()), "y26_cov": float(np.isfinite(z).mean()),
                                 "fold_med": float(np.nanmedian(x)) if np.isfinite(x).any() else None,
                                 "y26_med": float(np.nanmedian(z)) if np.isfinite(z).any() else None}
    r = out["y2026_median"] / out["fold_median"]
    out["ok"] = (out["fold_dump_coverage"] >= 0.5 and out["y2026_dump_coverage"] >= 0.5 and same_null >= 0.99
                 and 0.5 <= r <= 2.0)
    print("NOVELTY SANITY: " + json.dumps({k: v for k, v in out.items() if k != "per_airport"}, default=float), flush=True)
    for a, v in out["per_airport"].items():
        print(f"   {a}: fold cov {v['fold_cov']:.3f} med {v['fold_med']}  | 2026 cov {v['y26_cov']:.3f} med {v['y26_med']}", flush=True)
    return out


def c4_c9_corr():
    uz = np.load(I + "agent_sb2_unc_TX.npz")
    uid = uz["ids"]
    r = pl.read_parquet(I + "agent_offblock_rows_c238.parquet", columns=["MVT_ID_mvt", "day"])
    assert (r["MVT_ID_mvt"].cast(pl.Float64).to_numpy() == uid).all()
    dd = np.asarray(r["day"].to_list())
    ud = sorted(set(dd))
    fo = np.array([{d: i % 10 for i, d in enumerate(ud)}[d] for d in dd])
    from archive.agent_nov_lib import nov_folds
    nv = nov_folds(I + "agent_nov_fold.parquet", uid)
    u, n = np.full(len(uid), np.nan), np.full(len(uid), np.nan)
    for k in range(10):
        u[fo == k], n[fo == k] = uz[f"w{k}"][fo == k], nv[k][fo == k]
    m = np.isfinite(u) & np.isfinite(n)
    from scipy.stats import spearmanr
    out = {"n": int(m.sum()), "pearson": float(np.corrcoef(u[m], n[m])[0, 1]), "spearman": float(spearmanr(u[m], n[m])[0])}
    print(f"CORR C4 (unc_w) vs C9 (nov), test-fold values on {out['n']:,} dump rows: Pearson {out['pearson']:+.3f}, "
          f"Spearman {out['spearman']:+.3f}", flush=True)
    return out


def judge_bundle(name, comps):
    toks = {c for c in comps if c != "c2"}
    use_c2 = "c2" in comps
    out, ok = {"components": comps}, True
    for s in (0, 1, 2):
        pre = I + f"agent_sb2_{name}_preC2_s0.npz" if s == 0 else None
        if toks:
            p = compose(toks, s, c2=use_c2, pre_out=pre)
        else:
            p = c2_oof(pctrl(s), y, apts, days, HOUR, DOW)[0]
            if s == 0:
                np.savez(pre, ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=CTRL0)
        c = clauses(p, pctrl(s), f"PCTRL_{s}")
        show(f"{name} seed {s}", c)
        bl2 = P(I + f"agent_stagecap_RS_E9S_s{s}.npz") + DNBR
        c["vs_blend2_ctrl_all"] = rm(p, np.ones(N, bool)) - rm(bl2, np.ones(N, bool))
        out[f"seed{s}"] = c
        ok = ok and c["pass"]
        if s == 0:
            np.savez(I + f"agent_sb2_{name}_s0.npz", ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=p)
            dp = p - CTRL0
            out["fold_dp2_per_row"] = float((dp ** 2).sum() / N)
            out["fold_dp2_by_apt"] = {a: float((dp[apts == a] ** 2).sum()) for a in AIRPORTS}
            out["fold_rows_by_apt"] = {a: int((apts == a).sum()) for a in AIRPORTS}
    alls = [out[f"seed{s}"]["all"] for s in (0, 1, 2)]
    out["spread_all"] = max(alls) - min(alls)
    out["pass"] = ok
    return out


stage = sys.argv[1]
if stage == "repro":
    ref = {"bxTX": I + "agent_restack5_bxn_cap_evt_trk_bx_big_s0.npz", "bxE9f": I + "agent_stagecap_bxn_E9f_s0.npz",
           "sq": I + "agent_restack5_f_cap_evt_trk_seq_big_s0.npz"}
    rep = {pc: float(np.abs(P(fitfile(pc, set(), 0)) - P(rp)).max()) for pc, rp in ref.items()}
    for pc, v in rep.items():
        print(f"REPRO {pc}: sb2 no-token seed-0 fit vs {ref[pc]}: max |diff| {v}", flush=True)
    v9 = sp(sp(TXP, P(ref["bxTX"]), SBX), P(ref["sq"]), SQ)
    rep["V9TOP1_vs_composition"] = float(np.abs(P(I + "agent_restack5_V9TOP1.npz") - v9).max())
    rep["CTRL0_vs_sb2_composition"] = float(np.abs(compose(set(), 0) - CTRL0).max())
    print(f"REPRO V9TOP1 == composition: {rep['V9TOP1_vs_composition']}; CTRL_0 == sb2 composition + Dnbr: "
          f"{rep['CTRL0_vs_sb2_composition']}", flush=True)
    res["repro"] = rep
    res["repro_ok"] = all(v == 0.0 for v in rep.values())
    np.savez(I + "agent_sb2_ctrl_r0.npz", ids=c0["ids"], y=c0["y"], days=c0["days"], apts=c0["apts"], both_guard=CTRL0)
    res["hdow_coverage"] = float((HOUR >= 0).mean())
    res["nov_sanity"] = nov_sanity()
    save()
    print(f"REPRO_OK={int(res['repro_ok'])}  NOV_OK={int(res['nov_sanity']['ok'])}", flush=True)
elif stage == "screen0":
    res["c4_c9_corr"] = c4_c9_corr()
    comps = [c for c in COMPONENTS if c != "nov" or res["nov_sanity"]["ok"]]
    res.setdefault("screen", {})
    for comp in comps:
        c = screen(comp, 0)
        o = {"seed0": c, "survives0": c["all"] <= 0.0 and c["worst_apt"] <= 0.10}
        res["screen"][comp] = o
        save()
        print(f"SCREEN0 {CNAME[comp]}: {'survives' if o['survives0'] else 'DROPPED at seed 0'}", flush=True)
    res["survivors0"] = [c for c in comps if res["screen"][c]["survives0"]]
    save()
    print("SURVIVORS0=" + ",".join(res["survivors0"]), flush=True)
elif stage == "screen1":
    for comp in res["survivors0"]:
        c = screen(comp, 1)
        o = res["screen"][comp]
        o["seed1"] = c
        o["IN"] = c["all"] <= 0.0 and c["worst_apt"] <= 0.10
        save()
        print(f"SCREEN1 {CNAME[comp]}: {'IN' if o['IN'] else 'out at seed 1'}", flush=True)
    res["IN"] = [c for c in res["survivors0"] if res["screen"][c]["IN"]]
    if len(res["IN"]) >= 2:
        res["worst_IN"] = max(res["IN"], key=lambda c: (res["screen"][c]["seed0"]["all"], res["screen"][c]["seed1"]["all"]))
    save()
    print("IN=" + ",".join(res["IN"]), flush=True)
elif stage in ("bundle", "fallback"):
    comps = res["IN"] if stage == "bundle" else [c for c in res["IN"] if c != res["worst_IN"]]
    b = judge_bundle(stage, comps)
    if stage == "fallback":
        b["dropped"] = res["worst_IN"]
    res[stage] = b
    save()
    print(f"{stage.upper()}_PASS={int(b['pass'])}", flush=True)
