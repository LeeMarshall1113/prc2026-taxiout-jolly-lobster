"""experiment smallbundle 05: PRE-REGISTERED judge (docs/LAB-NOTES.md). Clause code copied from
archive/agent_expensive_06_judge.py (same d, same four clauses). No splice fallback, no second selection.

Inputs (from archive/agent_smallbundle_03_fold.py): data/interim/agent_smallbundle_{A,A_lebl,B,B_lebl}.npz.
Writes data/interim/agent_smallbundle_verdict.json and data/interim/agent_smallbundle_bundle.npz (+ member npz for M3).
"""
import glob
import json
import sys

sys.path.insert(0, ".")
import numpy as np
import polars as pl

I = "data/interim/"
C = pl.col
ID = "MVT_ID_mvt"


def load(p):
    z = np.load(p, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    keys = ("ids", "y", "days", "apts", "both_guard") + (("both_guard_prers",) if "both_guard_prers" in z else ())
    return {k: z[k][o] for k in keys}


q = load(I + "agent_restack3_CAB.npz")
y, apts = q["y"].astype(float), q["apts"].astype(str)
mon = np.array([d[5:7] for d in q["days"].astype(str)])
H1, H2 = np.isin(mon, ["01", "02", "03"]), np.isin(mon, ["07", "08", "09"])
pq = q["both_guard"].astype(float)
fin = q["both_guard_prers"].astype(float)
s6 = load(I + "modal/TX_S6.npz")
assert (s6["ids"] == q["ids"]).all() and (s6["both_guard"].astype(float) == fin).all(), "CAB prers != TX_S6"
rm = lambda p, m: float(np.sqrt(((p[m] - y[m]) ** 2).mean()))  # noqa: E731
ALL = np.ones(len(y), bool)


def clauses(pt):
    d = lambda m: rm(pt, m) - rm(pq, m)  # noqa: E731
    gain = (pq - y) ** 2 - (pt - y) ** 2
    keep = ALL.copy()
    keep[np.argsort(-gain)[:50]] = False
    ch = pt != pq
    h = [d(m) if (ch & m).any() else 0.0 for m in (H1, H2)]
    per = {a: d(apts == a) for a in sorted(set(apts))}
    c = {"all": d(ALL), "worst_apt": max(per.values()), "m50": d(keep), "h1": h[0], "h2": h[1], "per": per,
         "rows_changed": int(ch.sum())}
    c["pass"] = c["all"] <= -0.10 and c["worst_apt"] <= 0.10 and c["m50"] < 0 and c["h1"] < 0 and c["h2"] < 0
    c["included"] = c["all"] <= 0.0 and c["worst_apt"] <= 0.10
    return c


def show(name, c):
    print(f"{name}: all {c['all']:+.4f}  worst airport {c['worst_apt']:+.3f}  minus-top-50 {c['m50']:+.3f}  "
          f"Jan-Mar {c['h1']:+.3f}  Jul-Sep {c['h2']:+.3f}  rows changed {c['rows_changed']:,}  "
          f"4-clause {'PASS' if c['pass'] else 'FAIL'}  inclusion {'IN' if c['included'] else 'OUT'}")
    print("   by airport: " + "  ".join(f"{a} {v:+.3f}" for a, v in c["per"].items()))


# M3 cell: LIRF, in the dump, plan, fin in [1200, 2400)
dump = pl.read_parquet(I + "agent_offblock_rows_c238.parquet", columns=[ID])[ID].cast(pl.Float64).to_numpy()
ind = np.isin(q["ids"].astype(float), dump)
plan = (pl.DataFrame({ID: q["ids"].astype(float)})
        .join(pl.concat([pl.scan_parquet(p).select(C(ID).cast(pl.Float64), C("AOBT_3_flt").is_not_null().alias("plan"))
                         for p in sorted(glob.glob("data/raw/training_2025-*.parquet"))]).collect().unique(subset=ID),
              on=ID, how="left", maintain_order="left")["plan"].fill_null(False).to_numpy())
cell = (apts == "LIRF") & ind & plan & (fin >= 1200) & (fin < 2400)
print(f"M3 cell rows: {int(cell.sum()):,}")


def revert(p):
    return np.where(cell, fin, p)


A, AL, B, BL = (load(I + f"agent_smallbundle_{k}.npz") for k in ("A", "A_lebl", "B", "B_lebl"))
for k in (A, AL, B, BL):
    assert (k["ids"] == q["ids"]).all()
ident = float(np.abs(A["both_guard"].astype(float) - pq).max())
print(f"IDENTITY arm A (cap,evt,trk via the copy) vs CAB: max |diff| {ident:.3g} -> {'OK' if ident == 0 else 'NOT BIT-EQUAL'}")
res = {"identity_max_abs_diff": ident, "m3_cell_rows": int(cell.sum())}
arms = {"M1_seq": B["both_guard"].astype(float), "M2_lebl": AL["both_guard"].astype(float), "M3_lirf": revert(pq)}
print("== standalone members vs CAB")
for k, p in arms.items():
    c = clauses(p)
    show(k, c)
    res[k] = c
inc = [k for k in arms if res[k]["included"]]
print("INCLUDED:", inc or "none")
res["included"] = inc


def bundle(members):
    base = {(False, False): A, (False, True): AL, (True, False): B, (True, True): BL}[("M1_seq" in members, "M2_lebl" in members)]
    p = base["both_guard"].astype(float)
    return revert(p) if "M3_lirf" in members else p


if not inc:
    print("VERDICT: empty bundle, nothing to build")
    res["verdict"] = "EMPTY"
else:
    pb = bundle(inc)
    c = clauses(pb)
    print("== BUNDLE " + "+".join(inc) + " vs CAB (THE pre-registered test)")
    show("BUNDLE", c)
    res["bundle"] = c
    res["verdict"] = "PASS" if c["pass"] else "FAIL"
    print(f"VERDICT: {res['verdict']}")
    if len(inc) > 1:
        print("== leave-one-out inside the bundle (reported, not gating): bundle minus member, all-rows d vs CAB")
        res["loo"] = {}
        for k in inc:
            rest = [m for m in inc if m != k]
            cr = clauses(bundle(rest))
            res["loo"][k] = {"without_all": cr["all"], "member_contribution": c["all"] - cr["all"]}
            print(f"  without {k}: all {cr['all']:+.4f}  -> {k} contributes {c['all'] - cr['all']:+.4f}")
    z = np.load(I + "agent_restack3_CAB.npz", allow_pickle=True)
    Z = {kk: z[kk] for kk in z.keys()}
    o = np.argsort(Z["ids"], kind="stable")
    bg = np.empty_like(pb)
    bg[o] = pb
    Z["both_guard"] = bg
    np.savez(I + "agent_smallbundle_bundle.npz", **Z)
    print("wrote", I + "agent_smallbundle_bundle.npz")
    # T=0 fallback build has no v105 path, so M2 cannot be served there: its fold analogue (reported, not a new test)
    p0 = bundle([k for k in inc if k != "M2_lebl"]) if [k for k in inc if k != "M2_lebl"] else pq
    bg0 = np.empty_like(p0)
    bg0[o] = p0
    Z["both_guard"] = bg0
    np.savez(I + "agent_smallbundle_bundle_T0.npz", **Z)
    c0 = clauses(p0)
    show("T0 composition (bundle without M2; reported only)", c0)
    res["bundle_T0_composition"] = c0
json.dump(res, open(I + "agent_smallbundle_verdict.json", "w"), indent=1)
