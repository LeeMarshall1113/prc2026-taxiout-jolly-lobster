"""mega step 3: ingredient 4, per-airport mix of the refitted corrector (docs/LAB-NOTES.md sections 8-9).
Refit serving term (mean of the six serve fits in data/interim/bigc13_serve/C4ff7V5_s*.npz) at seven airports with the doses fitted on ALL eight old months
(agent_refit2_10_full.py, 'all-8 SSE-best'; log logs/agent_refit2_10_full.log): EDDF 1.25, EDDM 1.25, EGLL 1.0, LEBL 1.0, LEMD 1.0, LIRF 0.75, LSZH 1.25.
EHAM and LFPG keep the shipped term (kF_old, shipped doses 1.25 / 0.75: identical to what the log's 'ship' arm uses).
delta dR = kF_new - kF_old on the nine-airport rows that no later rule set (agent_refit2_06_assemble.py's nr = is9 & ~RR); zero elsewhere (336 rule rows, Istanbul, all other airports).
Guards (any failure -> no delta file, ingredient left out): corr(refit term, shipped term) >= 0.7 at each of the seven airports, |mean shift| <= 5 s per airport, |row move| <= 600 s.
Writes data/interim/agent_pcd_refit12_delta.npz when all pass."""
import glob, sys
import json
import numpy as np
import polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DOSE = {"EDDF": 1.25, "EDDM": 1.25, "EGLL": 1.0, "LEBL": 1.0, "LEMD": 1.0, "LIRF": 0.75, "LSZH": 1.25}
SHIP_KEEP = ["EHAM", "LFPG"]
sb = np.load(I + "agent_bigc13_serve_base.npz", allow_pickle=True); si = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True)
base = pl.read_parquet(I + "agent_basefix_final.parquet").sort(ID); ids = base[ID].to_numpy(); cur = base[T].cast(pl.Float64).to_numpy()
assert (sb["ids"] == ids).all() and (si["ids"] == ids).all()
apt = si["apt"].astype(str); is9, RR, B26, ehat, kold = sb["is9"], sb["RR"], sb["B26"], sb["ehat"], sb["kF_old"]
fs = sorted(glob.glob(I + (sys.argv[1] if len(sys.argv) > 1 else "bigc13_serve") + "/C4ff7V5_s*.npz")); assert len(fs) == 12, fs
zs = [np.load(f, allow_pickle=True) for f in fs]
print("serve fits:", [f.split("/")[-1] for f in fs], "rounds", [int(z["rounds"]) for z in zs])
P = np.mean([z["pred"] for z in zs], 0); assert P.shape == ids.shape and np.isfinite(P).all()
ship = json.load(open(I + "agent_m12_kapt_C4ff7V5.json")); assert ship["EHAM"] == 1.25 and ship["LFPG"] == 0.75
kap = np.array([DOSE.get(a, 0.0) for a in apt]); knew = kap * P; knew[~is9] = 0
keep = np.isin(apt, SHIP_KEEP); knew[keep] = kold[keep]
nr = is9 & ~RR
dR = np.where(nr, knew - kold, 0.0)
ok = True
print("\nairport | rows(nr) | dose | corr(refit term, shipped term) | corr(raw, shipped raw) | mean shift s | rms move s | max |move| s")
for a in sorted(DOSE) + SHIP_KEEP:
    m = nr & (apt == a)
    if a in SHIP_KEEP:
        print(f"{a} | {int(m.sum())} | shipped | (kept: delta {np.abs(dR[m]).max():.1f}) |"); continue
    c1 = float(np.corrcoef(knew[m], kold[m])[0, 1]); c2 = float(np.corrcoef(P[m], kold[m])[0, 1])
    d = dR[m]; row = f"{a} | {int(m.sum())} | {DOSE[a]} | {c1:.3f} | {c2:.3f} | {d.mean():+.2f} | {np.sqrt((d**2).mean()):.2f} | {np.abs(d).max():.1f}"
    bad = c1 < 0.7 or abs(d.mean()) > 5; ok &= not bad; print(row, "  <-- GUARD FAIL" if bad else "")
mx = np.abs(dR).max(); print("max |row move| over all rows %.1f s; rows with nonzero delta %d; later-rule rows touched %d; non-nine-airport rows touched %d" % (mx, int((dR != 0).sum()), int(((dR != 0) & RR).sum()), int(((dR != 0) & ~is9).sum())))
ok &= mx <= 600
# cross-check against the experiment's assembly formula
asm = np.rint(np.maximum(B26 + knew + 0.5 * ehat, 0)); oa = cur.copy(); oa[nr] = asm[nr]
mine = np.rint(np.maximum(cur + dR, 0))
print("delta-on-base vs experiment assembly formula (new term, rule rows held): max |difference| %.0f s, rows differing %d (rounding of the base only)" % (np.abs(mine - oa).max(), int((mine != oa).sum())))
if ok:
    np.savez(I + (sys.argv[2] if len(sys.argv) > 2 else "agent_pcd_refit12_delta.npz"), ids=ids, dR=dR); print("ALL GUARDS PASS: delta file written")
else:
    print("GUARD FAILED: ingredient 4 LEFT OUT (no delta file written)")
