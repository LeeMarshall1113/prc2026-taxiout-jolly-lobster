"""refit2 step 6: serve-side assembly. Current board file (agent_basefix_final.parquet) with the OLD corrector term swapped for the NEW one, then the later rules.
    python archive/agent_refit2_06_assemble.py selftest                                   # new term := old term; must reproduce the current file on every row
    python archive/agent_refit2_06_assemble.py build [SERVE_DIR=bigc13_serve] [KAPT=agent_m12_kapt_C4ff7V5.json] [OUT=agent_refit2_final.parquet] [MODE=freeze|prop]
Chain per nine-airport row (the fold chain of agent_basefix_lib.chain, in served order):  x = B26 + kF_new   (B26 = lv4r - D - kF_old, agent_refit2_04_serve.py: stage + rules + M + silent + meta, no blend)
   out = rint(max(x + 0.5 ehat, 0))                                   tail step
   rows set by a LATER rule (v138 != v132: Rome no-plan mixture / hidden-plan readings / lever B / duplicates / caps / long-gap step, 336 rows): MODE freeze keeps the current value unchanged
   (the corrector change is NOT propagated onto them; audit table C: corrector off on LIRF no-plan rows is neutral, Jan +0.038 / Jul -0.040, P 0.56). MODE prop adds (1 - w) x delta on the rows with
   3600 <= gap to schedule < 14400 (w = 0.5 below 7200, 0.85 above: the mixture weights, archive/agent_eb136_01_build.py) and keeps the fixed-value rows: UNVERIFIED at serve, fold-checked only.
Rows outside the nine airports (Istanbul) are never touched.  Guards printed: rows moved, mean shift, rms move, 2026 sum dp^2/row of (new - old term) against the fold's (needs FOLD_NPZ when given)."""
import glob
import json
import subprocess
import sys
import numpy as np
import polars as pl

I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
sd = sys.argv[2] if len(sys.argv) > 2 else "bigc13_serve"
kap_f = sys.argv[3] if len(sys.argv) > 3 else "agent_m12_kapt_C4ff7V5.json"
out_f = sys.argv[4] if len(sys.argv) > 4 else "agent_refit2_final.parquet"
mode = sys.argv[5] if len(sys.argv) > 5 else "freeze"
sb = np.load(I + "agent_bigc13_serve_base.npz", allow_pickle=True)
bf = pl.read_parquet(I + "agent_basefix_final.parquet").sort(ID); ids = bf[ID].to_numpy(); assert (ids == sb["ids"]).all()
cur = bf[T].cast(pl.Float64).to_numpy()
si = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True); apt = si["apt"].astype(str); assert (si["ids"] == ids).all()
is9, RR, B26, ehat, kF_old = sb["is9"], sb["RR"], sb["B26"], sb["ehat"], sb["kF_old"]
if cmd == "selftest":
    kF_new = kF_old.copy()
else:
    fs = sorted(glob.glob(I + f"{sd}/C4ff7V5_s*.npz")); assert fs, "no serve fits in data/interim/" + sd
    kap = json.load(open(I + kap_f)); kF_new = np.array([kap.get(a, 0.0) for a in apt]) * np.mean([np.load(f, allow_pickle=True)["pred"] for f in fs], 0); kF_new[~is9] = 0
    print("serve fits", len(fs), "rounds", [int(np.load(f, allow_pickle=True)["rounds"]) for f in fs])
x = B26 + kF_new
new = np.rint(np.maximum(x + 0.5 * ehat, 0))
out = cur.copy()
nr = is9 & ~RR
out[nr] = new[nr]
if mode == "prop":
    rk = pl.read_parquet("data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
    gs = dict(zip(rk[ID].to_list(), (rk["MVT_TIME_UTC_mvt"] - rk["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_list()))
    gg = np.array([gs.get(v, np.nan) for v in ids]); mid = RR & is9 & (gg >= 3600) & (gg < 14400); w = np.where(gg < 7200, 0.5, 0.85)
    out[mid] = np.rint(np.maximum(cur[mid] + (1 - w[mid]) * (kF_new - kF_old)[mid], 0))
    print("prop mode: mixture-gap rows among the later-rule rows:", int(mid.sum()))
if cmd == "selftest":
    d = out != cur
    print("SELFTEST new term := old term: rows differing from the current file:", int(d.sum()), "of", len(out), "| nine-airport rule-free rows rebuilt from pieces:", int(nr.sum()))
    assert not d.any(), "reassembly of the current file is not exact"
    print("SELFTEST OK: current file reassembled exactly from lv4r - D - kF_old + kF_old + 0.5 ehat, rule rows held")
    sys.exit(0)
bf.with_columns(pl.Series(T, out).cast(bf.schema[T])).write_parquet(I + out_f)
dlt = out - cur; mv = dlt != 0
print(f"rows moved {int(mv.sum()):,} (nine airports {int((mv & is9).sum()):,}, other airports {int((mv & ~is9).sum())}); mean shift {dlt.mean():+.3f}; rms move (all rows) {np.sqrt((dlt**2).mean()):.2f}; "
      f"rms (moved) {np.sqrt((dlt[mv]**2).mean()):.2f}; max {np.abs(dlt).max():.0f}; later-rule rows moved {int((mv & RR).sum())}")
dn = (kF_new - kF_old)[is9]
print(f"new-minus-old term: rms {np.sqrt((dn**2).mean()):.2f} s on nine airports; sum dp^2/row (all rows) {float((dlt**2).mean()):.2f}; corr(new term, old term) {np.corrcoef(kF_new[is9], kF_old[is9])[0, 1]:.3f}")
r = subprocess.run([sys.executable, "archive/agent_smallbundle_06_check.py", I + out_f, I + "agent_basefix_final.parquet"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                   env=dict(__import__("os").environ, PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage"))
print("\n".join(l for l in r.stdout.splitlines() if any(k in l for k in ("FAIL", "moved", "ALL", "shift", "corr"))))
