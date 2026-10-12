"""arith step 13: build data/interim/agent_arith_final.parquet = v146 + (candidate term - term served now) at the airports chosen on 2025 only (data/interim/agent_arith_airports.json from agent_arith_12_select.py).
Term served now: pcd (12 serve fits data/interim/agent_pcd_modal/serve/C4ff7V5_s0..s11) x dose at the seven refit airports and EHAM (1.25), the old term kF_old at LFPG (dose in kF_old). Candidate term: same doses x mean of the 12 serve fits
data/interim/agent_arith_serve/ABCDE-pw-s0-r0_s0..s11 ('pred'). Rows touched: nr = is9 & ~RR (agent_bigc13_serve_base) minus every row that v145, v146, v147 or v148 moved vs its parent (kept exactly as in v146).
Guards (any failure: no file): corr(new term, current term) >= 0.7 per changed airport, |mean shift| <= 5 s, |row move| <= 600 s, selftest delta=0 reproduces v146.
usage: python agent_arith_13_build.py [selftest|build]; then agent_smallbundle_06_check.py agent_arith_final.parquet agent_bundle_v146_final.parquet runs automatically."""
import glob, json, os, subprocess, sys
import numpy as np
import polars as pl
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; ARM = "ABCDE-pw-s0-r0"
cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
DOSE = {"EDDF": 1.25, "EDDM": 1.25, "EGLL": 1.0, "LEBL": 1.0, "LEMD": 1.0, "LIRF": 0.75, "LSZH": 1.25, "EHAM": 1.25}
sb = np.load(I + "agent_bigc13_serve_base.npz", allow_pickle=True); si = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True)
rd = lambda n: pl.read_parquet(I + n + ".parquet").sort(ID)   # noqa: E731
v146 = rd("agent_bundle_v146_final"); ids = v146[ID].to_numpy(); cur = v146[T].cast(pl.Float64).to_numpy()
assert (sb["ids"] == ids).all() and (si["ids"] == ids).all()
apt = si["apt"].astype(str); is9, RR, kold = sb["is9"], sb["RR"], sb["kF_old"]
nr = is9 & ~RR
moved_later = np.zeros(len(ids), bool)
for a, b in (("v144", "v145"), ("v145", "v146"), ("v146", "v147"), ("v146", "v148")):
    moved_later |= rd("agent_bundle_%s_final" % b)[T].cast(pl.Float64).to_numpy() != rd("agent_bundle_%s_final" % a)[T].cast(pl.Float64).to_numpy()
print("rows moved by v145/v146/v147/v148 vs parent (excluded from the swap):", int(moved_later.sum()))
v148 = rd("agent_bundle_v148_final")[T].cast(pl.Float64).to_numpy(); v147 = rd("agent_bundle_v147_final")[T].cast(pl.Float64).to_numpy()
ch = json.load(open(I + "agent_arith_airports.json")); print("airport choice", ch)
fp = sorted(glob.glob(I + "agent_pcd_modal/serve/C4ff7V5_s*.npz")); assert len(fp) == 12, fp
fa = sorted(glob.glob(I + "agent_arith_serve/%s_s*.npz" % ARM)); assert cmd == "selftest" or len(fa) == 12, fa
Pp = np.mean([np.load(f, allow_pickle=True)["pred"] for f in fp], 0)
Pa = Pp if cmd == "selftest" else np.mean([np.load(f, allow_pickle=True)["pred"] for f in fa], 0)
assert Pp.shape == ids.shape and np.isfinite(Pa).all()
kap = np.array([DOSE.get(a, 0.0) for a in apt])
term_now = np.where(apt == "LFPG", kold, kap * Pp)
term_new = np.where(apt == "LFPG", 0.75 * Pa, kap * Pa)
if cmd == "selftest":
    term_new = term_now.copy()
delta = np.zeros(len(ids)); ok = True
print("airport | rows | corr(new, now) | mean shift s | rms move s | max |move| s")
for a in sorted(DOSE) + ["LFPG"]:
    if not ch.get(a, False):
        continue
    m = nr & (apt == a) & ~moved_later; d = (term_new - term_now)[m]; delta[m] = d
    c = float(np.corrcoef(term_new[m], term_now[m])[0, 1]); bad = (c < 0.7 or abs(d.mean()) > 5) and cmd != "selftest"; ok &= not bad
    print("%s | %d | %.3f | %+.2f | %.2f | %.1f%s" % (a, m.sum(), c, d.mean(), np.sqrt((d ** 2).mean()), np.abs(d).max(), "  <-- GUARD FAIL" if bad else ""))
ok &= np.abs(delta).max() <= 600
if cmd == "selftest":
    assert (delta == 0).all(); print("SELFTEST OK (delta 0)"); sys.exit(0)
if not ok:
    print("GUARD FAILED: no file written"); sys.exit(1)
out = np.where(delta != 0, np.rint(np.maximum(cur + delta, 0)), cur)
v146.with_columns(pl.Series(T, out).cast(v146.schema[T])).write_parquet(I + "agent_arith_final.parquet")
dl = out - cur; mv = dl != 0
print("wrote agent_arith_final.parquet: rows moved %d; mean shift %+.3f; rms (moved) %.2f; max %.0f; RR rows moved %d; non-nine rows moved %d" % (mv.sum(), dl.mean(), np.sqrt((dl[mv] ** 2).mean()), np.abs(dl).max(), (mv & RR).sum(), (mv & ~is9).sum()))
print("overlap with v148's moved rows: %d of %d; with v147's: %d" % ((mv & (v148 != cur)).sum(), (v148 != cur).sum(), (mv & (v147 != cur)).sum()))
r = subprocess.run([sys.executable, "archive/agent_smallbundle_06_check.py", I + "agent_arith_final.parquet", I + "agent_bundle_v146_final.parquet"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                   env=dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage"))
print(r.stdout[-2500:]); print(r.stderr[-500:])
